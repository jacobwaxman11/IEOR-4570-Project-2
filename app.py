import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path

import uvicorn
from agents import (
    Agent,
    ModelSettings,
    Runner,
    RunHooks,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel
from agents.run_config import ModelInputData, RunConfig
from agents.run_state import RunState
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from present import (
    PREFETCH_MAPPINGS_CALL_ID,
    STATIC_DIR,
    add_favorite,
    cards_for,
    init_ui,
    list_favorites,
    list_threads,
    load_thread,
    remove_favorite,
    resolve_display,
)
import accounts
import observe
from rental_context import CURRENT, RentalContext, from_dict, to_dict
from scripts import bq
from session_store import claim_session, open_session, owns_session
from tools import (
    ANALYSIS_TOOLS,
    PLACES_TOOLS,
    get_enum_values,
    load_mappings,
    read_mappings,
    run_sql,
    start_read_mappings,
)

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# --- The Agent ---

_INSTRUCTIONS = """\
You are a data analyst. Answer questions about NYC rental listings.
The data is in a SQLite database. `listings` has one row per StreetEasy listing.
Related tables join on the keys below.

Tables:
- listings: one row per StreetEasy listing.
- listing_amenities: amenity tags. Join to listings on listing_id.
- listing_images: photo references. Join to listings on listing_id.
- image_scores: per-image scores. Join to listing_images on image_url.
- listing_scores: aggregated photo scores. Join to listings on listing_id.
- regression_results: predicted rent, actual rent, percent discount, and deal score for active listings. Join to listings on listing_id.

Which tool to use:
- Filtered search by neighborhood, beds, baths, sqft, rent, or amenities: search_listings.
- Cheapest units in a neighborhood: cheapest_in_neighborhood.
- Amenities for a building: get_building_amenities.
- Model deals: get_deals. Best deals in one neighborhood: best_deals_in_neighborhood.
- Compare rents across neighborhoods: compare_neighborhoods.
- A short profile of one neighborhood: summarize_neighborhood.
- A listing's location, or any other place question: search_places. Pass the listing street and neighborhood. Do not locate a unit with street LIKE or another fuzzy address match.
- Anything else: read_mappings, then get_enum_values before filtering an enum, then run_sql.

- When writing SQL, call read_mappings first.
- Before filtering on an enum column in SQL, call get_enum_values. Never guess values.
- Area names such as Upper East Side are not stored on listings.neighborhood. area_aliases in the column catalog lists each name and the neighborhoods to filter on.
- When run_sql returns listing rows, stop. Do not call run_sql again on that question. Answer from those rows. Select every column the answer needs in that one query. A location question calls search_places after that query, for the listings being answered, and does not run another SQL query.
- Missing values are NULL. Exclude them from filters, groups, and aggregates. A missing sqft should not hide a listing from a size filter.
- Return active listings unless the user asks about another status. Add `listing_status = 'ACTIVE'` to every query that returns or aggregates listings. Drop that filter only when the user asks about rented, delisted, in-contract, off-market, or another status.
- Quote an apostrophe in a SQL string as two apostrophes, as in 'Hell''s Kitchen'.
- Answer in plain English. Don't show the SQL unless asked.

Example queries:

-- Filtered search. A missing sqft does not hide the listing.
SELECT listing_id, name, unit, neighborhood, beds, baths, sqft, rent
FROM listings
WHERE listing_status = 'ACTIVE'
  AND neighborhood = 'Chelsea'
  AND beds >= 1
  AND baths >= 1
  AND (sqft IS NULL OR sqft >= 500)
  AND rent <= 6000
ORDER BY rent
LIMIT 20;

-- Amenities for one building, counted on active listings
SELECT a.amenity, COUNT(DISTINCT l.listing_id) AS units
FROM listings l
JOIN listing_amenities a ON l.listing_id = a.listing_id
WHERE l.listing_status = 'ACTIVE'
  AND (l.name = 'The Building' OR l.building_slug = 'the-building')
GROUP BY a.amenity
ORDER BY units DESC, a.amenity;

-- Model deals. percent_discount is 100 * (predicted - actual) / predicted.
SELECT l.listing_id, l.name, l.neighborhood, l.beds, l.rent,
       r.predicted_rent, r.percent_discount, r.deal_score
FROM listings l
JOIN regression_results r ON l.listing_id = r.listing_id
WHERE l.listing_status = 'ACTIVE'
  AND l.neighborhood = 'Hell''s Kitchen'
ORDER BY r.deal_score DESC, r.percent_discount DESC
LIMIT 10;

-- Neighborhood price comparison
SELECT neighborhood,
       COUNT(*) AS listings,
       ROUND(AVG(rent)) AS avg_rent,
       MIN(rent) AS min_rent,
       MAX(rent) AS max_rent,
       ROUND(AVG(CASE WHEN sqft > 0 THEN 1.0 * rent / sqft END), 2) AS avg_rent_per_sqft
FROM listings
WHERE listing_status = 'ACTIVE'
  AND neighborhood IN ('Chelsea', 'Hell''s Kitchen')
  AND rent IS NOT NULL
GROUP BY neighborhood
ORDER BY avg_rent;

Profile, charts, and places:
- When the user states a budget, a bedroom minimum, neighborhoods, or a work address, call save_profile. It does not save until the user approves.
- "My budget" and "to work" use the renter profile when one is present.
- Average rent by neighborhood: average_rent_by_neighborhood.
- Average rent by apartment type (studio, 1-bedroom, 2-bedroom, 3+): average_rent_by_unit_type.
- Share of listings by neighborhood or by apartment type: listing_mix.
- A neighborhood price map: neighborhood_price_map.
- Commute time from listings already in hand to an address or to work: commute_from_listing.
- Things to do in a neighborhood: neighborhood_activities.
- Show one unit on a map: show_listing_map.
- get_directions is only for two addresses when you do not have a listing id.
"""

if bq.prod():
    _INSTRUCTIONS = _INSTRUCTIONS.replace(
        "The data is in a SQLite database.",
        "The data is in BigQuery Standard SQL. Do not double-quote identifiers. "
        "Quote an identifier that contains an apostrophe or a space with backticks.",
        1,
    )

agent = Agent(
    name="NYC Rental Agent",
    instructions=_INSTRUCTIONS,
    model=LitellmModel(model="vertex_ai/gemini-3.8-flash", api_key="unused"),
    model_settings=ModelSettings(extra_args={"vertex_location": "global"}),
    tools=[read_mappings, get_enum_values, run_sql, *ANALYSIS_TOOLS, *PLACES_TOOLS],
)


def _tool_args(context) -> dict:
    raw = getattr(context, "tool_arguments", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _with_catalog(instructions: str, catalog: str) -> str:
    """Hand the model the columns so a new thread does not spend a turn loading them."""
    text = instructions.replace(
        "Anything else: read_mappings, then get_enum_values before filtering an enum, then run_sql.",
        "Anything else: use the column catalog below. Call get_enum_values before filtering "
        "an enum whose values are not in that catalog, then run_sql.",
        1,
    ).replace(
        "When writing SQL, call read_mappings first.",
        "When writing SQL, use the column catalog below. Do not call read_mappings.",
        1,
    )
    return (
        text
        + "\n\nColumn catalog. `columns` lists table, name, type, description, is_enum, "
        + "and enum_values when the enum is small. `area_aliases` maps an area name to the "
        + "neighborhood values stored on listings. Do not call read_mappings.\n"
        + catalog
    )


def _instructions(ctx, _agent) -> str:
    """Catalog plus the signed-in profile. Both are already loaded, so the model does not fetch them."""
    text = _with_catalog(_INSTRUCTIONS, load_mappings())
    rental = getattr(ctx, "context", None)
    block = accounts.profile_text(getattr(rental, "profile", None))
    if block:
        text += "\n\n" + block
    return text


agent.instructions = _instructions


def _prefetch_items(catalog: str) -> list[dict]:
    return [
        {
            "type": "function_call",
            "name": "read_mappings",
            "arguments": "{}",
            "call_id": PREFETCH_MAPPINGS_CALL_ID,
        },
        {
            "type": "function_call_output",
            "call_id": PREFETCH_MAPPINGS_CALL_ID,
            "output": catalog,
        },
    ]


def _call_id(item) -> str | None:
    if isinstance(item, dict):
        return item.get("call_id")
    return getattr(item, "call_id", None)


def _has_real_mappings(items) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == "read_mappings":
            if item.get("call_id") != PREFETCH_MAPPINGS_CALL_ID:
                return True
    return False


def _has_prefetch(items) -> bool:
    return any(_call_id(item) == PREFETCH_MAPPINGS_CALL_ID for item in items)


def _short_tool(name: str, output: str) -> str:
    try:
        parsed = json.loads(output) if output else None
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
        return f"{name}: {len(parsed['rows'])} rows"
    if isinstance(parsed, dict) and isinstance(parsed.get("listings"), list):
        return f"{name}: {len(parsed['listings'])} listings"
    if isinstance(parsed, dict) and parsed.get("error"):
        return f"{name}: error"
    return f"{name}: done"


def _trim_old_tools(items: list) -> list:
    """Keep the last two user turns intact and shorten older tool payloads."""
    user_at = []
    for index, item in enumerate(items):
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role == "user":
            user_at.append(index)
    if len(user_at) < 3:
        return items
    cutoff = user_at[-2]
    names = {}
    for item in items[:cutoff]:
        if isinstance(item, dict) and item.get("type") == "function_call":
            names[item.get("call_id")] = item.get("name") or "tool"
    trimmed = []
    for index, item in enumerate(items):
        if index >= cutoff or not isinstance(item, dict) or item.get("type") != "function_call_output":
            trimmed.append(item)
            continue
        copy = dict(item)
        copy["output"] = _short_tool(names.get(item.get("call_id"), "tool"), item.get("output") or "")
        trimmed.append(copy)
    return trimmed


def _prepare_model_input(data):
    """Drop the prefetched catalog read and shorten tool output from older turns.

    The catalog read is stored before the user message. Gemini rejects a tool call
    in that position. The catalog is already in the instructions.
    """
    kept = [item for item in data.model_data.input if _call_id(item) != PREFETCH_MAPPINGS_CALL_ID]
    return ModelInputData(input=_trim_old_tools(kept), instructions=data.model_data.instructions)


class RecordToolCalls(RunHooks):
    """Watches one run and publishes each tool call as it starts and finishes.

    Make a new one per request: it holds this run's calls.

    read_mappings starts before the user sends, on a new chat. Its row stays
    hidden until another tool is on screen, then it joins that list. Pass
    `mappings` when this turn should show that prefetched read.
    """

    def __init__(self, mappings: str | None = None, trace: observe.TurnTrace | None = None):
        self.tool_calls = []
        self.events = asyncio.Queue()
        self.prefetch_result = mappings
        self.trace = trace
        self._seq = 0
        self._indexes = {}
        self._started: dict[int, float] = {}
        self._held = []
        self._mappings_visible = False
        self._mappings_closed = False
        self._gate = asyncio.Lock()

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        text = "Writing the answer" if self.tool_calls or self._indexes else "Thinking"
        await self.events.put({"type": "status", "text": text})

    def _remember(self, name: str, args: dict, result: str):
        if name == "read_mappings" and any(call["name"] == "read_mappings" for call in self.tool_calls):
            return
        call = {"name": name, "args": args, "result": result}
        if name == "read_mappings":
            self.tool_calls.insert(0, call)
        else:
            self.tool_calls.append(call)

    async def _reveal(self):
        """Publish the column step into the list that already has the other tools."""
        if self._mappings_visible:
            return
        self._mappings_visible = True
        held = self._held
        self._held = []
        if held:
            for event in held:
                await self.events.put(event)
            self._mappings_closed = any(
                event.get("type") == "tool_end" and event.get("name") == "read_mappings"
                for event in held
            )
            return
        if not self.prefetch_result:
            return
        if any(call["name"] == "read_mappings" for call in self.tool_calls):
            self._mappings_closed = True
            return
        self._remember("read_mappings", {}, self.prefetch_result)
        self._mappings_closed = True
        await self.events.put({
            "type": "tool_start",
            "index": 0,
            "name": "read_mappings",
            "args": {},
        })
        await self.events.put({
            "type": "tool_end",
            "index": 0,
            "name": "read_mappings",
            "args": {},
            "result": self.prefetch_result,
        })

    async def on_tool_start(self, context, agent, tool):
        async with self._gate:
            if tool.name == "read_mappings" and (self._mappings_visible or self._mappings_closed):
                return
            self._seq += 1
            index = self._seq
            self._indexes[id(context)] = index
            self._started[index] = time.monotonic()
            event = {
                "type": "tool_start",
                "index": index,
                "name": tool.name,
                "args": _tool_args(context),
            }
            if tool.name == "read_mappings":
                self._held.append(event)
                return
            await self._reveal()
            await self.events.put(event)

    async def on_tool_end(self, context, agent, tool, result):
        async with self._gate:
            if tool.name == "read_mappings" and self._mappings_closed:
                self._indexes.pop(id(context), None)
                return
            args = _tool_args(context)
            # For a function tool, context is a ToolContext: it carries the call's arguments.
            self._remember(tool.name, args, str(result))
            index = self._indexes.pop(id(context), self._seq)
            started = self._started.pop(index, None)
            if self.trace is not None:
                seconds = time.monotonic() - started if started is not None else 0
                self.trace.tool(tool.name, args, str(result), seconds)
            event = {
                "type": "tool_end",
                "index": index,
                "name": tool.name,
                "args": args,
                "result": str(result),
            }
            if tool.name == "read_mappings" and not self._mappings_visible:
                self._held.append(event)
                return
            if tool.name == "read_mappings":
                self._mappings_closed = True
                await self.events.put(event)
                return
            await self._reveal()
            await self.events.put(event)

    async def release(self):
        """Show a column step that finished before any other tool, with the answer."""
        async with self._gate:
            await self._reveal()


# --- Session Store ---

init_ui()

# --- FastAPI App ---

app = FastAPI()

# The cookie is the Google account id. Chats, favorites, and the profile hang off it.
USER_COOKIE = "chat_user"
_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{6,128}$")


def _account_id(request: Request) -> str | None:
    current = request.cookies.get(USER_COOKIE, "")
    if not _ACCOUNT_ID.fullmatch(current):
        return None
    if accounts.get_account(current) is None:
        return None
    return current


def _require_account(request: Request) -> str:
    account_id = _account_id(request)
    if not account_id:
        raise HTTPException(status_code=401, detail="Sign in required")
    return account_id


def _set_user_cookie(response: Response, request: Request, user_id: str) -> None:
    forwarded = request.headers.get("x-forwarded-proto", "")
    secure = request.url.scheme == "https" or forwarded.split(",")[0].strip().lower() == "https"
    response.set_cookie(
        USER_COOKIE,
        user_id,
        max_age=60 * 60 * 24 * 400,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ResumeRequest(BaseModel):
    session_id: str
    approved: bool


class PreviewRequest(BaseModel):
    listing_ids: list[str]


class FavoriteRequest(BaseModel):
    listing_id: str


class GoogleRequest(BaseModel):
    credential: str


def _approval_arguments(item) -> dict:
    raw = getattr(item, "raw_item", None)
    if isinstance(raw, dict):
        args = raw.get("arguments") or "{}"
    else:
        args = getattr(raw, "arguments", None) or "{}"
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _event_stream(hooks: RecordToolCalls, run) -> StreamingResponse:
    task = asyncio.create_task(run())

    async def generate():
        try:
            while True:
                event = await hooks.events.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _emit_done(hooks, trace, question, response, session_id, account_id, profile) -> None:
    await hooks.release()
    if trace is not None:
        trace.finish(response or "")
    presented = await asyncio.to_thread(
        resolve_display, question, hooks.tool_calls, response, profile, account_id
    )
    await hooks.events.put({
        "type": "done",
        "response": response,
        "session_id": session_id,
        "tool_calls": hooks.tool_calls,
        "display": presented["display"],
        "listings": presented["listings"],
        "charts": presented["charts"],
        "map": presented["map"],
        "follow_ups": presented["follow_ups"],
        "trace_id": None if trace is None else trace.trace_id,
    })
    await hooks.events.put(None)


@app.get("/")
def index():
    # The empty page can load the column catalog before anyone is signed in.
    start_read_mappings()
    return FileResponse(
        Path(__file__).parent / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/auth/config")
def auth_config():
    return {
        "client_id": accounts.google_client_id(),
        "maps_api_key": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
    }


@app.get("/auth/me")
def auth_me(http_request: Request):
    account_id = _require_account(http_request)
    account = accounts.get_account(account_id) or {"account_id": account_id}
    return {**account, "profile": accounts.get_profile(account_id)}


@app.post("/auth/google")
def auth_google(body: GoogleRequest, http_request: Request):
    try:
        info = accounts.verify_google_token(body.credential)
    except accounts.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    account = accounts.upsert_account(str(info["sub"]), info.get("email"), info.get("name"))
    response = JSONResponse(account)
    _set_user_cookie(response, http_request, account["account_id"])
    return response


@app.post("/auth/logout")
def auth_logout(response: Response):
    response.delete_cookie(USER_COOKIE, path="/")
    return {"status": "ok"}


@app.post("/mappings/warm")
def warm_mappings():
    """Start read_mappings. Returns before the read finishes. The page shows nothing."""
    start_read_mappings()
    return {"status": "ok"}


async def _arm_mappings(session, hooks) -> None:
    """Store the column catalog on a new thread so the page can show that step."""
    prior = await session.get_items()
    if _has_real_mappings(prior):
        return
    catalog = await asyncio.to_thread(load_mappings)
    if not _has_prefetch(prior):
        await session.add_items(_prefetch_items(catalog))
        hooks.prefetch_result = catalog


def _open_turn(account_id: str, session_id: str, message: str) -> tuple[RentalContext, observe.TurnTrace]:
    profile = accounts.get_profile(account_id)
    ctx = RentalContext(account_id=account_id, profile=profile, user_message=message)
    CURRENT.set(ctx)
    trace = observe.start_trace(account_id, session_id, message)
    ctx.trace_id = trace.trace_id
    return ctx, trace


async def _pause_for_approval(result, hooks, trace, session_id, account_id) -> bool:
    interruptions = getattr(result, "interruptions", None) or []
    if not interruptions:
        return False
    state = result.to_state()
    await asyncio.to_thread(
        accounts.save_pending,
        session_id,
        account_id,
        state.to_json(context_serializer=to_dict),
    )
    item = interruptions[0]
    await hooks.events.put({
        "type": "approval",
        "session_id": session_id,
        "tool": item.tool_name or "save_profile",
        "arguments": _approval_arguments(item),
    })
    if trace is not None:
        trace.finish("Waiting for profile approval")
    await hooks.events.put(None)
    return True


@app.post("/chat")
async def chat(body: ChatRequest, http_request: Request):
    # Events stream as the tools run, so the page can show each step before the answer exists.
    account_id = _require_account(http_request)
    session_id = body.session_id or str(uuid.uuid4())
    allowed = await asyncio.to_thread(claim_session, session_id, account_id)
    if not allowed:
        raise HTTPException(status_code=404, detail="No chat with that id")
    await asyncio.to_thread(accounts.clear_pending, session_id)
    start_read_mappings()
    session = open_session(session_id)
    ctx, trace = _open_turn(account_id, session_id, body.message)
    hooks = RecordToolCalls(trace=trace)
    run_config = RunConfig(call_model_input_filter=_prepare_model_input)

    async def run():
        try:
            await _arm_mappings(session, hooks)
            result = await Runner.run(
                agent,
                body.message,
                session=session,
                hooks=hooks,
                max_turns=25,
                run_config=run_config,
                context=ctx,
            )
            if await _pause_for_approval(result, hooks, trace, session_id, account_id):
                return
            response = result.final_output or ""
        except Exception as e:
            # Auth, billing, a model that is not running: show it in the chat, not as a 500.
            response = f"Model call failed: {type(e).__name__}: {str(e)[:300]}"
        await _emit_done(hooks, trace, body.message, response, session_id, account_id, ctx.profile)

    return _event_stream(hooks, run)


@app.post("/chat/resume")
async def resume(body: ResumeRequest, http_request: Request):
    account_id = _require_account(http_request)
    payload = await asyncio.to_thread(accounts.load_pending, body.session_id, account_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No profile change is waiting")
    allowed = await asyncio.to_thread(owns_session, body.session_id, account_id)
    if not allowed:
        raise HTTPException(status_code=404, detail="No chat with that id")
    await asyncio.to_thread(accounts.clear_pending, body.session_id)
    session = open_session(body.session_id)

    def restore(data):
        ctx = from_dict(data if isinstance(data, dict) else {})
        ctx.account_id = account_id
        ctx.profile = accounts.get_profile(account_id)
        CURRENT.set(ctx)
        return ctx

    state = await RunState.from_json(agent, payload, context_deserializer=restore)
    ctx = CURRENT.get() or RentalContext(account_id=account_id, profile=accounts.get_profile(account_id))
    trace = observe.start_trace(account_id, body.session_id, ctx.user_message or "profile approval")
    ctx.trace_id = trace.trace_id
    hooks = RecordToolCalls(trace=trace)
    run_config = RunConfig(call_model_input_filter=_prepare_model_input)
    for item in state.get_interruptions():
        if body.approved:
            state.approve(item)
        else:
            state.reject(
                item,
                rejection_message="The user rejected this profile update. The profile was not saved.",
            )

    async def run():
        try:
            result = await Runner.run(
                agent,
                state,
                session=session,
                hooks=hooks,
                max_turns=25,
                run_config=run_config,
            )
            if await _pause_for_approval(result, hooks, trace, body.session_id, account_id):
                return
            response = result.final_output or ""
        except Exception as e:
            response = f"Model call failed: {type(e).__name__}: {str(e)[:300]}"
        await _emit_done(
            hooks, trace, ctx.user_message, response, body.session_id, account_id, ctx.profile
        )

    return _event_stream(hooks, run)


@app.get("/threads")
def threads(http_request: Request):
    return {"threads": list_threads(_require_account(http_request))}


@app.get("/threads/{session_id}")
def thread(session_id: str, http_request: Request):
    loaded = load_thread(session_id, _require_account(http_request))
    if loaded is None:
        raise HTTPException(status_code=404, detail="No chat with that id")
    return loaded


@app.post("/listings/preview")
def preview(request: PreviewRequest, http_request: Request):
    return {"listings": cards_for(request.listing_ids, _require_account(http_request))}


@app.get("/favorites")
def favorites(http_request: Request, response: Response, sort: str = "recent"):
    response.headers["Cache-Control"] = "no-store"
    return list_favorites(_require_account(http_request), sort)


@app.post("/favorites")
def favorite(request: FavoriteRequest, http_request: Request):
    card = add_favorite(_require_account(http_request), request.listing_id)
    if card is None:
        raise HTTPException(status_code=404, detail="No listing with that id")
    return card


@app.delete("/favorites/{listing_id}")
def unfavorite(listing_id: str, http_request: Request):
    if not remove_favorite(_require_account(http_request), listing_id):
        raise HTTPException(status_code=404, detail="That listing is not saved")
    return {"status": "ok"}


@app.post("/clear")
async def clear(http_request: Request, session_id: str | None = None):
    account_id = _require_account(http_request)
    if session_id:
        allowed = await asyncio.to_thread(owns_session, session_id, account_id)
        if not allowed:
            raise HTTPException(status_code=404, detail="No chat with that id")
        await open_session(session_id).clear_session()
    return {"status": "ok"}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
