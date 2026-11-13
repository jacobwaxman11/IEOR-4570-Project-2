import asyncio
import json
import os
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
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
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
from scripts import bq
from session_store import open_session
from tools import (
    ANALYSIS_TOOLS,
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
- Filtered search by neighborhood, beds, baths, sqft, or rent: search_listings.
- Cheapest units in a neighborhood: cheapest_in_neighborhood.
- Amenities for a building: get_building_amenities.
- Model deals: get_deals. Best deals in one neighborhood: best_deals_in_neighborhood.
- Compare rents across neighborhoods: compare_neighborhoods.
- A short profile of one neighborhood: summarize_neighborhood.
- Anything else: read_mappings, then get_enum_values before filtering an enum, then run_sql.

- When writing SQL, call read_mappings first.
- Before filtering on an enum column in SQL, call get_enum_values. Never guess values.
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
    tools=[read_mappings, get_enum_values, run_sql, *ANALYSIS_TOOLS],
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
        + "\n\nColumn catalog. Each entry has table, name, type, description, is_enum, "
        + "and enum_values when the enum is small. Do not call read_mappings.\n"
        + catalog
    )


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


def _drop_prefetch_mappings(data):
    """Keep the prefetched column read in the transcript, and out of the model request.

    It is stored before the user message. Gemini rejects a tool call in that position.
    The catalog is already in the instructions.
    """
    kept = [item for item in data.model_data.input if _call_id(item) != PREFETCH_MAPPINGS_CALL_ID]
    return ModelInputData(input=kept, instructions=data.model_data.instructions)


class RecordToolCalls(RunHooks):
    """Watches one run and publishes each tool call as it starts and finishes.

    Make a new one per request: it holds this run's calls.

    read_mappings starts before the user sends, on a new chat. Its row stays
    hidden until another tool is on screen, then it joins that list. Pass
    `mappings` when this turn should show that prefetched read.
    """

    def __init__(self, mappings: str | None = None):
        self.tool_calls = []
        self.events = asyncio.Queue()
        self.prefetch_result = mappings
        self._seq = 0
        self._indexes = {}
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
            event = {
                "type": "tool_end",
                "index": self._indexes.pop(id(context), self._seq),
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


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class PreviewRequest(BaseModel):
    listing_ids: list[str]


class FavoriteRequest(BaseModel):
    listing_id: str


@app.get("/")
def index():
    # The empty chat is on screen before anyone sends. Load the columns then.
    start_read_mappings()
    return FileResponse(
        Path(__file__).parent / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/mappings/warm")
def warm_mappings():
    """Start read_mappings. Returns before the read finishes. The page shows nothing."""
    start_read_mappings()
    return {"status": "ok"}


async def _arm_mappings(session, hooks) -> Agent:
    """Use a column catalog that was loaded before this message, when the thread has none."""
    prior = await session.get_items()
    if _has_real_mappings(prior):
        return agent
    catalog = await asyncio.to_thread(load_mappings)
    if not _has_prefetch(prior):
        await session.add_items(_prefetch_items(catalog))
        hooks.prefetch_result = catalog
    return agent.clone(instructions=_with_catalog(agent.instructions, catalog))


@app.post("/chat")
async def chat(request: ChatRequest):
    # Get or create the session. A new session on the same id sees the same history.
    # Events stream as the tools run, so the page can show each step before the answer exists.
    # The column read may already be running from the empty chat. Join it here, and
    # show that step only once the other tool calls are on screen.
    start_read_mappings()
    session_id = request.session_id or str(uuid.uuid4())
    session = open_session(session_id)
    hooks = RecordToolCalls()
    run_config = RunConfig(call_model_input_filter=_drop_prefetch_mappings)

    async def run():
        try:
            run_agent = await _arm_mappings(session, hooks)
            result = await Runner.run(
                run_agent,
                request.message,
                session=session,
                hooks=hooks,
                max_turns=25,
                run_config=run_config,
            )
            response = result.final_output
        except Exception as e:
            # Auth, billing, a model that is not running: show it in the chat, not as a 500.
            # Keep the calls that already finished so the page can still show those steps.
            response = f"Model call failed: {type(e).__name__}: {str(e)[:300]}"
        await hooks.release()
        tool_calls = hooks.tool_calls
        # Card lookup hits SQLite or BigQuery. Keep it off the event loop so the
        # page can keep painting tool steps and the sidebar stays clickable.
        presented = await asyncio.to_thread(
            resolve_display, request.message, tool_calls, response
        )
        await hooks.events.put({
            "type": "done",
            "response": response,
            "session_id": session_id,
            "tool_calls": tool_calls,
            "display": presented["display"],
            "listings": presented["listings"],
        })
        await hooks.events.put(None)

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


@app.get("/threads")
def threads():
    return {"threads": list_threads()}


@app.get("/threads/{session_id}")
def thread(session_id: str):
    loaded = load_thread(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="No chat with that id")
    return loaded


@app.post("/listings/preview")
def preview(request: PreviewRequest):
    return {"listings": cards_for(request.listing_ids)}


@app.get("/favorites")
def favorites(response: Response, sort: str = "recent"):
    response.headers["Cache-Control"] = "no-store"
    return list_favorites(sort)


@app.post("/favorites")
def favorite(request: FavoriteRequest):
    card = add_favorite(request.listing_id)
    if card is None:
        raise HTTPException(status_code=404, detail="No listing with that id")
    return card


@app.delete("/favorites/{listing_id}")
def unfavorite(listing_id: str):
    if not remove_favorite(listing_id):
        raise HTTPException(status_code=404, detail="That listing is not saved")
    return {"status": "ok"}


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        await open_session(session_id).clear_session()
    return {"status": "ok"}


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
