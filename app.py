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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from present import (
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
from tools import ANALYSIS_TOOLS, get_enum_values, read_mappings, run_sql

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


class RecordToolCalls(RunHooks):
    """Watches one run and publishes each tool call as it starts and finishes.

    Make a new one per request: it holds this run's calls.
    """

    def __init__(self):
        self.tool_calls = []
        self.events = asyncio.Queue()
        self._seq = 0
        self._indexes = {}

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        text = "Writing the answer" if self.tool_calls or self._indexes else "Thinking"
        await self.events.put({"type": "status", "text": text})

    async def on_tool_start(self, context, agent, tool):
        self._seq += 1
        index = self._seq
        self._indexes[id(context)] = index
        await self.events.put({
            "type": "tool_start",
            "index": index,
            "name": tool.name,
            "args": _tool_args(context),
        })

    async def on_tool_end(self, context, agent, tool, result):
        # For a function tool, context is a ToolContext: it carries the call's arguments.
        call = {
            "name": tool.name,
            "args": _tool_args(context),
            "result": str(result),
        }
        self.tool_calls += [call]
        await self.events.put({
            "type": "tool_end",
            "index": self._indexes.pop(id(context), self._seq),
            "name": call["name"],
            "args": call["args"],
            "result": call["result"],
        })


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
    return FileResponse(
        Path(__file__).parent / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    # Get or create the session. A new session on the same id sees the same history.
    # Events stream as the tools run, so the page can show each step before the answer exists.
    session_id = request.session_id or str(uuid.uuid4())
    session = open_session(session_id)
    hooks = RecordToolCalls()

    async def run():
        try:
            result = await Runner.run(
                agent,
                request.message,
                session=session,
                hooks=hooks,
                max_turns=25,
            )
            response, tool_calls = result.final_output, hooks.tool_calls
        except Exception as e:
            # Auth, billing, a model that is not running: show it in the chat, not as a 500.
            # Keep the calls that already finished so the page can still show those steps.
            response = f"Model call failed: {type(e).__name__}: {str(e)[:300]}"
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
def favorites(sort: str = "recent"):
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
