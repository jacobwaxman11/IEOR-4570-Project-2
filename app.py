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
    SQLiteSession,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import get_enum_values, read_mappings, run_sql

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# --- The Agent ---

agent = Agent(
    name="NYC Rental Analyst",
    instructions="""\
You are a data analyst. Answer questions about NYC rental listings.
The data is in a SQLite database. `listings` has one row per StreetEasy listing.
Related tables join on the keys below.

Tables:
- listings: one row per StreetEasy listing.
- listing_amenities: amenity tags. Join to listings on listing_id.
- listing_images: photo references. Join to listings on listing_id.
- image_scores: per-image scores. Join to listing_images on image_url.
- listing_scores: aggregated photo scores. Join to listings on listing_id.
- regression_results: predicted rent, actual rent, percent discount, and deal score. Join to listings on listing_id.
- regression_features: engineered predictors used to estimate rent. Join to listings on listing_id.

- Call read_mappings first to see the columns.
- Before filtering on an enum column, call get_enum_values. Never guess values.
- Missing values are NULL. Exclude them from filters, groups, and aggregates.
- Answer in plain English. Don't show the SQL unless asked.

Example queries will be added later.
""",
    model=LitellmModel(model="vertex_ai/gemini-3.8-flash", api_key="unused"),
    model_settings=ModelSettings(extra_args={"vertex_location": "global"}),
    tools=[read_mappings, get_enum_values, run_sql],
)


class RecordToolCalls(RunHooks):
    """Watches the run and records each tool call, for the page to show.

    Make a new one per request: it holds this run's calls.
    """

    def __init__(self):
        self.tool_calls = []

    async def on_tool_end(self, context, agent, tool, result):
        # For a function tool, context is a ToolContext: it carries the call's arguments.
        self.tool_calls += [{
            "name": tool.name,
            "args": json.loads(context.tool_arguments),
            "result": str(result),
        }]


# --- Session Store ---

SESSIONS_DB = str(Path(__file__).parent / "conversations.db")

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Get or create the session. A new SQLiteSession on the same file sees the same history.
    session_id = request.session_id or str(uuid.uuid4())
    session = SQLiteSession(session_id, SESSIONS_DB)
    hooks = RecordToolCalls()

    try:
        result = await Runner.run(agent, request.message, session=session, hooks=hooks)
        response, tool_calls = result.final_output, hooks.tool_calls
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        await SQLiteSession(session_id, SESSIONS_DB).clear_session()
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
