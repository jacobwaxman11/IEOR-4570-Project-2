"""Langfuse traces for a chat turn, and scores for each SQL call.

Missing keys leave every function as a no-op so the chat still runs.
LANGFUSE_HOST is accepted as an alias for the SDK's LANGFUSE_BASE_URL.
"""

from __future__ import annotations

import os
import re
from typing import Any

_installed = False
_client = None


def _enabled() -> bool:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public and secret)


def _host() -> None:
    host = os.environ.get("LANGFUSE_HOST", "").strip()
    if host and not os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = host


def client():
    global _client
    if not _enabled():
        return None
    if _client is None:
        _host()
        from langfuse import get_client

        _client = get_client()
    return _client


def install_litellm() -> None:
    """Record Gemini calls on the active Langfuse trace."""
    global _installed
    if _installed or not _enabled():
        return
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _Generations(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _record_generation(kwargs, response_obj)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            _record_generation(kwargs, response_obj)

    litellm.callbacks = list(getattr(litellm, "callbacks", None) or []) + [_Generations()]
    _installed = True


def _clip(text, limit: int = 500) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _message_preview(messages) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    content = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")
    if isinstance(content, list):
        content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return _clip(content, 300)


def _response_preview(response_obj) -> str:
    try:
        choice = response_obj.choices[0].message
        return _clip(getattr(choice, "content", "") or "")
    except Exception:
        return ""


def _record_generation(kwargs, response_obj) -> None:
    from rental_context import CURRENT

    rental = CURRENT.get()
    langfuse = client()
    if rental is None or not rental.trace_id or langfuse is None:
        return
    parent = _parents.get(rental.trace_id)
    try:
        span = langfuse.start_observation(
            trace_context={"trace_id": rental.trace_id, "parent_span_id": parent} if parent else {"trace_id": rental.trace_id},
            name="gemini",
            as_type="generation",
            model=str(kwargs.get("model") or ""),
            input={"preview": _message_preview(kwargs.get("messages"))},
            output={"preview": _response_preview(response_obj)},
        )
        span.end()
    except Exception:
        return


class TurnTrace:
    def __init__(self, trace_id: str | None, root: Any, user_message: str = ""):
        self.trace_id = trace_id
        self.root = root
        self.user_message = user_message

    def tool(self, name: str, args: dict, result: str, seconds: float) -> None:
        langfuse = client()
        if self.root is None or langfuse is None:
            return
        query = ""
        if isinstance(args, dict) and name == "run_sql":
            query = _clip(args.get("query") or "", 2000)
        cached = False
        try:
            import json
            parsed = json.loads(result) if result else None
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            cached = bool(parsed.get("cached"))
        try:
            span = self.root.start_observation(
                name=name or "tool",
                as_type="tool",
                input={"args": sorted((args or {}).keys()), "query": query, "cached": cached},
                metadata={"seconds": round(seconds, 3)},
            )
            summary = _result_summary(name, parsed, result)
            span.update(output=summary)
            if name == "run_sql":
                neighborhoods, amenities = _enums()
                for score in sql_scores(query, result, self.user_message, neighborhoods, amenities):
                    span.score(name=score["name"], value=score["value"], comment=score.get("comment") or None)
            span.end()
        except Exception:
            return

    def finish(self, output: str) -> None:
        if self.root is None:
            return
        try:
            self.root.update(output={"preview": _clip(output, 500)})
            self.root.end()
            langfuse = client()
            if langfuse is not None:
                langfuse.flush()
        except Exception:
            return
        if self.trace_id:
            _parents.pop(self.trace_id, None)


_parents: dict[str, str] = {}


def start_trace(account_id: str, session_id: str, message: str) -> TurnTrace:
    install_litellm()
    langfuse = client()
    if langfuse is None:
        return TurnTrace(None, None)
    try:
        from langfuse import propagate_attributes

        trace_id = langfuse.create_trace_id()
        with propagate_attributes(user_id=account_id, session_id=session_id):
            root = langfuse.start_observation(
                trace_context={"trace_id": trace_id},
                name="rental-chat",
                as_type="span",
                input={"message": _clip(message, 500)},
            )
        _parents[trace_id] = root.id
        return TurnTrace(trace_id, root, message)
    except Exception:
        return TurnTrace(None, None)


def _result_summary(name: str, parsed, result: str) -> dict:
    if not isinstance(parsed, dict):
        rejected = "rejected" in str(result or "").casefold()
        return {"tool": name, "rejected": rejected}
    if parsed.get("error"):
        return {"tool": name, "error": True}
    if isinstance(parsed.get("rows"), list):
        return {"tool": name, "row_count": len(parsed["rows"]), "truncated": bool(parsed.get("truncated"))}
    if isinstance(parsed.get("listings"), list):
        return {"tool": name, "row_count": len(parsed["listings"])}
    if parsed.get("chart"):
        return {"tool": name, "chart": parsed["chart"].get("type")}
    return {"tool": name, "ok": True}


_STATUS_WORDS = ("rented", "delisted", "in-contract", "in contract", "off-market", "off market")
_QUOTED = re.compile(r"'((?:[^']|'')*)'")
_EQ = {
    "neighborhood": re.compile(r"neighborhood\s*=\s*'((?:[^']|'')*)'", re.IGNORECASE),
    "amenity": re.compile(r"amenity\s*=\s*'((?:[^']|'')*)'", re.IGNORECASE),
}
_IN = {
    "neighborhood": re.compile(r"neighborhood\s+in\s*\(([^)]*)\)", re.IGNORECASE),
    "amenity": re.compile(r"amenity\s+in\s*\(([^)]*)\)", re.IGNORECASE),
}
_ACTIVE = re.compile(r"""listing_status\s*=\s*['"]active['"]""", re.IGNORECASE)


def _sql_literals(query: str, column: str) -> list[str]:
    """Equality and IN literals, with a doubled apostrophe turned back into one."""
    found = [raw.replace("''", "'") for raw in _EQ[column].findall(query or "")]
    for group in _IN[column].findall(query or ""):
        found.extend(raw.replace("''", "'") for raw in _QUOTED.findall(group))
    return found


def sql_scores(query: str, result: str, user_message: str, neighborhoods: set[str] | None = None, amenities: set[str] | None = None) -> list[dict]:
    """Deterministic scores for one run_sql call. Values are 0 or 1, plus row count."""
    import json

    text = result or ""
    rejected = text.startswith("Rejected:")
    parsed = None
    if not rejected:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    executed = isinstance(parsed, dict) and "columns" in parsed and "error" not in parsed
    rows = parsed.get("rows") if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list) else []
    lowered = (query or "").casefold()
    asked = (user_message or "").casefold()
    mentions_listings = "listings" in lowered
    other_status = any(word in asked for word in _STATUS_WORDS)
    if not mentions_listings or other_status or _ACTIVE.search(query or ""):
        active_score = 1
    else:
        active_score = 0

    known = 1
    comments = []
    if neighborhoods is not None:
        for name in _sql_literals(query or "", "neighborhood"):
            if name not in neighborhoods:
                known = 0
                comments.append(name)
    if amenities is not None:
        for name in _sql_literals(query or "", "amenity"):
            if name not in amenities:
                known = 0
                comments.append(name)

    return [
        {"name": "sql_read_only", "value": 0 if rejected else 1, "comment": "guardrail" if rejected else ""},
        {"name": "sql_executed", "value": 1 if executed else 0, "comment": ""},
        {"name": "sql_active_filter", "value": active_score, "comment": ""},
        {
            "name": "sql_known_enums",
            "value": known,
            "comment": ", ".join(comments[:8]),
        },
        {"name": "sql_row_count", "value": float(len(rows)), "comment": ""},
    ]


def _enums() -> tuple[set[str] | None, set[str] | None]:
    try:
        from tools import load_mappings
        import json
        catalog = json.loads(load_mappings())
    except Exception:
        return None, None
    neighborhoods = set()
    amenities = set()
    for column in catalog.get("columns") or []:
        values = column.get("enum_values") or []
        if column.get("name") == "neighborhood":
            neighborhoods.update(str(value) for value in values)
        if column.get("name") == "amenity":
            amenities.update(str(value) for value in values)
    return neighborhoods, amenities
