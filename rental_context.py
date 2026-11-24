"""Per-run state for the rental agent.

The object is shared with tool calls. Guardrails and Places tools mutate it.
A Langfuse span is not stored here, so a paused run can be serialized.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RentalContext:
    account_id: str
    profile: dict = field(default_factory=dict)
    sql_search_done: bool = False
    places_calls: int = 0
    user_message: str = ""
    trace_id: str | None = None


CURRENT: contextvars.ContextVar[RentalContext | None] = contextvars.ContextVar(
    "rental_context",
    default=None,
)


def as_rental(obj: Any) -> RentalContext | None:
    """Find the run context on a tool wrapper, a context wrapper, or this task."""
    if isinstance(obj, RentalContext):
        return obj
    inner = getattr(obj, "context", None)
    if isinstance(inner, RentalContext):
        return inner
    return CURRENT.get()


def bind(obj: Any) -> RentalContext | None:
    rental = as_rental(obj)
    if rental is not None:
        CURRENT.set(rental)
    return rental


def to_dict(ctx: RentalContext) -> dict:
    return {
        "account_id": ctx.account_id,
        "profile": ctx.profile,
        "sql_search_done": ctx.sql_search_done,
        "places_calls": ctx.places_calls,
        "user_message": ctx.user_message,
        "trace_id": ctx.trace_id,
    }


def from_dict(data: dict) -> RentalContext:
    profile = data.get("profile") if isinstance(data.get("profile"), dict) else {}
    return RentalContext(
        account_id=str(data.get("account_id") or ""),
        profile=profile,
        sql_search_done=bool(data.get("sql_search_done")),
        places_calls=int(data.get("places_calls") or 0),
        user_message=str(data.get("user_message") or ""),
        trace_id=data.get("trace_id"),
    )
