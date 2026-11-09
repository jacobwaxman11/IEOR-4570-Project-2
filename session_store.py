"""Chat history for the rental agent.

Local runs use the SQLite session that ships with the Agents SDK. A prod
run (``APP_ENV=prod``) stores the same tables in BigQuery, so a Cloud Run
instance does not need a disk.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from agents.memory.session import SessionABC
from agents.memory.session_settings import SessionSettings, resolve_session_limit

from scripts import bq


class BigQuerySession(SessionABC):
    """Session history in ``agent_sessions`` and ``agent_messages``."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_settings = SessionSettings()

    async def get_items(self, limit: int | None = None) -> list[Any]:
        session_limit = resolve_session_limit(limit, self.session_settings)
        return await asyncio.to_thread(self._get_items, session_limit)

    def _get_items(self, limit: int | None) -> list[Any]:
        if limit == 0:
            return []
        if limit is None or limit < 0:
            rows = bq.query_dicts(
                """
                SELECT message_data FROM agent_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (self.session_id,),
            )
        else:
            rows = bq.query_dicts(
                """
                SELECT message_data FROM agent_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.session_id, limit),
            )
            rows.reverse()
        items = []
        for row in rows:
            try:
                items.append(json.loads(row["message_data"]))
            except (json.JSONDecodeError, TypeError):
                continue
        return items

    async def add_items(self, items: list[Any]) -> None:
        if not items:
            return
        await asyncio.to_thread(self._add_items, items)

    def _add_items(self, items: list[Any]) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        bq.touch_session(self.session_id, now)
        bq.insert_messages(
            self.session_id,
            [json.dumps(item) for item in items],
            now,
        )

    async def pop_item(self) -> Any | None:
        return await asyncio.to_thread(self._pop_item)

    def _pop_item(self) -> Any | None:
        while True:
            latest = bq.latest_message(self.session_id)
            if latest is None:
                return None
            message_id, raw = latest
            bq.delete_message(self.session_id, message_id)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

    async def clear_session(self) -> None:
        await asyncio.to_thread(bq.clear_session, self.session_id)


def open_session(session_id: str):
    """Session bound to BigQuery in prod, and to ``conversations.db`` locally."""
    if bq.prod():
        return BigQuerySession(session_id)
    from agents import SQLiteSession
    from present import SESSIONS_DB

    return SQLiteSession(session_id, str(SESSIONS_DB))
