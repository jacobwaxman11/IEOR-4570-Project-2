"""Chat history for the rental agent.

Local runs use the SQLite session that ships with the Agents SDK. A prod
run (``APP_ENV=prod``) stores the same tables in BigQuery, so a Cloud Run
instance does not need a disk.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
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


_MAX_SESSION_ID = 128
_MAX_USER_ID = 64


def _ids_ok(session_id: str, user_id: str) -> bool:
    return (
        isinstance(session_id, str)
        and isinstance(user_id, str)
        and 1 <= len(session_id) <= _MAX_SESSION_ID
        and 1 <= len(user_id) <= _MAX_USER_ID
    )


def _sqlite():
    from present import SESSIONS_DB

    conn = sqlite3.connect(SESSIONS_DB, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_owner_column(conn: sqlite3.Connection) -> None:
    """Add ``user_id`` without disturbing the Agents SDK session tables."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT
        )
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_sessions)")}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE agent_sessions ADD COLUMN user_id TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES agent_sessions (session_id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_id
        ON agent_sessions (user_id)
        """
    )


def ensure_session_schema() -> None:
    """Make sure session rows can store an owner. No file is created just to list."""
    if bq.prod():
        bq.ensure_tables()
        return
    from present import SESSIONS_DB

    if not SESSIONS_DB.exists():
        return
    conn = _sqlite()
    try:
        _ensure_owner_column(conn)
        conn.commit()
    finally:
        conn.close()


def owns_session(session_id: str, user_id: str) -> bool:
    """True only when this user already owns the session."""
    if not _ids_ok(session_id, user_id):
        return False
    if bq.prod():
        return bq.owns_session(session_id, user_id)
    from present import SESSIONS_DB

    if not SESSIONS_DB.exists():
        return False
    conn = _sqlite()
    try:
        _ensure_owner_column(conn)
        conn.commit()
        row = conn.execute(
            """
            SELECT user_id FROM agent_sessions
            WHERE session_id = ? AND user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def claim_session(session_id: str, user_id: str) -> bool:
    """Reserve a new session for this user, or accept one they already own.

    Does not take over a session that has no owner, or one owned by someone else.
    """
    if not _ids_ok(session_id, user_id):
        return False
    if bq.prod():
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return bq.claim_session(session_id, user_id, now)
    conn = _sqlite()
    try:
        _ensure_owner_column(conn)
        row = conn.execute(
            "SELECT user_id FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None:
            conn.commit()
            return row["user_id"] == user_id
        try:
            conn.execute(
                "INSERT INTO agent_sessions (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            raced = conn.execute(
                "SELECT user_id FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return raced is not None and raced["user_id"] == user_id
    finally:
        conn.close()
