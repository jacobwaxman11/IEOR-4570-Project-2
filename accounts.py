"""Google accounts, renter profiles, and the tool cache.

Local runs use ui.db. Prod uses the same BigQuery dataset as chat history.
Favorites are stored per account. Older favorite rows without an account are left
in place and are not shown.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts import bq

UI_DB = Path(__file__).parent / "ui.db"
_lock = threading.Lock()

PLACES_TTL_SECONDS = 24 * 60 * 60
CHART_TTL_SECONDS = 60 * 60
SHARED_ACCOUNT = "__shared__"


class AuthError(Exception):
    """A sign-in token the server will not accept."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _now()).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(UI_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_args(args: dict) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def ensure_schema() -> None:
    """Create account tables. Prod also adds account_id to favorites."""
    if bq.prod():
        bq.ensure_tables()
        bq.execute(
            f"ALTER TABLE {bq.quote_ident('favorites')} "
            "ADD COLUMN IF NOT EXISTS account_id STRING"
        )
        return
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    email TEXT,
                    name TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS renter_profiles (
                    account_id TEXT PRIMARY KEY,
                    max_rent REAL,
                    min_beds REAL,
                    neighborhoods TEXT,
                    work_address TEXT,
                    commute_mode TEXT,
                    last_commute TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache (
                    account_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    args_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (account_id, tool, args_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_runs (
                    session_id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(favorites)")}
            if columns and "account_id" not in columns:
                conn.execute("ALTER TABLE favorites RENAME TO favorites_legacy")
                columns = set()
            if not columns:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS favorites (
                        account_id TEXT NOT NULL,
                        listing_id TEXT NOT NULL,
                        favorited_at TEXT NOT NULL,
                        PRIMARY KEY (account_id, listing_id)
                    )
                    """
                )
            conn.commit()
        finally:
            conn.close()


def google_client_id() -> str:
    return os.environ.get("GOOGLE_CLIENT_ID", "").strip()


def verify_google_token(token: str) -> dict:
    """Validate a Google Identity Services ID token and return its claims."""
    client_id = google_client_id()
    if not client_id:
        raise AuthError("GOOGLE_CLIENT_ID is not set")
    if not token:
        raise AuthError("Missing Google credential")
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    try:
        info = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
    except Exception as exc:
        raise AuthError("Google sign-in was rejected") from exc
    subject = str(info.get("sub") or "")
    if not subject:
        raise AuthError("Google account has no subject")
    return info


def upsert_account(account_id: str, email: str | None, name: str | None) -> dict:
    ensure_schema()
    created = _stamp()
    if bq.prod():
        bq.merge_row(
            "accounts",
            ["account_id"],
            {
                "account_id": account_id,
                "email": email,
                "name": name,
                "created_at": created,
            },
            update_columns=["email", "name"],
        )
        return {"account_id": account_id, "email": email, "name": name}
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO accounts (account_id, email, name, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    email = excluded.email,
                    name = excluded.name
                """,
                (account_id, email, name, created),
            )
            conn.commit()
        finally:
            conn.close()
    return {"account_id": account_id, "email": email, "name": name}


def get_account(account_id: str) -> dict | None:
    if not account_id:
        return None
    ensure_schema()
    if bq.prod():
        rows = bq.query_dicts(
            f"""
            SELECT account_id, email, name
            FROM {bq.quote_ident('accounts')}
            WHERE account_id = ?
            """,
            (account_id,),
        )
        return rows[0] if rows else None
    if not UI_DB.exists():
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT account_id, email, name FROM accounts WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"account_id": row["account_id"], "email": row["email"], "name": row["name"]}


def _profile_from(row: dict) -> dict:
    neighborhoods = row.get("neighborhoods")
    if isinstance(neighborhoods, str) and neighborhoods:
        try:
            parsed = json.loads(neighborhoods)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in neighborhoods.split(",") if part.strip()]
        neighborhoods = parsed
    if not isinstance(neighborhoods, list):
        neighborhoods = []
    return {
        "max_rent": row.get("max_rent"),
        "min_beds": row.get("min_beds"),
        "neighborhoods": neighborhoods,
        "work_address": row.get("work_address") or "",
        "commute_mode": row.get("commute_mode") or "transit",
        "last_commute": row.get("last_commute") or "",
    }


def get_profile(account_id: str) -> dict:
    empty = _profile_from({})
    if not account_id:
        return empty
    ensure_schema()
    if bq.prod():
        rows = bq.query_dicts(
            f"SELECT * FROM {bq.quote_ident('renter_profiles')} WHERE account_id = ?",
            (account_id,),
        )
        return _profile_from(rows[0]) if rows else empty
    if not UI_DB.exists():
        return empty
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM renter_profiles WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    finally:
        conn.close()
    return _profile_from(dict(row)) if row is not None else empty


def save_profile(account_id: str, fields: dict) -> dict:
    """Merge the provided fields into the account profile. None leaves a field as it is."""
    current = get_profile(account_id)
    for key in ("max_rent", "min_beds", "neighborhoods", "work_address", "commute_mode", "last_commute"):
        if key in fields and fields[key] is not None:
            current[key] = fields[key]
    stored = {
        "account_id": account_id,
        "max_rent": current.get("max_rent"),
        "min_beds": current.get("min_beds"),
        "neighborhoods": json.dumps(current.get("neighborhoods") or []),
        "work_address": current.get("work_address") or "",
        "commute_mode": current.get("commute_mode") or "transit",
        "last_commute": current.get("last_commute") or "",
        "updated_at": _stamp(),
    }
    ensure_schema()
    if bq.prod():
        bq.merge_row("renter_profiles", ["account_id"], stored)
    else:
        with _lock:
            conn = _connect()
            try:
                conn.execute(
                    """
                    INSERT INTO renter_profiles (
                        account_id, max_rent, min_beds, neighborhoods, work_address,
                        commute_mode, last_commute, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_id) DO UPDATE SET
                        max_rent = excluded.max_rent,
                        min_beds = excluded.min_beds,
                        neighborhoods = excluded.neighborhoods,
                        work_address = excluded.work_address,
                        commute_mode = excluded.commute_mode,
                        last_commute = excluded.last_commute,
                        updated_at = excluded.updated_at
                    """,
                    (
                        stored["account_id"],
                        stored["max_rent"],
                        stored["min_beds"],
                        stored["neighborhoods"],
                        stored["work_address"],
                        stored["commute_mode"],
                        stored["last_commute"],
                        stored["updated_at"],
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    return get_profile(account_id)


def set_last_commute(account_id: str, summary: str) -> None:
    if not account_id or account_id == SHARED_ACCOUNT:
        return
    save_profile(account_id, {"last_commute": summary})


def cache_get(account_id: str, tool: str, args: dict) -> dict | None:
    if not account_id:
        return None
    ensure_schema()
    digest = _hash_args(args)
    now = _stamp()
    if bq.prod():
        rows = bq.query_dicts(
            f"""
            SELECT payload, expires_at
            FROM {bq.quote_ident('tool_cache')}
            WHERE account_id = ? AND tool = ? AND args_hash = ?
            """,
            (account_id, tool, digest),
        )
        if not rows:
            return None
        if str(rows[0].get("expires_at") or "") <= now:
            return None
        try:
            payload = json.loads(rows[0]["payload"])
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None
    if not UI_DB.exists():
        return None
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT payload, expires_at FROM tool_cache
            WHERE account_id = ? AND tool = ? AND args_hash = ?
            """,
            (account_id, tool, digest),
        ).fetchone()
    finally:
        conn.close()
    if row is None or str(row["expires_at"]) <= now:
        return None
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def cache_put(account_id: str, tool: str, args: dict, payload: dict, ttl_seconds: int) -> None:
    if not account_id or not isinstance(payload, dict):
        return
    stored = {key: value for key, value in payload.items() if key != "cached"}
    blob = json.dumps(stored, default=str)
    expires = _stamp(_now() + timedelta(seconds=ttl_seconds))
    digest = _hash_args(args)
    ensure_schema()
    if bq.prod():
        bq.merge_row(
            "tool_cache",
            ["account_id", "tool", "args_hash"],
            {
                "account_id": account_id,
                "tool": tool,
                "args_hash": digest,
                "payload": blob,
                "expires_at": expires,
            },
        )
        return
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO tool_cache (account_id, tool, args_hash, payload, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id, tool, args_hash) DO UPDATE SET
                    payload = excluded.payload,
                    expires_at = excluded.expires_at
                """,
                (account_id, tool, digest, blob, expires),
            )
            conn.commit()
        finally:
            conn.close()


def save_pending(session_id: str, account_id: str, state: dict) -> None:
    ensure_schema()
    blob = json.dumps(state)
    created = _stamp()
    if bq.prod():
        bq.merge_row(
            "pending_runs",
            ["session_id"],
            {
                "session_id": session_id,
                "account_id": account_id,
                "state_json": blob,
                "created_at": created,
            },
        )
        return
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO pending_runs (session_id, account_id, state_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    account_id = excluded.account_id,
                    state_json = excluded.state_json,
                    created_at = excluded.created_at
                """,
                (session_id, account_id, blob, created),
            )
            conn.commit()
        finally:
            conn.close()


def load_pending(session_id: str, account_id: str) -> dict | None:
    ensure_schema()
    if bq.prod():
        rows = bq.query_dicts(
            f"""
            SELECT state_json FROM {bq.quote_ident('pending_runs')}
            WHERE session_id = ? AND account_id = ?
            """,
            (session_id, account_id),
        )
        raw = rows[0]["state_json"] if rows else None
    else:
        if not UI_DB.exists():
            return None
        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT state_json FROM pending_runs
                WHERE session_id = ? AND account_id = ?
                """,
                (session_id, account_id),
            ).fetchone()
        finally:
            conn.close()
        raw = row["state_json"] if row is not None else None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def clear_pending(session_id: str) -> None:
    if not session_id:
        return
    ensure_schema()
    if bq.prod():
        bq.execute(
            f"DELETE FROM {bq.quote_ident('pending_runs')} WHERE session_id = ?",
            (session_id,),
        )
        return
    if not UI_DB.exists():
        return
    with _lock:
        conn = _connect()
        try:
            conn.execute("DELETE FROM pending_runs WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()


def profile_text(profile: dict | None) -> str:
    """Short block appended to the system prompt."""
    if not profile:
        return ""
    lines = []
    if profile.get("max_rent") is not None:
        lines.append(f"Budget ceiling ${int(profile['max_rent']):,} per month.")
    if profile.get("min_beds") is not None:
        beds = profile["min_beds"]
        label = "studio" if beds == 0 else f"{int(beds)}+ bedrooms"
        lines.append(f"Minimum size: {label}.")
    neighborhoods = profile.get("neighborhoods") or []
    if neighborhoods:
        lines.append("Neighborhoods: " + ", ".join(str(name) for name in neighborhoods) + ".")
    if profile.get("work_address"):
        mode = profile.get("commute_mode") or "transit"
        lines.append(f"Work address: {profile['work_address']}. Commute by {mode}.")
    if profile.get("last_commute"):
        lines.append(f"Last commute: {profile['last_commute']}.")
    if not lines:
        return ""
    return (
        "Renter profile for this signed-in account. Use it when the user says "
        "\"my budget\" or \"to work\". Change it only by calling save_profile.\n"
        + "\n".join(lines)
    )
