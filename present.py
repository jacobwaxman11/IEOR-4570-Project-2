"""Turn tool results into listing cards, and store saved listings.

The agent prompt is unchanged. This module only reads what a turn already
returned and the rental database, then decides whether the page shows a grid,
one listing, or neither.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
RENTALS_DB = ROOT / "scripts" / "rentals.db"
IMAGES_DIR = ROOT / "scripts" / "images"
SESSIONS_DB = ROOT / "conversations.db"
UI_DB = ROOT / "ui.db"
STATIC_DIR = ROOT / "static"

LISTING_TOOLS = {
    "search_listings",
    "cheapest_in_neighborhood",
    "get_deals",
    "best_deals_in_neighborhood",
}

SORTS = {"recent", "rent_asc", "rent_desc", "neighborhood", "beds", "deal"}

# A singular ask wins even when the sentence also says "show me".
_SINGULAR = re.compile(
    r"\b("
    r"the cheapest(?!\s+(apartments|listings|units|options|studios|deals)\b)|"
    r"cheapest (apartment|unit|listing|studio|one)\b|"
    r"the most expensive(?!\s+(apartments|listings|units|options|studios)\b)|"
    r"most expensive (apartment|unit|listing|studio)\b|"
    r"the best deal(?!s\b)|"
    r"best deal(?!s\b)|"
    r"top deal(?!s\b)|"
    r"this apartment|this unit|this listing|"
    r"that apartment|that unit|"
    r"tell me about|"
    r"an example"
    r")\b",
    re.IGNORECASE,
)
_OPTIONS = re.compile(
    r"\b(apartments|listings|units|options|studios|places|deals)\b|"
    r"\b(show me|find me|find|list|compare)\b",
    re.IGNORECASE,
)
_UNIT = re.compile(r"(?:#\s*|unit\s+)([A-Za-z0-9]+)", re.IGNORECASE)
_CHEAP = re.compile(r"\b(cheapest|least expensive|lowest rent)\b", re.IGNORECASE)
_EXPENSIVE = re.compile(r"\b(most expensive|highest rent|priciest)\b", re.IGNORECASE)
_DEAL = re.compile(r"\b(best deal|top deal|deal score)\b", re.IGNORECASE)

_ui_lock = threading.Lock()


def init_ui() -> None:
    with _ui_lock:
        conn = sqlite3.connect(UI_DB)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS favorites (
                    listing_id TEXT PRIMARY KEY,
                    favorited_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def _rentals():
    uri = RENTALS_DB.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _ui():
    conn = sqlite3.connect(UI_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _sessions():
    conn = sqlite3.connect(SESSIONS_DB, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def favorite_ids() -> set[str]:
    if not UI_DB.exists():
        return set()
    conn = _ui()
    try:
        return {row["listing_id"] for row in conn.execute("SELECT listing_id FROM favorites")}
    finally:
        conn.close()


def _clip(text, limit: int = 320) -> str | None:
    if not text:
        return None
    text = " ".join(str(text).split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _image_src(local_path: str | None) -> str | None:
    if not local_path:
        return None
    rel = str(local_path).replace("\\", "/").lstrip("/")
    if rel.startswith("images/"):
        rel = rel[len("images/") :]
    root = IMAGES_DIR.resolve()
    full = (root / rel).resolve()
    if not str(full).startswith(str(root) + "/") and full != root:
        return None
    if not full.is_file():
        return None
    return "/media/" + rel


def _number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number.is_integer():
        return int(number)
    return number


def _cards(ids: list[str], saved: set[str] | None = None) -> list[dict]:
    if not ids or not RENTALS_DB.exists():
        return []
    saved = favorite_ids() if saved is None else saved
    placeholders = ",".join("?" * len(ids))
    conn = _rentals()
    try:
        rows = conn.execute(
            f"""
            SELECT l.listing_id, l.name AS building, l.unit, l.neighborhood,
                   l.beds, l.baths, l.sqft, l.rent, l.url, l.listing_status,
                   l.description, r.deal_score,
                   (
                       SELECT i.local_image_path
                       FROM listing_images i
                       WHERE i.listing_id = l.listing_id
                       ORDER BY i.position
                       LIMIT 1
                   ) AS image_path
            FROM listings l
            LEFT JOIN regression_results r ON r.listing_id = l.listing_id
            WHERE l.listing_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
    finally:
        conn.close()
    by_id = {row["listing_id"]: row for row in rows}
    cards = []
    for listing_id in ids:
        row = by_id.get(listing_id)
        if row is None:
            continue
        status = row["listing_status"]
        link = row["url"] if row["url"] and status in (None, "", "ACTIVE") else None
        cards.append({
            "listing_id": listing_id,
            "building": row["building"],
            "unit": row["unit"],
            "neighborhood": row["neighborhood"],
            "beds": _number(row["beds"]),
            "baths": _number(row["baths"]),
            "sqft": _number(row["sqft"]),
            "rent": _number(row["rent"]),
            "url": link,
            "deal_score": _number(row["deal_score"]),
            "image": _image_src(row["image_path"]),
            "description": _clip(row["description"]),
            "favorited": listing_id in saved,
        })
    return cards


def cards_for(ids: list[str]) -> list[dict]:
    seen = []
    for listing_id in ids:
        if listing_id and listing_id not in seen:
            seen.append(str(listing_id))
    return _cards(seen)


def _parse_result(result) -> object | None:
    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str) or not result:
        return None
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return None


def _refs_from_call(call: dict) -> list[dict]:
    data = _parse_result(call.get("result"))
    if not isinstance(data, dict) or data.get("error"):
        return []
    name = call.get("name")
    if name in LISTING_TOOLS:
        refs = []
        for row in data.get("listings") or []:
            if isinstance(row, dict) and row.get("listing_id"):
                refs.append({"listing_id": str(row["listing_id"])})
        return refs
    if name != "run_sql":
        return []
    columns = [str(column).lower() for column in data.get("columns") or []]
    rows = data.get("rows") or []
    if not columns or not isinstance(rows, list):
        return []

    def index(*names: str) -> int:
        for candidate in names:
            if candidate in columns:
                return columns.index(candidate)
        return -1

    id_at = index("listing_id")
    url_at = index("url")
    name_at = index("name", "building")
    unit_at = index("unit")
    refs = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        if id_at >= 0 and id_at < len(row) and row[id_at]:
            refs.append({"listing_id": str(row[id_at])})
        elif url_at >= 0 and url_at < len(row) and row[url_at]:
            refs.append({"url": str(row[url_at])})
        elif (
            name_at >= 0
            and unit_at >= 0
            and name_at < len(row)
            and unit_at < len(row)
            and row[name_at]
            and row[unit_at] not in (None, "")
        ):
            refs.append({"name": str(row[name_at]), "unit": str(row[unit_at])})
    return refs


def _resolve_refs(refs: list[dict]) -> list[str]:
    if not refs or not RENTALS_DB.exists():
        return []
    wanted_ids = [ref["listing_id"] for ref in refs if ref.get("listing_id")]
    wanted_urls = [ref["url"] for ref in refs if ref.get("url")]
    pairs = [(ref["name"], ref["unit"]) for ref in refs if ref.get("name") and ref.get("unit") is not None]
    conn = _rentals()
    try:
        known = set()
        if wanted_ids:
            marks = ",".join("?" * len(wanted_ids))
            known = {
                row["listing_id"]
                for row in conn.execute(
                    f"SELECT listing_id FROM listings WHERE listing_id IN ({marks})",
                    wanted_ids,
                )
            }
        url_map = {}
        if wanted_urls:
            marks = ",".join("?" * len(wanted_urls))
            url_map = {
                row["url"]: row["listing_id"]
                for row in conn.execute(
                    f"SELECT listing_id, url FROM listings WHERE url IN ({marks})",
                    wanted_urls,
                )
            }
        pair_map = {}
        for building, unit in pairs:
            row = conn.execute(
                """
                SELECT listing_id FROM listings
                WHERE lower(name) = lower(?) AND lower(unit) = lower(?)
                ORDER BY listing_id
                LIMIT 1
                """,
                (building, unit),
            ).fetchone()
            if row is not None:
                pair_map[(building.casefold(), unit.casefold())] = row["listing_id"]
    finally:
        conn.close()

    ordered = []
    seen = set()
    for ref in refs:
        listing_id = None
        if ref.get("listing_id") and ref["listing_id"] in known:
            listing_id = ref["listing_id"]
        elif ref.get("url"):
            listing_id = url_map.get(ref["url"])
        elif ref.get("name") and ref.get("unit") is not None:
            listing_id = pair_map.get((ref["name"].casefold(), ref["unit"].casefold()))
        if listing_id and listing_id not in seen:
            seen.add(listing_id)
            ordered.append(listing_id)
    return ordered


def classify(question: str) -> str:
    """Return 'single', 'grid', or 'auto' from the wording of the question."""
    if _SINGULAR.search(question or ""):
        return "single"
    if _OPTIONS.search(question or ""):
        return "grid"
    return "auto"


def _superlative(question: str) -> str | None:
    if _EXPENSIVE.search(question or ""):
        return "rent_desc"
    if _DEAL.search(question or ""):
        return "deal"
    if _CHEAP.search(question or ""):
        return "rent_asc"
    return None


def _metrics(ids: list[str]) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    conn = _rentals()
    try:
        rows = conn.execute(
            f"""
            SELECT l.listing_id, l.unit, l.rent, r.deal_score
            FROM listings l
            LEFT JOIN regression_results r ON r.listing_id = l.listing_id
            WHERE l.listing_id IN ({marks})
            """,
            ids,
        ).fetchall()
    finally:
        conn.close()
    return {row["listing_id"]: row for row in rows}


def _choose_single(question: str, groups: list[list[str]]) -> str:
    for group in reversed(groups):
        if len(group) == 1:
            return group[0]
    pool = max(groups, key=len)
    ordered = []
    seen = set()
    for group in groups:
        for listing_id in group:
            if listing_id not in seen:
                seen.add(listing_id)
                ordered.append(listing_id)
    metrics = _metrics(ordered)
    unit_match = _UNIT.search(question or "")
    if unit_match:
        token = unit_match.group(1).casefold()
        for listing_id in ordered:
            unit = str((metrics.get(listing_id) or {"unit": ""})["unit"] or "").casefold()
            if unit == token or unit.lstrip("0") == token.lstrip("0"):
                return listing_id
    kind = _superlative(question)
    ranked = [listing_id for listing_id in pool if listing_id in metrics] or pool

    def rent(listing_id: str):
        value = metrics[listing_id]["rent"] if listing_id in metrics else None
        return value

    def deal(listing_id: str):
        value = metrics[listing_id]["deal_score"] if listing_id in metrics else None
        return value

    if kind == "rent_desc":
        priced = [listing_id for listing_id in ranked if rent(listing_id) is not None]
        if priced:
            return max(priced, key=rent)
    elif kind == "deal":
        scored = [listing_id for listing_id in ranked if deal(listing_id) is not None]
        if scored:
            return max(scored, key=deal)
    elif kind == "rent_asc":
        priced = [listing_id for listing_id in ranked if rent(listing_id) is not None]
        if priced:
            return min(priced, key=rent)
    return ranked[0]


def resolve_display(question: str, tool_calls: list[dict]) -> dict:
    """Decide the card layout for one turn and load the listing cards."""
    groups = []
    try:
        for call in tool_calls or []:
            ids = _resolve_refs(_refs_from_call(call))
            if ids:
                groups.append(ids)
        if not groups:
            return {"display": "none", "listings": []}
        mode = classify(question)
        unique = []
        seen = set()
        for group in groups:
            for listing_id in group:
                if listing_id not in seen:
                    seen.add(listing_id)
                    unique.append(listing_id)
        single = mode == "single" or (mode == "auto" and len(unique) == 1)
        if single:
            cards = _cards([_choose_single(question, groups)])
            if not cards:
                return {"display": "none", "listings": []}
            return {"display": "single", "listings": cards}
        largest = max(groups, key=len)
        # A later set of the same size is the refined search.
        for group in groups:
            if len(group) == len(largest):
                largest = group
        cards = _cards(largest)
        if not cards:
            return {"display": "none", "listings": []}
        return {"display": "grid", "listings": cards}
    except sqlite3.Error:
        return {"display": "none", "listings": []}


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def _load_items(session_id: str) -> list[dict]:
    if not SESSIONS_DB.exists():
        return []
    conn = _sessions()
    try:
        rows = conn.execute(
            "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    items = []
    for row in rows:
        try:
            items.append(json.loads(row["message_data"]))
        except json.JSONDecodeError:
            continue
    return items


def _turns(items: list[dict]) -> list[dict]:
    turns = []
    current = None
    pending: dict[str, dict] = {}

    def close():
        nonlocal current
        if current is None:
            return
        presented = resolve_display(current["user"], current["tool_calls"])
        turns.append({
            "user": current["user"],
            "response": current["response"],
            "tool_calls": current["tool_calls"],
            "display": presented["display"],
            "listings": presented["listings"],
        })
        current = None

    for item in items:
        kind = item.get("type")
        if kind == "function_call":
            if current is None:
                current = {"user": "", "response": "", "tool_calls": []}
            args = item.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"query": args} if item.get("name") == "run_sql" else {}
            call = {
                "name": item.get("name") or "tool",
                "args": args if isinstance(args, dict) else {},
                "result": "",
            }
            current["tool_calls"].append(call)
            if item.get("call_id"):
                pending[item["call_id"]] = call
            continue
        if kind == "function_call_output":
            call = pending.get(item.get("call_id"))
            if call is not None:
                call["result"] = item.get("output") or ""
            continue
        if item.get("role") == "user":
            close()
            current = {"user": _text_content(item.get("content")), "response": "", "tool_calls": []}
            continue
        if item.get("role") == "assistant" or kind == "message":
            if current is None:
                current = {"user": "", "response": "", "tool_calls": []}
            text = _text_content(item.get("content"))
            current["response"] = (current["response"] + "\n\n" + text).strip() if current["response"] else text
    close()
    return [turn for turn in turns if turn["user"] or turn["response"] or turn["tool_calls"]]


def _title_from(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "Chat"
    if len(text) > 72:
        return text[:71].rstrip() + "…"
    return text


def list_threads() -> list[dict]:
    if not SESSIONS_DB.exists():
        return []
    conn = _sessions()
    try:
        rows = conn.execute(
            """
            SELECT s.session_id, s.updated_at, m.message_data
            FROM agent_sessions s
            JOIN agent_messages m ON m.session_id = s.session_id
            ORDER BY s.updated_at DESC, m.id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    threads = []
    seen = set()
    titles: dict[str, str] = {}
    updated: dict[str, str] = {}
    order = []
    for row in rows:
        session_id = row["session_id"]
        if session_id not in seen:
            seen.add(session_id)
            order.append(session_id)
            updated[session_id] = row["updated_at"]
        if session_id in titles:
            continue
        try:
            item = json.loads(row["message_data"])
        except json.JSONDecodeError:
            continue
        if item.get("role") == "user":
            titles[session_id] = _title_from(_text_content(item.get("content")))
    for session_id in order:
        threads.append({
            "session_id": session_id,
            "title": titles.get(session_id, "Chat"),
            "updated_at": updated.get(session_id),
        })
    return threads


def load_thread(session_id: str) -> dict | None:
    if not SESSIONS_DB.exists():
        return None
    conn = _sessions()
    try:
        row = conn.execute(
            "SELECT session_id FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    turns = _turns(_load_items(session_id))
    title = _title_from(next((turn["user"] for turn in turns if turn["user"]), ""))
    return {"session_id": session_id, "title": title, "turns": turns}


def _missing_last(value):
    return (value is None, value)


def list_favorites(sort: str) -> dict:
    chosen = sort if sort in SORTS else "recent"
    if not UI_DB.exists() or not RENTALS_DB.exists():
        return {"sort": chosen, "listings": []}
    conn = _ui()
    try:
        stamps = {
            row["listing_id"]: row["favorited_at"]
            for row in conn.execute("SELECT listing_id, favorited_at FROM favorites")
        }
    finally:
        conn.close()
    if not stamps:
        return {"sort": chosen, "listings": []}
    cards = _cards(list(stamps), saved=set(stamps))

    def neighborhood(card):
        return (card.get("neighborhood") or "", card.get("building") or "")

    def beds(card):
        return (card.get("beds") is None, card.get("beds") if card.get("beds") is not None else 0, card.get("rent") is None, card.get("rent") or 0)

    if chosen == "rent_asc":
        cards.sort(key=lambda card: _missing_last(card.get("rent")))
    elif chosen == "rent_desc":
        cards.sort(key=lambda card: (card.get("rent") is None, -(card.get("rent") or 0)))
    elif chosen == "neighborhood":
        cards.sort(key=neighborhood)
    elif chosen == "beds":
        cards.sort(key=beds)
    elif chosen == "deal":
        cards.sort(key=lambda card: (card.get("deal_score") is None, -(card.get("deal_score") or 0)))
    else:
        cards.sort(key=lambda card: stamps.get(card["listing_id"]) or "", reverse=True)
    return {"sort": chosen, "listings": cards}


def add_favorite(listing_id: str) -> dict | None:
    init_ui()
    cards = _cards([listing_id], saved={listing_id})
    if not cards:
        return None
    stamp = datetime.now(timezone.utc).isoformat()
    with _ui_lock:
        conn = _ui()
        try:
            conn.execute(
                """
                INSERT INTO favorites (listing_id, favorited_at) VALUES (?, ?)
                ON CONFLICT(listing_id) DO UPDATE SET favorited_at = excluded.favorited_at
                """,
                (listing_id, stamp),
            )
            conn.commit()
        finally:
            conn.close()
    cards[0]["favorited"] = True
    return cards[0]


def remove_favorite(listing_id: str) -> bool:
    if not UI_DB.exists():
        return False
    with _ui_lock:
        conn = _ui()
        try:
            cursor = conn.execute("DELETE FROM favorites WHERE listing_id = ?", (listing_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
