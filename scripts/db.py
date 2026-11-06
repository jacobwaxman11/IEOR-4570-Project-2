"""SQLite schema + helpers for the NYCRentalRankings dataset.

The database is the single source of truth. Each script (scraper, backfill,
scorer) reads/writes through the helpers here instead of through JSON files.

Tables:
  - listings           : one row per StreetEasy listing
  - listing_amenities  : normalized amenity tags
  - listing_images     : per-listing image references (positional)
  - image_scores       : Gemini scores keyed by image_url (one per unique image)
  - listing_scores     : aggregated scores per listing (rebuildable from image_scores)

Design notes:
  - `listing_id` is the canonical key (building_slug + unit).
  - `image_url` is the canonical image key. The same URL can be referenced by
    multiple listings (shared building photo) but is scored only once.
  - listing_amenities + listing_images are "child" tables: ON DELETE CASCADE so
    `DELETE FROM listings ...` cleans up automatically.
  - `image_scores` and `listing_scores` are NOT cascaded — keeping them across
    listing deletions would make any future revival cheap.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional


DEFAULT_DB_PATH = "rentals.db"


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS listings (
    listing_id              TEXT PRIMARY KEY,
    url                     TEXT,
    name                    TEXT,
    street                  TEXT,
    neighborhood            TEXT,
    zip                     TEXT,
    lat                     REAL,
    lng                     REAL,
    beds                    REAL,
    baths                   REAL,
    sqft                    INTEGER,
    rent                    INTEGER,
    lease_months            INTEGER,
    furnished               TEXT,
    building_type           TEXT,
    building_slug           TEXT,
    unit                    TEXT,
    description             TEXT,
    available_from          TEXT,
    floor_plan_url          TEXT,
    local_floor_plan_path   TEXT,
    listing_status          TEXT,    -- StreetEasy status: ACTIVE, RENTED, IN_CONTRACT, DELISTED, ...
    first_scraped_at        TEXT,
    last_updated_date       TEXT,    -- as-of date of the rent and status currently stored
    detail_fetched_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_listings_neighborhood ON listings(neighborhood);
CREATE INDEX IF NOT EXISTS idx_listings_building     ON listings(building_slug);

CREATE TABLE IF NOT EXISTS listing_amenities (
    listing_id  TEXT NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    amenity     TEXT NOT NULL,
    PRIMARY KEY (listing_id, amenity)
);
CREATE INDEX IF NOT EXISTS idx_amenity ON listing_amenities(amenity);

CREATE TABLE IF NOT EXISTS listing_images (
    listing_id        TEXT NOT NULL REFERENCES listings(listing_id) ON DELETE CASCADE,
    position          INTEGER NOT NULL,
    image_url         TEXT NOT NULL,
    local_image_path  TEXT,
    PRIMARY KEY (listing_id, position)
);
CREATE INDEX IF NOT EXISTS idx_listing_images_url ON listing_images(image_url);

CREATE TABLE IF NOT EXISTS image_scores (
    image_url         TEXT PRIMARY KEY,
    image_category    TEXT,    -- 'apartment' | 'common_space' | 'irrelevant'
    -- apartment-only fields
    room_type         TEXT,
    natural_light     INTEGER,
    space_feeling     INTEGER,
    view_quality      INTEGER,
    -- common-space-only fields
    space_type        TEXT,
    appeal            INTEGER,
    -- shared between apartment + common_space
    finish_quality    INTEGER,
    condition_score   INTEGER,
    -- irrelevant-only
    irrelevant_reason TEXT,
    -- bookkeeping
    model             TEXT,
    scored_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_image_scores_category ON image_scores(image_category);
CREATE INDEX IF NOT EXISTS idx_image_scores_room     ON image_scores(room_type);
CREATE INDEX IF NOT EXISTS idx_image_scores_space    ON image_scores(space_type);

CREATE TABLE IF NOT EXISTS listing_scores (
    listing_id              TEXT PRIMARY KEY REFERENCES listings(listing_id) ON DELETE CASCADE,
    -- apartment aggregates
    avg_natural_light       REAL,
    avg_finish_quality      REAL,
    avg_space_feeling       REAL,
    avg_condition           REAL,
    max_view_quality        INTEGER,
    pct_bright_rooms        REAL,
    has_good_view           INTEGER,  -- 0/1
    -- common-space aggregates
    common_avg_finish_quality REAL,
    common_avg_condition      REAL,
    common_avg_appeal         REAL,
    -- counts
    photos_total            INTEGER,
    photos_scored           INTEGER,
    photos_apartment        INTEGER,
    photos_common           INTEGER,
    photos_irrelevant       INTEGER,
    -- nested distributions kept as JSON (small, not feature-y)
    room_type_counts_json   TEXT,
    space_type_counts_json  TEXT,
    aggregated_at           TEXT
);
"""


# ── connection ────────────────────────────────────────────────────────────────


def connect(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _listing_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(listings)")}


def migrate_listings(conn: sqlite3.Connection) -> None:
    """Bring an existing rentals.db up to the current listings columns.

    ``CREATE TABLE IF NOT EXISTS`` does not add columns to a table that was
    created by an older version of this file. Renames ``scraped_at`` and
    backfills ``last_updated_date`` from it so a listing scraped today is not
    treated as stale.
    """
    cols = _listing_columns(conn)
    if not cols:
        return
    if "scraped_at" in cols and "first_scraped_at" not in cols:
        conn.execute("ALTER TABLE listings RENAME COLUMN scraped_at TO first_scraped_at")
        cols.discard("scraped_at")
        cols.add("first_scraped_at")
    if "listing_status" not in cols:
        conn.execute("ALTER TABLE listings ADD COLUMN listing_status TEXT")
    if "first_scraped_at" not in cols:
        conn.execute("ALTER TABLE listings ADD COLUMN first_scraped_at TEXT")
    if "last_updated_date" not in cols:
        conn.execute("ALTER TABLE listings ADD COLUMN last_updated_date TEXT")
    conn.execute(
        """
        UPDATE listings
        SET last_updated_date = first_scraped_at
        WHERE last_updated_date IS NULL
          AND first_scraped_at IS NOT NULL
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_listings_last_updated ON listings(last_updated_date)"
    )


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    migrate_listings(conn)
    conn.commit()


@contextmanager
def open_db(path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    init_schema(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── listings ──────────────────────────────────────────────────────────────────

LISTING_COLS = (
    "listing_id", "url", "name", "street", "neighborhood", "zip", "lat", "lng",
    "beds", "baths", "sqft", "rent", "lease_months", "furnished", "building_type",
    "building_slug", "unit", "description", "available_from", "listing_status",
    "floor_plan_url", "local_floor_plan_path",
)


def upsert_listing(conn: sqlite3.Connection, listing: dict) -> None:
    """Insert or update a listing's core columns. Does not touch amenities or
    images — those have their own helpers."""
    values = [listing.get(c) for c in LISTING_COLS]
    placeholders = ",".join("?" * len(LISTING_COLS))
    cols = ",".join(LISTING_COLS)

    set_clause = ",".join(f"{c}=excluded.{c}" for c in LISTING_COLS if c != "listing_id")

    conn.execute(
        f"""
        INSERT INTO listings ({cols}, first_scraped_at, last_updated_date)
        VALUES ({placeholders}, ?, ?)
        ON CONFLICT(listing_id) DO UPDATE SET {set_clause}
        """,
        values + [_utcnow(), _utcnow()],
    )


def update_listing_fields(
    conn: sqlite3.Connection, listing_id: str, fields: dict
) -> None:
    """Update specific columns on an existing listing (used by backfill)."""
    if not fields:
        return
    cols = ",".join(f"{k}=?" for k in fields)
    conn.execute(
        f"UPDATE listings SET {cols} WHERE listing_id=?",
        list(fields.values()) + [listing_id],
    )


def listings_due_for_refresh(
    conn: sqlite3.Connection, stale_before: str, limit: Optional[int] = None
) -> list[dict]:
    """Listings whose rent and status are older than ``stale_before``.

    ``last_updated_date`` is the as-of date. Rows that have never been
    refreshed (NULL) are included first, then the oldest dates. ``limit``
    caps how many StreetEasy detail pages a refresh run will request.
    """
    sql = """
        SELECT * FROM listings
        WHERE last_updated_date IS NULL OR last_updated_date < ?
        ORDER BY last_updated_date IS NOT NULL, last_updated_date, listing_id
    """
    params: list[Any] = [stale_before]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(r) for r in conn.execute(sql, params)]


def refresh_listing_market(
    conn: sqlite3.Connection,
    listing_id: str,
    *,
    rent: Optional[int] = None,
    listing_status: Optional[str] = None,
) -> None:
    """Write a fresh rent and/or status and stamp ``last_updated_date``.

    ``first_scraped_at`` is left alone. Rent or status that could not be
    read (None) are not overwritten.
    """
    fields: dict[str, Any] = {"last_updated_date": _utcnow()}
    if rent is not None:
        fields["rent"] = rent
    if listing_status is not None:
        fields["listing_status"] = listing_status
    update_listing_fields(conn, listing_id, fields)


def mark_detail_fetched(conn: sqlite3.Connection, listing_id: str) -> None:
    conn.execute(
        "UPDATE listings SET detail_fetched_at=? WHERE listing_id=?",
        (_utcnow(), listing_id),
    )


def get_listing(conn: sqlite3.Connection, listing_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM listings WHERE listing_id=?", (listing_id,)
    ).fetchone()
    return dict(row) if row else None


def get_listing_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT listing_id FROM listings")}


def listings_missing_amenities(conn: sqlite3.Connection) -> list[dict]:
    """Listings whose amenity table is empty — i.e. detail page hasn't been
    backfilled yet."""
    rows = conn.execute(
        """
        SELECT l.*
        FROM listings l
        LEFT JOIN listing_amenities a ON l.listing_id = a.listing_id
        WHERE a.amenity IS NULL
        ORDER BY l.listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def listings_missing_scores(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT l.*
        FROM listings l
        LEFT JOIN listing_scores s ON l.listing_id = s.listing_id
        WHERE s.listing_id IS NULL
        ORDER BY l.listing_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


# ── amenities ────────────────────────────────────────────────────────────────


def set_amenities(
    conn: sqlite3.Connection, listing_id: str, amenities: Iterable[str]
) -> None:
    conn.execute("DELETE FROM listing_amenities WHERE listing_id=?", (listing_id,))
    rows = [(listing_id, a) for a in amenities if a]
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO listing_amenities (listing_id, amenity) VALUES (?, ?)",
            rows,
        )


def get_amenities(conn: sqlite3.Connection, listing_id: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT amenity FROM listing_amenities WHERE listing_id=? ORDER BY amenity",
            (listing_id,),
        )
    ]


# ── images ───────────────────────────────────────────────────────────────────


def set_listing_images(
    conn: sqlite3.Connection,
    listing_id: str,
    image_urls: list[str],
    local_image_paths: Optional[list[str]] = None,
) -> None:
    """Replace the entire image list for a listing."""
    conn.execute("DELETE FROM listing_images WHERE listing_id=?", (listing_id,))
    paths = local_image_paths or []
    rows = [
        (
            listing_id,
            i,
            url,
            paths[i] if i < len(paths) else None,
        )
        for i, url in enumerate(image_urls)
    ]
    if rows:
        conn.executemany(
            "INSERT INTO listing_images (listing_id, position, image_url, local_image_path) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )


def listings_needing_image_download(conn: sqlite3.Connection) -> list[str]:
    """Listings whose photo URLs are unusable or whose files were never saved.

    StreetEasy's search JSON currently emits ``https://undefined/fp/...``. The
    path is real, but the host is not, so those rows have to be rewritten and
    downloaded before they can be scored.
    """
    return [
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT listing_id
            FROM listing_images
            WHERE image_url LIKE 'https://undefined/%'
               OR image_url LIKE 'http://undefined/%'
               OR local_image_path IS NULL
            ORDER BY listing_id
            """
        )
    ]


def get_listing_images(conn: sqlite3.Connection, listing_id: str) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT position, image_url, local_image_path "
            "FROM listing_images WHERE listing_id=? ORDER BY position",
            (listing_id,),
        )
    ]


# ── image scores ─────────────────────────────────────────────────────────────


def get_existing_image_scores(
    conn: sqlite3.Connection, image_urls: Iterable[str]
) -> dict[str, dict]:
    """Return {image_url: row_dict} for any URLs we've already scored."""
    urls = list(image_urls)
    if not urls:
        return {}
    placeholders = ",".join("?" * len(urls))
    rows = conn.execute(
        f"SELECT * FROM image_scores WHERE image_url IN ({placeholders})", urls
    ).fetchall()
    return {r["image_url"]: dict(r) for r in rows}


def upsert_image_score(
    conn: sqlite3.Connection,
    image_url: str,
    score: dict,
    model: str,
) -> None:
    """Persist a Gemini score for one image. ``score`` is the full nested dict
    returned by the model (with image_category, apartment_scores,
    common_space_scores, irrelevant_reason)."""
    apt = score.get("apartment_scores") or {}
    com = score.get("common_space_scores") or {}

    # The "shared" fields (finish_quality, condition) come from whichever sub-
    # object is populated. apt and com are mutually exclusive per the schema.
    finish = apt.get("finish_quality") if apt else com.get("finish_quality")
    cond = apt.get("condition") if apt else com.get("condition")

    conn.execute(
        """
        INSERT OR REPLACE INTO image_scores (
            image_url, image_category,
            room_type, natural_light, space_feeling, view_quality,
            space_type, appeal,
            finish_quality, condition_score,
            irrelevant_reason,
            model, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_url,
            score.get("image_category"),
            apt.get("room_type"),
            apt.get("natural_light"),
            apt.get("space_feeling"),
            apt.get("view_quality"),
            com.get("space_type"),
            com.get("appeal"),
            finish,
            cond,
            score.get("irrelevant_reason"),
            model,
            _utcnow(),
        ),
    )


def image_score_to_dict(row: dict) -> dict:
    """Convert a flat image_scores row back into the nested shape the
    aggregator expects."""
    cat = row.get("image_category")
    out: dict[str, Any] = {
        "image_category": cat,
        "apartment_scores": None,
        "common_space_scores": None,
        "irrelevant_reason": row.get("irrelevant_reason"),
    }
    if cat == "apartment":
        out["apartment_scores"] = {
            "room_type": row.get("room_type"),
            "natural_light": row.get("natural_light"),
            "finish_quality": row.get("finish_quality"),
            "space_feeling": row.get("space_feeling"),
            "view_quality": row.get("view_quality"),
            "condition": row.get("condition_score"),
        }
    elif cat == "common_space":
        out["common_space_scores"] = {
            "space_type": row.get("space_type"),
            "finish_quality": row.get("finish_quality"),
            "condition": row.get("condition_score"),
            "appeal": row.get("appeal"),
        }
    return out


# ── listing-level aggregated scores ──────────────────────────────────────────


LISTING_SCORE_COLS = (
    "avg_natural_light", "avg_finish_quality", "avg_space_feeling", "avg_condition",
    "max_view_quality", "pct_bright_rooms", "has_good_view",
    "common_avg_finish_quality", "common_avg_condition", "common_avg_appeal",
    "photos_total", "photos_scored", "photos_apartment", "photos_common",
    "photos_irrelevant",
)


def upsert_listing_scores(
    conn: sqlite3.Connection, listing_id: str, summary: dict
) -> None:
    """Persist aggregated per-listing scores (output of aggregate_scores)."""
    apt = summary.get("apartment") or {}
    com = summary.get("common_space") or {}

    values = (
        listing_id,
        apt.get("avg_natural_light"),
        apt.get("avg_finish_quality"),
        apt.get("avg_space_feeling"),
        apt.get("avg_condition"),
        apt.get("max_view_quality"),
        apt.get("pct_bright_rooms"),
        1 if apt.get("has_good_view") else 0,
        com.get("avg_finish_quality"),
        com.get("avg_condition"),
        com.get("avg_appeal"),
        summary.get("photos_total"),
        summary.get("photos_scored"),
        summary.get("photos_apartment"),
        summary.get("photos_common"),
        summary.get("photos_irrelevant"),
        json.dumps(apt.get("room_type_counts") or {}),
        json.dumps(com.get("space_type_counts") or {}),
        _utcnow(),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO listing_scores (
            listing_id,
            avg_natural_light, avg_finish_quality, avg_space_feeling, avg_condition,
            max_view_quality, pct_bright_rooms, has_good_view,
            common_avg_finish_quality, common_avg_condition, common_avg_appeal,
            photos_total, photos_scored, photos_apartment, photos_common,
            photos_irrelevant,
            room_type_counts_json, space_type_counts_json,
            aggregated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )


# ── DB-level summary (useful for CLI tools) ──────────────────────────────────


def stats(conn: sqlite3.Connection) -> dict:
    def cnt(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    return {
        "listings": cnt("SELECT COUNT(*) FROM listings"),
        "listings_with_amenities": cnt(
            "SELECT COUNT(DISTINCT listing_id) FROM listing_amenities"
        ),
        "listings_with_scores": cnt("SELECT COUNT(*) FROM listing_scores"),
        "image_refs": cnt("SELECT COUNT(*) FROM listing_images"),
        "unique_images": cnt("SELECT COUNT(DISTINCT image_url) FROM listing_images"),
        "scored_images": cnt("SELECT COUNT(*) FROM image_scores"),
    }
