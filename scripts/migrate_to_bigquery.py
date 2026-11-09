"""Create the BigQuery dataset and replace its tables with the local SQLite files.

Run from the project root or from ``scripts/``:

    uv run scripts/migrate_to_bigquery.py

A second run truncates each table and loads it again, so prod matches the
local files at the moment the script runs. ``regression_features`` is created
from the columns currently in SQLite, including one-hot names such as
``nh_Hell's Kitchen``.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import bq

ROOT = Path(__file__).resolve().parent.parent
RENTALS = ROOT / "scripts" / "rentals.db"
CONVERSATIONS = ROOT / "conversations.db"
UI = ROOT / "ui.db"

RENTAL_TABLES = (
    "listings",
    "listing_amenities",
    "listing_images",
    "image_scores",
    "listing_scores",
    "regression_results",
)


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"No database at {path}.")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _load_fixed(conn: sqlite3.Connection, table: str) -> int:
    schema = bq.SCHEMAS[table]
    if not _table_exists(conn, table):
        bq.load_rows(table, [], schema)
        return 0
    names = [field.name for field in schema]
    quoted = ", ".join(f'"{name}"' for name in names)
    records = [dict(row) for row in conn.execute(f"SELECT {quoted} FROM {table}")]
    rows = bq.rows_for_schema(records, schema)
    return bq.load_rows(table, rows, schema)


def _load_features(conn: sqlite3.Connection) -> int:
    if not _table_exists(conn, "regression_features"):
        print("regression_features is missing locally; skipped.")
        return 0
    info = conn.execute("PRAGMA table_info(regression_features)").fetchall()
    source_names = []
    kinds = []
    for column in info:
        name, declared = column[1], (column[2] or "TEXT").upper()
        source_names.append(name)
        if "INT" in declared:
            kinds.append("INT64")
        elif any(token in declared for token in ("REAL", "FLOA", "DOUB")):
            kinds.append("FLOAT64")
        else:
            kinds.append("STRING")
    renamed = bq.legal_names(source_names)
    changed = [
        f"{old} -> {new}"
        for old, new in zip(source_names, renamed)
        if old != new
    ]
    if changed:
        print("  regression_features column names BigQuery cannot store:")
        for line in changed:
            print(f"    {line}")
    schema = [
        bq.bigquery.SchemaField(name, kind)
        for name, kind in zip(renamed, kinds)
    ]
    quoted = ", ".join(f'"{name}"' for name in source_names)
    records = []
    for row in conn.execute(f"SELECT {quoted} FROM regression_features"):
        records.append({
            legal: row[source]
            for source, legal in zip(source_names, renamed)
        })
    rows = bq.rows_for_schema(records, schema)
    return bq.load_rows("regression_features", rows, schema, replace_schema=True)


def main() -> int:
    print(f"Project {bq.project()}  dataset {bq.dataset()}  location {bq.location()}")
    bq.ensure_tables()

    rentals = _connect(RENTALS)
    try:
        for table in RENTAL_TABLES:
            loaded = _load_fixed(rentals, table)
            print(f"  {table}: loaded {loaded}")
        loaded = _load_features(rentals)
        print(f"  regression_features: loaded {loaded}")
    finally:
        rentals.close()

    conversations = _connect(CONVERSATIONS)
    try:
        for table in ("agent_sessions", "agent_messages"):
            loaded = _load_fixed(conversations, table)
            print(f"  {table}: loaded {loaded}")
    finally:
        conversations.close()

    ui = _connect(UI)
    try:
        loaded = _load_fixed(ui, "favorites")
        print(f"  favorites: loaded {loaded}")
    finally:
        ui.close()

    print("BigQuery row counts:")
    for table in (
        *RENTAL_TABLES,
        "regression_features",
        "agent_sessions",
        "agent_messages",
        "favorites",
    ):
        print(f"  {table}: {bq.count_rows(table)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
