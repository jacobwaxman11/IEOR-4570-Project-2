"""BigQuery dataset for the rental tables, chat history, and saved listings.

Local scripts keep writing SQLite, then mirror each write here when
``BQ_MIRROR`` is not ``0``. The app reads and writes this dataset only when
``APP_ENV=prod``. Otherwise it stays on the local SQLite files.

Dataset ids cannot contain hyphens, so the default id is
``IEOR_4570_Project_2``. Location defaults to ``US`` because
``GOOGLE_CLOUD_LOCATION`` is a Vertex region (often ``global``), which
BigQuery does not accept.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_DATASET = "IEOR_4570_Project_2"
DEFAULT_LOCATION = "US"

RENTAL_TABLES = frozenset({
    "listings",
    "listing_amenities",
    "listing_images",
    "image_scores",
    "listing_scores",
    "regression_results",
    "regression_features",
})


def _columns(*pairs: tuple[str, str]) -> list[bigquery.SchemaField]:
    return [bigquery.SchemaField(name, kind) for name, kind in pairs]


SCHEMAS: dict[str, list[bigquery.SchemaField]] = {
    "listings": _columns(
        ("listing_id", "STRING"),
        ("url", "STRING"),
        ("name", "STRING"),
        ("street", "STRING"),
        ("neighborhood", "STRING"),
        ("zip", "STRING"),
        ("lat", "FLOAT64"),
        ("lng", "FLOAT64"),
        ("beds", "FLOAT64"),
        ("baths", "FLOAT64"),
        ("sqft", "INT64"),
        ("rent", "INT64"),
        ("lease_months", "INT64"),
        ("furnished", "STRING"),
        ("building_type", "STRING"),
        ("building_slug", "STRING"),
        ("unit", "STRING"),
        ("description", "STRING"),
        ("available_from", "STRING"),
        ("floor_plan_url", "STRING"),
        ("local_floor_plan_path", "STRING"),
        ("listing_status", "STRING"),
        ("first_scraped_at", "STRING"),
        ("last_updated_date", "STRING"),
        ("detail_fetched_at", "STRING"),
    ),
    "listing_amenities": _columns(
        ("listing_id", "STRING"),
        ("amenity", "STRING"),
    ),
    "listing_images": _columns(
        ("listing_id", "STRING"),
        ("position", "INT64"),
        ("image_url", "STRING"),
        ("local_image_path", "STRING"),
    ),
    "image_scores": _columns(
        ("image_url", "STRING"),
        ("image_category", "STRING"),
        ("room_type", "STRING"),
        ("natural_light", "INT64"),
        ("space_feeling", "INT64"),
        ("view_quality", "INT64"),
        ("space_type", "STRING"),
        ("appeal", "INT64"),
        ("finish_quality", "INT64"),
        ("condition_score", "INT64"),
        ("irrelevant_reason", "STRING"),
        ("model", "STRING"),
        ("scored_at", "STRING"),
    ),
    "listing_scores": _columns(
        ("listing_id", "STRING"),
        ("avg_natural_light", "FLOAT64"),
        ("avg_finish_quality", "FLOAT64"),
        ("avg_space_feeling", "FLOAT64"),
        ("avg_condition", "FLOAT64"),
        ("max_view_quality", "INT64"),
        ("pct_bright_rooms", "FLOAT64"),
        ("has_good_view", "INT64"),
        ("common_avg_finish_quality", "FLOAT64"),
        ("common_avg_condition", "FLOAT64"),
        ("common_avg_appeal", "FLOAT64"),
        ("photos_total", "INT64"),
        ("photos_scored", "INT64"),
        ("photos_apartment", "INT64"),
        ("photos_common", "INT64"),
        ("photos_irrelevant", "INT64"),
        ("room_type_counts_json", "STRING"),
        ("space_type_counts_json", "STRING"),
        ("aggregated_at", "STRING"),
    ),
    "regression_results": _columns(
        ("listing_id", "STRING"),
        ("actual_rent", "INT64"),
        ("predicted_rent", "INT64"),
        ("percent_discount", "FLOAT64"),
        ("deal_score", "FLOAT64"),
        ("model", "STRING"),
        ("scored_at", "STRING"),
    ),
    "favorites": _columns(
        ("listing_id", "STRING"),
        ("favorited_at", "STRING"),
    ),
    "agent_sessions": _columns(
        ("session_id", "STRING"),
        ("created_at", "STRING"),
        ("updated_at", "STRING"),
    ),
    "agent_messages": _columns(
        ("id", "INT64"),
        ("session_id", "STRING"),
        ("message_data", "STRING"),
        ("created_at", "STRING"),
    ),
}

_LISTING_UPDATE = [
    field.name
    for field in SCHEMAS["listings"]
    if field.name not in {"listing_id", "first_scraped_at", "last_updated_date"}
]

_client: bigquery.Client | None = None


def prod() -> bool:
    """True when the app should use BigQuery instead of the local SQLite files.

    ``APP_ENV=prod`` selects BigQuery. Cloud Run sets ``K_SERVICE`` and does not
    ship ``rentals.db``, so an unset ``APP_ENV`` there is prod as well.
    ``APP_ENV=local`` stays on SQLite even on Cloud Run.
    """
    choice = os.environ.get("APP_ENV", "").strip().lower()
    if choice == "prod":
        return True
    if choice == "local":
        return False
    return bool(os.environ.get("K_SERVICE"))


def mirror_enabled() -> bool:
    """Scripts mirror SQLite writes unless ``BQ_MIRROR=0``."""
    return os.environ.get("BQ_MIRROR", "1").strip() != "0"


def project() -> str:
    value = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if value:
        return value
    import google.auth

    _, detected = google.auth.default()
    if not detected:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not set.")
    return detected


def dataset() -> str:
    return os.environ.get("BQ_DATASET", DEFAULT_DATASET).strip() or DEFAULT_DATASET


def location() -> str:
    return os.environ.get("BQ_LOCATION", DEFAULT_LOCATION).strip() or DEFAULT_LOCATION


def quote_ident(name: str) -> str:
    """Quote a table or column. Standard SQL treats double quotes as strings."""
    return "`" + name.replace("`", "``") + "`"


def bigquery_column(name: str) -> str:
    """Make a column name BigQuery will accept.

    Letters, numbers, and underscores only. An apostrophe is dropped so
    ``nh_Hell's Kitchen`` becomes ``nh_Hells_Kitchen``. Anything else, including
    spaces and slashes, becomes an underscore.
    """
    cleaned = name.replace("'", "")
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned[:300]


def legal_names(names: Sequence[str]) -> list[str]:
    """Return BigQuery column names, unique within ``names``."""
    used: set[str] = set()
    result: list[str] = []
    for name in names:
        base = bigquery_column(str(name))
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"[:300]
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=project(), location=location())
    return _client


def _table_ref(table: str) -> str:
    return f"{project()}.{dataset()}.{table}"


def _field_type(table: str, column: str) -> str:
    schema = SCHEMAS.get(table)
    if schema is None:
        raise KeyError(f"No fixed schema for {table}")
    for field in schema:
        if field.name == column:
            return field.field_type
    raise KeyError(f"No column {column} on {table}")


def coerce(value: Any, kind: str) -> Any:
    """Turn a SQLite or numpy value into a JSON value BigQuery will accept."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    if kind == "INT64":
        return int(value)
    if kind == "FLOAT64":
        return float(value)
    if kind == "BOOL":
        return bool(value)
    if isinstance(value, str):
        return value
    return str(value)


def _job_config(
    parameters: Sequence[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] = (),
) -> bigquery.QueryJobConfig:
    config = bigquery.QueryJobConfig(default_dataset=f"{project()}.{dataset()}")
    if parameters:
        config.query_parameters = list(parameters)
    return config


def bind_params(sql: str, params: Sequence[Any]) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Replace positional ``?`` placeholders with ``@p0``, ``@p1``, ..."""
    if not params:
        return sql, []
    pieces: list[str] = []
    index = 0
    for char in sql:
        if char != "?":
            pieces.append(char)
            continue
        if index >= len(params):
            raise ValueError(f"SQL has more '?' placeholders than parameters: {sql}")
        pieces.append(f"@p{index}")
        index += 1
    if index != len(params):
        raise ValueError(
            f"SQL has {index} '?' placeholders but {len(params)} parameters were passed."
        )
    bound = [
        bigquery.ScalarQueryParameter(f"p{i}", _param_type(value), _param_value(value))
        for i, value in enumerate(params)
    ]
    return "".join(pieces), bound


def _param_type(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    return "STRING"


def _param_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return None if value != value else float(value)
    return value if isinstance(value, str) else str(value)


def query(
    sql: str,
    params: Sequence[Any] = (),
    *,
    max_rows: int | None = None,
) -> tuple[list[str], list[list[Any]]]:
    """Run a read and return ``(column_names, rows)``."""
    bound, parameters = bind_params(sql, params)
    job = client().query(bound, job_config=_job_config(parameters))
    result = job.result(max_results=max_rows)
    columns = [field.name for field in result.schema]
    rows = [[record[name] for name in columns] for record in result]
    return columns, rows


def query_dicts(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    columns, rows = query(sql, params)
    return [dict(zip(columns, row)) for row in rows]


def execute(sql: str, params: Sequence[Any] = ()) -> int:
    """Run one DML statement and return how many rows it changed."""
    bound, parameters = bind_params(sql, params)
    job = client().query(bound, job_config=_job_config(parameters))
    job.result()
    return int(job.num_dml_affected_rows or 0)


def ensure_dataset() -> None:
    ref = bigquery.Dataset(f"{project()}.{dataset()}")
    ref.location = location()
    client().create_dataset(ref, exists_ok=True)


def ensure_tables() -> None:
    """Create the fixed tables. ``regression_features`` is loaded with its own columns."""
    ensure_dataset()
    bq = client()
    for name, schema in SCHEMAS.items():
        table = bigquery.Table(_table_ref(name), schema=schema)
        bq.create_table(table, exists_ok=True)


def load_rows(
    table: str,
    rows: list[dict[str, Any]],
    schema: list[bigquery.SchemaField],
    *,
    replace_schema: bool = False,
) -> int:
    """Replace ``table`` with ``rows``. An empty list leaves an empty table."""
    ensure_dataset()
    bq = client()
    ref = _table_ref(table)
    if replace_schema:
        bq.delete_table(ref, not_found_ok=True)
    if not rows:
        bq.create_table(bigquery.Table(ref, schema=schema), exists_ok=True)
        execute(f"DELETE FROM {quote_ident(table)} WHERE TRUE")
        return 0

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    path = _write_jsonl(rows)
    try:
        with open(path, "rb") as handle:
            job = bq.load_table_from_file(handle, ref, job_config=job_config)
        job.result()
    finally:
        Path(path).unlink(missing_ok=True)
    return len(rows)


def _write_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    with handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False))
            handle.write("\n")
    return handle.name


def rows_for_schema(
    records: Iterable[dict[str, Any]],
    schema: list[bigquery.SchemaField],
) -> list[dict[str, Any]]:
    kinds = {field.name: field.field_type for field in schema}
    names = [field.name for field in schema]
    return [
        {name: coerce(record.get(name), kinds[name]) for name in names}
        for record in records
    ]


def count_rows(table: str) -> int:
    columns, rows = query(f"SELECT COUNT(*) AS n FROM {quote_ident(table)}")
    return int(rows[0][columns.index("n")])


def merge_row(
    table: str,
    keys: list[str],
    row: dict[str, Any],
    update_columns: list[str] | None = None,
) -> None:
    """Insert ``row`` or update it when ``keys`` match. Other columns stay put on update
    when ``update_columns`` is set.
    """
    schema = SCHEMAS[table]
    kinds = {field.name: field.field_type for field in schema}
    columns = [field.name for field in schema]
    stored = {name: coerce(row.get(name), kinds[name]) for name in columns}
    updates = update_columns if update_columns is not None else [name for name in columns if name not in keys]
    if not updates:
        raise ValueError(f"MERGE into {table} has no columns to update.")

    select_list = ", ".join(f"@{name} AS {quote_ident(name)}" for name in columns)
    on = " AND ".join(f"T.{quote_ident(key)} = S.{quote_ident(key)}" for key in keys)
    set_clause = ", ".join(f"{quote_ident(name)} = S.{quote_ident(name)}" for name in updates)
    insert_cols = ", ".join(quote_ident(name) for name in columns)
    insert_vals = ", ".join(f"S.{quote_ident(name)}" for name in columns)
    sql = f"""
        MERGE {quote_ident(table)} T
        USING (SELECT {select_list}) S
        ON {on}
        WHEN MATCHED THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """
    parameters = [
        bigquery.ScalarQueryParameter(name, kinds[name], stored[name]) for name in columns
    ]
    job = client().query(sql, job_config=_job_config(parameters))
    job.result()


def update_fields(table: str, key_column: str, key_value: Any, fields: dict[str, Any]) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{quote_ident(name)} = @{name}" for name in fields)
    sql = (
        f"UPDATE {quote_ident(table)} SET {assignments} "
        f"WHERE {quote_ident(key_column)} = @key"
    )
    parameters = [
        bigquery.ScalarQueryParameter(
            name, _field_type(table, name), coerce(value, _field_type(table, name))
        )
        for name, value in fields.items()
    ]
    key_type = _field_type(table, key_column)
    parameters.append(
        bigquery.ScalarQueryParameter("key", key_type, coerce(key_value, key_type))
    )
    job = client().query(sql, job_config=_job_config(parameters))
    job.result()


def mirror_upsert_listing(listing: dict[str, Any], scraped_at: str) -> None:
    if not mirror_enabled():
        return
    ensure_tables()
    row = {field.name: listing.get(field.name) for field in SCHEMAS["listings"]}
    row["first_scraped_at"] = scraped_at
    row["last_updated_date"] = scraped_at
    merge_row("listings", ["listing_id"], row, update_columns=_LISTING_UPDATE)


def mirror_update_listing(listing_id: str, fields: dict[str, Any]) -> None:
    if not mirror_enabled() or not fields:
        return
    ensure_tables()
    update_fields("listings", "listing_id", listing_id, fields)


def replace_amenities(listing_id: str, amenities: list[str]) -> None:
    if not mirror_enabled():
        return
    ensure_tables()
    execute(f"DELETE FROM {quote_ident('listing_amenities')} WHERE listing_id = ?", (listing_id,))
    clean: list[str] = []
    for amenity in amenities:
        if amenity and amenity not in clean:
            clean.append(amenity)
    if not clean:
        return
    sql = f"""
        INSERT INTO {quote_ident('listing_amenities')} (listing_id, amenity)
        SELECT @listing_id, amenity FROM UNNEST(@amenities) AS amenity
    """
    parameters = [
        bigquery.ScalarQueryParameter("listing_id", "STRING", listing_id),
        bigquery.ArrayQueryParameter("amenities", "STRING", clean),
    ]
    client().query(sql, job_config=_job_config(parameters)).result()


def replace_images(
    listing_id: str,
    image_urls: list[str],
    local_image_paths: list[str | None],
) -> None:
    if not mirror_enabled():
        return
    ensure_tables()
    execute(f"DELETE FROM {quote_ident('listing_images')} WHERE listing_id = ?", (listing_id,))
    if not image_urls:
        return
    paths = list(local_image_paths) + [None] * (len(image_urls) - len(local_image_paths))
    sql = f"""
        INSERT INTO {quote_ident('listing_images')}
            (listing_id, position, image_url, local_image_path)
        SELECT @listing_id, position, image_url, local_image_path
        FROM UNNEST(@positions) AS position WITH OFFSET i
        JOIN UNNEST(@urls) AS image_url WITH OFFSET j ON i = j
        JOIN UNNEST(@paths) AS local_image_path WITH OFFSET k ON i = k
    """
    parameters = [
        bigquery.ScalarQueryParameter("listing_id", "STRING", listing_id),
        bigquery.ArrayQueryParameter("positions", "INT64", list(range(len(image_urls)))),
        bigquery.ArrayQueryParameter("urls", "STRING", list(image_urls)),
        bigquery.ArrayQueryParameter("paths", "STRING", paths[: len(image_urls)]),
    ]
    client().query(sql, job_config=_job_config(parameters)).result()


def mirror_image_score(row: dict[str, Any]) -> None:
    if not mirror_enabled():
        return
    ensure_tables()
    merge_row("image_scores", ["image_url"], row)


def mirror_listing_scores(row: dict[str, Any]) -> None:
    if not mirror_enabled():
        return
    ensure_tables()
    merge_row("listing_scores", ["listing_id"], row)


def replace_regression(features: Any, results: Any) -> None:
    """Replace the feature frame and the deal-score table. ``features`` and
    ``results`` are pandas DataFrames from ``features.py``.
    """
    if not mirror_enabled():
        return
    ensure_tables()
    feature_schema, feature_rows = _frame_payload(features)
    load_rows("regression_features", feature_rows, feature_schema, replace_schema=True)
    result_rows = rows_for_schema(results.to_dict(orient="records"), SCHEMAS["regression_results"])
    load_rows("regression_results", result_rows, SCHEMAS["regression_results"])


def _frame_payload(frame: Any) -> tuple[list[bigquery.SchemaField], list[dict[str, Any]]]:
    import pandas as pd

    original = [str(name) for name in frame.columns]
    renamed = legal_names(original)
    schema: list[bigquery.SchemaField] = []
    kinds: dict[str, str] = {}
    for source, name in zip(original, renamed):
        kind = _series_kind(frame[source])
        kinds[name] = kind
        schema.append(bigquery.SchemaField(name, kind))
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append({
            legal: coerce(_plain(record[source], pd), kinds[legal])
            for source, legal in zip(original, renamed)
        })
    return schema, rows


def _plain(value: Any, pd: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _series_kind(series: Any) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return "BOOL"
    if pd.api.types.is_integer_dtype(series):
        return "INT64"
    if pd.api.types.is_float_dtype(series):
        values = series.dropna()
        if len(values) and (values % 1 == 0).all():
            return "INT64"
        return "FLOAT64"
    return "STRING"


def upsert_favorite(listing_id: str, favorited_at: str) -> None:
    ensure_tables()
    merge_row(
        "favorites",
        ["listing_id"],
        {"listing_id": listing_id, "favorited_at": favorited_at},
    )


def delete_favorite(listing_id: str) -> bool:
    ensure_tables()
    changed = execute(
        f"DELETE FROM {quote_ident('favorites')} WHERE listing_id = ?",
        (listing_id,),
    )
    return changed > 0


def touch_session(session_id: str, now: str) -> None:
    """Insert the session, or bump ``updated_at`` and leave ``created_at`` alone."""
    ensure_tables()
    sql = f"""
        MERGE {quote_ident('agent_sessions')} T
        USING (
            SELECT @session_id AS session_id, @now AS created_at, @now AS updated_at
        ) S
        ON T.session_id = S.session_id
        WHEN MATCHED THEN UPDATE SET updated_at = S.updated_at
        WHEN NOT MATCHED THEN INSERT (session_id, created_at, updated_at)
            VALUES (S.session_id, S.created_at, S.updated_at)
    """
    parameters = [
        bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
        bigquery.ScalarQueryParameter("now", "STRING", now),
    ]
    client().query(sql, job_config=_job_config(parameters)).result()


def next_message_id() -> int:
    """First id for a new message. BigQuery has no autoincrement."""
    columns, rows = query(f"SELECT IFNULL(MAX(id), 0) AS n FROM {quote_ident('agent_messages')}")
    return int(rows[0][columns.index("n")]) + 1 if rows else 1


def insert_messages(session_id: str, messages: list[str], now: str) -> None:
    if not messages:
        return
    ensure_tables()
    start = next_message_id()
    ids = list(range(start, start + len(messages)))
    sql = f"""
        INSERT INTO {quote_ident('agent_messages')} (id, session_id, message_data, created_at)
        SELECT id, @session_id, message_data, @created_at
        FROM UNNEST(@ids) AS id WITH OFFSET i
        JOIN UNNEST(@messages) AS message_data WITH OFFSET j ON i = j
    """
    parameters = [
        bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
        bigquery.ScalarQueryParameter("created_at", "STRING", now),
        bigquery.ArrayQueryParameter("ids", "INT64", ids),
        bigquery.ArrayQueryParameter("messages", "STRING", messages),
    ]
    client().query(sql, job_config=_job_config(parameters)).result()


def latest_message(session_id: str) -> tuple[int, str] | None:
    rows = query_dicts(
        f"""
        SELECT id, message_data
        FROM {quote_ident('agent_messages')}
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (session_id,),
    )
    if not rows:
        return None
    return int(rows[0]["id"]), rows[0]["message_data"]


def delete_message(session_id: str, message_id: int) -> None:
    execute(
        f"DELETE FROM {quote_ident('agent_messages')} WHERE session_id = ? AND id = ?",
        (session_id, message_id),
    )


def clear_session(session_id: str) -> None:
    ensure_tables()
    execute(
        f"DELETE FROM {quote_ident('agent_messages')} WHERE session_id = ?",
        (session_id,),
    )
    execute(
        f"DELETE FROM {quote_ident('agent_sessions')} WHERE session_id = ?",
        (session_id,),
    )
