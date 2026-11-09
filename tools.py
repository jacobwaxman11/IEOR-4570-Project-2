"""The tools the harness can run, and the JSON that describes them to the model."""

import functools
import json
import os
import re
import sqlite3
import statistics
import threading
import time
from pathlib import Path

import requests
from agents import ToolGuardrailFunctionOutput, function_tool, tool_input_guardrail
from dotenv import load_dotenv

from scripts import bq

load_dotenv()

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

CACHE_TTL_SECONDS = 60
METERS_PER_MILE = 1609.344

# (lat, lng, radius meters). Bias keeps a query like "pizza" inside New York.
BOROUGH_CENTERS = {
    "manhattan": (40.7831, -73.9712, 8000),
    "brooklyn": (40.6782, -73.9442, 9000),
    "queens": (40.7282, -73.7949, 12000),
    "bronx": (40.8448, -73.8648, 8000),
    "staten island": (40.5795, -74.1502, 10000),
}
CITY_CENTER = (40.7128, -74.0060, 20000)

TRAVEL_MODES = {"transit": "TRANSIT", "walk": "WALK", "drive": "DRIVE", "bicycle": "BICYCLE"}
PRICE_LEVELS = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}


class ToolError(Exception):
    """A failure the model should see as {"error": ...} instead of a crash."""


# (url, body) -> (fetched_at, data). Keeps repeat calls inside the API quota.
_cache: dict[tuple, tuple[float, object]] = {}


def _post(service: str, url: str, body: dict, headers: dict) -> object:
    key = (url, json.dumps(body, sort_keys=True))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except requests.HTTPError as e:
        detail = e.response.text[:200] if e.response is not None else str(e)
        status = e.response.status_code if e.response is not None else "?"
        raise ToolError(f"{service} request failed ({status}): {detail}")
    except (requests.RequestException, ValueError) as e:
        raise ToolError(f"{service} request failed: {e}")
    _cache[key] = (time.time(), payload)
    return payload


def _require_google_key() -> None:
    if not GOOGLE_MAPS_API_KEY:
        raise ToolError("GOOGLE_MAPS_API_KEY is not set in .env")


def _tool(fn):
    """Turn ToolErrors into the {"error": ...} JSON the model can reason about."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return json.dumps(fn(*args, **kwargs))
        except ToolError as e:
            return json.dumps({"error": str(e)})
    return wrapper


def _clip(text, limit: int = 240):
    if not text:
        return None
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _text(value):
    """Places and Routes sometimes return a string and sometimes {text: ...}."""
    if isinstance(value, dict):
        return value.get("text") or value.get("name")
    return value if isinstance(value, str) else None


def _clamp_int(value, default: int, low: int, high: int) -> int:
    if value is None or value == "":
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"Expected an integer, got {value!r}")
    return max(low, min(value, high))


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _seconds(value) -> int:
    """Routes durations look like '123s'."""
    if value is None:
        return 0
    text = str(value).rstrip("s")
    try:
        return int(float(text))
    except ValueError:
        return 0


def _minutes(seconds: int) -> int:
    if seconds <= 0:
        return 0
    return max(1, round(seconds / 60))


def _miles(meters) -> float:
    miles = (meters or 0) / METERS_PER_MILE
    return round(miles, 2) if miles < 1 else round(miles, 1)


def _bias(location: str) -> dict:
    lowered = location.lower()
    for name, (lat, lng, radius) in BOROUGH_CENTERS.items():
        if name in lowered:
            break
    else:
        lat, lng, radius = CITY_CENTER
    return {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": float(radius)}}


def _drop_empty(row: dict) -> dict:
    return {key: value for key, value in row.items() if value not in (None, "", [])}


def _google_headers(field_mask: str) -> dict:
    return {"X-Goog-Api-Key": GOOGLE_MAPS_API_KEY, "X-Goog-FieldMask": field_mask}


# --- Tools ---

@_tool
def search_places(query: str, location: str = "New York, NY", limit: int = 5) -> dict:
    """Find places in NYC, with ratings and up to two review excerpts each."""
    query = _clean(query)
    location = _clean(location) or "New York, NY"
    if not query:
        raise ToolError("search_places needs a query, e.g. 'museums' or 'ramen'")
    limit = _clamp_int(limit, 5, 1, 5)
    _require_google_key()
    body = _post("Google Places", PLACES_URL, {
        "textQuery": f"{query} in {location}",
        "pageSize": limit,
        "languageCode": "en",
        "regionCode": "US",
        "locationBias": _bias(location),
    }, _google_headers(
        "places.displayName,places.formattedAddress,places.rating,"
        "places.userRatingCount,places.priceLevel,places.types,places.location,"
        "places.googleMapsUri,places.reviews"
    ))

    places = []
    for place in (body.get("places") or [])[:limit]:
        loc = place.get("location") or {}
        reviews = []
        for review in (place.get("reviews") or [])[:2]:
            text = _clip(_text(review.get("text")) or _text(review.get("originalText")))
            if not text:
                continue
            reviews.append(_drop_empty({
                "rating": review.get("rating"),
                "author": (review.get("authorAttribution") or {}).get("displayName"),
                "when": review.get("relativePublishTimeDescription"),
                "text": text,
            }))
        types = [t for t in (place.get("types") or []) if t not in ("point_of_interest", "establishment")][:4]
        places.append(_drop_empty({
            "name": _text(place.get("displayName")),
            "address": place.get("formattedAddress"),
            "rating": place.get("rating"),
            "rating_count": place.get("userRatingCount"),
            "price": PRICE_LEVELS.get(place.get("priceLevel"), place.get("priceLevel")),
            "types": types,
            "lat": round(loc["latitude"], 5) if loc.get("latitude") is not None else None,
            "lng": round(loc["longitude"], 5) if loc.get("longitude") is not None else None,
            "maps_url": place.get("googleMapsUri"),
            "reviews": reviews,
        }))
    return {"query": query, "location": location, "places": places}


def _step_summary(step: dict) -> str:
    minutes = _minutes(_seconds(step.get("staticDuration")))
    transit = step.get("transitDetails") or {}
    if transit:
        line = transit.get("transitLine") or {}
        vehicle = _text((line.get("vehicle") or {}).get("name")) or (line.get("vehicle") or {}).get("type") or "Transit"
        name = line.get("nameShort") or _text(line.get("name")) or transit.get("headsign") or vehicle
        return f"{name} ({vehicle}) {minutes} min" if minutes else f"{name} ({vehicle})"
    instruction = _text((step.get("navigationInstruction") or {}).get("instructions"))
    mode = (step.get("travelMode") or "WALK").title()
    if instruction:
        return f"{instruction} ({minutes} min)" if minutes else instruction
    return f"{mode} {minutes} min" if minutes else mode


@_tool
def get_directions(origin: str, destination: str, mode: str = "transit") -> dict:
    """Travel distance and time between two places, with a short step list."""
    origin, destination = _clean(origin), _clean(destination)
    if not origin or not destination:
        raise ToolError("get_directions needs both an origin and a destination")
    mode_key = _clean(mode).lower() or "transit"
    if mode_key not in TRAVEL_MODES:
        raise ToolError(f"Unknown mode '{mode}'. Use one of: {', '.join(TRAVEL_MODES)}.")
    _require_google_key()
    body = _post("Google Routes", ROUTES_URL, {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": TRAVEL_MODES[mode_key],
        "languageCode": "en-US",
        "units": "IMPERIAL",
    }, _google_headers(
        "routes.duration,routes.distanceMeters,routes.legs.steps.staticDuration,"
        "routes.legs.steps.travelMode,routes.legs.steps.navigationInstruction.instructions,"
        "routes.legs.steps.transitDetails.transitLine,routes.legs.steps.transitDetails.headsign"
    ))
    routes = body.get("routes") or []
    if not routes:
        raise ToolError(f"No {mode_key} route from '{origin}' to '{destination}'")
    route = routes[0]
    steps = []
    for leg in route.get("legs") or []:
        steps.extend(_step_summary(step) for step in leg.get("steps") or [])
    hidden = max(0, len(steps) - 10)
    result = {
        "origin": origin,
        "destination": destination,
        "mode": mode_key,
        "distance_miles": _miles(route.get("distanceMeters")),
        "duration_minutes": _minutes(_seconds(route.get("duration"))),
        "steps": steps[:10],
    }
    if hidden:
        result["steps_note"] = f"{hidden} more steps omitted"
    return result


# What the model sees.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": (
                "Find restaurants, bars, attractions, museums, and activities in New York City via Google Places. "
                "Returns address, rating, rating count, price, types, coordinates, a Maps link, and up to two review excerpts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to find, e.g. 'ramen', 'art museums', or 'jazz bar'."},
                    "location": {
                        "type": "string",
                        "description": "Neighborhood or borough. Defaults to 'New York, NY'. Example: 'Williamsburg, Brooklyn'.",
                    },
                    "limit": {"type": "integer", "description": "How many places to return, 1 to 5. Defaults to 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": (
                "Get distance, travel time, and a short step list between two New York locations via Google Routes. "
                "Use transit by default, and walking for short hops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Starting address, place, or neighborhood."},
                    "destination": {"type": "string", "description": "Ending address, place, or neighborhood."},
                    "mode": {
                        "type": "string",
                        "enum": ["transit", "walk", "drive", "bicycle"],
                        "description": "How to travel. Defaults to transit.",
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
]

TOOL_MAP = {
    "search_places": search_places,
    "get_directions": get_directions,
}

# Composite tools would list nested calls under this key. None of these tools do.
SUB_CALLS_KEY = "_sub_calls"


def run_tool(name: str, args: dict) -> str:
    """Run one tool call. Models invent tool names and arguments; never let that crash the loop."""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool '{name}'. Available: {list(TOOL_MAP)}"})
    try:
        return TOOL_MAP[name](**args)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{name} failed unexpectedly: {type(e).__name__}: {e}"})


# --- Text2SQL ---
#
# The Maps tools above stay in this file for later. The agent registers only
# read_mappings, get_enum_values, and run_sql.

DB_PATH = Path(__file__).parent / "scripts" / "rentals.db"
MAX_ROWS = 100
MAX_ENUM_VALUES = 50

# Identifiers, free text, paths, and timestamps. A small sample would otherwise
# make these look like enums.
_NON_ENUM = frozenset({
    "listing_id", "url", "name", "street", "description", "building_slug", "unit",
    "floor_plan_url", "local_floor_plan_path", "image_url", "local_image_path",
    "room_type_counts_json", "space_type_counts_json", "irrelevant_reason",
    "available_from", "last_updated_date",
})

_WRITE_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|CREATE|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

# (table, column) overrides. Shared meanings live in _SHARED_DESCRIPTIONS.
_DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("listings", "listing_id"): "Canonical listing key (building slug + unit).",
    ("listing_amenities", "listing_id"): "Listing this amenity belongs to. Join to listings.listing_id.",
    ("listing_images", "listing_id"): "Listing this photo belongs to. Join to listings.listing_id.",
    ("listing_scores", "listing_id"): "Listing these photo aggregates belong to. Join to listings.listing_id.",
    ("regression_results", "listing_id"): "Listing this deal score belongs to. Join to listings.listing_id.",
    ("regression_features", "listing_id"): "Listing this feature row belongs to. Join to listings.listing_id.",
    ("listing_images", "image_url"): "Canonical image URL. Join to image_scores.image_url.",
    ("image_scores", "image_url"): "Canonical image URL. Join to listing_images.image_url.",
    ("image_scores", "model"): "Model that scored the image.",
    ("regression_results", "model"): "Regression model name that produced the score.",
    ("image_scores", "scored_at"): "When the image was scored.",
    ("regression_results", "scored_at"): "When the deal score was written.",
    ("listings", "listing_status"): "StreetEasy status, such as ACTIVE, RENTED, IN_CONTRACT, or DELISTED.",
    ("listings", "last_updated_date"): "As-of date of the rent and status currently stored.",
    ("regression_results", "actual_rent"): "Listed monthly rent in dollars.",
    ("regression_results", "predicted_rent"): "Model fair rent in dollars.",
    ("regression_results", "percent_discount"): (
        "100 * (predicted_rent - actual_rent) / predicted_rent. Positive means listed below the model."
    ),
    ("regression_results", "deal_score"): (
        "Deal score from 0 to 10. 5 is fairly priced, 10 is at least 25% under the model, "
        "and 0 is at least 25% over."
    ),
}

_SHARED_DESCRIPTIONS = {
    "url": "StreetEasy listing URL.",
    "name": "Building or listing name.",
    "street": "Street address.",
    "neighborhood": "NYC neighborhood.",
    "zip": "ZIP code.",
    "lat": "Latitude.",
    "lng": "Longitude.",
    "beds": "Number of bedrooms.",
    "baths": "Number of bathrooms.",
    "sqft": "Interior square feet.",
    "rent": "Monthly rent in dollars.",
    "lease_months": "Lease length in months.",
    "furnished": "Whether the unit is furnished.",
    "building_type": "Building type, such as rental.",
    "building_slug": "StreetEasy building identifier.",
    "unit": "Unit number.",
    "description": "Listing description text.",
    "available_from": "Date the unit is available.",
    "floor_plan_url": "URL of the floor plan image.",
    "local_floor_plan_path": "Local path of the downloaded floor plan.",
    "first_scraped_at": "When the listing was first scraped.",
    "detail_fetched_at": "When the detail page was last fetched.",
    "amenity": "Normalized amenity tag, such as gym or doorman.",
    "position": "Photo order on the listing, starting at 0.",
    "local_image_path": "Local path of the downloaded photo.",
    "image_category": "Photo class: apartment, common_space, or irrelevant.",
    "room_type": "Apartment photo room type, such as living_room or kitchen.",
    "natural_light": "Apartment natural light, scored 1 to 10.",
    "space_feeling": "Apartment space feeling, scored 1 to 10.",
    "view_quality": "Apartment view quality, scored 1 to 10.",
    "space_type": "Common-space type, such as lobby or gym.",
    "appeal": "Common-space appeal, scored 1 to 10.",
    "finish_quality": "Finish quality, scored 1 to 10.",
    "condition_score": "Condition, scored 1 to 10.",
    "irrelevant_reason": "Why the photo was marked irrelevant.",
    "avg_natural_light": "Mean apartment natural-light score.",
    "avg_finish_quality": "Mean apartment finish-quality score.",
    "avg_space_feeling": "Mean apartment space-feeling score.",
    "avg_condition": "Mean apartment condition score.",
    "max_view_quality": "Best apartment view-quality score.",
    "pct_bright_rooms": "Share of apartment photos scored as bright.",
    "has_good_view": "1 if the listing has a good view, else 0.",
    "common_avg_finish_quality": "Mean common-space finish quality.",
    "common_avg_condition": "Mean common-space condition.",
    "common_avg_appeal": "Mean common-space appeal.",
    "photos_total": "Number of photos on the listing.",
    "photos_scored": "Number of photos that were scored.",
    "photos_apartment": "Number of photos classified as the apartment.",
    "photos_common": "Number of photos classified as common space.",
    "photos_irrelevant": "Number of photos classified as irrelevant.",
    "room_type_counts_json": "JSON counts of apartment room types.",
    "space_type_counts_json": "JSON counts of common-space types.",
    "aggregated_at": "When these photo aggregates were built.",
    "log_rent": "Natural log of monthly rent. This is the regression target.",
    "apt_quality": "Mean of the apartment image scores for light, finish, space, and condition.",
    "common_quality": "Mean of the common-space image scores for finish, condition, and appeal.",
    "has_floor_plan": "1 if a floor plan URL is present, else 0.",
    "has_sqft": "1 if square feet was listed rather than imputed, else 0.",
    "amen_tier_a": "Count of premium amenities such as a pool, gym, doorman, or washer/dryer.",
    "amen_tier_b": "Count of standard amenities such as an elevator, dishwasher, or laundry.",
    "amen_tier_c": "Count of low-signal amenity tags.",
}


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _describe(table: str, name: str, sql_type: str) -> str:
    specific = _DESCRIPTIONS.get((table, name))
    if specific:
        return specific
    shared = _SHARED_DESCRIPTIONS.get(name)
    if shared:
        return shared
    if name.startswith("nh_"):
        label = name.removeprefix("nh_").replace("_", " ")
        return f"Neighborhood indicator for {label}. 1 if the listing is in that neighborhood."
    if name.startswith("bt_"):
        label = name.removeprefix("bt_").replace("_", " ")
        return f"Building-type indicator for {label}. 1 if the listing has that building type."
    if name.startswith("amen_"):
        label = name.removeprefix("amen_").replace("_", " ")
        return f"Amenity indicator for {label}. 1 if the listing has that amenity."
    return f"{sql_type} column {name} on {table}."


def _enum_candidate(name: str, sql_type: str) -> bool:
    if (sql_type or "").upper() not in {"TEXT", "STRING"}:
        return False
    if name in _NON_ENUM or name.lower().endswith(("_at", "_json", "_url", "_path")):
        return False
    return True


def _load_columns_bq() -> list[dict]:
    """Column catalog from BigQuery. Chat history and favorites stay out of it,
    the same way the local app only opens rentals.db.
    """
    project, dataset = bq.project(), bq.dataset()
    listed = bq.query_dicts(
        f"""
        SELECT table_name, column_name, data_type
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name, ordinal_position
        """
    )
    columns = []
    for info in listed:
        table, name, sql_type = info["table_name"], info["column_name"], info["data_type"]
        if table not in bq.RENTAL_TABLES:
            continue
        column = {
            "table": table,
            "name": name,
            "type": sql_type,
            "description": _describe(table, name, sql_type),
        }
        columns.append(column)
    _fill_enums_bq(columns)
    return columns


def _fill_enums_bq(columns: list[dict]) -> None:
    """One query per table for enum values. A query per column would run before
    the process can listen, and Cloud Run would time out the startup check.
    """
    by_table: dict[str, list[dict]] = {}
    for column in columns:
        if _enum_candidate(column["name"], column["type"]):
            by_table.setdefault(column["table"], []).append(column)
    for table, candidates in by_table.items():
        parts = []
        for column in candidates:
            quoted = bq.quote_ident(column["name"])
            literal = column["name"].replace("'", "''")
            parts.append(
                f"SELECT '{literal}' AS column_name, CAST({quoted} AS STRING) AS v "
                f"FROM {bq.quote_ident(table)} WHERE {quoted} IS NOT NULL"
            )
        grouped: dict[str, set] = {column["name"]: set() for column in candidates}
        for row in bq.query_dicts(" UNION ALL ".join(parts)):
            grouped[row["column_name"]].add(row["v"])
        for column in candidates:
            values = grouped[column["name"]]
            if 0 < len(values) <= MAX_ENUM_VALUES:
                column["enum_values"] = sorted(values, key=lambda value: (value is None, str(value)))


def _open_db() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise SystemExit(f"No database found at {DB_PATH}.")
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


def _load_columns(conn: sqlite3.Connection) -> list[dict]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    columns = []
    for table in tables:
        for info in conn.execute(f"PRAGMA table_info({_quote(table)})"):
            name, sql_type = info[1], info[2] or "TEXT"
            column = {
                "table": table,
                "name": name,
                "type": sql_type,
                "description": _describe(table, name, sql_type),
            }
            if _enum_candidate(name, sql_type):
                quoted_table, quoted_name = _quote(table), _quote(name)
                count = conn.execute(
                    f"SELECT COUNT(DISTINCT {quoted_name}) FROM {quoted_table} "
                    f"WHERE {quoted_name} IS NOT NULL"
                ).fetchone()[0]
                if 0 < count <= MAX_ENUM_VALUES:
                    column["enum_values"] = [
                        row[0]
                        for row in conn.execute(
                            f"SELECT DISTINCT {quoted_name} AS v FROM {quoted_table} "
                            f"WHERE {quoted_name} IS NOT NULL ORDER BY v"
                        )
                    ]
            columns.append(column)
    return columns


_db_lock = threading.Lock()
_db = None
_columns: list[dict] | None = None


def _sqlite() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = _open_db()
    return _db


def _columns_list() -> list[dict]:
    """Load the catalog on first use so the server can bind $PORT first."""
    global _columns
    if _columns is None:
        with _db_lock:
            if _columns is None:
                _columns = _load_columns_bq() if bq.prod() else _load_columns(_sqlite())
    return _columns


def _read_mappings() -> str:
    rows = []
    for column in _columns_list():
        # One-hot training columns. The chat joins regression_results, not this table.
        if column["table"] == "regression_features":
            continue
        row = {
            "table": column["table"],
            "name": column["name"],
            "type": column["type"],
            "description": column["description"],
            "is_enum": "enum_values" in column,
        }
        if "enum_values" in column:
            row["enum_values"] = column["enum_values"]
        rows.append(row)
    return json.dumps(rows)


def _split_column(column: str) -> tuple[str | None, str]:
    if "." in column:
        table, name = column.split(".", 1)
        return table, name
    return None, column


def _get_enum_values(column: str) -> str:
    table, name = _split_column(column.strip())
    if table is not None:
        match = next((c for c in _columns_list() if c["table"] == table and c["name"] == name), None)
        if match is None:
            return json.dumps({"error": f"No column '{column}'. Call read_mappings for the columns."})
        if "enum_values" not in match:
            return json.dumps({"error": f"'{column}' is not an enum column."})
        return json.dumps(
            {"table": table, "column": name, "enum_values": match["enum_values"]},
            default=str,
        )

    matches = [c for c in _columns_list() if c["name"] == name]
    if not matches:
        return json.dumps({"error": f"No column '{column}'. Call read_mappings for the columns."})
    if len(matches) > 1:
        tables = ", ".join(c["table"] for c in matches)
        return json.dumps({
            "error": f"'{name}' is on more than one table ({tables}). Pass table.column.",
        })
    match = matches[0]
    if "enum_values" not in match:
        return json.dumps({"error": f"'{column}' is not an enum column."})
    return json.dumps(
        {"table": match["table"], "column": name, "enum_values": match["enum_values"]},
        default=str,
    )


def _is_read_only(query: str) -> bool:
    text = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    text = re.sub(r"--[^\n]*", " ", text)
    text = text.strip().rstrip(";").strip()
    if not text or ";" in text:
        return False
    if not re.match(r"(?is)^(SELECT|WITH)\b", text):
        return False
    return _WRITE_SQL.search(text) is None


def _run_sql(query: str) -> str:
    """The SDK runs tools on worker threads. One lock keeps the shared connection safe."""
    if bq.prod():
        try:
            columns, rows = bq.query(query, max_rows=MAX_ROWS + 1)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps(
            {
                "columns": columns,
                "rows": rows[:MAX_ROWS],
                "truncated": len(rows) > MAX_ROWS,
            },
            default=str,
        )
    with _db_lock:
        try:
            cursor = _sqlite().execute(query)
            rows = cursor.fetchmany(MAX_ROWS + 1)
            columns = [desc[0] for desc in cursor.description or []]
        except sqlite3.Error as e:
            return json.dumps({"error": str(e)})
    return json.dumps(
        {
            "columns": columns,
            "rows": [list(row) for row in rows[:MAX_ROWS]],
            "truncated": len(rows) > MAX_ROWS,
        },
        default=str,
    )


@function_tool
def read_mappings() -> str:
    """Read rental table columns: table, name, type, a description, and enum values for a small enum.

    Call this only when no analysis tool can answer the question. Small enums such as
    neighborhood and listing_status include their values here. Call get_enum_values only
    when a column is an enum and this list left its values out.
    """
    return _read_mappings()


@function_tool
def get_enum_values(column: str) -> str:
    """Get the exact valid values of an enum column. Match them exactly: equality is case-sensitive.

    Call this only when read_mappings did not include enum_values for the column.

    Args:
        column: A column name, e.g. 'neighborhood', or 'listings.neighborhood' when the name is shared.
    """
    return _get_enum_values(column)


@tool_input_guardrail
def read_only_sql(data) -> ToolGuardrailFunctionOutput:
    """Reject SQL that is not a single read-only SELECT or WITH query."""
    query = json.loads(data.context.tool_arguments or "{}").get("query", "")
    if not _is_read_only(query):
        return ToolGuardrailFunctionOutput.reject_content(
            "Rejected: only a single read-only SELECT or WITH query is allowed. "
            "Do not use INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, or PRAGMA."
        )
    return ToolGuardrailFunctionOutput.allow()


@function_tool(tool_input_guardrails=[read_only_sql])
def run_sql(query: str) -> str:
    """Run one read-only query against the rental database and return the rows.

    Args:
        query: A single SELECT or WITH query.
    """
    return _run_sql(query)


# --- Analysis tools ---
#
# These call the read-only database, then rank or summarize the rows. Composite
# tools call another tool's __wrapped__ function so they share one query path.


def _query_rows(sql: str, params: tuple = ()) -> list[dict]:
    """Run a parameterized read and return dict rows. Names stay out of the SQL text."""
    if bq.prod():
        try:
            return bq.query_dicts(sql, params)
        except Exception as e:
            raise ToolError(str(e)) from e
    with _db_lock:
        try:
            cursor = _sqlite().execute(sql, params)
            columns = [desc[0] for desc in cursor.description or []]
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            raise ToolError(str(e))
    return [dict(zip(columns, row)) for row in rows]


def _neighborhoods() -> list[str]:
    match = next(
        (c for c in _columns_list() if c["table"] == "listings" and c["name"] == "neighborhood"),
        None,
    )
    values = (match or {}).get("enum_values")
    if values:
        return list(values)
    return [
        row["neighborhood"]
        for row in _query_rows(
            "SELECT DISTINCT neighborhood FROM listings "
            "WHERE neighborhood IS NOT NULL ORDER BY neighborhood"
        )
    ]


def _resolve_neighborhood(name: str) -> str:
    cleaned = _clean(name)
    if not cleaned:
        raise ToolError("A neighborhood is required.")
    neighborhoods = _neighborhoods()
    for neighborhood in neighborhoods:
        if neighborhood.casefold() == cleaned.casefold():
            return neighborhood
    raise ToolError(
        f"No neighborhood '{cleaned}'. Valid neighborhoods: {', '.join(neighborhoods)}."
    )


# Umbrella names are not stored on listings.neighborhood. UES has no row of its own.
_AREA_ALIASES = {
    "ues": ("Carnegie Hill", "Lenox Hill", "Yorkville", "Sutton Place"),
    "upper east side": ("Carnegie Hill", "Lenox Hill", "Yorkville", "Sutton Place"),
    "uws": ("Upper West Side", "Lincoln Square", "Manhattan Valley", "Morningside Heights"),
    "upper west side": ("Upper West Side", "Lincoln Square", "Manhattan Valley", "Morningside Heights"),
}


def _resolve_areas(neighborhoods) -> list[str]:
    """Expand area aliases and neighborhood names into canonical neighborhood values."""
    if neighborhoods is None or neighborhoods == "":
        return []
    known = {name.casefold(): name for name in _neighborhoods()}
    resolved = []
    for part in _split_names(neighborhoods):
        key = part.casefold()
        if key in _AREA_ALIASES:
            matched = []
            for alias_name in _AREA_ALIASES[key]:
                canonical = known.get(alias_name.casefold())
                if canonical and canonical not in resolved and canonical not in matched:
                    matched.append(canonical)
            if not matched:
                raise ToolError(
                    f"No neighborhood '{part}'. Valid neighborhoods: {', '.join(known.values())}."
                )
            resolved.extend(matched)
            continue
        canonical = known.get(key)
        if canonical is None:
            raise ToolError(
                f"No neighborhood '{part}'. Valid neighborhoods: {', '.join(known.values())}."
            )
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def _optional_number(value, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ToolError(f"{label} must be a number, got {value!r}")


def _split_names(neighborhoods) -> list[str]:
    if isinstance(neighborhoods, (list, tuple)):
        names = []
        for item in neighborhoods:
            names.extend(_split_names(item))
        return names
    parts = []
    for chunk in str(neighborhoods or "").split(","):
        parts.extend(chunk.split(" and "))
    return [_clean(part) for part in parts if _clean(part)]


def _price_stats(rows: list[dict]) -> dict:
    rents = [row["rent"] for row in rows if row.get("rent") is not None]
    per_sqft = [
        row["rent"] / row["sqft"]
        for row in rows
        if row.get("rent") is not None and row.get("sqft")
    ]
    stats = {
        "listings": len(rows),
        "min_rent": min(rents) if rents else None,
        "max_rent": max(rents) if rents else None,
        "avg_rent": round(sum(rents) / len(rents)) if rents else None,
        "median_rent": round(statistics.median(rents)) if rents else None,
        "avg_rent_per_sqft": round(sum(per_sqft) / len(per_sqft), 2) if per_sqft else None,
    }
    return stats


def _listing_rows(sql: str, params: tuple) -> list[dict]:
    rows = _query_rows(sql, params)
    for row in rows:
        for key in ("beds", "baths"):
            if isinstance(row.get(key), float) and row[key].is_integer():
                row[key] = int(row[key])
    return rows


@_tool
def search_listings(
    neighborhood: str | None = None,
    min_beds: float | None = None,
    max_beds: float | None = None,
    min_baths: float | None = None,
    min_sqft: float | None = None,
    max_sqft: float | None = None,
    min_rent: float | None = None,
    max_rent: float | None = None,
    limit: int = 10,
) -> dict:
    """Find listings by neighborhood, bedrooms, bathrooms, square feet, and rent.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea' or 'Hell's Kitchen'. Optional.
        min_beds: Minimum bedrooms. Optional.
        max_beds: Maximum bedrooms. Optional.
        min_baths: Minimum bathrooms. Optional.
        min_sqft: Minimum square feet. Listings with no sqft are kept. Optional.
        max_sqft: Maximum square feet. Listings with no sqft are kept. Optional.
        min_rent: Minimum monthly rent. Optional.
        max_rent: Maximum monthly rent. Optional.
        limit: How many listings to return, 1 to 25. Defaults to 10.
    """
    limit = _clamp_int(limit, 10, 1, 25)
    min_beds = _optional_number(min_beds, "min_beds")
    max_beds = _optional_number(max_beds, "max_beds")
    min_baths = _optional_number(min_baths, "min_baths")
    min_sqft = _optional_number(min_sqft, "min_sqft")
    max_sqft = _optional_number(max_sqft, "max_sqft")
    min_rent = _optional_number(min_rent, "min_rent")
    max_rent = _optional_number(max_rent, "max_rent")

    clauses = ["(listing_status IS NULL OR listing_status = 'ACTIVE')"]
    params: list = []
    resolved = None
    if neighborhood:
        resolved = _resolve_neighborhood(neighborhood)
        clauses.append("neighborhood = ?")
        params.append(resolved)
    if min_beds is not None:
        clauses.append("beds >= ?")
        params.append(min_beds)
    if max_beds is not None:
        clauses.append("beds <= ?")
        params.append(max_beds)
    if min_baths is not None:
        clauses.append("baths >= ?")
        params.append(min_baths)
    if min_sqft is not None:
        clauses.append("(sqft IS NULL OR sqft >= ?)")
        params.append(min_sqft)
    if max_sqft is not None:
        clauses.append("(sqft IS NULL OR sqft <= ?)")
        params.append(max_sqft)
    if min_rent is not None:
        clauses.append("rent >= ?")
        params.append(min_rent)
    if max_rent is not None:
        clauses.append("rent <= ?")
        params.append(max_rent)

    rows = _listing_rows(
        f"""
        SELECT listing_id, name AS building, unit, neighborhood, beds, baths, sqft, rent, url
        FROM listings
        WHERE {' AND '.join(clauses)}
        ORDER BY rent IS NULL, rent, listing_id
        LIMIT ?
        """,
        tuple(params) + (limit,),
    )
    filters = {
        "neighborhood": resolved,
        "min_beds": min_beds,
        "max_beds": max_beds,
        "min_baths": min_baths,
        "min_sqft": min_sqft,
        "max_sqft": max_sqft,
        "min_rent": min_rent,
        "max_rent": max_rent,
    }
    return {
        "neighborhood": resolved,
        "filters": {key: value for key, value in filters.items() if value is not None},
        "count": len(rows),
        "listings": rows,
    }


@_tool
def get_building_amenities(building: str) -> dict:
    """List the amenities of one building, and how many of its units have each.

    Args:
        building: Building name or StreetEasy building slug.
    """
    building = _clean(building)
    if not building:
        raise ToolError("get_building_amenities needs a building name or slug.")

    def matches(where: str) -> list[dict]:
        return _query_rows(
            f"""
            SELECT building_slug, MIN(name) AS name, COUNT(*) AS units
            FROM listings
            WHERE {where}
            GROUP BY building_slug
            ORDER BY units DESC, name
            """,
            (building, building),
        )

    found = matches("lower(name) = lower(?) OR lower(building_slug) = lower(?)")
    if not found:
        fuzzy = (
            "STRPOS(LOWER(name), LOWER(?)) > 0 OR STRPOS(LOWER(building_slug), LOWER(?)) > 0"
            if bq.prod()
            else "instr(lower(name), lower(?)) > 0 OR instr(lower(building_slug), lower(?)) > 0"
        )
        found = matches(fuzzy)
    if not found:
        raise ToolError(f"No building matching '{building}'.")
    if len(found) > 1:
        return {
            "error": f"'{building}' matches more than one building. Pass a more specific name or slug.",
            "matches": [{"building": row["name"], "building_slug": row["building_slug"], "units": row["units"]} for row in found],
        }

    chosen = found[0]
    amenities = _query_rows(
        """
        SELECT a.amenity, COUNT(DISTINCT l.listing_id) AS units
        FROM listings l
        JOIN listing_amenities a ON l.listing_id = a.listing_id
        WHERE l.building_slug = ?
        GROUP BY a.amenity
        ORDER BY units DESC, a.amenity
        """,
        (chosen["building_slug"],),
    )
    return {
        "building": chosen["name"],
        "building_slug": chosen["building_slug"],
        "units": chosen["units"],
        "amenities": amenities,
    }


@_tool
def get_deals(
    neighborhood: str | None = None,
    min_beds: float | None = None,
    max_beds: float | None = None,
    max_rent: float | None = None,
    limit: int = 10,
) -> dict:
    """List model deals: predicted rent, actual rent, percent discount, and deal score.

    Use this for a deal question with a neighborhood, an area, bedrooms, or a max rent.
    Do not write SQL for that. 'UES' and 'Upper East Side' cover Carnegie Hill, Lenox Hill,
    Yorkville, and Sutton Place. 'UWS' and 'Upper West Side' cover Upper West Side,
    Lincoln Square, Manhattan Valley, and Morningside Heights. Several names can be
    comma-separated, as in 'Chelsea, Hell's Kitchen'.

    Args:
        neighborhood: Neighborhood, area alias, or comma-separated names. Optional.
        min_beds: Minimum bedrooms. Optional.
        max_beds: Maximum bedrooms. Optional.
        max_rent: Maximum monthly rent. Optional.
        limit: How many deals to return, 1 to 25. Defaults to 10.
    """
    limit = _clamp_int(limit, 10, 1, 25)
    min_beds = _optional_number(min_beds, "min_beds")
    max_beds = _optional_number(max_beds, "max_beds")
    max_rent = _optional_number(max_rent, "max_rent")
    scored = _query_rows("SELECT COUNT(*) AS n FROM regression_results")[0]["n"]
    if not scored:
        raise ToolError(
            "Deal scores have not been written yet. regression_results is empty. "
            "Run features.py to score listings before asking for deals."
        )

    resolved = _resolve_areas(neighborhood)
    clauses = []
    params: list = []
    if resolved:
        marks = ", ".join("?" for _ in resolved)
        clauses.append(f"l.neighborhood IN ({marks})")
        params.extend(resolved)
    if min_beds is not None:
        clauses.append("l.beds >= ?")
        params.append(min_beds)
    if max_beds is not None:
        clauses.append("l.beds <= ?")
        params.append(max_beds)
    if max_rent is not None:
        clauses.append("l.rent <= ?")
        params.append(max_rent)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = _listing_rows(
        f"""
        SELECT l.listing_id, l.name AS building, l.street, l.unit, l.neighborhood,
               l.beds, l.baths, l.sqft, l.rent AS actual_rent,
               r.predicted_rent, r.percent_discount, r.deal_score
        FROM listings l
        JOIN regression_results r ON l.listing_id = r.listing_id
        {where}
        ORDER BY r.deal_score DESC, r.percent_discount DESC, l.listing_id
        LIMIT ?
        """,
        tuple(params) + (limit,),
    )
    filters = {
        "min_beds": min_beds,
        "max_beds": max_beds,
        "max_rent": max_rent,
    }
    return {
        "neighborhoods": resolved or None,
        "filters": {key: value for key, value in filters.items() if value is not None},
        "ranked_by": "deal_score",
        "count": len(rows),
        "listings": rows,
    }


@_tool
def cheapest_in_neighborhood(
    neighborhood: str, min_beds: float | None = None, limit: int = 5
) -> dict:
    """Cheapest listings in one neighborhood, ranked by listed rent.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea'.
        min_beds: Minimum bedrooms. Optional.
        limit: How many listings to return, 1 to 25. Defaults to 5.
    """
    found = search_listings.__wrapped__(neighborhood=neighborhood, min_beds=min_beds, limit=limit)
    return {
        "neighborhood": found["neighborhood"],
        "ranked_by": "listed rent",
        "count": found["count"],
        "listings": found["listings"],
    }


@_tool
def best_deals_in_neighborhood(neighborhood: str, limit: int = 5) -> dict:
    """Best model deals in one neighborhood or area, ranked by deal score.

    Args:
        neighborhood: Neighborhood or area name, such as 'Chelsea' or 'UES'.
        limit: How many deals to return, 1 to 25. Defaults to 5.
    """
    return get_deals.__wrapped__(neighborhood=neighborhood, limit=limit)


@_tool
def compare_neighborhoods(neighborhoods: str) -> dict:
    """Compare listing counts and rents for two or more neighborhoods.

    Args:
        neighborhoods: Comma-separated names, such as 'Chelsea, Hell's Kitchen'.
    """
    names = []
    for name in _split_names(neighborhoods):
        resolved = _resolve_neighborhood(name)
        if resolved not in names:
            names.append(resolved)
    if len(names) < 2:
        raise ToolError("compare_neighborhoods needs at least 2 neighborhoods.")

    summaries = []
    for name in names:
        rows = _query_rows(
            "SELECT rent, sqft FROM listings WHERE neighborhood = ?",
            (name,),
        )
        summaries.append({"neighborhood": name, **_price_stats(rows)})

    priced = [row for row in summaries if row["median_rent"] is not None]
    lowest = min(priced, key=lambda row: row["median_rent"])["neighborhood"] if priced else None
    return {"neighborhoods": summaries, "lowest_median_rent": lowest}


@_tool
def summarize_neighborhood(neighborhood: str) -> dict:
    """Price, bedroom mix, and most common amenities for one neighborhood.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea'.
    """
    name = _resolve_neighborhood(neighborhood)
    rows = _query_rows(
        "SELECT rent, sqft, beds FROM listings WHERE neighborhood = ?",
        (name,),
    )
    bed_counts: dict = {}
    for row in rows:
        bed_counts[row["beds"]] = bed_counts.get(row["beds"], 0) + 1
    bed_mix = [
        {"beds": beds, "listings": count}
        for beds, count in sorted(bed_counts.items(), key=lambda item: (item[0] is None, item[0] or 0))
    ]
    amenities = _query_rows(
        """
        SELECT a.amenity, COUNT(DISTINCT l.listing_id) AS units
        FROM listings l
        JOIN listing_amenities a ON l.listing_id = a.listing_id
        WHERE l.neighborhood = ?
        GROUP BY a.amenity
        ORDER BY units DESC, a.amenity
        LIMIT 8
        """,
        (name,),
    )
    return {"neighborhood": name, **_price_stats(rows), "bed_mix": bed_mix, "top_amenities": amenities}


def _agent_tool(fn):
    """Expose a crypto-style tool to the Agents SDK without hiding its __wrapped__ function."""
    return function_tool(fn, strict_mode=False)


ANALYSIS_TOOLS = [
    _agent_tool(search_listings),
    _agent_tool(get_building_amenities),
    _agent_tool(get_deals),
    _agent_tool(cheapest_in_neighborhood),
    _agent_tool(best_deals_in_neighborhood),
    _agent_tool(compare_neighborhoods),
    _agent_tool(summarize_neighborhood),
]
