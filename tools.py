"""The tools the harness can run, and the JSON that describes them to the model."""

import asyncio
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
from agents import (
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    function_tool,
    tool_input_guardrail,
    tool_output_guardrail,
)
from dotenv import load_dotenv

import accounts
import charts
from rental_context import as_rental, bind
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
    _note_places_call()
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


def _note_places_call() -> None:
    rental = as_rental(None)
    if rental is None:
        return
    if rental.places_calls >= 6:
        raise ToolError("Places and Routes call limit reached for this turn.")
    rental.places_calls += 1


def _cache_account() -> str:
    rental = as_rental(None)
    if rental and rental.account_id:
        return rental.account_id
    return accounts.SHARED_ACCOUNT


def _cached_tool(tool: str, args: dict, ttl: int, fetch, account: str | None = None):
    owner = account or _cache_account()
    hit = accounts.cache_get(owner, tool, args)
    if hit is not None:
        hit = dict(hit)
        hit["cached"] = True
        return hit
    result = fetch()
    accounts.cache_put(owner, tool, args, result, ttl)
    return result


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
    """Find a place in New York via Google Places, with its address and coordinates.

    Use this for a listing's location. Pass the listing street as the query and the
    neighborhood as the location, with limit 1. Use the returned address and coordinates.
    Do not match the address with a street LIKE filter or another fuzzy address match.

    Also use this for any other place question, such as a restaurant, a subway stop,
    or what is nearby. Pass that place as the query and the neighborhood as the location.
    Call it for the listings you are answering about, not for every row in a large result.

    Args:
        query: A listing street, or the place to find, such as '225 East 63rd Street' or 'ramen'.
        location: Neighborhood or borough. Defaults to 'New York, NY'. Example: 'Yorkville, New York, NY'.
        limit: How many places to return, 1 to 5. Use 1 for a listing address. Defaults to 5.
    """
    query = _clean(query)
    location = _clean(location) or "New York, NY"
    if not query:
        raise ToolError("search_places needs a query, e.g. 'museums' or 'ramen'")
    limit = _clamp_int(limit, 5, 1, 5)
    cache_key = {"query": query, "location": location, "limit": limit}

    def fetch() -> dict:
        return _search_places_live(query, location, limit)

    return _cached_tool("search_places", cache_key, accounts.PLACES_TTL_SECONDS, fetch)


def _search_places_live(query: str, location: str, limit: int) -> dict:
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
    cache_key = {"origin": origin, "destination": destination, "mode": mode_key}

    def fetch() -> dict:
        return _directions_live(origin, destination, mode_key)

    return _cached_tool("get_directions", cache_key, accounts.PLACES_TTL_SECONDS, fetch)


def _directions_live(origin: str, destination: str, mode_key: str) -> dict:
    _require_google_key()
    body = _post("Google Routes", ROUTES_URL, {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": TRAVEL_MODES[mode_key],
        "languageCode": "en-US",
        "units": "IMPERIAL",
    }, _google_headers(
        "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline,routes.legs.steps.staticDuration,"
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
        "encoded_polyline": (route.get("polyline") or {}).get("encodedPolyline"),
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
                "Find a place in New York via Google Places, including a listing's location. "
                "Pass the listing street as the query and the neighborhood as the location. "
                "Also use this for a restaurant, a subway stop, or what is nearby. "
                "Returns address, coordinates, rating, a Maps link, and up to two review excerpts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A listing street, or the place to find, such as '225 East 63rd Street' or 'ramen'.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Neighborhood or borough. Defaults to 'New York, NY'. Example: 'Yorkville, New York, NY'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many places to return, 1 to 5. Use 1 for a listing address. Defaults to 5.",
                    },
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
# search_places is registered on the agent. get_directions stays in this file.

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
    ("listings", "listing_status"): (
        "StreetEasy status, such as ACTIVE, RENTED, IN_CONTRACT, DELISTED, or OFF_MARKET. "
        "Listing results default to listing_status = 'ACTIVE' unless another status was requested."
    ),
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
    "neighborhood": (
        "NYC neighborhood stored on the listing. Area names such as Upper East Side "
        "are not values of this column. Use area_aliases from read_mappings."
    ),
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


# Re-entrant: a query holds this lock and then opens the connection on first use.
_db_lock = threading.RLock()
_columns_lock = threading.Lock()
_mappings_lock = threading.Lock()
_db = None
_columns: list[dict] | None = None
_mappings_json: str | None = None
_mappings_started = False


def _sqlite() -> sqlite3.Connection:
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = _open_db()
    return _db


def _columns_list() -> list[dict]:
    """Load the catalog on first use so the server can bind $PORT first.

    The chat page starts this before anyone sends a message. A private
    connection keeps that scan off the lock that query tools use.
    """
    global _columns
    if _columns is not None:
        return _columns
    with _columns_lock:
        if _columns is None:
            if bq.prod():
                _columns = _load_columns_bq()
            else:
                conn = _open_db()
                try:
                    _columns = _load_columns(conn)
                finally:
                    conn.close()
    return _columns


def _read_mappings() -> str:
    global _mappings_json
    if _mappings_json is not None:
        return _mappings_json
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
    payload = json.dumps({
        "columns": rows,
        "area_aliases": _area_alias_catalog(),
    })
    with _columns_lock:
        if _mappings_json is None:
            _mappings_json = payload
        return _mappings_json


def _warm_mappings() -> None:
    global _mappings_started
    try:
        _read_mappings()
    except BaseException:
        with _mappings_lock:
            _mappings_started = False


def start_read_mappings() -> None:
    """Read the column catalog on a background thread.

    The new-chat page calls this before the user sends. A later read waits
    on the same load instead of starting a second one.
    """
    global _mappings_started
    if _mappings_json is not None:
        return
    with _mappings_lock:
        if _mappings_started or _mappings_json is not None:
            return
        _mappings_started = True
    threading.Thread(target=_warm_mappings, name="read-mappings", daemon=True).start()


def load_mappings() -> str:
    """Return the column catalog, joining a background load if one is running."""
    return _read_mappings()


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


def _is_listing_search(query: str, columns: list, rows: list) -> bool:
    """True when a query returned listing rows the answer can use."""
    if not rows or not re.search(r"\blistings\b", query, re.IGNORECASE):
        return False
    names = {str(column).lower() for column in columns}
    return "listing_id" in names and "rent" in names


def _run_sql(query: str) -> str:
    """The SDK runs tools on worker threads. One lock keeps the shared connection safe."""
    if bq.prod():
        try:
            columns, rows = bq.query(query, max_rows=MAX_ROWS + 1)
        except Exception as e:
            return json.dumps({"error": str(e)})
        payload = {
            "columns": columns,
            "rows": rows[:MAX_ROWS],
            "truncated": len(rows) > MAX_ROWS,
        }
    else:
        with _db_lock:
            try:
                cursor = _sqlite().execute(query)
                rows = cursor.fetchmany(MAX_ROWS + 1)
                columns = [desc[0] for desc in cursor.description or []]
            except sqlite3.Error as e:
                return json.dumps({"error": str(e)})
        payload = {
            "columns": columns,
            "rows": [list(row) for row in rows[:MAX_ROWS]],
            "truncated": len(rows) > MAX_ROWS,
        }
    if _is_listing_search(query, payload["columns"], payload["rows"]):
        payload["stop"] = (
            "These rows are the search result. Do not call run_sql again. "
            "A location question calls search_places for the listings you are answering about."
        )
    else:
        rental = as_rental(None)
        question = rental.user_message if rental else ""
        chart = charts.infer_chart(payload["columns"], payload["rows"], question)
        if chart is not None:
            if chart["type"] == "map":
                _fill_map_coords(chart)
            payload["chart"] = chart
    return json.dumps(payload, default=str)


# One flag per agent run. Tool calls in that run share the run's usage object.
# The next user message starts a new run, so a later question can query again.
_SEARCH_DONE = "_sql_search_done"


def _mark_search_done(ctx) -> None:
    rental = as_rental(ctx)
    if rental is not None:
        rental.sql_search_done = True
        return
    usage = getattr(ctx, "usage", None)
    if usage is not None:
        object.__setattr__(usage, _SEARCH_DONE, True)


def _search_is_done(ctx) -> bool:
    rental = as_rental(ctx)
    if rental is not None:
        return rental.sql_search_done
    usage = getattr(ctx, "usage", None)
    return bool(usage is not None and getattr(usage, _SEARCH_DONE, False))


@tool_input_guardrail
def bind_rental(data) -> ToolGuardrailFunctionOutput:
    """Publish this run's context for Places cache and the call cap."""
    bind(data.context)
    return ToolGuardrailFunctionOutput.allow()


@tool_input_guardrail
def places_budget(data) -> ToolGuardrailFunctionOutput:
    """Stop a turn that has already spent its Places and Routes calls."""
    rental = as_rental(data.context)
    if rental is not None and rental.places_calls >= 6:
        return ToolGuardrailFunctionOutput.reject_content(
            "Places and Routes call limit reached for this turn. Answer from the calls already made."
        )
    return ToolGuardrailFunctionOutput.allow()


@tool_input_guardrail
def finished_search(data) -> ToolGuardrailFunctionOutput:
    """Refuse another query once this run already has its listing rows."""
    if _search_is_done(data.context):
        return ToolGuardrailFunctionOutput.reject_content(
            "The search already returned listings. This call was refused. "
            "Do not call run_sql again. For a location, call search_places."
        )
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def remember_search(data) -> ToolGuardrailFunctionOutput:
    """Remember a listing search that returned rows, so the next call can be refused."""
    output = data.output
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            output = None
    if isinstance(output, dict) and output.get("stop"):
        _mark_search_done(data.context)
    return ToolGuardrailFunctionOutput.allow()


@function_tool
def read_mappings() -> str:
    """Read rental table columns and area aliases.

    Columns include table, name, type, a description, and enum values for a small enum.
    area_aliases maps a name such as Upper East Side to the neighborhood values stored
    on listings. That name is not itself a neighborhood. Filter with the listed neighborhoods.

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


@function_tool(
    tool_input_guardrails=[bind_rental, read_only_sql, finished_search],
    tool_output_guardrails=[remember_search],
)
def run_sql(query: str) -> str:
    """Run one read-only query against the rental database and return the rows.

    Queries that return or aggregate listings must filter listing_status = 'ACTIVE'
    unless the user asked for another status.

    When the result includes listing rows, stop. Do not call run_sql again for this
    question. A later call is refused. Put every column the answer needs in this one
    query. A location question still calls search_places after that.

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
    alias = _AREA_ALIASES.get(cleaned.casefold())
    if alias:
        raise ToolError(
            f"'{cleaned}' covers more than one neighborhood: {', '.join(alias)}. "
            "search_listings takes one neighborhood. Call it once per neighborhood, "
            "or filter SQL with those names."
        )
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


def _alias_label(alias: str) -> str:
    if len(alias) <= 3 and alias.isalpha():
        return alias.upper()
    return str(alias).title()


def _area_alias_catalog() -> dict:
    """Area names resolved to the neighborhood values stored on listings."""
    grouped: dict[tuple[str, ...], list[str]] = {}
    for alias, names in _AREA_ALIASES.items():
        grouped.setdefault(tuple(names), []).append(alias)
    areas = []
    for aliases in grouped.values():
        try:
            neighborhoods = _resolve_areas(aliases[0])
        except ToolError:
            continue
        if not neighborhoods:
            continue
        areas.append({
            "names": [_alias_label(alias) for alias in sorted(aliases, key=len, reverse=True)],
            "neighborhoods": neighborhoods,
        })
    return {
        "description": (
            "These names are not stored on listings.neighborhood. "
            "Filter with the neighborhoods listed for the name."
        ),
        "areas": areas,
    }


def _amenity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _amenity_values() -> list[str]:
    match = next(
        (c for c in _columns_list() if c["table"] == "listing_amenities" and c["name"] == "amenity"),
        None,
    )
    values = (match or {}).get("enum_values")
    if values:
        return list(values)
    return [
        row["amenity"]
        for row in _query_rows(
            "SELECT DISTINCT amenity FROM listing_amenities "
            "WHERE amenity IS NOT NULL ORDER BY amenity"
        )
    ]


def _resolve_amenities(amenities) -> list[str]:
    """Match requested amenities to stored tags. A listing must have every one."""
    if amenities is None or amenities == "" or amenities == []:
        return []
    known = {_amenity_key(name): name for name in _amenity_values()}
    resolved = []
    for part in _split_names(amenities):
        canonical = known.get(_amenity_key(part))
        if canonical is None:
            raise ToolError(
                f"No amenity '{part}'. Valid amenities: {', '.join(sorted(known.values()))}."
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
    amenities: str | None = None,
    limit: int = 10,
) -> dict:
    """Find active listings by neighborhood, bedrooms, bathrooms, square feet, rent, and amenities.

    Use this when the user wants units that have specific amenities, such as a doorman
    and an elevator. A listing must have every amenity named here.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea' or 'Hell's Kitchen'. Optional.
        min_beds: Minimum bedrooms. Optional.
        max_beds: Maximum bedrooms. Optional.
        min_baths: Minimum bathrooms. Optional.
        min_sqft: Minimum square feet. Listings with no sqft are kept. Optional.
        max_sqft: Maximum square feet. Listings with no sqft are kept. Optional.
        min_rent: Minimum monthly rent. Optional.
        max_rent: Maximum monthly rent. Optional.
        amenities: Required amenities, such as 'doorman, elevator'. Optional.
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
    required = _resolve_amenities(amenities)

    clauses = ["listing_status = 'ACTIVE'"]
    params: list = []
    resolved = None
    if neighborhood:
        resolved = _resolve_neighborhood(neighborhood)
        clauses.append("neighborhood = ?")
        params.append(resolved)
    for amenity in required:
        clauses.append(
            "EXISTS (SELECT 1 FROM listing_amenities a "
            "WHERE a.listing_id = listings.listing_id AND a.amenity = ?)"
        )
        params.append(amenity)
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
        "amenities": required or None,
    }
    return {
        "neighborhood": resolved,
        "filters": {key: value for key, value in filters.items() if value is not None},
        "count": len(rows),
        "listings": rows,
    }


@_tool
def get_building_amenities(building: str) -> dict:
    """List the amenities of one building, and how many of its active units have each.

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
            WHERE listing_status = 'ACTIVE' AND ({where})
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
        WHERE l.building_slug = ? AND l.listing_status = 'ACTIVE'
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
    """List model deals for active listings: predicted rent, actual rent, percent discount, and deal score.

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
    scored = _query_rows(
        """
        SELECT COUNT(*) AS n
        FROM regression_results r
        JOIN listings l ON l.listing_id = r.listing_id
        WHERE l.listing_status = 'ACTIVE'
        """
    )[0]["n"]
    if not scored:
        raise ToolError(
            "Deal scores have not been written yet for active listings. "
            "Run features.py to score listings before asking for deals."
        )

    resolved = _resolve_areas(neighborhood)
    clauses = ["l.listing_status = 'ACTIVE'"]
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
    """Cheapest active listings in one neighborhood, ranked by listed rent.

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
    """Best model deals among active listings in one neighborhood or area, ranked by deal score.

    Args:
        neighborhood: Neighborhood or area name, such as 'Chelsea' or 'UES'.
        limit: How many deals to return, 1 to 25. Defaults to 5.
    """
    return get_deals.__wrapped__(neighborhood=neighborhood, limit=limit)


@_tool
def compare_neighborhoods(neighborhoods: str) -> dict:
    """Compare active listing counts and rents for two or more neighborhoods.

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
            "SELECT rent, sqft FROM listings WHERE neighborhood = ? AND listing_status = 'ACTIVE'",
            (name,),
        )
        summaries.append({"neighborhood": name, **_price_stats(rows)})

    priced = [row for row in summaries if row["median_rent"] is not None]
    lowest = min(priced, key=lambda row: row["median_rent"])["neighborhood"] if priced else None
    points = [
        charts._point(row["neighborhood"], row["avg_rent"])
        for row in summaries
        if row.get("avg_rent") is not None
    ]
    chart = charts.bar_chart("Average rent by neighborhood", points, unit="usd")
    return {"neighborhoods": summaries, "lowest_median_rent": lowest, "chart": chart}


@_tool
def summarize_neighborhood(neighborhood: str) -> dict:
    """Price, bedroom mix, and most common amenities for active listings in one neighborhood.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea'.
    """
    name = _resolve_neighborhood(neighborhood)
    rows = _query_rows(
        "SELECT rent, sqft, beds FROM listings WHERE neighborhood = ? AND listing_status = 'ACTIVE'",
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
        WHERE l.neighborhood = ? AND l.listing_status = 'ACTIVE'
        GROUP BY a.amenity
        ORDER BY units DESC, a.amenity
        LIMIT 8
        """,
        (name,),
    )
    mix_points = [
        charts._point(charts.unit_label(row["beds"]), row["listings"])
        for row in bed_mix
    ]
    chart = charts.bar_chart(f"Listings by apartment type in {name}", mix_points, unit="count")
    return {
        "neighborhood": name,
        **_price_stats(rows),
        "bed_mix": bed_mix,
        "top_amenities": amenities,
        "chart": chart,
    }


def _fill_map_coords(chart: dict) -> None:
    missing = [point for point in chart.get("points") or [] if "lat" not in point]
    if not missing:
        return
    rows = _query_rows(
        """
        SELECT neighborhood, AVG(lat) AS lat, AVG(lng) AS lng
        FROM listings
        WHERE lat IS NOT NULL AND lng IS NOT NULL AND neighborhood IS NOT NULL
        GROUP BY neighborhood
        """
    )
    by_name = {row["neighborhood"]: row for row in rows}
    for point in chart.get("points") or []:
        found = by_name.get(point["label"])
        if found and found.get("lat") is not None and found.get("lng") is not None:
            point["lat"] = round(float(found["lat"]), 5)
            point["lng"] = round(float(found["lng"]), 5)


def _active_where(neighborhood: str | None) -> tuple[str, list]:
    clauses = ["listing_status = 'ACTIVE'", "rent IS NOT NULL"]
    params: list = []
    if neighborhood:
        resolved = _resolve_areas(neighborhood) or [_resolve_neighborhood(neighborhood)]
        marks = ", ".join("?" for _ in resolved)
        clauses.append(f"neighborhood IN ({marks})")
        params.extend(resolved)
    return " AND ".join(clauses), params


_UNIT_SQL = """
CASE
    WHEN beds IS NULL THEN 'Unknown'
    WHEN beds < 1 THEN 'Studio'
    WHEN beds < 2 THEN '1-bedroom'
    WHEN beds < 3 THEN '2-bedroom'
    ELSE '3+ bedrooms'
END
"""


@_tool
def average_rent_by_neighborhood() -> dict:
    """Average listed rent for active listings in each neighborhood, with a bar chart.

    Use this when the user asks for average rent by neighborhood.
    """
    def fetch() -> dict:
        rows = _query_rows(
            """
            SELECT neighborhood, ROUND(AVG(rent)) AS avg_rent, COUNT(*) AS listings,
                   AVG(lat) AS lat, AVG(lng) AS lng
            FROM listings
            WHERE listing_status = 'ACTIVE' AND rent IS NOT NULL AND neighborhood IS NOT NULL
            GROUP BY neighborhood
            ORDER BY avg_rent
            """
        )
        points = [
            charts._point(row["neighborhood"], row["avg_rent"], row.get("lat"), row.get("lng"))
            for row in rows
        ]
        chart = charts.bar_chart("Average rent by neighborhood", points, unit="usd")
        return {"count": len(rows), "neighborhoods": rows, "chart": chart}

    return _cached_tool(
        "average_rent_by_neighborhood", {}, accounts.CHART_TTL_SECONDS, fetch, accounts.SHARED_ACCOUNT
    )


@_tool
def average_rent_by_unit_type(neighborhood: str | None = None) -> dict:
    """Average listed rent by apartment type: studio, 1-bedroom, 2-bedroom, and 3+.

    Args:
        neighborhood: Optional neighborhood or area name, such as 'Chelsea' or 'UWS'.
    """
    where, params = _active_where(neighborhood)
    cache_key = {"neighborhood": neighborhood or ""}

    def fetch() -> dict:
        rows = _query_rows(
            f"""
            SELECT {_UNIT_SQL} AS unit_type,
                   ROUND(AVG(rent)) AS avg_rent,
                   COUNT(*) AS listings
            FROM listings
            WHERE {where}
            GROUP BY unit_type
            ORDER BY MIN(beds)
            """,
            tuple(params),
        )
        points = [charts._point(row["unit_type"], row["avg_rent"]) for row in rows]
        title = "Average rent by apartment type"
        if neighborhood:
            title += f" in {neighborhood}"
        chart = charts.bar_chart(title, points, unit="usd")
        return {"neighborhood": neighborhood, "unit_types": rows, "chart": chart}

    return _cached_tool(
        "average_rent_by_unit_type", cache_key, accounts.CHART_TTL_SECONDS, fetch, accounts.SHARED_ACCOUNT
    )


@_tool
def listing_mix(by: str = "neighborhood", neighborhood: str | None = None) -> dict:
    """Share of active listings as a pie chart, by neighborhood or by apartment type.

    Args:
        by: 'neighborhood' or 'unit_type'. Defaults to neighborhood.
        neighborhood: Optional neighborhood filter when by is 'unit_type'.
    """
    kind = _clean(by).lower().replace(" ", "_") or "neighborhood"
    if kind in {"apartment_type", "beds", "unit", "bedrooms"}:
        kind = "unit_type"
    if kind not in {"neighborhood", "unit_type"}:
        raise ToolError("listing_mix by must be 'neighborhood' or 'unit_type'.")
    where, params = _active_where(neighborhood if kind == "unit_type" else None)
    label = _UNIT_SQL if kind == "unit_type" else "neighborhood"
    cache_key = {"by": kind, "neighborhood": neighborhood or ""}

    def fetch() -> dict:
        rows = _query_rows(
            f"""
            SELECT {label} AS label, COUNT(*) AS listings
            FROM listings
            WHERE {where} AND {label} IS NOT NULL
            GROUP BY label
            ORDER BY listings DESC
            """,
            tuple(params),
        )
        points = [charts._point(row["label"], row["listings"]) for row in rows]
        title = "Listings by neighborhood" if kind == "neighborhood" else "Listings by apartment type"
        if neighborhood and kind == "unit_type":
            title = f"Listings by apartment type in {neighborhood}"
        chart = charts.pie_chart(title, points)
        return {"by": kind, "rows": rows, "chart": chart}

    return _cached_tool(
        "listing_mix", cache_key, accounts.CHART_TTL_SECONDS, fetch, accounts.SHARED_ACCOUNT
    )


@_tool
def neighborhood_price_map() -> dict:
    """Neighborhood map colored by average rent. Each point is one neighborhood."""
    def fetch() -> dict:
        rows = _query_rows(
            """
            SELECT neighborhood, ROUND(AVG(rent)) AS avg_rent, COUNT(*) AS listings,
                   AVG(lat) AS lat, AVG(lng) AS lng
            FROM listings
            WHERE listing_status = 'ACTIVE' AND rent IS NOT NULL AND neighborhood IS NOT NULL
              AND lat IS NOT NULL AND lng IS NOT NULL
            GROUP BY neighborhood
            ORDER BY avg_rent
            """
        )
        points = [
            charts._point(row["neighborhood"], row["avg_rent"], row.get("lat"), row.get("lng"))
            for row in rows
        ]
        chart = charts.map_chart("Average rent by neighborhood", points)
        return {"neighborhoods": rows, "chart": chart}

    return _cached_tool(
        "neighborhood_price_map", {}, accounts.CHART_TTL_SECONDS, fetch, accounts.SHARED_ACCOUNT
    )


def _listing_place(listing_id: str) -> dict:
    rows = _query_rows(
        """
        SELECT listing_id, name, street, unit, neighborhood, lat, lng
        FROM listings
        WHERE listing_id = ?
        """,
        (listing_id,),
    )
    if not rows:
        raise ToolError(f"No listing '{listing_id}'.")
    row = rows[0]
    if row.get("lat") is None or row.get("lng") is None:
        street = _clean(row.get("street") or row.get("name"))
        neighborhood = _clean(row.get("neighborhood") or "New York, NY")
        if street:
            found = search_places.__wrapped__(street, neighborhood, 1)
            if isinstance(found, str):
                found = json.loads(found)
            places = (found or {}).get("places") or []
            if places and places[0].get("lat") is not None:
                row["lat"] = places[0]["lat"]
                row["lng"] = places[0]["lng"]
                row["geocoded"] = True
    return row


@_tool
def show_listing_map(listing_id: str) -> dict:
    """Return the map pin for one listing. Uses stored coordinates, or geocodes the street once.

    Args:
        listing_id: The listing_id from a previous search or deal result.
    """
    row = _listing_place(_clean(listing_id))
    if row.get("lat") is None or row.get("lng") is None:
        raise ToolError(f"No coordinates for listing '{listing_id}'.")
    return {
        "listing_id": row["listing_id"],
        "building": row.get("name"),
        "street": row.get("street"),
        "unit": row.get("unit"),
        "neighborhood": row.get("neighborhood"),
        "lat": row.get("lat"),
        "lng": row.get("lng"),
        "geocoded": bool(row.get("geocoded")),
    }


@_tool
def neighborhood_activities(neighborhood: str) -> dict:
    """Places to go in a neighborhood: parks, restaurants, and museums.

    Args:
        neighborhood: Neighborhood name, such as 'Chelsea' or 'Hell's Kitchen'.
    """
    try:
        name = _resolve_neighborhood(neighborhood)
    except ToolError:
        resolved = _resolve_areas(neighborhood)
        name = resolved[0] if resolved else _clean(neighborhood)
    found = search_places.__wrapped__(f"things to do in {name}", f"{name}, New York, NY", 5)
    if isinstance(found, str):
        found = json.loads(found)
    if isinstance(found, dict) and found.get("error"):
        raise ToolError(found["error"])
    return {"neighborhood": name, "places": (found or {}).get("places") or []}


def _commute_one(rental, listing_id: str, destination: str, mode: str) -> dict:
    from rental_context import CURRENT

    token = CURRENT.set(rental)
    try:
        row = _listing_place(listing_id)
        origin = ", ".join(
            part for part in (row.get("street"), row.get("neighborhood"), "New York, NY") if part
        )
        route = get_directions.__wrapped__(origin, destination, mode)
        return {
            "listing_id": row["listing_id"],
            "building": row.get("name"),
            "street": row.get("street"),
            "neighborhood": row.get("neighborhood"),
            "lat": row.get("lat"),
            "lng": row.get("lng"),
            "origin": origin,
            "destination": route.get("destination"),
            "mode": route.get("mode"),
            "duration_minutes": route.get("duration_minutes"),
            "distance_miles": route.get("distance_miles"),
            "encoded_polyline": route.get("encoded_polyline"),
        }
    except ToolError as exc:
        return {"listing_id": listing_id, "error": str(exc)}
    finally:
        CURRENT.reset(token)


def _split_ids(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = str(value or "").replace(";", ",").split(",")
    ids = []
    for part in parts:
        cleaned = _clean(part)
        if cleaned and cleaned not in ids:
            ids.append(cleaned)
    return ids[:4]


async def commute_from_listing(
    ctx: RunContextWrapper,
    listing_ids: str,
    destination: str = "",
    mode: str = "",
) -> str:
    """Travel time from one or more listings to an address. Say 'work' to use the saved work address.

    Up to four listings are routed in parallel.

    Args:
        listing_ids: One listing_id, or several separated by commas.
        destination: Street or place. Use 'work' for the profile work address. Optional when the profile has one.
        mode: transit, walk, drive, or bicycle. Optional. Defaults to the profile, then transit.
    """
    rental = as_rental(ctx)
    bind(ctx)
    profile = rental.profile if rental else {}
    dest = _clean(destination)
    if not dest or dest.casefold() in {"work", "my work", "the office"}:
        dest = _clean(profile.get("work_address"))
    if not dest:
        return json.dumps({"error": "A destination or a saved work address is required."})
    travel = _clean(mode).lower() or _clean(profile.get("commute_mode")).lower() or "transit"
    ids = _split_ids(listing_ids)
    if not ids:
        return json.dumps({"error": "commute_from_listing needs at least one listing_id."})
    rows = await asyncio.gather(
        *[asyncio.to_thread(_commute_one, rental, listing_id, dest, travel) for listing_id in ids]
    )
    ok = next((row for row in rows if row.get("duration_minutes") is not None), None)
    if ok and rental and rental.account_id:
        summary = (
            f"{ok.get('street') or ok['listing_id']} to {dest}: "
            f"{ok['duration_minutes']} min by {ok.get('mode') or travel}"
        )
        accounts.set_last_commute(rental.account_id, summary)
        rental.profile = accounts.get_profile(rental.account_id)
    return json.dumps({
        "destination": dest,
        "mode": travel,
        "listings": list(rows),
    }, default=str)


@function_tool(needs_approval=True, strict_mode=False, tool_input_guardrails=[bind_rental])
def save_profile(
    ctx: RunContextWrapper,
    max_rent: float | None = None,
    min_beds: float | None = None,
    neighborhoods: str | None = None,
    work_address: str | None = None,
    commute_mode: str | None = None,
) -> str:
    """Save budget, size, neighborhoods, or work address on the signed-in account.

    Call this when the user states a budget, a bedroom count, neighborhoods they want,
    or where they work. The user must approve the change before it is stored.

    Args:
        max_rent: Monthly budget ceiling in dollars. Omit to leave it unchanged.
        min_beds: Minimum bedrooms. Use 0 for a studio. Omit to leave it unchanged.
        neighborhoods: Comma-separated neighborhood names. Omit to leave them unchanged.
        work_address: Street or place they commute to. Omit to leave it unchanged.
        commute_mode: transit, walk, drive, or bicycle. Omit to leave it unchanged.
    """
    rental = as_rental(ctx)
    if rental is None or not rental.account_id:
        return json.dumps({"error": "Sign in before saving a profile."})
    fields: dict = {}
    if max_rent is not None:
        fields["max_rent"] = max_rent
    if min_beds is not None:
        fields["min_beds"] = min_beds
    if neighborhoods is not None:
        fields["neighborhoods"] = [part.strip() for part in str(neighborhoods).split(",") if part.strip()]
    if work_address is not None:
        fields["work_address"] = _clean(work_address)
    if commute_mode is not None:
        mode = _clean(commute_mode).lower()
        if mode and mode not in TRAVEL_MODES:
            return json.dumps({"error": f"Unknown commute mode '{commute_mode}'."})
        fields["commute_mode"] = mode or "transit"
    profile = accounts.save_profile(rental.account_id, fields)
    rental.profile = profile
    return json.dumps({"saved": True, "profile": profile})


def _commute_enabled(ctx, agent) -> bool:
    rental = as_rental(ctx)
    message = (rental.user_message if rental else "").casefold()
    profile = rental.profile if rental else {}
    words = (
        "commute", "directions", "how long", "how far", "travel time",
        "route", "to work", "from this", "from the",
    )
    if any(word in message for word in words):
        return True
    return bool(profile.get("work_address") and "work" in message)


def _agent_tool(fn, **kwargs):
    """Expose a tool to the Agents SDK without hiding its __wrapped__ function."""
    guards = [bind_rental, *list(kwargs.pop("tool_input_guardrails", []) or [])]
    return function_tool(fn, strict_mode=False, tool_input_guardrails=guards, **kwargs)


ANALYSIS_TOOLS = [
    _agent_tool(search_listings),
    _agent_tool(get_building_amenities),
    _agent_tool(get_deals),
    _agent_tool(cheapest_in_neighborhood),
    _agent_tool(best_deals_in_neighborhood),
    _agent_tool(compare_neighborhoods),
    _agent_tool(summarize_neighborhood),
    _agent_tool(average_rent_by_neighborhood),
    _agent_tool(average_rent_by_unit_type),
    _agent_tool(listing_mix),
    _agent_tool(neighborhood_price_map),
    save_profile,
]

PLACES_TOOLS = [
    _agent_tool(search_places, tool_input_guardrails=[places_budget]),
    _agent_tool(
        get_directions,
        is_enabled=_commute_enabled,
        tool_input_guardrails=[places_budget],
    ),
    _agent_tool(show_listing_map, tool_input_guardrails=[places_budget]),
    _agent_tool(neighborhood_activities, tool_input_guardrails=[places_budget]),
    function_tool(
        commute_from_listing,
        strict_mode=False,
        tool_input_guardrails=[bind_rental, places_budget],
    ),
]
