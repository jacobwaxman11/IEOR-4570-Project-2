"""The tools the harness can run, and the JSON that describes them to the model."""

import functools
import json
import os
import time

import requests
from dotenv import load_dotenv

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
