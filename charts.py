"""Chart specs built from query rows. The model does not invent the numbers."""

from __future__ import annotations

UNIT_ORDER = ["Studio", "1-bedroom", "2-bedroom", "3+ bedrooms"]


def unit_label(beds) -> str:
    if beds is None or beds == "":
        return "Unknown"
    try:
        count = float(beds)
    except (TypeError, ValueError):
        return str(beds)
    if count < 1:
        return "Studio"
    if count < 2:
        return "1-bedroom"
    if count < 3:
        return "2-bedroom"
    return "3+ bedrooms"


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point(label, value, lat=None, lng=None) -> dict | None:
    number = _number(value)
    if number is None or label in (None, ""):
        return None
    point = {"label": str(label), "value": round(number, 2)}
    lat_n, lng_n = _number(lat), _number(lng)
    if lat_n is not None and lng_n is not None:
        point["lat"] = round(lat_n, 5)
        point["lng"] = round(lng_n, 5)
    return point


def bar_chart(title: str, points: list[dict], unit: str = "usd") -> dict:
    return {"type": "bar", "title": title, "unit": unit, "points": [p for p in points if p]}


def pie_chart(title: str, points: list[dict]) -> dict:
    return {"type": "pie", "title": title, "unit": "count", "points": [p for p in points if p]}


def map_chart(title: str, points: list[dict]) -> dict:
    return {"type": "map", "title": title, "unit": "usd", "points": [p for p in points if p]}


def _column_index(columns: list, *names: str) -> int:
    folded = [str(column).lower() for column in columns]
    for name in names:
        if name in folded:
            return folded.index(name)
    return -1


def _cell(row, index):
    if index < 0 or index >= len(row):
        return None
    return row[index]


def infer_chart(columns: list, rows: list, question: str = "") -> dict | None:
    """Pick a bar, pie, or map spec from an aggregate result. Listing rows return None."""
    if not columns or not rows or len(rows) < 2:
        return None
    names = {str(column).lower() for column in columns}
    if "listing_id" in names:
        return None
    label_at = _column_index(
        columns, "neighborhood", "unit_type", "label", "amenity", "beds", "name"
    )
    if label_at < 0:
        for index, column in enumerate(columns):
            sample = _cell(rows[0], index)
            if sample is not None and _number(sample) is None:
                label_at = index
                break
    value_at = _column_index(
        columns, "avg_rent", "average_rent", "median_rent", "rent", "listings", "units", "count", "value"
    )
    if value_at < 0:
        for index, column in enumerate(columns):
            if str(column).lower() in {"lat", "lng", "latitude", "longitude"}:
                continue
            if _number(_cell(rows[0], index)) is not None:
                value_at = index
                break
    if label_at < 0 or value_at < 0 or label_at == value_at:
        return None
    lat_at = _column_index(columns, "lat", "latitude")
    lng_at = _column_index(columns, "lng", "longitude")
    points = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        label = _cell(row, label_at)
        if str(columns[label_at]).lower() == "beds":
            label = unit_label(label)
        point = _point(label, _cell(row, value_at), _cell(row, lat_at), _cell(row, lng_at))
        if point:
            points.append(point)
    if len(points) < 2:
        return None
    asked = (question or "").casefold()
    title = "Result"
    if "neighborhood" in asked and "rent" in asked:
        title = "Average rent by neighborhood"
    elif "apartment type" in asked or "unit type" in asked or "bedroom" in asked:
        title = "Average rent by apartment type"
    unit = "count" if "count" in str(columns[value_at]).lower() or str(columns[value_at]).lower() in {"listings", "units"} else "usd"
    if any(word in asked for word in ("heat map", "heatmap", "price map")):
        return map_chart(title if title != "Result" else "Average rent by neighborhood", points)
    if any(word in asked for word in ("pie", "break down", "breakdown", "share of")):
        return pie_chart(title if title != "Result" else "Listing mix", points)
    return bar_chart(title, points, unit=unit)
