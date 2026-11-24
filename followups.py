"""Follow-up prompts built from the turn that just finished.

The text only names streets, neighborhoods, and chart types present in the tool
results, so a click cannot ask about a unit the answer did not return.
"""

from __future__ import annotations


def _clean(text) -> str:
    return " ".join(str(text or "").split())


def suggest(question: str, tool_calls: list[dict], listings: list[dict], charts: list[dict], profile: dict | None) -> list[str]:
    profile = profile or {}
    work = _clean(profile.get("work_address"))
    names = {str(call.get("name") or "") for call in tool_calls or []}
    ideas: list[str] = []
    top = listings[0] if listings else None
    street = _clean((top or {}).get("street") or (top or {}).get("building"))
    neighborhood = _clean((top or {}).get("neighborhood"))
    if top and work and street and "commute_from_listing" not in names:
        ideas.append(f"How long from {street} to {work}?")
    if neighborhood and "neighborhood_activities" not in names:
        ideas.append(f"What's there to do in {neighborhood}?")
    if top and top.get("lat") is not None and street and "show_listing_map" not in names:
        ideas.append(f"Show {street} on a map")
    chart_titles = " ".join(_clean(chart.get("title")) for chart in charts or []).casefold()
    compared = "compare_neighborhoods" in names or "average_rent_by_neighborhood" in names or "neighborhood" in chart_titles
    if compared and "average_rent_by_unit_type" not in names:
        ideas.append("Show average rent by unit type")
    if compared and "listing_mix" not in names:
        ideas.append("Show the share of listings by apartment type")
    if not ideas:
        ideas.append("What is the average rent by neighborhood?")
        ideas.append("Show a neighborhood heat map of average rent")
    asked = _clean(question).casefold()
    unique = []
    for idea in ideas:
        if idea.casefold() == asked or idea in unique:
            continue
        unique.append(idea)
        if len(unique) == 3:
            break
    return unique
