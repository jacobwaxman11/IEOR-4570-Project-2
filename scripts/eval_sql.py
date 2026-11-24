"""Gold checks for rental SQL, charts, and the optional agent run.

`uv run scripts/eval_sql.py` checks oracle SQL and the chart tools against the
local database. Add `--agent` to also run the chat agent and score whether its
tool payload contains the oracle listing or the expected chart type.

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, each case is written
to the Langfuse dataset `rental-sql-gold` and scored as `sql_expected_hit`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import observe  # noqa: E402
from tools import (  # noqa: E402
    _query_rows,
    average_rent_by_neighborhood,
    average_rent_by_unit_type,
    listing_mix,
    neighborhood_price_map,
)

DATASET = "rental-sql-gold"

LISTING_CASES = [
    {
        "id": "chelsea-cheapest-1bed",
        "question": "What is the cheapest active 1-bedroom in Chelsea? Give the address, square feet, and monthly rent.",
        "sql": """
            SELECT listing_id, street, unit, rent
            FROM listings
            WHERE listing_status = 'ACTIVE' AND neighborhood = 'Chelsea' AND beds = 1 AND rent IS NOT NULL
            ORDER BY rent, listing_id
            LIMIT 1
        """,
    },
    {
        "id": "hells-kitchen-best-deal",
        "question": "What is the single best model deal in Hell's Kitchen? Give the address, bedrooms, listed rent, predicted rent, percent discount, and deal score.",
        "sql": """
            SELECT l.listing_id, l.street, l.unit, r.deal_score
            FROM listings l
            JOIN regression_results r ON l.listing_id = r.listing_id
            WHERE l.listing_status = 'ACTIVE' AND l.neighborhood = 'Hell''s Kitchen'
            ORDER BY r.deal_score DESC, r.percent_discount DESC
            LIMIT 1
        """,
    },
    {
        "id": "chelsea-amenities",
        "question": "What are the five most common amenities among Chelsea listings, and how many Chelsea listings have each?",
        "sql": """
            SELECT a.amenity AS label, COUNT(DISTINCT l.listing_id) AS listings
            FROM listings l
            JOIN listing_amenities a ON l.listing_id = a.listing_id
            WHERE l.listing_status = 'ACTIVE' AND l.neighborhood = 'Chelsea'
            GROUP BY a.amenity
            ORDER BY listings DESC, a.amenity
            LIMIT 5
        """,
        "labels": True,
    },
    {
        "id": "uws-1bed-deal",
        "question": "What is the best 1-bedroom deal on the Upper West Side with rent at or below $5000? Give the address, the neighborhood stored on the listing, the listed rent, the predicted rent, and the deal score.",
        "sql": """
            SELECT l.listing_id, l.street, l.neighborhood, r.deal_score
            FROM listings l
            JOIN regression_results r ON l.listing_id = r.listing_id
            WHERE l.listing_status = 'ACTIVE' AND l.beds = 1 AND l.rent <= 5000
              AND l.neighborhood IN ('Upper West Side', 'Lincoln Square', 'Manhattan Valley', 'Morningside Heights')
            ORDER BY r.deal_score DESC
            LIMIT 1
        """,
    },
    {
        "id": "uws-2bed-deal",
        "question": "What is the best 2-bedroom deal on the Upper West Side? Give the address, the neighborhood stored on the listing, the listed rent, the predicted rent, and the deal score.",
        "sql": """
            SELECT l.listing_id, l.street, l.neighborhood, r.deal_score
            FROM listings l
            JOIN regression_results r ON l.listing_id = r.listing_id
            WHERE l.listing_status = 'ACTIVE' AND l.beds = 2
              AND l.neighborhood IN ('Upper West Side', 'Lincoln Square', 'Manhattan Valley', 'Morningside Heights')
            ORDER BY r.deal_score DESC
            LIMIT 1
        """,
    },
]

CHART_CASES = [
    {
        "id": "avg-rent-neighborhood",
        "question": "What is the average rent by neighborhood?",
        "tool": average_rent_by_neighborhood,
        "chart": "bar",
    },
    {
        "id": "avg-rent-unit-type",
        "question": "What is the average rent by apartment type?",
        "tool": average_rent_by_unit_type,
        "chart": "bar",
    },
    {
        "id": "mix-neighborhood",
        "question": "Break down listings by neighborhood",
        "tool": lambda: listing_mix("neighborhood"),
        "chart": "pie",
    },
    {
        "id": "price-map",
        "question": "Show a neighborhood heat map of average rent",
        "tool": neighborhood_price_map,
        "chart": "map",
    },
]


def _parse(raw):
    if isinstance(raw, dict):
        return raw
    return json.loads(raw)


def _hit_text(blob: str, expected: dict) -> bool:
    folded = blob.casefold()
    listing_id = str(expected.get("listing_id") or "")
    if listing_id and listing_id.casefold() in folded:
        return True
    street = str(expected.get("street") or "")
    if street and street.casefold() in folded:
        return True
    chart = expected.get("chart")
    if chart and chart in folded:
        return True
    labels = expected.get("labels") or []
    return bool(labels) and all(str(label).casefold() in folded for label in labels[:2])


def _publish(case: dict, expected: dict, hit: bool, output: str) -> None:
    langfuse = observe.client()
    if langfuse is None:
        return
    try:
        try:
            langfuse.create_dataset(
                name=DATASET,
                description="Gold rental questions for SQL, charts, and commute answers.",
            )
        except Exception:
            pass
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            id=f"rental-sql-{case['id']}",
            input={"question": case["question"]},
            expected_output=expected,
        )
        span = langfuse.start_observation(
            name="sql_eval",
            as_type="span",
            input={"question": case["question"]},
            output={"preview": output[:500]},
            metadata={"dataset": DATASET, "case": case["id"]},
        )
        span.score(
            name="sql_expected_hit",
            value=1 if hit else 0,
            comment=case["id"],
        )
        span.end()
        langfuse.flush()
    except Exception as exc:
        print(f"  langfuse: {type(exc).__name__}: {exc}")


def _listing_checks() -> list[tuple[str, bool]]:
    results = []
    for case in LISTING_CASES:
        rows = _query_rows(case["sql"])
        if case.get("labels"):
            labels = [row.get("label") for row in rows if row.get("label")]
            hit = len(labels) >= 5
            expected = {"labels": labels}
            print(f"{'ok' if hit else 'miss'}  {case['id']}: {', '.join(labels)}")
        else:
            hit = bool(rows and rows[0].get("listing_id"))
            expected = {"listing_id": (rows[0] if rows else {}).get("listing_id"), "street": (rows[0] if rows else {}).get("street")}
            print(f"{'ok' if hit else 'miss'}  {case['id']}: {expected.get('listing_id')}")
        _publish(case, expected, hit, json.dumps(expected, default=str))
        results.append((case["id"], hit))
    return results


def _chart_checks() -> list[tuple[str, bool]]:
    results = []
    for case in CHART_CASES:
        payload = _parse(case["tool"].__wrapped__() if hasattr(case["tool"], "__wrapped__") else case["tool"]())
        chart = payload.get("chart") or {}
        points = chart.get("points") or []
        hit = chart.get("type") == case["chart"] and len(points) >= 2
        print(f"{'ok' if hit else 'miss'}  {case['id']}: {chart.get('type')} ({len(points)} points)")
        _publish(
            case,
            {"chart": case["chart"], "labels": [point.get("label") for point in points[:3]]},
            hit,
            json.dumps({"chart": chart.get("type"), "points": len(points)}),
        )
        results.append((case["id"], hit))
    return results


def _commute_check() -> list[tuple[str, bool]]:
    """The commute tool asks for a work address before it calls Google."""
    from rental_context import CURRENT, RentalContext
    from tools import commute_from_listing

    class Ctx:
        def __init__(self, rental):
            self.context = rental

    rental = RentalContext(
        account_id="eval-commute",
        profile={},
        user_message="How long from this apartment to work?",
    )
    CURRENT.set(rental)

    async def run():
        return await commute_from_listing(Ctx(rental), "chelsea-tower_6h", "work", "transit")

    payload = _parse(asyncio.run(run()))
    hit = "work address" in str(payload.get("error") or "").casefold()
    print(f"{'ok' if hit else 'miss'}  commute-needs-work-address")
    _publish(
        {
            "id": "commute-to-work",
            "question": "How long from this apartment to work?",
        },
        {"needs": "work_address"},
        hit,
        json.dumps(payload),
    )
    return [("commute-needs-work-address", hit)]


async def _agent_checks() -> list[tuple[str, bool]]:
    from agents import Runner

    from app import agent
    from rental_context import CURRENT, RentalContext

    results = []
    cases = list(LISTING_CASES) + list(CHART_CASES)
    for case in cases:
        if "sql" in case:
            rows = _query_rows(case["sql"])
            if case.get("labels"):
                expected = {"labels": [row.get("label") for row in rows if row.get("label")]}
            else:
                expected = rows[0] if rows else {}
        else:
            expected = {"chart": case["chart"]}
        ctx = RentalContext(account_id="eval", profile={}, user_message=case["question"])
        CURRENT.set(ctx)
        result = await Runner.run(agent, case["question"], context=ctx, max_turns=8)
        blob = json.dumps(result.to_input_list() if hasattr(result, "to_input_list") else {}, default=str)
        blob += "\n" + str(result.final_output or "")
        hit = _hit_text(blob, expected)
        print(f"{'ok' if hit else 'miss'}  agent {case['id']}")
        _publish(case, expected, hit, str(result.final_output or "")[:500])
        results.append((case["id"], hit))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Score gold rental questions.")
    parser.add_argument("--agent", action="store_true", help="Also run the chat agent.")
    args = parser.parse_args()
    results = _listing_checks() + _chart_checks() + _commute_check()
    if args.agent:
        results.extend(asyncio.run(_agent_checks()))
    missed = [name for name, hit in results if not hit]
    print(f"{len(results) - len(missed)}/{len(results)} passed")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
