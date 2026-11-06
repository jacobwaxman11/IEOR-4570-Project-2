"""Refresh rent and listing status for rows that have gone stale.

``last_updated_date`` is the as-of date of the rent and status stored on a
listing. This script selects rows whose as-of date is older than a week
(or missing), fetches each listing's StreetEasy page, and writes the latest
rent and status back. ``first_scraped_at`` is never changed.

One detail page is requested per listing, with a pause between requests.
The first 403 stops the run so a blocked IP is not hit again. Pass
``--max-listings-checked`` to cap how many pages a single run requests.

Example (from this directory; uv uses the project environment):
    uv run update_listings.py
    uv run update_listings.py --max-listings-checked 25
    uv run update_listings.py --stale-days 0 --max-listings-checked 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

import db as dbm
from scrape_listings import HEADERS


# Current listing objects look like "status":"ACTIVE","statusChanges":[...]
# Historical events also contain "status", so the statusChanges neighbor is
# what identifies the live value.
_STATUS_RE = re.compile(r'"status":"([A-Z_]+)","statusChanges"')
_PRICE_RE = re.compile(r'"pricing":\{"price":(\d+)')
_OFF_MARKET_RE = re.compile(r'"offMarketAt":"(\d{4}-[^"]*)"')

STALE_DAYS = 7


def _unit_slug(listing: dict) -> str:
    path = urlparse(listing.get("url") or "").path.rstrip("/")
    if path:
        return path.split("/")[-1].lower()
    return str(listing.get("unit") or "").lstrip("#").lower()


def _rent_from_ldjson(html: str) -> Optional[int]:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", {"type": "application/ld+json"})
    if not tag or not tag.string:
        return None
    try:
        graph = json.loads(tag.string).get("@graph", [])
    except json.JSONDecodeError:
        return None
    for item in graph:
        if item.get("@type") != "Apartment":
            continue
        for prop in item.get("additionalProperty") or []:
            if prop.get("name") != "Monthly Rent":
                continue
            digits = (
                str(prop.get("value") or "")
                .replace("$", "")
                .replace(",", "")
                .replace("/mo", "")
                .strip()
            )
            if digits.isdigit():
                return int(digits)
    return None


def parse_market_fields(html: str, unit_slug: str) -> dict:
    """Pull the live status and asking rent out of a listing detail page.

    Returns a dict with whichever of ``listing_status`` and ``rent`` could
    be read. An off-market page with no live status contributes
    ``listing_status="OFF_MARKET"``.
    """
    raw = html.replace('\\"', '"')
    window = raw
    if unit_slug:
        anchor = raw.lower().find(f'"slug":"{unit_slug}"')
        if anchor >= 0:
            window = raw[max(0, anchor - 25000): anchor + 400]

    found: dict = {}
    status = _STATUS_RE.search(window) or _STATUS_RE.search(raw)
    if status:
        found["listing_status"] = status.group(1)
    elif _OFF_MARKET_RE.search(window) or _OFF_MARKET_RE.search(raw):
        found["listing_status"] = "OFF_MARKET"

    price = _PRICE_RE.search(window) or _PRICE_RE.search(raw)
    if price:
        found["rent"] = int(price.group(1))
    else:
        rent = _rent_from_ldjson(html)
        if rent is not None:
            found["rent"] = rent
    return found


def fetch_market(
    url: str, unit_slug: str, client: httpx.Client, timeout: int
) -> tuple[Optional[dict], int]:
    """GET one detail page. ``(None, status)`` on a non-200 or a transport error."""
    try:
        response = client.get(url, timeout=timeout)
    except Exception as e:
        print(f"    ✗ request failed: {e}")
        return None, -1

    status = response.status_code
    if status in (404, 410):
        return {"listing_status": "OFF_MARKET"}, status
    if status != 200:
        return None, status
    return parse_market_fields(response.text, unit_slug), status


def run(
    db_path: str,
    stale_days: int,
    max_listings_checked: Optional[int],
    min_delay: float,
    max_delay: float,
    timeout: int,
) -> int:
    """Returns 0 on success, 2 when StreetEasy answers 403."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
    stale_before = cutoff.isoformat(timespec="seconds")

    with dbm.open_db(db_path) as conn:
        due = dbm.listings_due_for_refresh(conn, stale_before, max_listings_checked)
        print(
            f"DB: {db_path} — {len(due)} listing(s) with last_updated_date "
            f"older than {stale_days} day(s) (before {stale_before})."
        )
        if max_listings_checked is not None:
            print(f"Capped at --max-listings-checked {max_listings_checked}.")
        if not due:
            print("Nothing to update.")
            return 0

        checked = 0
        updated = 0
        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            for n, listing in enumerate(due, start=1):
                url = listing.get("url")
                name = listing.get("name") or listing["listing_id"]
                print(f"── [{n}/{len(due)}] {name} ──")
                if not url:
                    print("  ✗ no url, skipping")
                    continue

                delay = random.uniform(min_delay, max_delay)
                print(f"  sleeping {delay:.1f}s...")
                time.sleep(delay)

                print(f"  → GET {url}")
                fields, status = fetch_market(url, _unit_slug(listing), client, timeout)
                checked += 1

                if status == 403:
                    print(
                        "\n!!! 403 — stopping so this IP is not requested again. !!!\n"
                        f"Updated {updated} listing(s) before the block."
                    )
                    return 2

                if not fields or (
                    fields.get("listing_status") is None and fields.get("rent") is None
                ):
                    print(f"  ✗ status={status}, rent and listing status not found")
                    continue

                dbm.refresh_listing_market(
                    conn,
                    listing["listing_id"],
                    rent=fields.get("rent"),
                    listing_status=fields.get("listing_status"),
                )
                conn.commit()
                updated += 1
                print(
                    f"  ✓ status={fields.get('listing_status') or '(unchanged)'} "
                    f"rent={fields.get('rent') if fields.get('rent') is not None else '(unchanged)'}"
                )

        print(f"── Done. Updated {updated}/{checked} listing(s). ──")
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refresh rent and status for listings older than a week.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument(
        "--stale-days",
        type=int,
        default=STALE_DAYS,
        help="Refresh listings whose last_updated_date is older than this many days",
    )
    p.add_argument(
        "--max-listings-checked",
        type=int,
        default=None,
        help="Cap how many stale listings to request from StreetEasy this run",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=5.0,
        help="Min sleep before each detail-page request (seconds)",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=12.0,
        help="Max sleep before each detail-page request (seconds)",
    )
    p.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_listings_checked is not None and args.max_listings_checked < 0:
        raise SystemExit("--max-listings-checked must be >= 0")
    if args.stale_days < 0:
        raise SystemExit("--stale-days must be >= 0")
    code = run(
        db_path=args.db,
        stale_days=args.stale_days,
        max_listings_checked=args.max_listings_checked,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        timeout=args.timeout,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
