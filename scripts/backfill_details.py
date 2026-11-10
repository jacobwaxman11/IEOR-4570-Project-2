"""Backfill amenities/description/availability/floor plan from StreetEasy detail pages.

This is the second stage of the pipeline. The scraper (`scrape_listings.py`)
collects basic listing data + images from search pages and writes them to
``rentals.db``. This script then walks through listings whose detail-page
fields haven't been filled in yet, fetches each detail page, and updates the DB.

Design notes:
  - Listings are NEVER deleted. We only add to fields that are not yet populated.
  - A listing is considered "needs backfill" when its `listing_amenities` table
    has no rows for it.
  - We use `curl_cffi` (mimics Chrome's TLS fingerprint). Do not pass a custom
    User-Agent: on curl_cffi 0.17 that replaces the impersonation headers, and
    PerimeterX 403s the mismatched fingerprint. The library's own headers work.
  - Default delays are intentionally generous (4–8s) to avoid bans.
  - On the first 403 we **terminate immediately** — that signals the IP has
    been flagged, and continuing risks a longer block. Re-run later.
  - Saves are incremental (one commit per listing), so a Ctrl-C or 403 mid-run
    keeps everything we've already backfilled.

Example (from this directory; uv uses the project environment):
    uv run backfill_details.py
    uv run backfill_details.py --max-listings 50 --min-delay 8 --max-delay 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

# `_download_one` is a generic HTTP-download helper that lives next to the
# scraper's image downloads; reusing it here keeps that logic in one place.
from scrape_listings import _download_one

import db as dbm


# ── Detail-page parsing ──────────────────────────────────────────────────────


def find_floor_plan(soup: BeautifulSoup) -> Optional[str]:
    """Best-effort extraction of a floor plan image URL from a detail page.

    StreetEasy's HTML changes over time. We look for common signals:
      - <img> with alt/class containing "floor plan" / "floorplan"
      - <a> with class/data attribute pointing to a floor plan
      - Any zillowstatic image whose alt text mentions floor plan
    Returns None if nothing looks like a floor plan.
    """
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").lower()
        cls = " ".join(img.get("class") or []).lower()
        if "floor plan" in alt or "floorplan" in cls or "floor-plan" in cls:
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                return src

    for a in soup.find_all("a"):
        cls = " ".join(a.get("class") or []).lower()
        data_cat = (a.get("data-gtm-category") or "").lower()
        if "floor-plan" in cls or "floorplan" in cls or "floorplan" in data_cat:
            href = a.get("href")
            if href and href.startswith("http"):
                return href

    return None


def parse_detail(graph: list, soup: BeautifulSoup) -> dict:
    """Extract amenities/description/available_from/floor_plan from a detail page."""
    result: dict = {}
    for item in graph:
        t = item.get("@type")
        if t == "Apartment" and "amenityFeature" in item:
            result["amenities"] = [
                f["name"]
                for f in item["amenityFeature"]
                if f.get("value") is True
            ]
        if t == "ItemPage":
            about = item.get("about", {}) or {}
            result["description"] = about.get("description")
            offers = about.get("offers") or {}
            result["available_from"] = offers.get("validFrom")

    floor_plan_url = find_floor_plan(soup)
    if floor_plan_url:
        result["floor_plan_url"] = floor_plan_url
    return result


# ── Listing-level helpers ────────────────────────────────────────────────────


def fetch_detail(
    url: str, session: "cffi_requests.Session", timeout: int = 20
) -> tuple[Optional[dict], int]:
    """Fetch and parse a detail page. Returns (detail_dict_or_None, status_code).

    A non-200 response yields (None, status). We don't raise on 4xx so the
    caller can decide what to do (e.g. terminate on 403).
    """
    # Leave headers to the impersonation profile. Passing our own User-Agent
    # (Chrome/122) makes curl_cffi drop sec-ch-ua / sec-fetch-* and PerimeterX
    # answers 403, even though the same URL returns the listing with the
    # profile's own headers.
    try:
        response = session.get(url, timeout=timeout)
    except Exception as e:
        print(f"    ✗ request failed: {e}")
        return None, -1

    status = response.status_code
    if status != 200:
        return None, status

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("script", {"type": "application/ld+json"})
    graph: list = []
    if tag:
        try:
            graph = json.loads(tag.string).get("@graph", [])
        except json.JSONDecodeError:
            graph = []

    return parse_detail(graph, soup), status


def maybe_download_floor_plan(
    listing: dict, image_dir: str, session: "cffi_requests.Session"
) -> Optional[str]:
    """Download the listing's floor plan if we have a URL but no local file."""
    url = listing.get("floor_plan_url")
    if not url or listing.get("local_floor_plan_path"):
        return listing.get("local_floor_plan_path")

    building = listing.get("building_slug") or listing["listing_id"]
    unit = listing.get("unit") or listing["listing_id"]
    unit_dir = os.path.join(image_dir, building, unit)
    os.makedirs(unit_dir, exist_ok=True)

    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    dest = os.path.join(unit_dir, f"floor_plan{ext}")

    if _download_one(url, dest, session):
        return dest
    return None


def merge_detail(listing: dict, detail: dict) -> dict:
    """Compute the subset of detail fields that should be written to the DB.

    We only fill in fields that are currently missing, so re-running the
    backfill after a partial success can't clobber good data.
    """
    updates: dict = {}
    for key in ("description", "available_from", "floor_plan_url"):
        if not listing.get(key) and detail.get(key):
            updates[key] = detail[key]
    return updates


# ── Main ─────────────────────────────────────────────────────────────────────


def run(
    db_path: str,
    image_dir: str,
    max_listings: Optional[int],
    min_delay: float,
    max_delay: float,
    impersonate: str,
    timeout: int,
) -> int:
    """Returns the exit code (0 = ok, 2 = stopped on 403)."""
    with dbm.open_db(db_path) as conn:
        todo = dbm.listings_missing_amenities(conn)
        total = dbm.stats(conn)["listings"]
        print(
            f"DB: {db_path} — {total} listings total; "
            f"{len(todo)} need detail backfill."
        )

        if max_listings is not None:
            todo = todo[:max_listings]
            print(f"Capped at --max-listings {max_listings}.")

        if not todo:
            print("Nothing to do. Exiting.")
            return 0

        session = cffi_requests.Session(impersonate=impersonate)
        print(f"Using curl_cffi with impersonate={impersonate!r}\n")

        processed = 0
        succeeded = 0

        for n, listing in enumerate(todo, start=1):
            url = listing.get("url")
            name = listing.get("name") or listing["listing_id"]
            print(f"── [{n}/{len(todo)}] {name} ──")

            if not url:
                print("  ✗ no url, skipping")
                continue

            delay = random.uniform(min_delay, max_delay)
            print(f"  sleeping {delay:.1f}s...")
            time.sleep(delay)

            print(f"  → GET {url}")
            detail, status = fetch_detail(url, session, timeout=timeout)
            processed += 1

            if status == 403:
                print(
                    "\n!!! 403 (PerimeterX) — terminating to avoid further bans. !!!\n"
                    f"Processed {processed - 1} successful, "
                    f"{n - processed} skipped, before block.\n"
                    "Re-run later (try a longer delay or a different IP/VPN)."
                )
                return 2

            if detail is None:
                print(f"  ✗ status={status}, no detail extracted")
                continue

            # Apply updates to the listings row (description / available_from /
            # floor_plan_url) and persist amenities into their dedicated table.
            updates = merge_detail(listing, detail)
            if updates:
                dbm.update_listing_fields(conn, listing["listing_id"], updates)

            if detail.get("amenities"):
                dbm.set_amenities(conn, listing["listing_id"], detail["amenities"])
                print(f"  ✓ {len(detail['amenities'])} amenities")
            if "description" in updates and updates["description"]:
                print(f"  ✓ description ({len(updates['description'])} chars)")
            if "available_from" in updates:
                print(f"  ✓ available_from: {updates['available_from']}")
            if "floor_plan_url" in updates:
                print("  ✓ floor_plan_url found")

            # Floor plan download (best-effort, doesn't count against 403 logic).
            floor_plan_url = updates.get("floor_plan_url") or listing.get("floor_plan_url")
            if floor_plan_url and not listing.get("local_floor_plan_path"):
                fp_payload = {
                    **listing,
                    "floor_plan_url": floor_plan_url,
                }
                fp_path = maybe_download_floor_plan(fp_payload, image_dir, session)
                if fp_path:
                    dbm.update_listing_fields(
                        conn, listing["listing_id"], {"local_floor_plan_path": fp_path}
                    )
                    print(f"  ✓ floor plan saved: {fp_path}")

            dbm.mark_detail_fetched(conn, listing["listing_id"])
            conn.commit()
            succeeded += 1
            print()

        print(
            f"── Done. {succeeded}/{processed} backfills succeeded "
            f"({len(todo) - processed} not attempted) ──"
        )
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill amenities + description + floor plan into the SQLite DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Where to save floor-plan images",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Cap on listings to process per run (useful for testing)",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=4.0,
        help="Min sleep before each detail fetch (seconds)",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=8.0,
        help="Max sleep before each detail fetch (seconds)",
    )
    p.add_argument(
        "--impersonate",
        type=str,
        default="chrome",
        help="curl_cffi browser fingerprint. 'chrome' tracks the library's current Chrome profile",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-request timeout in seconds",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    code = run(
        db_path=args.db,
        image_dir=args.image_dir,
        max_listings=args.max_listings,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        impersonate=args.impersonate,
        timeout=args.timeout,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
