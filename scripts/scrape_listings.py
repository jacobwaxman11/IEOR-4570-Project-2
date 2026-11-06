"""Scrape StreetEasy rental listings (search pages) and download images.

This is stage 1 of the pipeline. It walks search-result pages, parses the
schema.org Apartment objects, downloads each listing's photos, and writes the
results to ``rentals.db``. Each new row records StreetEasy's listing status
(``ACTIVE`` on a for-rent search) plus ``first_scraped_at`` and
``last_updated_date``. Detail-page fields (amenities, description,
available_from, floor plan) are filled in afterwards by ``backfill_details.py``.

Re-runs are additive and idempotent: existing listing_ids are skipped, and only
newly-discovered listings get processed.

Example (from this directory; uv uses the project environment):
    uv run scrape_listings.py \\
        --price-min 4000 --price-max 6000 \\
        --areas 115,158,113,146,140 \\
        --max-listings 100

Area IDs (a few common ones):
    115 = Gramercy, 158 = Flatiron, 113 = Chelsea,
    146 = Hudson Yards, 140 = Upper East Side
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from typing import Optional
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

import db as dbm


# StreetEasy's ld+json currently interpolates an unset CDN host, so every
# photo URL looks like https://undefined/fp/<hash>-se_medium_500_250.webp.
# The hash and path are valid; the photos are served by the Zillow photo CDN.
PHOTO_CDN_HOST = "photos.zillowstatic.com"


def normalize_image_url(url: str) -> str:
    """Turn a StreetEasy image URL into one that can actually be fetched.

    Leaves already-absolute, working URLs unchanged. Rewrites a literal
    ``undefined`` host, and prefixes a bare ``/fp/...`` path.
    """
    if not url:
        return url
    if url.startswith("/fp/"):
        return f"https://{PHOTO_CDN_HOST}{url}"
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https") and parsed.netloc == "undefined":
        return urlunparse(parsed._replace(scheme="https", netloc=PHOTO_CDN_HOST))
    return url


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://streeteasy.com/",
}


# ── URL + parsing helpers ────────────────────────────────────────────────────


def build_url(price_min: int, price_max: int, area_ids: list[int], page: int = 1) -> str:
    areas = ",".join(str(a) for a in area_ids)
    path = f"/for-rent/nyc/price:{price_min}-{price_max}|area:{areas}"
    if page > 1:
        path += f"?page={page}"
    return f"https://streeteasy.com{path}"


def _street_key(value: str) -> str:
    return " ".join(str(value or "").lower().split("#")[0].split())


def _unit_key(value: str) -> str:
    text = "".join(ch for ch in str(value or "").lower() if ch.isalnum())
    return text.lstrip("0") or text


def listing_nodes_by_path(html: str) -> dict[str, dict]:
    """Index the embedded search results by listing path.

    Schema.org ``Apartment`` objects on the search page have rent and photos,
    but the StreetEasy status (``ACTIVE``, and later ``RENTED`` / ``DELISTED``)
    only appears in the Next.js payload as ``node.status``. That payload is a
    quoted JSON string, so the quotes are escaped.
    """
    marker = '\\"edges\\":['
    start = html.find(marker)
    if start < 0:
        return {}
    raw = "{" + html[start:].replace('\\"', '"')
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse embedded listing nodes: {e}")
        return {}

    nodes: dict[str, dict] = {}
    for edge in obj.get("edges") or []:
        node = (edge or {}).get("node") or {}
        path = (node.get("urlPath") or "").rstrip("/")
        if path:
            nodes[path] = node
    return nodes


def parse_listing(apt: dict) -> dict:
    """Extract fields from a schema.org Apartment object on the search page."""
    props = {p["name"]: p["value"] for p in apt.get("additionalProperty", [])}

    rent_str = props.get("Monthly Rent", "")
    rent = (
        int(rent_str.replace("$", "").replace(",", "").replace("/mo", ""))
        if rent_str
        else None
    )

    sqft = apt.get("floorSize", {}).get("value") if apt.get("floorSize") else None
    lease = apt.get("leaseLength", {}).get("value") if apt.get("leaseLength") else None
    address = apt.get("address", {}) or {}
    image_urls = []
    for img in apt.get("image", []):
        raw = img.get("contentUrl") or img.get("url") if isinstance(img, dict) else img
        if isinstance(raw, str) and raw:
            image_urls.append(normalize_image_url(raw))

    # Split the URL path into building slug + unit.
    # e.g. ".../building/stonehenge-gardens/006j" -> building="stonehenge-gardens", unit="006j"
    id_parts = [p for p in apt.get("@id", "").split("/") if p]
    building_slug = id_parts[-2] if len(id_parts) >= 2 else ""
    unit = id_parts[-1] if id_parts else ""
    listing_id = f"{building_slug}_{unit}" if building_slug else unit

    return {
        "listing_id": listing_id,
        "building_slug": building_slug,
        "unit": unit,
        "url": apt.get("url"),
        "name": apt.get("name"),
        "street": address.get("streetAddress"),
        "neighborhood": address.get("addressLocality"),
        "zip": address.get("postalCode"),
        "lat": apt.get("geo", {}).get("latitude") if apt.get("geo") else None,
        "lng": apt.get("geo", {}).get("longitude") if apt.get("geo") else None,
        "beds": apt.get("numberOfBedrooms"),
        "baths": apt.get("numberOfBathroomsTotal"),
        "sqft": sqft,
        "rent": rent,
        "lease_months": lease,
        "furnished": props.get("Furnished"),
        "building_type": props.get("Building Type"),
        # Filled in by fetch_search_page from the embedded search node.
        # StreetEasy's values include ACTIVE, RENTED, IN_CONTRACT, DELISTED.
        "listing_status": None,
        # Image URLs are kept on the dict during scraping so download_images
        # can consume them; persist_listing then writes them to listing_images.
        "image_urls": image_urls,
    }


# ── Fetchers ─────────────────────────────────────────────────────────────────


def fetch_search_page(url: str, client: httpx.Client) -> list[dict]:
    print(f"  → GET {url}")
    try:
        response = client.get(url, timeout=15)
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return []

    print(f"  ← {response.status_code} ({len(response.text):,} chars)")
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    tag = soup.find("script", {"type": "application/ld+json"})
    if not tag:
        print("  ✗ No ld+json tag found")
        return []

    try:
        graph = json.loads(tag.string).get("@graph", [])
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse ld+json: {e}")
        return []

    listings = [parse_listing(it) for it in graph if it.get("@type") == "Apartment"]
    nodes = listing_nodes_by_path(response.text)
    by_address: dict[tuple[str, str], dict] = {}
    for path, node in nodes.items():
        unit = _unit_key(node.get("unit") or path.rstrip("/").split("/")[-1])
        by_address[(_street_key(node.get("street") or ""), unit)] = node

    matched = 0
    for listing in listings:
        path = urlparse(listing.get("url") or "").path.rstrip("/")
        node = nodes.get(path)
        if node is None and path:
            unit = _unit_key(path.split("/")[-1])
            node = by_address.get((_street_key(listing.get("street") or ""), unit))
        status = (node or {}).get("status")
        if status:
            matched += 1
        # Search results are the open (for-rent) inventory. When a card is not
        # in the embedded payload, StreetEasy still served it as an active listing.
        listing["listing_status"] = status or "ACTIVE"
    print(
        f"  ✓ Parsed {len(listings)} listings from page "
        f"(status ACTIVE; {matched} confirmed in page data)"
    )
    return listings


def _download_one(url: str, dest_path: str, client) -> bool:
    """Download a single URL to dest_path. Returns True on success or if it already exists.

    Generic enough that ``backfill_details.py`` reuses it for floor-plan downloads
    (passing in its own curl_cffi session in place of an httpx.Client)."""
    if os.path.exists(dest_path):
        return True
    try:
        r = client.get(url, timeout=15)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"    ✗ download failed ({url}): {e}")
        return False


def download_images(
    listing: dict,
    base_dir: str,
    client: httpx.Client,
    min_delay: float = 0.2,
    max_delay: float = 0.5,
) -> list[Optional[str]]:
    """Download all listing images into images/<building>/<unit>/.

    Filenames use the Zillow CDN hash (last path segment of the URL).
    Returns a list of local paths in the same order as listing['image_urls'];
    failed downloads produce a None placeholder so positional alignment is
    preserved when we write to listing_images.

    Files already on disk are skipped. A short delay is applied only after a
    real download so a listing with dozens of photos does not hit the CDN in
    one burst. These requests go to the photo CDN, not to StreetEasy.
    """
    building = listing.get("building_slug") or listing["listing_id"]
    unit = listing.get("unit") or listing["listing_id"]
    unit_dir = os.path.join(base_dir, building, unit)
    os.makedirs(unit_dir, exist_ok=True)

    local_paths: list[Optional[str]] = []
    urls = listing.get("image_urls", [])
    for i, img_url in enumerate(urls):
        filename = img_url.split("/")[-1].split("?")[0]  # e.g. "<hash>-se_medium_500_250.webp"
        filepath = os.path.join(unit_dir, filename)

        if os.path.exists(filepath):
            local_paths.append(filepath)
            continue

        ok = _download_one(img_url, filepath, client)
        local_paths.append(filepath if ok else None)
        if ok and max_delay > 0 and i < len(urls) - 1:
            time.sleep(random.uniform(min_delay, max_delay))

    return local_paths


def repair_stored_images(
    conn: sqlite3.Connection,
    image_dir: str,
    client: httpx.Client,
    min_delay: float,
    max_delay: float,
) -> int:
    """Fix photo URLs already saved with an ``undefined`` host and download them.

    Uses only the rows in ``listing_images`` plus the photo CDN. It does not
    request StreetEasy search or detail pages.
    """
    listing_ids = dbm.listings_needing_image_download(conn)
    if not listing_ids:
        return 0

    print(
        f"Repairing photos for {len(listing_ids)} listings already in the DB "
        f"(photo CDN only, no StreetEasy requests)"
    )
    repaired = 0
    for listing_id in listing_ids:
        listing = dbm.get_listing(conn, listing_id) or {"listing_id": listing_id}
        images = dbm.get_listing_images(conn, listing_id)
        urls = [normalize_image_url(im["image_url"]) for im in images]
        payload = {**listing, "image_urls": urls}
        local_paths = download_images(payload, image_dir, client, min_delay, max_delay)
        dbm.set_listing_images(conn, listing_id, urls, local_paths)
        conn.commit()
        n_ok = sum(1 for p in local_paths if p)
        name = listing.get("name") or listing_id
        print(f"  ✓ {name}: {n_ok}/{len(local_paths)} images saved")
        repaired += 1
    return repaired


# ── Persist ──────────────────────────────────────────────────────────────────


def persist_listing(conn: sqlite3.Connection, listing: dict) -> None:
    """Write a listing's core columns + image refs to the DB.

    Detail-page fields (amenities, description, available_from, floor plan)
    are intentionally not touched here — they belong to backfill_details.py.
    """
    dbm.upsert_listing(conn, listing)

    if listing.get("image_urls"):
        dbm.set_listing_images(
            conn,
            listing["listing_id"],
            listing["image_urls"],
            listing.get("local_image_paths"),
        )

    conn.commit()


# ── Main ─────────────────────────────────────────────────────────────────────


def scrape(
    db_path: str,
    price_min: int,
    price_max: int,
    area_ids: list[int],
    max_listings: Optional[int],
    max_pages: int,
    image_dir: str,
    skip_images: bool,
    min_delay: float,
    max_delay: float,
    image_min_delay: float,
    image_max_delay: float,
) -> None:
    os.makedirs(image_dir, exist_ok=True)

    with dbm.open_db(db_path) as conn:
        existing_ids: set[str] = dbm.get_listing_ids(conn)
        if existing_ids:
            print(f"Resuming — {len(existing_ids)} listings already in {db_path}")
        else:
            print(f"Starting fresh (no existing entries in {db_path})")

        new_to_process: list[dict] = []

        with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
            # Photos already stored with a broken host are fetched from the
            # photo CDN here, before any StreetEasy search request.
            if not skip_images:
                repair_stored_images(
                    conn, image_dir, client, image_min_delay, image_max_delay
                )

            # ── Stage 1: search pages ────────────────────────────────────────
            # Keep paginating until we've queued max_listings NEW listings (or
            # we run out of pages). Listings already in the DB are skipped.
            # max_listings == 0 repairs photos and does not request StreetEasy.
            for page in range(1, max_pages + 1):
                if max_listings is not None and len(new_to_process) >= max_listings:
                    break

                print(f"\n── Search page {page} ──────────────────────────────────")
                url = build_url(price_min, price_max, area_ids, page)
                page_listings = fetch_search_page(url, client)

                if not page_listings:
                    print("  (no listings returned — stopping pagination)")
                    break

                dup_count = 0
                for listing in page_listings:
                    lid = listing.get("listing_id")
                    if lid and lid in existing_ids:
                        dup_count += 1
                        continue
                    new_to_process.append(listing)
                    existing_ids.add(lid)  # avoid in-page duplicates too
                    if max_listings is not None and len(new_to_process) >= max_listings:
                        break

                if dup_count:
                    print(f"  ↺ {dup_count} already in DB, skipping")

                if page < max_pages:
                    delay = random.uniform(min_delay, max_delay)
                    print(f"  sleeping {delay:.1f}s before next page...")
                    time.sleep(delay)

            print(f"\n── {len(new_to_process)} new listings to process ──")

            # ── Stage 2: image downloads per NEW listing ─────────────────────
            for i, listing in enumerate(new_to_process, start=1):
                print(
                    f"\n── [{i}/{len(new_to_process)}] "
                    f"{listing.get('name') or listing.get('listing_id')} "
                    f"────────────────"
                )

                if not skip_images and listing.get("image_urls"):
                    print(f"  downloading {len(listing['image_urls'])} images...")
                    local_paths = download_images(
                        listing, image_dir, client, image_min_delay, image_max_delay
                    )
                    listing["local_image_paths"] = local_paths
                    n_ok = sum(1 for p in local_paths if p)
                    print(f"  ✓ {n_ok}/{len(local_paths)} images saved")

                # Incremental persist: a Ctrl-C between listings keeps progress.
                persist_listing(conn, listing)

        st = dbm.stats(conn)
        print(
            f"\n── Done. {st['listings']} listings in {db_path} "
            f"({len(new_to_process)} new this run) ──"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape StreetEasy rental listings and download images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--price-min", type=int, default=3000, help="Minimum rent ($/mo)")
    p.add_argument("--price-max", type=int, default=7000, help="Maximum rent ($/mo)")
    p.add_argument(
        "--areas",
        type=str,
        # FULL LIST:
        # Financial District, Tribeca, Stuyvesant Town/PCV, Gramercy Park, Chelsea
        # Greenwich Village, East Village, Murray Hill, Sutton Place, Turtle Bay
        # Kips Bay, Upper East Side, Hudson Yards, Hell's Kitchen, West Village, Flatiron, Nolita
        default="104,105,106,113,115,116,117,157,158,162,133,130,131,132,152,146,140",
        help="Comma-separated StreetEasy area IDs",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=250,
        help="Stop after this many NEW listings (useful for testing, e.g. 100)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Hard cap on search-result pages to crawl",
    )
    p.add_argument("--image-dir", type=str, default="images", help="Directory for downloaded images")
    p.add_argument("--skip-images", action="store_true", help="Skip image downloads")
    p.add_argument("--min-delay", type=float, default=2.0, help="Min delay between StreetEasy search pages (s)")
    p.add_argument("--max-delay", type=float, default=4.0, help="Max delay between StreetEasy search pages (s)")
    p.add_argument(
        "--image-min-delay",
        type=float,
        default=0.2,
        help="Min delay between photo CDN downloads (s)",
    )
    p.add_argument(
        "--image-max-delay",
        type=float,
        default=0.5,
        help="Max delay between photo CDN downloads (s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    area_ids = [int(a.strip()) for a in args.areas.split(",") if a.strip()]

    scrape(
        db_path=args.db,
        price_min=args.price_min,
        price_max=args.price_max,
        area_ids=area_ids,
        max_listings=args.max_listings,
        max_pages=args.max_pages,
        image_dir=args.image_dir,
        skip_images=args.skip_images,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        image_min_delay=args.image_min_delay,
        image_max_delay=args.image_max_delay,
    )


if __name__ == "__main__":
    main()
