"""Score apartment photos with Gemini and persist results to SQLite.

Reads listings (and their image URLs/local paths) from ``rentals.db``, scores
each unseen image with Gemini, and writes per-image scores + per-listing
aggregates back into the same DB.

Example (from this directory; uv uses the project environment):
    uv run score_listings.py --max-listings 10

Authenticates to Vertex AI with Application Default Credentials. Requires
GOOGLE_APPLICATION_CREDENTIALS (path to the credentials JSON) and
GOOGLE_CLOUD_PROJECT in the environment or in a .env file. Location defaults
to ``global``; set GOOGLE_CLOUD_LOCATION to override it.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

import db as dbm


SYSTEM_PROMPT = (
    "You are a real estate analyst evaluating listing photos for NYC rental "
    "apartments. Be precise and return only valid JSON."
)

SCORE_PROMPT = """You will receive one or more listing photos in a single request.
For EACH image, classify it, then score it.

Return a single JSON ARRAY with one object per image, in the SAME ORDER the
images were provided. Do not wrap the array in any other object, do not add
commentary. If a single image is provided, still return an array of length 1.

Each array element MUST have this exact shape:
{
  "image_category": "apartment" | "common_space" | "irrelevant",
  "apartment_scores": { ... } | null,
  "common_space_scores": { ... } | null,
  "irrelevant_reason": "..." | null
}

Classification rules:
- "apartment": inside the specific rental unit (living room, bedroom, kitchen,
  bathroom, private balcony/terrace, in-unit laundry closet, floor plan of the unit).
- "common_space": shared building areas (lobby, hallway, elevator, gym, pool,
  rooftop/roof deck, courtyard, shared laundry room, building exterior / facade,
  doorman desk, bike room, package room, mail room).
- "irrelevant": anything not useful for judging this listing — map/street view,
  neighborhood photo, logo, floor plan diagram for an unrelated unit, stock photo,
  text-only advertisement, watermarked placeholder, or unclear/unusable image.

If image_category = "apartment", populate "apartment_scores" and set the other two to null:
{
  "room_type": "living_room" | "kitchen" | "bedroom" | "bathroom" |
               "balcony_terrace" | "floor_plan" | "other_apartment",
  "natural_light": 1-10,
  "finish_quality": 1-10,
  "space_feeling": 1-10,
  "view_quality": 1-10,
  "condition": 1-10
}

If image_category = "common_space", populate "common_space_scores" and set the others to null:
{
  "space_type": "lobby" | "hallway" | "gym" | "pool" | "rooftop" | "courtyard" |
                "laundry_room" | "exterior" | "package_room" | "bike_room" | "other_common",
  "finish_quality": 1-10,
  "condition": 1-10,
  "appeal": 1-10
}

If image_category = "irrelevant", set both *_scores to null and fill "irrelevant_reason"
with a short phrase (e.g. "neighborhood map", "watermark placeholder", "street photo").

Scoring guide (apartment):
- natural_light: 1=dark/no windows, 10=bright/floor-to-ceiling windows
- finish_quality: 1=carpet/basic, 10=hardwood/stone/high-end finishes
- space_feeling: 1=cramped/cluttered, 10=open/airy/well-proportioned
- view_quality: 1=brick wall/no view, 10=skyline/water/park view (if no window visible, score 5)
- condition: 1=worn/dated, 10=brand new/pristine

Scoring guide (common_space):
- finish_quality: 1=basic/dated/builder-grade, 10=luxury materials and design
- condition: 1=worn/dirty, 10=pristine
- appeal: 1=uninviting/cramped, 10=impressive/desirable amenity

IMPORTANT: the length of the returned array MUST equal the number of images
provided. Do not combine, skip, or merge images."""


# ── Scorer ───────────────────────────────────────────────────────────────────


def score_image_batch(
    paths: list[str],
    client: genai.Client,
    model_name: str,
    timeout_ms: int,
) -> list[Optional[dict]]:
    """Send up to N images in one Gemini call and return one score dict per
    image in the same order. On any failure, returns [None] * len(paths)."""
    if not paths:
        return []

    try:
        parts: list = [SCORE_PROMPT]
        for path in paths:
            data = Path(path).read_bytes()
            mime = mimetypes.guess_type(path)[0] or "image/webp"
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))

        response = client.models.generate_content(
            model=model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=timeout_ms),
            ),
        )

        parsed = json.loads(response.text)
        if not isinstance(parsed, list):
            # Model occasionally returns a lone object for batch-of-one despite
            # the instructions — tolerate it.
            parsed = [parsed]

        if len(parsed) != len(paths):
            print(
                f"    ! expected {len(paths)} results, got {len(parsed)} — "
                f"padding with nulls"
            )
            parsed = list(parsed)[: len(paths)] + [None] * max(
                0, len(paths) - len(parsed)
            )

        return parsed

    except Exception as e:
        print(f"    ✗ Batch API error: {e}")
        return [None] * len(paths)


def aggregate_scores(image_scores: list[Optional[dict]]) -> dict:
    """Roll up per-image scores into per-listing features.

    Separates apartment photos from common-space photos so averages over
    apartment features are not polluted by lobby/gym/exterior shots, and
    counts irrelevant photos separately so we can flag low-quality listings.
    """
    scored = [s for s in image_scores if s]

    apartment = [s["apartment_scores"] for s in scored
                 if s.get("image_category") == "apartment" and s.get("apartment_scores")]
    common = [s["common_space_scores"] for s in scored
              if s.get("image_category") == "common_space" and s.get("common_space_scores")]
    irrelevant = [s for s in scored if s.get("image_category") == "irrelevant"]

    def _avg(items: list[dict], key: str) -> Optional[float]:
        vals = [it[key] for it in items if it.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    def _max(items: list[dict], key: str):
        vals = [it[key] for it in items if it.get(key) is not None]
        return max(vals) if vals else None

    # ── Apartment metrics ────────────────────────────────────────────────────
    room_types = [a.get("room_type") for a in apartment if a.get("room_type")]
    room_type_counts = {r: room_types.count(r) for r in set(room_types)}

    apartment_summary = {
        "avg_natural_light":  _avg(apartment, "natural_light"),
        "avg_finish_quality": _avg(apartment, "finish_quality"),
        "avg_space_feeling":  _avg(apartment, "space_feeling"),
        "avg_condition":      _avg(apartment, "condition"),
        "max_view_quality":   _max(apartment, "view_quality"),
        "pct_bright_rooms":   round(
            sum(1 for a in apartment if a.get("natural_light", 0) >= 7) / len(apartment), 2
        ) if apartment else None,
        "has_good_view":      (_max(apartment, "view_quality") or 0) >= 7,
        "room_type_counts":   room_type_counts,
    } if apartment else {}

    # ── Common-space metrics ─────────────────────────────────────────────────
    # NOTE: we deliberately do NOT roll a "detected_amenities" boolean list out
    # of the photos — StreetEasy's `amenities` (populated by backfill_details.py)
    # is the canonical source. Including a Gemini-inferred copy would create
    # autocorrelated features for downstream modeling. We keep `space_type_counts`
    # because the *quantity* of photos per amenity carries a different signal
    # (how prominently the listing showcases each space) than a flat existence flag.
    space_types = [c.get("space_type") for c in common if c.get("space_type")]
    space_type_counts = {s: space_types.count(s) for s in set(space_types)}

    common_summary = {
        "avg_finish_quality": _avg(common, "finish_quality"),
        "avg_condition":      _avg(common, "condition"),
        "avg_appeal":         _avg(common, "appeal"),
        "space_type_counts":  space_type_counts,
    } if common else {}

    return {
        "apartment":         apartment_summary,
        "common_space":      common_summary,
        "photos_total":      len(image_scores),
        "photos_scored":     len(scored),
        "photos_apartment":  len(apartment),
        "photos_common":     len(common),
        "photos_irrelevant": len(irrelevant),
    }


def _log_score(idx: int, total: int, path: str, scores: Optional[dict]) -> None:
    prefix = f"    [{idx}/{total}] {os.path.basename(path)}"
    if not scores:
        print(f"{prefix}  ✗ no score")
        return
    cat = scores.get("image_category")
    if cat == "apartment":
        a = scores.get("apartment_scores") or {}
        print(
            f"{prefix}  ✓ apartment/{a.get('room_type')} "
            f"light={a.get('natural_light')} "
            f"finish={a.get('finish_quality')} "
            f"condition={a.get('condition')}"
        )
    elif cat == "common_space":
        c = scores.get("common_space_scores") or {}
        print(
            f"{prefix}  ✓ common/{c.get('space_type')} "
            f"finish={c.get('finish_quality')} "
            f"appeal={c.get('appeal')}"
        )
    elif cat == "irrelevant":
        print(f"{prefix}  ⊘ irrelevant: {scores.get('irrelevant_reason')}")
    else:
        print(f"{prefix}  ? unknown category: {cat}")


def score_listing(
    conn: sqlite3.Connection,
    listing: dict,
    client: genai.Client,
    model_name: str,
    sleep_s: float,
    timeout_ms: int,
    batch_size: int,
    max_images: int,
) -> dict:
    """Score images for one listing. Persists each new image score immediately
    (so a crash mid-listing keeps work) and aggregates at the end into
    ``listing_scores``.

    Returns the aggregated summary dict (same shape as before)."""
    listing_id = listing["listing_id"]

    # Pull the listing's images from the DB.
    images = dbm.get_listing_images(conn, listing_id)
    urls = [im["image_url"] for im in images]
    paths = [im["local_image_path"] for im in images]

    if not images:
        print("  ✗ no images to score")
        summary = aggregate_scores([])
        dbm.upsert_listing_scores(conn, listing_id, summary)
        return summary

    if len(images) > max_images:
        print(f"  capping {len(images)} → {max_images} images")
        images = images[:max_images]
        urls = urls[:max_images]
        paths = paths[:max_images]

    # Cache lookup is now keyed by image_url against the image_scores table.
    cached = dbm.get_existing_image_scores(conn, urls)
    raw_scores: list[Optional[dict]] = [None] * len(images)
    miss_indices: list[int] = []

    for i, url in enumerate(urls):
        if url in cached:
            raw_scores[i] = dbm.image_score_to_dict(cached[url])
        elif paths[i] and os.path.exists(paths[i]):
            miss_indices.append(i)
        else:
            print(f"    ! image {i + 1}: no local file at {paths[i]!r}, skipping")

    cache_hits = sum(1 for s in raw_scores if s is not None)
    if cache_hits:
        print(f"  ↺ {cache_hits}/{len(images)} images already scored in DB")

    # Send misses in batches.
    for batch_start in range(0, len(miss_indices), batch_size):
        batch_idxs = miss_indices[batch_start:batch_start + batch_size]
        batch_paths = [paths[i] for i in batch_idxs]
        batch_urls = [urls[i] for i in batch_idxs]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(miss_indices) + batch_size - 1) // batch_size
        print(
            f"  → batch {batch_num}/{total_batches} "
            f"({len(batch_paths)} image{'s' if len(batch_paths) != 1 else ''})"
        )

        batch_scores = score_image_batch(batch_paths, client, model_name, timeout_ms)

        for idx, path, url, scores in zip(batch_idxs, batch_paths, batch_urls, batch_scores):
            raw_scores[idx] = scores
            if scores:
                dbm.upsert_image_score(conn, url, scores, model_name)
            _log_score(idx + 1, len(images), path, scores)

        # Commit progress after every batch so a crash doesn't lose the API
        # spend we just incurred.
        conn.commit()

        if sleep_s > 0 and batch_start + batch_size < len(miss_indices):
            time.sleep(sleep_s)

    summary = aggregate_scores(raw_scores)
    dbm.upsert_listing_scores(conn, listing_id, summary)
    conn.commit()
    return summary


# ── Main ─────────────────────────────────────────────────────────────────────


def _print_summary(s: dict) -> None:
    if not s:
        return
    print("  Summary:")
    print(
        f"    photos: {s.get('photos_apartment', 0)} apt / "
        f"{s.get('photos_common', 0)} common / "
        f"{s.get('photos_irrelevant', 0)} irrelevant "
        f"(total {s.get('photos_total', 0)})"
    )
    apt = s.get("apartment") or {}
    if apt:
        print(
            f"    apt:    light={apt.get('avg_natural_light')} "
            f"finish={apt.get('avg_finish_quality')} "
            f"space={apt.get('avg_space_feeling')} "
            f"cond={apt.get('avg_condition')} "
            f"view_max={apt.get('max_view_quality')} "
            f"bright%={apt.get('pct_bright_rooms')}"
        )
        print(f"    rooms:  {apt.get('room_type_counts')}")
    common = s.get("common_space") or {}
    if common:
        print(
            f"    common: finish={common.get('avg_finish_quality')} "
            f"cond={common.get('avg_condition')} "
            f"appeal={common.get('avg_appeal')}"
        )
        print(f"    spaces: {common.get('space_type_counts')}")


def run(
    db_path: str,
    model_name: str,
    max_listings: Optional[int],
    sleep_s: float,
    timeout_ms: int,
    rescore: bool,
    batch_size: int,
    max_images: int,
) -> None:
    load_dotenv()

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    missing = [
        name
        for name, value in (
            ("GOOGLE_APPLICATION_CREDENTIALS", creds_path),
            ("GOOGLE_CLOUD_PROJECT", project),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing "
            + " and ".join(missing)
            + ". Set them in the environment or in .env."
        )
    if not Path(creds_path).is_file():
        raise SystemExit(
            f"GOOGLE_APPLICATION_CREDENTIALS file not found: {creds_path}"
        )

    # google-auth reads GOOGLE_APPLICATION_CREDENTIALS itself. Passing the
    # project explicitly is required: user ADC files do not carry a project id.
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    client = genai.Client(vertexai=True, project=project, location=location)

    with dbm.open_db(db_path) as conn:
        # Pick the queue. ``rescore`` re-aggregates everything (image scores
        # already in image_scores are still re-used; we just rebuild
        # listing_scores). Default is "only listings without aggregates".
        if rescore:
            rows = conn.execute("SELECT * FROM listings ORDER BY listing_id").fetchall()
            queue = [dict(r) for r in rows]
        else:
            queue = dbm.listings_missing_scores(conn)

        if max_listings is not None:
            queue = queue[:max_listings]

        already = dbm.stats(conn)["scored_images"]
        print(
            f"DB: {db_path} — {already} images already scored, "
            f"{len(queue)} listing{'s' if len(queue) != 1 else ''} queued."
        )

        for i, listing in enumerate(queue, start=1):
            name = listing.get("name") or listing["listing_id"]
            print(f"\n── [{i}/{len(queue)}] {name} ──────────────────")

            summary = score_listing(
                conn,
                listing,
                client,
                model_name,
                sleep_s,
                timeout_ms,
                batch_size=batch_size,
                max_images=max_images,
            )
            _print_summary(summary)

        st = dbm.stats(conn)
        print(
            f"\n── Done. {st['listings_with_scores']}/{st['listings']} listings scored, "
            f"{st['scored_images']} unique images scored in {db_path} ──"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score apartment photos with Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", type=str, default=dbm.DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3.8-flash",
        help="Gemini model name (e.g. gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro)",
    )
    p.add_argument(
        "--max-listings",
        type=int,
        default=100,
        help="Only score the first N listings in the queue (useful for testing)",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=4,
        help="Seconds to sleep between Gemini batch calls",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of images to send per Gemini API call",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Cap images scored per listing (takes the first N)",
    )
    p.add_argument(
        "--timeout-ms",
        type=int,
        default=60_000,
        help="Per-call Gemini HTTP timeout in milliseconds",
    )
    p.add_argument(
        "--rescore",
        action="store_true",
        help="Re-aggregate every listing's listing_scores even if already present "
             "(individual image_scores rows are still cached, so this is cheap)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(
        db_path=args.db,
        model_name=args.model,
        max_listings=args.max_listings,
        sleep_s=args.sleep,
        timeout_ms=args.timeout_ms,
        rescore=args.rescore,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
