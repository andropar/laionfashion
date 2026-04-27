#!/usr/bin/env python3
"""Download fashion images from Pexels API.

Usage:
    python scripts/20_download_pexels.py <output_dir> --api-key <KEY>
    python scripts/20_download_pexels.py <output_dir>  # reads PEXELS_API_KEY env var

Downloads images matching fashion/outfit keywords, saves full-resolution
images and a metadata CSV for later merging into a bundle.

Rate limit: 200 requests/minute (Pexels is generous).
Each request returns up to 80 photos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "street style fashion",
    "outfit of the day",
    "fashion portrait full body",
    "casual outfit",
    "streetwear look",
    "men fashion outfit",
    "women fashion outfit",
    "summer outfit",
    "winter outfit",
    "autumn fashion",
    "spring fashion look",
    "minimalist outfit",
    "formal outfit",
    "business casual outfit",
    "vintage fashion",
    "urban fashion",
    "elegant outfit",
    "sporty outfit",
    "bohemian fashion",
    "smart casual",
    "fashion week street style",
    "daily outfit",
    "workwear outfit",
    "date night outfit",
    "weekend outfit casual",
    "denim outfit",
    "monochrome outfit",
    "layered outfit",
    "oversized fashion",
    "tailored outfit",
    "fashion blogger outfit",
    "fashion model full body",
    "stylish man walking",
    "stylish woman walking",
    "outfit inspiration",
    "trendy outfit",
    "retro fashion look",
    "grunge fashion",
    "preppy outfit",
    "athleisure outfit",
    "fashion editorial",
    "outfit flatlay",
    "korean fashion",
    "scandinavian fashion",
    "japanese street fashion",
    "parisian style",
    "new york street style",
    "london fashion",
    "outfit mirror selfie",
    "ootd fashion",
]

PEXELS_API_URL = "https://api.pexels.com/v1/search"
PER_PAGE = 80  # max allowed by Pexels
MAX_PAGES_PER_QUERY = 50  # Pexels caps at ~4000 results per query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fashion images from Pexels.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--api-key", type=str, default=None,
                        help="Pexels API key (or set PEXELS_API_KEY env var)")
    parser.add_argument("--max-per-query", type=int, default=2000,
                        help="Max images to download per search query")
    parser.add_argument("--min-width", type=int, default=512,
                        help="Minimum image width")
    parser.add_argument("--min-height", type=int, default=512,
                        help="Minimum image height")
    parser.add_argument("--orientation", type=str, default="portrait",
                        choices=["portrait", "landscape", "square", "all"],
                        help="Preferred orientation filter")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already downloaded")
    return parser.parse_args()


def download_image(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.debug("Failed to download %s: %s", url, e)
        return False


def search_pexels(
    api_key: str,
    query: str,
    per_page: int = PER_PAGE,
    page: int = 1,
    orientation: str | None = None,
) -> dict:
    """Execute a single Pexels search request."""
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    if orientation and orientation != "all":
        params["orientation"] = orientation
    resp = requests.get(PEXELS_API_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    api_key = args.api_key or os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set PEXELS_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.csv"
    seen_ids: set[int] = set()
    existing_rows = []

    # Resume support: load existing metadata
    if args.resume and meta_path.exists():
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_ids.add(int(row["pexels_id"]))
                existing_rows.append(row)
        logger.info("Resuming: %d images already downloaded", len(seen_ids))

    fieldnames = [
        "pexels_id", "photographer", "photographer_url", "url_page",
        "url_original", "url_large2x", "width", "height",
        "query", "filename", "caption",
    ]

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        total_downloaded = len(seen_ids)
        for qi, query in enumerate(SEARCH_QUERIES):
            query_count = 0
            logger.info("[%d/%d] Searching: %s", qi + 1, len(SEARCH_QUERIES), query)

            for page in range(1, MAX_PAGES_PER_QUERY + 1):
                if query_count >= args.max_per_query:
                    break

                try:
                    data = search_pexels(
                        api_key, query, per_page=PER_PAGE, page=page,
                        orientation=args.orientation if args.orientation != "all" else None,
                    )
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        logger.warning("Rate limited, sleeping 60s...")
                        time.sleep(60)
                        continue
                    logger.error("HTTP error for query '%s' page %d: %s", query, page, e)
                    break
                except Exception as e:
                    logger.error("Request failed for query '%s' page %d: %s", query, page, e)
                    break

                photos = data.get("photos", [])
                if not photos:
                    break

                for photo in photos:
                    pid = photo["id"]
                    if pid in seen_ids:
                        continue

                    w, h = photo["width"], photo["height"]
                    if w < args.min_width or h < args.min_height:
                        continue

                    # Prefer large2x (up to ~1880px wide), fall back to original
                    src = photo.get("src", {})
                    img_url = src.get("large2x") or src.get("original")
                    if not img_url:
                        continue

                    ext = Path(urlparse(img_url).path).suffix or ".jpeg"
                    filename = f"pexels_{pid}{ext}"
                    dest = img_dir / filename

                    if args.resume and dest.exists():
                        seen_ids.add(pid)
                        query_count += 1
                        continue

                    if download_image(img_url, dest):
                        seen_ids.add(pid)
                        query_count += 1
                        total_downloaded += 1

                        row = {
                            "pexels_id": pid,
                            "photographer": photo.get("photographer", ""),
                            "photographer_url": photo.get("photographer_url", ""),
                            "url_page": photo.get("url", ""),
                            "url_original": src.get("original", ""),
                            "url_large2x": src.get("large2x", ""),
                            "width": w,
                            "height": h,
                            "query": query,
                            "filename": filename,
                            "caption": photo.get("alt", ""),
                        }
                        writer.writerow(row)
                        f.flush()

                        if total_downloaded % 100 == 0:
                            logger.info("  Total downloaded: %d", total_downloaded)

                # Small delay to stay well under rate limits
                time.sleep(0.35)

                if not data.get("next_page"):
                    break

            logger.info("  Query '%s': %d new images", query, query_count)

    logger.info("Done. Total images: %d in %s", total_downloaded, img_dir)


if __name__ == "__main__":
    main()
