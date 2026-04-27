#!/usr/bin/env python3
"""Download fashion images from Unsplash API.

Usage:
    python scripts/21_download_unsplash.py <output_dir> --api-key <KEY>
    python scripts/21_download_unsplash.py <output_dir>  # reads UNSPLASH_ACCESS_KEY env var

Downloads images matching fashion/outfit keywords at full resolution.
Saves images and metadata CSV for later merging into a bundle.

Rate limit: 50 requests/hour (demo), 5000 requests/hour (production).
Each request returns up to 30 photos.

At demo rates: ~1500 image URLs/hr. This script handles rate limiting
automatically and can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "street style",
    "outfit of the day",
    "fashion portrait",
    "casual outfit",
    "streetwear",
    "men fashion",
    "women fashion",
    "summer outfit",
    "winter outfit",
    "autumn fashion",
    "spring outfit",
    "minimalist fashion",
    "formal wear",
    "business casual",
    "vintage fashion",
    "urban style",
    "elegant outfit",
    "sporty fashion",
    "bohemian style",
    "fashion week",
    "daily outfit",
    "workwear",
    "denim fashion",
    "monochrome outfit",
    "layered outfit",
    "fashion blogger",
    "fashion model",
    "stylish person",
    "outfit inspiration",
    "trendy outfit",
    "retro fashion",
    "preppy style",
    "athleisure",
    "fashion editorial",
    "korean fashion",
    "scandinavian style",
    "japanese street fashion",
    "parisian fashion",
    "street fashion",
    "clothing style",
]

UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
PER_PAGE = 30  # max allowed by Unsplash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fashion images from Unsplash.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--api-key", type=str, default=None,
                        help="Unsplash Access Key (or set UNSPLASH_ACCESS_KEY env var)")
    parser.add_argument("--max-per-query", type=int, default=1000,
                        help="Max images to download per search query")
    parser.add_argument("--min-width", type=int, default=512,
                        help="Minimum image width")
    parser.add_argument("--min-height", type=int, default=512,
                        help="Minimum image height")
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


def search_unsplash(api_key: str, query: str, page: int = 1) -> tuple[dict, dict]:
    """Execute a single Unsplash search request. Returns (data, rate_limit_info)."""
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": query, "per_page": PER_PAGE, "page": page, "order_by": "relevant"}
    resp = requests.get(UNSPLASH_SEARCH_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    rate_info = {
        "remaining": int(resp.headers.get("X-Ratelimit-Remaining", 0)),
        "limit": int(resp.headers.get("X-Ratelimit-Limit", 50)),
    }
    return resp.json(), rate_info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    api_key = args.api_key or os.environ.get("UNSPLASH_ACCESS_KEY")
    if not api_key:
        print("Error: provide --api-key or set UNSPLASH_ACCESS_KEY env var", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.csv"
    seen_ids: set[str] = set()
    existing_rows = []

    if args.resume and meta_path.exists():
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_ids.add(row["unsplash_id"])
                existing_rows.append(row)
        logger.info("Resuming: %d images already downloaded", len(seen_ids))

    fieldnames = [
        "unsplash_id", "photographer", "photographer_url", "url_page",
        "url_raw", "url_regular", "width", "height",
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

            for page in range(1, 60):  # Unsplash caps at ~50 pages
                if query_count >= args.max_per_query:
                    break

                try:
                    data, rate_info = search_unsplash(api_key, query, page=page)
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 403:
                        logger.warning("Rate limited. Sleeping 3600s (1 hour)...")
                        time.sleep(3600)
                        continue
                    logger.error("HTTP error: %s", e)
                    break
                except Exception as e:
                    logger.error("Request failed: %s", e)
                    break

                results = data.get("results", [])
                if not results:
                    break

                # Adaptive rate limiting
                remaining = rate_info["remaining"]
                if remaining < 5:
                    wait = 3600 if rate_info["limit"] <= 50 else 60
                    logger.warning("Only %d requests remaining, sleeping %ds...", remaining, wait)
                    time.sleep(wait)
                elif remaining < 10:
                    time.sleep(30)

                for photo in results:
                    pid = photo["id"]
                    if pid in seen_ids:
                        continue

                    w = photo.get("width", 0)
                    h = photo.get("height", 0)
                    if w < args.min_width or h < args.min_height:
                        continue

                    urls = photo.get("urls", {})
                    # Use regular (1080px wide) to balance quality and bandwidth
                    # raw is original but can be enormous
                    img_url = urls.get("regular") or urls.get("raw")
                    if not img_url:
                        continue

                    filename = f"unsplash_{pid}.jpeg"
                    dest = img_dir / filename

                    if args.resume and dest.exists():
                        seen_ids.add(pid)
                        query_count += 1
                        continue

                    if download_image(img_url, dest):
                        seen_ids.add(pid)
                        query_count += 1
                        total_downloaded += 1

                        desc = photo.get("description") or photo.get("alt_description") or ""
                        user = photo.get("user", {})
                        row = {
                            "unsplash_id": pid,
                            "photographer": user.get("name", ""),
                            "photographer_url": user.get("links", {}).get("html", ""),
                            "url_page": photo.get("links", {}).get("html", ""),
                            "url_raw": urls.get("raw", ""),
                            "url_regular": urls.get("regular", ""),
                            "width": w,
                            "height": h,
                            "query": query,
                            "filename": filename,
                            "caption": desc,
                        }
                        writer.writerow(row)
                        f.flush()

                        if total_downloaded % 100 == 0:
                            logger.info("  Total downloaded: %d", total_downloaded)

                # Respect rate limits — be gentle
                time.sleep(1.5)

                total_pages = data.get("total_pages", 0)
                if page >= total_pages:
                    break

            logger.info("  Query '%s': %d new images", query, query_count)

    logger.info("Done. Total images: %d in %s", total_downloaded, img_dir)


if __name__ == "__main__":
    main()
