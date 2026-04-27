#!/usr/bin/env python3
"""Re-download LAION images at original resolution from source URLs.

Usage:
    python scripts/22_redownload_laion_urls.py <bundle_dir> <output_dir>
    python scripts/22_redownload_laion_urls.py <bundle_dir> <output_dir> --candidates <candidates.parquet>

Reads URLs from the bundle's records.parquet (and optionally from a larger
candidates parquet), attempts to re-download each at full resolution.
Outputs images + metadata CSV in the same format as the Pexels/Unsplash scripts.

Many URLs will be dead — expect ~30-60% success rate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-download LAION images from URLs.")
    parser.add_argument("bundle_dir", type=Path, help="Existing bundle with records.parquet")
    parser.add_argument("output_dir", type=Path, help="Output directory for re-downloaded images")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Additional candidates parquet with more URLs")
    parser.add_argument("--min-size", type=int, default=512,
                        help="Minimum dimension (width or height) to keep")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Download timeout per image in seconds")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of concurrent download workers")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already downloaded")
    return parser.parse_args()


def download_one(url: str, dest: Path, timeout: int) -> bool:
    """Download a single image. Returns True on success."""
    try:
        resp = requests.get(
            url, timeout=timeout, stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot)"},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet" not in content_type:
            return False
        data = b""
        for chunk in resp.iter_content(8192):
            data += chunk
            if len(data) > 50_000_000:  # skip files > 50MB
                return False
        if len(data) < 1000:  # skip tiny/broken files
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Load URLs from bundle
    records_path = args.bundle_dir / "records.parquet"
    if not records_path.exists():
        records_path = args.bundle_dir / "records.csv"
    if records_path.suffix == ".parquet":
        records = pd.read_parquet(records_path)
    else:
        records = pd.read_csv(records_path)

    if "url" not in records.columns:
        print("Error: records missing 'url' column", file=sys.stderr)
        sys.exit(1)

    urls_df = records[["url", "caption"]].dropna(subset=["url"]).copy()

    # Add candidates if provided
    if args.candidates and args.candidates.exists():
        cand = pd.read_parquet(args.candidates)
        if "url" in cand.columns:
            cand_urls = cand[["url", "caption"]].dropna(subset=["url"])
            urls_df = pd.concat([urls_df, cand_urls]).drop_duplicates(subset=["url"])
            logger.info("Added candidates: %d total URLs", len(urls_df))

    logger.info("Attempting to re-download %d URLs", len(urls_df))

    out = args.output_dir
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out / "metadata.csv"
    seen_urls: set[str] = set()
    existing_rows = []

    if args.resume and meta_path.exists():
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_urls.add(row["url"])
                existing_rows.append(row)
        logger.info("Resuming: %d images already downloaded", len(seen_urls))

    fieldnames = ["url", "filename", "caption", "width", "height"]

    # Pre-filter already seen
    tasks = []
    for _, row in urls_df.iterrows():
        url = row["url"]
        if url in seen_urls:
            continue
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        filename = f"laion_{url_hash}.jpeg"
        tasks.append((url, filename, str(row.get("caption", ""))))

    logger.info("Downloading %d new images (skipping %d already done)", len(tasks), len(seen_urls))

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        success = len(existing_rows)
        failed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for url, filename, caption in tasks:
                dest = img_dir / filename
                if args.resume and dest.exists():
                    continue
                fut = pool.submit(download_one, url, dest, args.timeout)
                futures[fut] = (url, filename, caption, dest)

            for fut in as_completed(futures):
                url, filename, caption, dest = futures[fut]
                ok = fut.result()
                if ok:
                    # Check dimensions
                    try:
                        from PIL import Image
                        with Image.open(dest) as img:
                            w, h = img.size
                        if min(w, h) < args.min_size:
                            dest.unlink()
                            failed += 1
                            continue
                    except Exception:
                        dest.unlink()
                        failed += 1
                        continue

                    writer.writerow({
                        "url": url,
                        "filename": filename,
                        "caption": caption,
                        "width": w,
                        "height": h,
                    })
                    f.flush()
                    success += 1
                else:
                    failed += 1

                total = success + failed
                if total % 200 == 0:
                    logger.info("  Progress: %d success, %d failed (%.0f%% success rate)",
                                success, failed, 100 * success / max(1, total))

    logger.info("Done. %d images saved, %d failed (%.0f%% success rate)",
                success, failed, 100 * success / max(1, success + failed))


if __name__ == "__main__":
    main()
