#!/usr/bin/env python3
"""Harvest fashion-candidate URLs from LAION-natural without downloading images.

Usage (on Raven):
    python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
      /u/rothj/laionfashion/scripts/24_harvest_laion_urls.py \
      --candidate-scan 50000000 --n-candidates 500000

This scans LAION-natural captions for fashion/outfit keywords, scores them,
and exports a lightweight candidates.parquet with URLs and captions.
No images are downloaded — that happens locally with 22_redownload_laion_urls.py.

Output: scripts/outputs/24_harvest_laion_urls/<timestamp>/candidates.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.config import load_data_paths
from laionfashion.data_access import NaturalSubsetIndex, LaionTarReader
from laionfashion.filtering import score_caption, SELECTION_MODES
from laionfashion.outputs import make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest fashion URLs from LAION-natural captions."
    )
    parser.add_argument("--candidate-scan", type=int, default=50_000_000,
                        help="Number of captions to scan (default: 50M)")
    parser.add_argument("--n-candidates", type=int, default=500_000,
                        help="Max candidates to keep (default: 500k)")
    parser.add_argument("--min-score", type=float, default=1.5,
                        help="Minimum caption score to keep (default: 1.5, 'strict' tier)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_data_paths()
    index = NaturalSubsetIndex.from_paths(paths)
    rng = np.random.default_rng(args.seed)

    out_dir = make_output_dir(REPO_ROOT / "scripts" / "outputs" / "24_harvest_laion_urls")
    print(f"Output: {out_dir}")
    print(f"Scanning {args.candidate_scan:,} captions, keeping up to {args.n_candidates:,} "
          f"with score >= {args.min_score}")
    print(f"Total shards available: {len(index.shards)}")

    candidates = []
    scanned = 0
    metadata_errors = 0

    shard_order = rng.permutation(len(index.shards))

    with tqdm(total=args.candidate_scan, desc="Scanning captions") as pbar:
        for shard_index in shard_order:
            if scanned >= args.candidate_scan or len(candidates) >= args.n_candidates:
                break

            shard = index.shards[int(shard_index)]
            try:
                with LaionTarReader(shard.tar_path) as reader:
                    n_available = min(shard.n_images, len(reader))
                    within_order = rng.permutation(n_available)

                    for within_shard_index in within_order:
                        if scanned >= args.candidate_scan or len(candidates) >= args.n_candidates:
                            break

                        scanned += 1
                        pbar.update(1)

                        try:
                            metadata = reader.read_metadata(int(within_shard_index))
                        except Exception:
                            metadata_errors += 1
                            continue

                        caption = metadata.get("caption", "")
                        if not caption:
                            continue

                        result = score_caption(caption)
                        if result.score < args.min_score:
                            continue

                        url = metadata.get("url", "")
                        if not url:
                            continue

                        candidates.append({
                            "url": url,
                            "caption": caption,
                            "caption_score": float(result.score),
                            "width": metadata.get("width", 0),
                            "height": metadata.get("height", 0),
                            "global_index": shard.start_index + int(within_shard_index),
                        })

                        if len(candidates) % 10000 == 0:
                            elapsed = pbar.format_dict["elapsed"]
                            print(f"  Found {len(candidates):,} candidates "
                                  f"from {scanned:,} scanned")
            except Exception as e:
                # Skip unreadable shards
                continue

    print(f"\nScanned {scanned:,} captions")
    print(f"Found {len(candidates):,} candidates with score >= {args.min_score}")
    print(f"Metadata errors: {metadata_errors:,}")

    # Sort by caption score descending
    candidates.sort(key=lambda x: x["caption_score"], reverse=True)

    df = pd.DataFrame(candidates)
    df.to_parquet(out_dir / "candidates.parquet", index=False)

    # Save summary
    summary = {
        "scanned": scanned,
        "n_candidates": len(candidates),
        "min_score": args.min_score,
        "candidate_scan": args.candidate_scan,
        "metadata_errors": metadata_errors,
        "accept_rate": round(len(candidates) / max(1, scanned), 6),
        "score_distribution": {
            "mean": round(float(df["caption_score"].mean()), 3) if len(df) else 0,
            "median": round(float(df["caption_score"].median()), 3) if len(df) else 0,
            "p25": round(float(df["caption_score"].quantile(0.25)), 3) if len(df) else 0,
            "p75": round(float(df["caption_score"].quantile(0.75)), 3) if len(df) else 0,
        },
        "urls_with_content": int(df["url"].str.len().gt(0).sum()) if len(df) else 0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    print(f"\nSaved to {out_dir / 'candidates.parquet'}")
    print(f"\nNext steps:")
    print(f"  1. scp raven:{out_dir / 'candidates.parquet'} .")
    print(f"  2. python scripts/22_redownload_laion_urls.py <bundle> <output> "
          f"--candidates candidates.parquet")


if __name__ == "__main__":
    main()
