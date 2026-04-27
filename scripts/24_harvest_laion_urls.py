#!/usr/bin/env python3
"""Harvest fashion-candidate URLs from LAION-natural without downloading images.

Usage (on Raven):
    python /u/rothj/laion_natural/scripts/start_as_slurm_job.py \
      /u/rothj/laionfashion/scripts/24_harvest_laion_urls.py \
      --candidate-scan 50000000 --n-candidates 500000

Scans LAION-natural tar shards SEQUENTIALLY (no random access) for speed.
Reads only JSON metadata — never opens images. Exports candidates.parquet.

Output: scripts/outputs/24_harvest_laion_urls/<timestamp>/candidates.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

# Force unbuffered output for SLURM
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.config import load_data_paths
from laionfashion.data_access import NaturalSubsetIndex
from laionfashion.filtering import score_caption
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


def iter_shard_metadata(tar_path: Path):
    """Iterate through all JSON metadata in a tar shard sequentially.

    Yields dicts. Skips non-JSON members and malformed entries.
    Much faster than random-access via LaionTarReader.
    """
    try:
        with tarfile.open(tar_path, "r:*", ignore_zeros=True) as tf:
            for member in tf:
                if not member.name.endswith(".json"):
                    continue
                try:
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    data = json.load(f)
                    yield data
                except Exception:
                    continue
    except Exception:
        return


def main() -> None:
    args = parse_args()
    paths = load_data_paths()
    index = NaturalSubsetIndex.from_paths(paths)
    rng = np.random.default_rng(args.seed)

    out_dir = make_output_dir(REPO_ROOT / "scripts" / "outputs" / "24_harvest_laion_urls")
    print(f"Output: {out_dir}")
    print(f"Scanning up to {args.candidate_scan:,} captions, keeping up to {args.n_candidates:,} "
          f"with score >= {args.min_score}")
    print(f"Total shards: {len(index.shards)}")

    candidates = []
    scanned = 0
    t0 = time.time()

    # Shuffle shard order for diversity, but iterate sequentially within each shard
    shard_order = rng.permutation(len(index.shards))
    print(f"Starting scan of {len(shard_order)} shards...", flush=True)

    for si, shard_idx in enumerate(shard_order):
        if scanned >= args.candidate_scan or len(candidates) >= args.n_candidates:
            break

        shard = index.shards[int(shard_idx)]
        shard_scanned = 0
        shard_found = 0

        for metadata in iter_shard_metadata(shard.tar_path):
            if scanned >= args.candidate_scan or len(candidates) >= args.n_candidates:
                break

            scanned += 1
            shard_scanned += 1
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
            })
            shard_found += 1

        elapsed = time.time() - t0
        rate = scanned / max(1, elapsed)
        print(f"  Shard {si+1}/{len(shard_order)}: "
              f"+{shard_scanned} scanned, +{shard_found} found | "
              f"Total: {scanned:,} scanned ({rate:,.0f}/s), "
              f"{len(candidates):,} candidates", flush=True)

    elapsed = time.time() - t0
    rate = scanned / max(1, elapsed)
    print(f"\nDone in {elapsed:.0f}s ({rate:,.0f} captions/s)")
    print(f"Scanned {scanned:,} captions, found {len(candidates):,} candidates")

    # Sort by caption score descending
    candidates.sort(key=lambda x: x["caption_score"], reverse=True)

    df = pd.DataFrame(candidates)
    df.to_parquet(out_dir / "candidates.parquet", index=False)

    summary = {
        "scanned": scanned,
        "n_candidates": len(candidates),
        "min_score": args.min_score,
        "candidate_scan": args.candidate_scan,
        "elapsed_seconds": round(elapsed, 1),
        "rate_per_second": round(rate, 1),
        "accept_rate": round(len(candidates) / max(1, scanned), 6),
        "score_distribution": {
            "mean": round(float(df["caption_score"].mean()), 3) if len(df) else 0,
            "median": round(float(df["caption_score"].median()), 3) if len(df) else 0,
            "p25": round(float(df["caption_score"].quantile(0.25)), 3) if len(df) else 0,
            "p75": round(float(df["caption_score"].quantile(0.75)), 3) if len(df) else 0,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {out_dir / 'candidates.parquet'}")


if __name__ == "__main__":
    main()
