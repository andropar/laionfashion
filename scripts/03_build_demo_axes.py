#!/usr/bin/env python3
"""Build demo/proxy style-axis scores for a debug bundle.

Usage:
    python scripts/03_build_demo_axes.py <bundle_dir> [--seed 42]

Writes axis_scores.parquet into the bundle directory and updates manifest.json.

These are placeholder axes derived from embedding PCA components and caption
keywords.  They exist to bootstrap the explorer UI until real prompt-direction
scores are computed from a matching contrastive text encoder on the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.axes import (
    axis_names,
    build_demo_axes,
    save_axis_scores,
    validate_axis_scores,
)
from laionfashion.bundle import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute demo/proxy style-axis scores for a debug bundle."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Path to a debug bundle directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images, embeddings {bundle.embeddings.shape}")

    scores = build_demo_axes(
        bundle.embeddings,
        bundle.records,
        random_state=args.seed,
    )
    validate_axis_scores(scores, bundle.n_images)

    axes = axis_names(scores)
    print(f"Built {len(axes)} demo axes: {', '.join(axes)}")

    out_path = save_axis_scores(scores, args.bundle_dir)
    print(f"Wrote {out_path}")

    # Update manifest
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["axis_scores"] = {
            "path": str(out_path),
            "axes": axes,
            "method": "demo_proxy",
            "seed": args.seed,
            "caveat": (
                "These are proxy axes derived from embedding PCA and caption keywords, "
                "not real prompt-direction scores."
            ),
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with axis info.")


if __name__ == "__main__":
    main()
