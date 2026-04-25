#!/usr/bin/env python3
"""Generate pairwise annotation tasks for style axis calibration.

Usage:
    python scripts/15_generate_annotation_tasks.py <bundle_dir> [--pairs-per-axis 100]

Creates annotation_tasks.csv with pairs of images to compare on each style
axis.  A human annotator fills in the 'choice' column (a/b/tie) and the
'annotator' column, then saves the file.

After annotation, run scripts/16_evaluate_annotations.py to measure
agreement between human labels and model axis scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.annotation import generate_annotation_pairs, pairs_to_dataframe, save_annotation_tasks
from laionfashion.axes import axis_names, load_axis_scores
from laionfashion.bundle import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pairwise annotation tasks.")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--pairs-per-axis", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    ax_scores = load_axis_scores(args.bundle_dir)

    if ax_scores is not None:
        axes = axis_names(ax_scores)
    else:
        axes = ["formal_vs_casual", "colorful_vs_neutral", "streetwear_vs_classic",
                "minimalist_vs_maximalist", "sporty_vs_dressy"]

    print(f"Bundle: {bundle.n_images} images")
    print(f"Axes: {', '.join(axes)}")
    print(f"Pairs per axis: {args.pairs_per_axis}")

    pairs = generate_annotation_pairs(
        n_images=bundle.n_images,
        axes=axes,
        pairs_per_axis=args.pairs_per_axis,
        seed=args.seed,
    )

    df = pairs_to_dataframe(pairs)
    out_path = args.bundle_dir / "annotation_tasks.csv"
    save_annotation_tasks(df, out_path)
    print(f"\nWrote {len(df)} pairs to {out_path}")
    print(f"Fill in 'choice' (a/b/tie) and 'annotator' columns, then run:")
    print(f"  python scripts/16_evaluate_annotations.py {args.bundle_dir}")


if __name__ == "__main__":
    main()
