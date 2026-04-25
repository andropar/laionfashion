#!/usr/bin/env python3
"""Evaluate human annotations against model axis scores.

Usage:
    python scripts/16_evaluate_annotations.py <bundle_dir>

Reads annotation_tasks.csv (with filled 'choice' column) and compares
human pairwise judgments against model prompt-axis scores.  Reports
per-axis accuracy and saves results to annotation_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.annotation import evaluate_axis_agreement, load_annotations
from laionfashion.axes import axis_names, load_axis_scores
from laionfashion.bundle import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate human annotations.")
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)

    ann_path = args.bundle_dir / "annotation_tasks.csv"
    if not ann_path.exists():
        print(f"No annotation_tasks.csv found in {args.bundle_dir}")
        print("Run scripts/15_generate_annotation_tasks.py first.")
        sys.exit(1)

    annotations = load_annotations(ann_path)
    filled = annotations[annotations["choice"].isin(["a", "b", "tie"])]
    print(f"Loaded {len(annotations)} pairs, {len(filled)} annotated")

    if filled.empty:
        print("No annotations filled in yet. Fill the 'choice' column in annotation_tasks.csv.")
        sys.exit(0)

    ax_scores = load_axis_scores(args.bundle_dir)
    if ax_scores is None:
        print("No axis_scores found. Run 05_build_clip_axes.py first.")
        sys.exit(1)

    axes = axis_names(ax_scores)
    results = {}

    print(f"\n{'='*50}")
    print(f"Annotation evaluation")
    print(f"{'='*50}")

    for axis in axes:
        agreement = evaluate_axis_agreement(annotations, ax_scores, axis)
        results[axis] = agreement
        if agreement["n_evaluated"] > 0:
            print(f"  {axis}: accuracy={agreement['accuracy']:.3f} "
                  f"({agreement['n_agree']}/{agreement['n_evaluated']})")
        else:
            print(f"  {axis}: no annotations")

    out_path = args.bundle_dir / "annotation_eval.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
