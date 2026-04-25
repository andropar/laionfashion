#!/usr/bin/env python3
"""Run the full Fashion Embedding Explorer pipeline end-to-end.

Usage:
    python scripts/run_full_pipeline.py <bundle_dir>

Runs all post-build steps on an existing bundle:
  1. Garment detection (DETR)
  2. Garment embedding (CLIP)
  3. UMAP projection
  4. CLIP style axes
  5. FAISS search index
  6. Clustering
  7. Evaluation
  8. Review sheets
  9. Failure report
 10. Annotation task generation
 11. Validation

Assumes scripts/01_build_debug_subset.py has already been run to create
the bundle with records, embeddings, and thumbnails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full pipeline on a bundle.")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip (by number, e.g. --skip 1 2)")
    return parser.parse_args()


STEPS = [
    ("1", "Garment detection", ["scripts/06_extract_garments.py"]),
    ("2", "Garment embedding", ["scripts/07_embed_garments.py"]),
    ("3", "UMAP projection", ["scripts/02_build_projection.py"]),
    ("4", "CLIP style axes", ["scripts/05_build_clip_axes.py"]),
    ("5", "FAISS search index", ["scripts/12_build_search_index.py"]),
    ("6", "Clustering", ["scripts/14_cluster_and_label.py"]),
    ("7", "Evaluation", ["scripts/08_evaluate_retrieval.py"]),
    ("8", "Contact sheet", ["scripts/04_make_review_contact_sheet.py"]),
    ("9", "Garment review", ["scripts/09_garment_review.py"]),
    ("10", "Failure report", ["scripts/18_failure_report.py"]),
    ("11", "Annotation tasks", ["scripts/15_generate_annotation_tasks.py"]),
    ("12", "Validation", ["scripts/11_validate_portable_bundle.py"]),
]


def main() -> None:
    args = parse_args()
    bundle = str(args.bundle_dir)
    skip = set(args.skip)

    print(f"Running full pipeline on: {bundle}")
    print(f"{'='*60}")

    total_start = time.time()
    results = []

    for step_id, name, script_parts in STEPS:
        if step_id in skip:
            print(f"\n[{step_id}] {name} — SKIPPED")
            results.append((step_id, name, "skipped", 0))
            continue

        print(f"\n[{step_id}] {name}...")
        start = time.time()
        cmd = [sys.executable] + script_parts + [bundle]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            elapsed = time.time() - start
            if result.returncode == 0:
                print(f"    OK ({elapsed:.1f}s)")
                results.append((step_id, name, "ok", elapsed))
            else:
                print(f"    FAILED ({elapsed:.1f}s)")
                print(f"    stderr: {result.stderr[:500]}")
                results.append((step_id, name, "failed", elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print(f"    ERROR: {e}")
            results.append((step_id, name, "error", elapsed))

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total_elapsed:.1f}s\n")

    for step_id, name, status, elapsed in results:
        icon = {"ok": "OK", "failed": "FAIL", "error": "ERR", "skipped": "SKIP"}[status]
        print(f"  [{icon:>4}] {step_id:>2}. {name} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
