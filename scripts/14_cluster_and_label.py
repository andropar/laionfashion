#!/usr/bin/env python3
"""Cluster bundle embeddings and generate human-readable labels.

Usage:
    python scripts/14_cluster_and_label.py <bundle_dir>
    python scripts/14_cluster_and_label.py <bundle_dir> --n-clusters 12

Writes ``clusters.parquet`` (columns: row_id, cluster_id, cluster_label) and
updates manifest.json with cluster metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.axes import load_axis_scores
from laionfashion.bundle import load_bundle
from laionfashion.clusters import cluster_embeddings, cluster_exemplars, label_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster bundle embeddings and generate labels."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of clusters (default: 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images")

    # Cluster
    cluster_df = cluster_embeddings(
        bundle.embeddings, n_clusters=args.n_clusters
    )
    n_actual = cluster_df["cluster_id"].nunique()
    print(f"Clustered into {n_actual} clusters (requested {args.n_clusters})")

    # Load axis scores if available
    axis_scores = load_axis_scores(args.bundle_dir)
    if axis_scores is not None:
        print(f"Using axis scores for labelling ({len(axis_scores.columns) - 1} axes)")
    else:
        print("No axis_scores found — using numeric labels")

    # Generate labels
    cluster_ids = cluster_df["cluster_id"].values
    labels = label_clusters(bundle.embeddings, cluster_ids, axis_scores)
    cluster_df["cluster_label"] = cluster_df["cluster_id"].map(labels)

    # Print summary
    for cid in sorted(labels):
        count = (cluster_ids == cid).sum()
        print(f"  Cluster {cid}: {labels[cid]} ({count} images)")

    # Exemplars
    exemplars = cluster_exemplars(bundle.embeddings, cluster_ids, n=5)

    # Save
    out_path = args.bundle_dir / "clusters.parquet"
    try:
        cluster_df.to_parquet(out_path, index=False)
    except Exception:
        out_path = args.bundle_dir / "clusters.csv"
        cluster_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Update manifest
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["clusters"] = {
        "path": str(out_path.name),
        "n_clusters": n_actual,
        "method": "kmeans",
        "labels": labels,
        "exemplars": {str(k): v for k, v in exemplars.items()},
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print("Updated manifest.json with cluster info.")


if __name__ == "__main__":
    main()
