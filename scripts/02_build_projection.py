#!/usr/bin/env python3
"""Build a 2D projection from a debug bundle's embeddings.

Usage:
    python scripts/02_build_projection.py <bundle_dir> [--method umap|pca|trivial]

Writes projection.parquet (or .csv) into the bundle directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.projection import project_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D projection for a debug bundle's embeddings."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Path to a debug bundle directory.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "pca", "trivial"],
        default=None,
        help="Projection method. Default: auto-select based on bundle size.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images, embeddings {bundle.embeddings.shape}")

    projection, method_used = project_embeddings(
        bundle.embeddings,
        method=args.method,
        random_state=args.seed,
    )
    print(f"Projection method: {method_used.value}")

    out_path = args.bundle_dir / "projection.parquet"
    try:
        projection.to_parquet(out_path, index=False)
    except Exception:
        out_path = args.bundle_dir / "projection.csv"
        projection.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(projection)} rows)")

    # Update manifest if it exists
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["projection"] = {
            "method": method_used.value,
            "path": str(out_path),
            "n_points": len(projection),
            "seed": args.seed,
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with projection info.")


if __name__ == "__main__":
    main()
