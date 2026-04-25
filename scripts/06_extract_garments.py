#!/usr/bin/env python3
"""Extract garment regions from a debug bundle's thumbnails.

Usage:
    python scripts/06_extract_garments.py <bundle_dir> [--method region_split_v0]

Writes garment_crops/ and garments.parquet into the bundle directory.

The v0 method is a crude upper/lower body split — it exists to unblock the
retrieval and evaluation pipeline.  Replace with a real garment detector
(YOLOv8, Grounding DINO, etc.) when available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.garments import (
    extract_garments_from_bundle,
    save_garments,
    validate_garments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract garment regions from bundle thumbnails."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument(
        "--method",
        type=str,
        default="region_split_v0",
        help="Extraction method (default: region_split_v0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images")

    print(f"Extracting garments with method={args.method}...")
    garments = extract_garments_from_bundle(
        bundle.records,
        args.bundle_dir,
        method=args.method,
    )
    validate_garments(garments, bundle.n_images)

    n_garments = len(garments)
    categories = garments["category"].value_counts().to_dict()
    print(f"Extracted {n_garments} garment regions:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    out_path = save_garments(garments, args.bundle_dir)
    print(f"Wrote {out_path}")

    # Update manifest
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["garments"] = {
            "path": str(out_path),
            "n_garments": n_garments,
            "method": args.method,
            "categories": categories,
            "caveat": (
                "v0 uses a crude upper/lower body split, not a real garment detector. "
                "Region boundaries are approximate."
            ),
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with garment info.")


if __name__ == "__main__":
    main()
