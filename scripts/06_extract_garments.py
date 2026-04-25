#!/usr/bin/env python3
"""Extract garment regions from a debug bundle's thumbnails.

Usage:
    python scripts/06_extract_garments.py <bundle_dir> [--method detr]

Writes garment_crops/ and garments.parquet into the bundle directory.

Methods:
- detr (default): Uses yainage90/fashion-object-detection (Conditional DETR).
  Categories: top, bottom, dress, outer, shoes, hat, bag.
  Requires: transformers, torch.
- region_split_v0: Crude upper/lower body split. No model required.
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
        choices=["detr", "region_split_v0"],
        default="detr",
        help="Detection method (default: detr).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence for DETR detections (default: 0.5).",
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
        confidence_threshold=args.confidence_threshold,
    )
    validate_garments(garments, bundle.n_images)

    n_garments = len(garments)
    categories = garments["category"].value_counts().to_dict()
    print(f"\nExtracted {n_garments} garment regions:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    if "confidence" in garments.columns:
        conf = garments["confidence"].dropna()
        if len(conf) > 0:
            print(f"\nConfidence: min={conf.min():.3f}, mean={conf.mean():.3f}, max={conf.max():.3f}")

    # Per-outfit stats
    per_outfit = garments.groupby("outfit_id").size()
    print(f"\nGarments per outfit: min={per_outfit.min()}, mean={per_outfit.mean():.1f}, max={per_outfit.max()}")
    no_garments = bundle.n_images - garments["outfit_id"].nunique()
    if no_garments > 0:
        print(f"Outfits with no detected garments: {no_garments}")

    out_path = save_garments(garments, args.bundle_dir)
    print(f"\nWrote {out_path}")

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
            "garments_per_outfit": {
                "min": int(per_outfit.min()),
                "mean": round(float(per_outfit.mean()), 1),
                "max": int(per_outfit.max()),
            },
        }
        if args.method == "detr":
            manifest["garments"]["model"] = "yainage90/fashion-object-detection"
            manifest["garments"]["confidence_threshold"] = args.confidence_threshold
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with garment info.")


if __name__ == "__main__":
    main()
