#!/usr/bin/env python3
"""Evaluate cross-category garment retrieval on a bundle.

Usage:
    python scripts/08_evaluate_retrieval.py <bundle_dir>

Builds hold-out queries from multi-garment outfits, retrieves from other
outfits, and measures hit@K and MRR.  Results are printed and saved to
eval_results.json in the bundle directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.evaluation import build_eval_queries, evaluate_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-category garment retrieval."
    )
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument(
        "--min-outfit-garments", type=int, default=2,
        help="Minimum garments per outfit to include (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)

    if not bundle.has_garments or bundle.garment_embeddings is None:
        raise RuntimeError(
            "Bundle needs garments.parquet and garment_embeddings.npy. "
            "Run 06_extract_garments.py and 07_embed_garments.py first."
        )

    print(f"Bundle: {bundle.n_images} images, {bundle.n_garments} garments")
    print(f"Garment embeddings: {bundle.garment_embeddings.shape}")
    print(f"Categories: {dict(bundle.garments['category'].value_counts())}")

    queries = build_eval_queries(
        bundle.garments,
        min_outfit_garments=args.min_outfit_garments,
    )
    print(f"\nBuilt {len(queries)} evaluation queries")

    if not queries:
        print("No valid queries — need outfits with >= 2 garments.")
        return

    metrics = evaluate_retrieval(
        queries,
        bundle.garments,
        bundle.garment_embeddings,
    )

    print(f"\n{'='*50}")
    print(f"CLIP baseline retrieval evaluation")
    print(f"{'='*50}")
    print(f"Queries:   {metrics.n_queries}")
    print(f"Hit@1:     {metrics.hit_at_1:.4f}")
    print(f"Hit@5:     {metrics.hit_at_5:.4f}")
    print(f"Hit@10:    {metrics.hit_at_10:.4f}")
    print(f"MRR:       {metrics.mrr:.4f}")

    if metrics.per_category:
        print(f"\nPer-category breakdown:")
        for cat, cm in sorted(metrics.per_category.items()):
            print(f"  {cat} (n={cm['n_queries']}): "
                  f"hit@5={cm['hit_at_5']:.3f} MRR={cm['mrr']:.3f}")

    # Save results
    out_path = args.bundle_dir / "eval_results.json"
    with out_path.open("w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
