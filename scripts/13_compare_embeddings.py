#!/usr/bin/env python3
"""Compare embeddings from multiple CLIP-family models on a debug bundle.

Usage:
    python scripts/13_compare_embeddings.py <bundle_dir> [--models clip-vit-b-32 fashionclip]

For each selected model, encodes all bundle thumbnails, computes prompt-direction
axis scores, and saves results alongside the bundle.  Prints a comparison table
of per-axis score ranges.

This script is for analysis/exploration, not the main pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.axes import (
    DEFAULT_PROMPT_AXES,
    build_clip_axes,
)
from laionfashion.bundle import load_bundle
from laionfashion.models import AVAILABLE_MODELS, encode_images, encode_texts, load_clip_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CLIP-family model embeddings on a debug bundle."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model keys to compare (from AVAILABLE_MODELS). "
            f"Available: {', '.join(AVAILABLE_MODELS.keys())}. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for image encoding (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images")

    # Resolve model list
    model_keys = args.models or list(AVAILABLE_MODELS.keys())
    for key in model_keys:
        if key not in AVAILABLE_MODELS:
            print(f"Error: unknown model '{key}'. Available: {', '.join(AVAILABLE_MODELS.keys())}")
            sys.exit(1)

    # Load all thumbnail images once
    print("Loading thumbnails...")
    pil_images: list[Image.Image] = []
    for _, row in bundle.records.iterrows():
        thumb = bundle.thumbnail_path(row["row_id"])
        if thumb is None:
            raise FileNotFoundError(f"Missing thumbnail for row_id={row['row_id']}")
        pil_images.append(Image.open(thumb).convert("RGB"))
    print(f"  {len(pil_images)} thumbnails loaded.")

    # Collect all prompt texts
    axes = DEFAULT_PROMPT_AXES
    all_prompts: list[str] = []
    for ax in axes:
        all_prompts.append(ax.positive)
        all_prompts.append(ax.negative)

    # Per-model results for comparison table
    summary_rows: list[dict] = []

    for key in model_keys:
        config = AVAILABLE_MODELS[key]
        print(f"\n{'=' * 60}")
        print(f"Model: {config.name} ({config.description})")
        print(f"{'=' * 60}")

        model_tuple = load_clip_model(config)

        # Encode images
        print(f"Encoding {len(pil_images)} images...")
        img_emb = encode_images(model_tuple, pil_images, batch_size=args.batch_size)
        emb_path = args.bundle_dir / f"embeddings_{config.name}.npy"
        np.save(emb_path, img_emb)
        print(f"  Saved {emb_path} (shape={img_emb.shape})")

        # Encode texts and compute axes
        print(f"Encoding {len(all_prompts)} prompts...")
        text_emb = encode_texts(model_tuple, all_prompts)

        scores = build_clip_axes(
            image_embeddings=img_emb,
            text_embeddings=text_emb,
            axes=axes,
        )

        scores_path = args.bundle_dir / f"axis_scores_{config.name}.parquet"
        scores.to_parquet(scores_path, index=False)
        print(f"  Saved {scores_path}")

        # Collect stats
        for ax in axes:
            vals = scores[ax.name]
            summary_rows.append({
                "model": config.name,
                "axis": ax.name,
                "min": vals.min(),
                "max": vals.max(),
                "mean": vals.mean(),
                "std": vals.std(),
                "range": vals.max() - vals.min(),
            })

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("Comparison Table: per-axis score statistics")
    print(f"{'=' * 80}")
    summary = pd.DataFrame(summary_rows)
    header = f"{'Model':<20} {'Axis':<30} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Range':>8}"
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        print(
            f"{row['model']:<20} {row['axis']:<30} "
            f"{row['min']:>8.4f} {row['max']:>8.4f} {row['mean']:>8.4f} "
            f"{row['std']:>8.4f} {row['range']:>8.4f}"
        )


if __name__ == "__main__":
    main()
