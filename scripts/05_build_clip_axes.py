#!/usr/bin/env python3
"""Compute real CLIP prompt-direction style axes for a debug bundle.

Usage:
    python scripts/05_build_clip_axes.py <bundle_dir> [options]

Encodes bundle thumbnails and axis prompts with a CLIP model, then scores
each image by dot product with the prompt-direction vectors.  Writes
axis_scores.parquet in the same format the Streamlit app consumes.

Requires open_clip and torch (available in the server/app extras).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.axes import (
    DEFAULT_PROMPT_AXES,
    axis_names,
    build_clip_axes,
    encode_images_with_clip,
    encode_texts_with_clip,
    save_axis_scores,
    validate_axis_scores,
)
from laionfashion.bundle import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLIP prompt-direction style axes for a debug bundle."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument(
        "--clip-model-name", type=str, default="ViT-B-32",
        help="Open CLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--clip-pretrained", type=str, default="laion400m_e31",
        help="Open CLIP pretrained weights (default: laion400m_e31).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for image encoding (default: 32).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images")

    axes = DEFAULT_PROMPT_AXES
    print(f"Computing {len(axes)} prompt-direction axes:")
    for ax in axes:
        print(f"  {ax.name}: {ax.positive!r} vs {ax.negative!r}")

    # Collect all prompt texts
    all_prompts = []
    for ax in axes:
        all_prompts.append(ax.positive)
        all_prompts.append(ax.negative)

    # Encode texts
    print(f"\nEncoding {len(all_prompts)} prompts with {args.clip_model_name}/{args.clip_pretrained}...")
    text_embeddings = encode_texts_with_clip(
        all_prompts,
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
    )

    # Encode images
    image_paths = []
    for _, row in bundle.records.iterrows():
        thumb = bundle.thumbnail_path(row["row_id"])
        if thumb is None:
            raise FileNotFoundError(f"Missing thumbnail for row_id={row['row_id']}")
        image_paths.append(thumb)

    print(f"Encoding {len(image_paths)} images...")
    image_embeddings = encode_images_with_clip(
        image_paths,
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
    )

    # Compute axes
    scores = build_clip_axes(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        axes=axes,
    )
    validate_axis_scores(scores, bundle.n_images)

    axes_list = axis_names(scores)
    print(f"\nComputed {len(axes_list)} axes: {', '.join(axes_list)}")
    for ax_name in axes_list:
        vals = scores[ax_name]
        print(f"  {ax_name}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    out_path = save_axis_scores(scores, args.bundle_dir)
    print(f"\nWrote {out_path}")

    # Update manifest
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["axis_scores"] = {
            "path": str(out_path),
            "axes": axes_list,
            "method": "clip_prompt_direction",
            "clip_model": args.clip_model_name,
            "clip_pretrained": args.clip_pretrained,
            "prompts": {ax.name: {"positive": ax.positive, "negative": ax.negative} for ax in axes},
            "caveat": (
                "Prompt-direction axes are exploratory — they reflect CLIP's "
                "text-image alignment, not ground-truth style categories."
            ),
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with CLIP axis info.")


if __name__ == "__main__":
    main()
