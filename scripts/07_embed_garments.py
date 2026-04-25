#!/usr/bin/env python3
"""Embed garment crops with CLIP for cross-category retrieval.

Usage:
    python scripts/07_embed_garments.py <bundle_dir> [options]

Reads garments.parquet, loads each crop, embeds with OpenCLIP, and writes:
- garment_embeddings.npy — (n_garments, dim) L2-normalized float32 matrix
- Updates manifest.json with embedding info

Requires open_clip and torch (server extras).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.garments import embed_garment_crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed garment crops with CLIP."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument(
        "--clip-model-name", type=str, default="ViT-B-32",
        help="OpenCLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--clip-pretrained", type=str, default="laion400m_e31",
        help="OpenCLIP pretrained weights (default: laion400m_e31).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for embedding (default: 32).",
    )
    return parser.parse_args()


class CLIPGarmentEmbedder:
    """Embed garment crops with OpenCLIP."""

    def __init__(self, model_name: str, pretrained: str) -> None:
        import open_clip
        import torch

        self._torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
        except Exception:
            print(f"Failed to load {model_name}/{pretrained}, falling back to openai")
            pretrained = "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", device=device
            )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._device = device
        self.model_name = model_name
        self.pretrained = pretrained
        print(f"CLIPGarmentEmbedder ready: {model_name}/{pretrained} on {device}")

    def embed_batch(self, images: list) -> np.ndarray:
        import torch

        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)

    if not bundle.has_garments:
        raise RuntimeError(
            "No garments.parquet found. Run scripts/06_extract_garments.py first."
        )

    print(f"Loaded bundle: {bundle.n_images} images, {bundle.n_garments} garments")

    embedder = CLIPGarmentEmbedder(args.clip_model_name, args.clip_pretrained)

    embeddings = embed_garment_crops(
        bundle.garments,
        bundle.bundle_dir,
        embedder,
        batch_size=args.batch_size,
    )

    out_path = args.bundle_dir / "garment_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"Wrote {out_path}: shape={embeddings.shape}, dtype={embeddings.dtype}")

    # Update manifest
    manifest_path = args.bundle_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
        manifest["garment_embeddings"] = {
            "path": str(out_path),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "clip_model": args.clip_model_name,
            "clip_pretrained": embedder.pretrained,
            "n_garments": len(embeddings),
        }
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        print("Updated manifest.json with garment embedding info.")


if __name__ == "__main__":
    main()
