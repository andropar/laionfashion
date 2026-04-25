#!/usr/bin/env python3
"""Build FAISS search indices for a portable bundle.

Usage:
    python scripts/12_build_search_index.py <bundle_dir>

Loads embeddings.npy (and optionally garment_embeddings.npy) from the bundle,
builds IndexFlatIP indices, and saves them alongside the embeddings.
Updates manifest.json if present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.search import build_faiss_index, save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS search indices for a bundle.")
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args()


def _update_manifest(bundle_dir: Path, new_artifacts: list[str]) -> None:
    """Add new artifact entries to manifest.json if it exists."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts", {})
    for name in new_artifacts:
        artifacts[name] = {"present": True}
    manifest["artifacts"] = artifacts
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Updated manifest.json with: {', '.join(new_artifacts)}")


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()

    if not bundle_dir.is_dir():
        print(f"ERROR: Bundle directory not found: {bundle_dir}", file=sys.stderr)
        sys.exit(1)

    new_artifacts: list[str] = []

    # --- Main embeddings ---
    emb_path = bundle_dir / "embeddings.npy"
    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embeddings from {emb_path} ...")
    embeddings = np.load(emb_path)
    print(f"  shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    index_path = bundle_dir / "search_index.faiss"
    save_index(index, index_path)
    new_artifacts.append("search_index.faiss")
    print(f"  Saved {index_path} ({index.ntotal} vectors)")

    # --- Garment embeddings (optional) ---
    garment_emb_path = bundle_dir / "garment_embeddings.npy"
    if garment_emb_path.exists():
        print(f"Loading garment embeddings from {garment_emb_path} ...")
        garment_embeddings = np.load(garment_emb_path)
        print(f"  shape: {garment_embeddings.shape}")

        garment_index = build_faiss_index(garment_embeddings)
        garment_index_path = bundle_dir / "garment_search_index.faiss"
        save_index(garment_index, garment_index_path)
        new_artifacts.append("garment_search_index.faiss")
        print(f"  Saved {garment_index_path} ({garment_index.ntotal} vectors)")
    else:
        print("No garment_embeddings.npy found, skipping garment index.")

    _update_manifest(bundle_dir, new_artifacts)
    print("\nDone.")


if __name__ == "__main__":
    main()
