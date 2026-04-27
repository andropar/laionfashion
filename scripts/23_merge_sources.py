#!/usr/bin/env python3
"""Merge multiple image sources into a unified bundle for the pipeline.

Usage:
    python scripts/23_merge_sources.py <output_bundle> \
        --pexels <pexels_dir> \
        --unsplash <unsplash_dir> \
        --laion <laion_dir> \
        [--clip-model ViT-B-32 --clip-pretrained laion400m_e31]

Takes the output directories from scripts 20/21/22 and produces a single
bundle with:
  - records.parquet (unified metadata)
  - embeddings.npy (CLIP embeddings computed from images)
  - thumbnails/ (160px resized copies)
  - detection_images/ (768px copies for garment detection)
  - manifest.json

After this, run the full pipeline:
  python scripts/run_full_pipeline.py <output_bundle>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge image sources into a bundle.")
    parser.add_argument("output_bundle", type=Path)
    parser.add_argument("--pexels", type=Path, default=None, help="Pexels download directory")
    parser.add_argument("--unsplash", type=Path, default=None, help="Unsplash download directory")
    parser.add_argument("--laion", type=Path, default=None, help="LAION re-download directory")
    parser.add_argument("--thumbnail-size", type=int, default=160)
    parser.add_argument("--detection-image-size", type=int, default=768)
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion400m_e31")
    parser.add_argument("--deduplicate", action="store_true", default=True,
                        help="Remove perceptual duplicates")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_source(source_dir: Path, source_name: str) -> list[dict]:
    """Load metadata CSV from a download script output directory."""
    meta_path = source_dir / "metadata.csv"
    if not meta_path.exists():
        logger.warning("No metadata.csv in %s, skipping", source_dir)
        return []

    img_dir = source_dir / "images"
    records = []
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            filename = row.get("filename", "")
            img_path = img_dir / filename
            if not img_path.exists():
                continue
            records.append({
                "source": source_name,
                "source_id": row.get("pexels_id") or row.get("unsplash_id") or row.get("url", ""),
                "original_path": str(img_path),
                "caption": row.get("caption", ""),
                "original_width": int(row.get("width", 0)),
                "original_height": int(row.get("height", 0)),
                "photographer": row.get("photographer", ""),
                "url": row.get("url_page") or row.get("url", ""),
            })
    logger.info("Loaded %d images from %s", len(records), source_name)
    return records


def write_resized(src: Path, dst: Path, max_size: int) -> tuple[int, int]:
    """Resize image so longest side <= max_size. Returns (width, height)."""
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        img.save(dst, "JPEG", quality=85)
        return img.size


def compute_phash(img_path: Path, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash for deduplication."""
    with Image.open(img_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(str(int(b)) for b in diff.flatten())


def compute_clip_embeddings(
    image_paths: list[Path],
    model_name: str,
    pretrained: str,
    batch_size: int,
) -> np.ndarray:
    """Compute CLIP embeddings for a list of images."""
    import open_clip
    import torch
    from PIL import Image as PILImage

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CLIP model %s/%s on %s", model_name, pretrained, device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = PILImage.open(p).convert("RGB")
                images.append(preprocess(img))
            except Exception:
                # Use a blank image as fallback
                images.append(preprocess(PILImage.new("RGB", (224, 224))))

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info("  Embedded %d/%d images", min(i + batch_size, len(image_paths)), len(image_paths))

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Load all sources
    all_records = []
    if args.pexels and args.pexels.exists():
        all_records.extend(load_source(args.pexels, "pexels"))
    if args.unsplash and args.unsplash.exists():
        all_records.extend(load_source(args.unsplash, "unsplash"))
    if args.laion and args.laion.exists():
        all_records.extend(load_source(args.laion, "laion"))

    if not all_records:
        print("Error: no images found from any source", file=sys.stderr)
        sys.exit(1)

    logger.info("Total images before dedup: %d", len(all_records))

    # Deduplication by perceptual hash
    if args.deduplicate:
        logger.info("Computing perceptual hashes for deduplication...")
        seen_hashes: set[str] = set()
        deduped = []
        for rec in all_records:
            try:
                ph = compute_phash(Path(rec["original_path"]))
                if ph not in seen_hashes:
                    seen_hashes.add(ph)
                    deduped.append(rec)
            except Exception:
                deduped.append(rec)  # keep on hash failure
        logger.info("After dedup: %d images (removed %d duplicates)",
                     len(deduped), len(all_records) - len(deduped))
        all_records = deduped

    # Create bundle structure
    out = args.output_bundle
    out.mkdir(parents=True, exist_ok=True)
    thumb_dir = out / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    det_dir = out / "detection_images"
    det_dir.mkdir(exist_ok=True)

    # Process images: create thumbnails and detection images
    logger.info("Creating thumbnails and detection images...")
    final_records = []
    for i, rec in enumerate(all_records):
        src = Path(rec["original_path"])
        row_id = i
        fname = f"{row_id:06d}.jpg"

        try:
            # Thumbnail
            thumb_path = thumb_dir / fname
            write_resized(src, thumb_path, args.thumbnail_size)

            # Detection image
            det_path = det_dir / fname
            w, h = write_resized(src, det_path, args.detection_image_size)

            final_records.append({
                "row_id": row_id,
                "source": rec["source"],
                "source_id": rec["source_id"],
                "caption": rec["caption"],
                "url": rec["url"],
                "width": w,
                "height": h,
                "original_width": rec["original_width"],
                "original_height": rec["original_height"],
                "photographer": rec["photographer"],
                "thumbnail_path": f"thumbnails/{fname}",
                "detection_image_path": f"detection_images/{fname}",
            })
        except Exception as e:
            logger.debug("Failed to process %s: %s", src, e)
            continue

        if (i + 1) % 1000 == 0:
            logger.info("  Processed %d/%d images", i + 1, len(all_records))

    logger.info("Final bundle: %d images", len(final_records))

    # Save records
    df = pd.DataFrame(final_records)
    df.to_parquet(out / "records.parquet", index=False)

    # Compute CLIP embeddings
    logger.info("Computing CLIP embeddings...")
    det_paths = [out / rec["detection_image_path"] for rec in final_records]
    embeddings = compute_clip_embeddings(det_paths, args.clip_model, args.clip_pretrained, args.batch_size)
    np.save(out / "embeddings.npy", embeddings)
    logger.info("Saved embeddings: %s", embeddings.shape)

    # Manifest
    manifest = {
        "n_exported": len(final_records),
        "sources": {
            source: sum(1 for r in final_records if r["source"] == source)
            for source in sorted(set(r["source"] for r in final_records))
        },
        "feature": {
            "model": args.clip_model,
            "pretrained": args.clip_pretrained,
            "dim": int(embeddings.shape[1]),
        },
        "thumbnail_size": args.thumbnail_size,
        "detection_image_size": args.detection_image_size,
        "caveats": [
            "Mixed-source bundle: Pexels, Unsplash, and/or LAION re-downloads.",
            "Pexels/Unsplash images are licensed under their respective terms.",
            "LAION re-downloads retain original web image licensing (unknown/mixed).",
        ],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Bundle created at %s", out)
    logger.info("Next: python scripts/run_full_pipeline.py %s", out)


if __name__ == "__main__":
    main()
