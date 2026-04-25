"""Garment-level data model and region extraction.

A garment record maps a bounding-box region within a source outfit image to a
category label, a crop file, and (optionally) an embedding row.

The v0 extractor uses a simple heuristic: split each image into upper-body and
lower-body regions.  This is intentionally crude — it unblocks the downstream
retrieval and evaluation pipeline while we decide on a real detector (YOLOv8,
Grounding DINO, etc.).

Schema (``garments.parquet``)::

    outfit_id       int     — row_id of the source image in records.parquet
    garment_id      int     — unique garment ID across the bundle
    category        str     — "upper_body", "lower_body", "full_body", or finer labels later
    bbox_x          int     — crop bounding box, pixels in the source thumbnail
    bbox_y          int
    bbox_w          int
    bbox_h          int
    crop_path       str     — relative path to the garment crop JPEG
    method          str     — extraction method ("region_split_v0", "yolov8", etc.)

Embedding columns (added later by a separate encoding step):

    clip_embedding_row  int — row index into garment_embeddings.npy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Category labels used by the v0 region splitter.
CATEGORY_UPPER = "upper_body"
CATEGORY_LOWER = "lower_body"
CATEGORY_FULL = "full_body"

# All recognized categories (will grow with real detectors).
ALL_CATEGORIES = (CATEGORY_UPPER, CATEGORY_LOWER, CATEGORY_FULL)


@dataclass(frozen=True)
class GarmentRegion:
    """A detected garment region within a source image."""

    category: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


# ---------------------------------------------------------------------------
# Region extraction — v0 heuristic split
# ---------------------------------------------------------------------------

# Vertical split ratios for upper/lower body.
# Upper body: top 55% of the image (includes head/shoulders context).
# Lower body: bottom 55% (overlaps intentionally for waist coverage).
_UPPER_RATIO = 0.55
_LOWER_START_RATIO = 0.40


def extract_regions_v0(image: Image.Image) -> list[GarmentRegion]:
    """Split an image into upper-body and lower-body regions.

    This is a crude heuristic baseline — it assumes the image contains a
    roughly centered person.  Replace with a real detector for production use.
    """
    w, h = image.size
    if h < 20 or w < 20:
        return []

    upper_h = max(1, int(h * _UPPER_RATIO))
    lower_y = int(h * _LOWER_START_RATIO)
    lower_h = h - lower_y

    regions = [
        GarmentRegion(
            category=CATEGORY_UPPER,
            bbox_x=0, bbox_y=0,
            bbox_w=w, bbox_h=upper_h,
        ),
        GarmentRegion(
            category=CATEGORY_LOWER,
            bbox_x=0, bbox_y=lower_y,
            bbox_w=w, bbox_h=lower_h,
        ),
    ]
    return regions


# ---------------------------------------------------------------------------
# Crop writing
# ---------------------------------------------------------------------------


def write_garment_crop(
    source_image: Image.Image,
    region: GarmentRegion,
    output_path: Path,
    *,
    quality: int = 90,
) -> None:
    """Crop *region* from *source_image* and save to *output_path*."""
    box = (
        region.bbox_x,
        region.bbox_y,
        region.bbox_x + region.bbox_w,
        region.bbox_y + region.bbox_h,
    )
    crop = source_image.crop(box)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=quality)


# ---------------------------------------------------------------------------
# Bundle-level extraction
# ---------------------------------------------------------------------------


def extract_garments_from_bundle(
    records: pd.DataFrame,
    bundle_dir: Path,
    *,
    method: str = "region_split_v0",
) -> pd.DataFrame:
    """Extract garment regions from all thumbnails in a bundle.

    Parameters
    ----------
    records:
        Bundle records DataFrame (must have ``row_id`` and ``thumbnail_path``).
    bundle_dir:
        Root directory of the bundle.
    method:
        Extraction method name.  Currently only ``"region_split_v0"`` is
        implemented.

    Returns
    -------
    DataFrame with the garments.parquet schema.
    """
    crop_dir = bundle_dir / "garment_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    garment_rows: list[dict] = []
    garment_id = 0

    for _, row in records.iterrows():
        outfit_id = int(row["row_id"])
        thumb_rel = row.get("thumbnail_path", "")
        thumb_path = bundle_dir / thumb_rel if thumb_rel else None

        if thumb_path is None or not thumb_path.exists():
            logger.warning("Missing thumbnail for outfit_id=%d, skipping", outfit_id)
            continue

        image = Image.open(thumb_path).convert("RGB")

        if method == "region_split_v0":
            regions = extract_regions_v0(image)
        else:
            raise ValueError(f"Unknown garment extraction method: {method!r}")

        for region in regions:
            crop_name = f"{garment_id:06d}_{outfit_id}_{region.category}.jpg"
            crop_rel = f"garment_crops/{crop_name}"
            crop_path = bundle_dir / crop_rel

            write_garment_crop(image, region, crop_path)

            garment_rows.append({
                "outfit_id": outfit_id,
                "garment_id": garment_id,
                "category": region.category,
                "bbox_x": region.bbox_x,
                "bbox_y": region.bbox_y,
                "bbox_w": region.bbox_w,
                "bbox_h": region.bbox_h,
                "crop_path": crop_rel,
                "method": method,
            })
            garment_id += 1

    return pd.DataFrame(garment_rows)


def load_garments(bundle_dir: str | Path) -> pd.DataFrame | None:
    """Load ``garments.parquet`` or ``.csv`` from *bundle_dir*, or return *None*."""
    d = Path(bundle_dir)
    parquet = d / "garments.parquet"
    csv = d / "garments.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def save_garments(garments: pd.DataFrame, bundle_dir: str | Path) -> Path:
    """Write *garments* to ``garments.parquet`` in *bundle_dir*."""
    d = Path(bundle_dir)
    out = d / "garments.parquet"
    try:
        garments.to_parquet(out, index=False)
    except Exception:
        out = d / "garments.csv"
        garments.to_csv(out, index=False)
    return out


def validate_garments(garments: pd.DataFrame, n_outfits: int) -> None:
    """Validate garments DataFrame against the bundle."""
    required = {"outfit_id", "garment_id", "category", "crop_path"}
    missing = required - set(garments.columns)
    if missing:
        raise ValueError(f"garments missing required columns: {missing}")

    outfit_ids = set(garments["outfit_id"].unique())
    expected_ids = set(range(n_outfits))
    # Not all outfits need garments (some may fail extraction), but
    # no outfit_id should be outside the valid range.
    invalid = outfit_ids - expected_ids
    if invalid:
        raise ValueError(
            f"garments contain outfit_ids not in records: {sorted(invalid)[:10]}"
        )

    if garments["garment_id"].duplicated().any():
        raise ValueError("garment_id must be unique")
