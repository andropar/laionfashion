"""Garment-level data model and detection.

A garment record maps a bounding-box region within a source outfit image to a
category label, a crop file, and (optionally) an embedding row.

Two extraction methods are available:

- **detr** (default): Uses ``yainage90/fashion-object-detection``, a Conditional
  DETR fine-tuned on ModaNet + Fashionpedia.  Produces bounding boxes with
  category labels: top, bottom, dress, outer, shoes, hat, bag.
- **region_split_v0**: Crude upper/lower body split.  No model required.

Schema (``garments.parquet``)::

    outfit_id       int     — row_id of the source image in records.parquet
    garment_id      int     — unique garment ID across the bundle
    category        str     — "top", "bottom", "dress", "outer", "shoes", etc.
    bbox_x          int     — crop bounding box, pixels in the source thumbnail
    bbox_y          int
    bbox_w          int
    bbox_h          int
    confidence      float   — detector confidence (NaN for region_split_v0)
    crop_path       str     — relative path to the garment crop JPEG
    method          str     — extraction method ("detr", "region_split_v0")

Embedding columns (added later by a separate encoding step):

    clip_embedding_row  int — row index into garment_embeddings.npy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Categories from the fashion-object-detection model.
DETR_CATEGORIES = ("top", "bottom", "dress", "outer", "shoes", "hat", "bag")

# v0 region split categories (kept for backward compat / fallback).
CATEGORY_UPPER = "upper_body"
CATEGORY_LOWER = "lower_body"
CATEGORY_FULL = "full_body"


@dataclass(frozen=True)
class GarmentRegion:
    """A detected garment region within a source image."""

    category: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float = float("nan")


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------


class GarmentDetector(Protocol):
    """Protocol for garment detection implementations."""

    def detect(self, image: Image.Image) -> list[GarmentRegion]: ...


# ---------------------------------------------------------------------------
# DETR fashion detector
# ---------------------------------------------------------------------------


class FashionDETRDetector:
    """Garment detector using yainage90/fashion-object-detection (Conditional DETR).

    Categories: top, bottom, dress, outer, shoes, hat, bag.
    """

    def __init__(
        self,
        model_name: str = "yainage90/fashion-object-detection",
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        import torch

        self._threshold = confidence_threshold
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForObjectDetection.from_pretrained(model_name).to(self._device)
        self._model.eval()
        self._torch = torch

        logger.info(
            "FashionDETRDetector ready: %s on %s, threshold=%.2f",
            model_name, self._device, confidence_threshold,
        )

    def detect(self, image: Image.Image) -> list[GarmentRegion]:
        """Detect garments in a PIL image.  Returns list of GarmentRegion."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = self._torch.tensor([image.size[::-1]]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self._threshold
        )[0]

        regions = []
        for score, label_id, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            category = self._model.config.id2label[label_id.item()]
            regions.append(
                GarmentRegion(
                    category=category,
                    bbox_x=int(x1),
                    bbox_y=int(y1),
                    bbox_w=int(x2 - x1),
                    bbox_h=int(y2 - y1),
                    confidence=float(score.item()),
                )
            )

        return regions


# ---------------------------------------------------------------------------
# Region extraction — v0 heuristic split (fallback)
# ---------------------------------------------------------------------------

_UPPER_RATIO = 0.55
_LOWER_START_RATIO = 0.40


def extract_regions_v0(image: Image.Image) -> list[GarmentRegion]:
    """Split an image into upper-body and lower-body regions.

    Crude heuristic baseline — assumes a roughly centered person.
    """
    w, h = image.size
    if h < 20 or w < 20:
        return []

    upper_h = max(1, int(h * _UPPER_RATIO))
    lower_y = int(h * _LOWER_START_RATIO)
    lower_h = h - lower_y

    return [
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


# ---------------------------------------------------------------------------
# Mock detector (for testing)
# ---------------------------------------------------------------------------


class MockDetector:
    """Returns fixed garment regions.  Useful for testing without a model."""

    def __init__(self, regions: list[GarmentRegion] | None = None) -> None:
        self._regions = regions or [
            GarmentRegion("top", 0, 0, 80, 60, confidence=0.95),
            GarmentRegion("bottom", 0, 50, 80, 70, confidence=0.90),
        ]

    def detect(self, image: Image.Image) -> list[GarmentRegion]:
        return self._regions


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
    w, h = source_image.size
    box = (
        max(0, region.bbox_x),
        max(0, region.bbox_y),
        min(w, region.bbox_x + region.bbox_w),
        min(h, region.bbox_y + region.bbox_h),
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
    method: str = "detr",
    detector: GarmentDetector | None = None,
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """Extract garment regions from all thumbnails in a bundle.

    Parameters
    ----------
    records:
        Bundle records DataFrame (must have ``row_id`` and ``thumbnail_path``).
    bundle_dir:
        Root directory of the bundle.
    method:
        ``"detr"`` (default) uses the fashion DETR model.
        ``"region_split_v0"`` uses the crude heuristic split.
    detector:
        Optional pre-initialized detector.  If *None*, one is created
        based on *method*.  Pass a ``MockDetector`` for testing.
    confidence_threshold:
        Minimum confidence for DETR detections (ignored for v0).
    """
    crop_dir = bundle_dir / "garment_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector if needed
    if detector is None and method == "detr":
        detector = FashionDETRDetector(confidence_threshold=confidence_threshold)

    garment_rows: list[dict] = []
    garment_id = 0

    has_detection_images = "detection_image_path" in records.columns

    for _, row in tqdm(records.iterrows(), total=len(records), desc="Detecting garments"):
        outfit_id = int(row["row_id"])

        # Prefer detection images (higher resolution) over thumbnails
        source_rel = None
        if has_detection_images and pd.notna(row.get("detection_image_path")):
            candidate = bundle_dir / row["detection_image_path"]
            if candidate.exists():
                source_rel = row["detection_image_path"]
        if source_rel is None:
            source_rel = row.get("thumbnail_path", "")

        source_path = bundle_dir / source_rel if source_rel else None
        if source_path is None or not source_path.exists():
            logger.warning("Missing image for outfit_id=%d, skipping", outfit_id)
            continue

        image = Image.open(source_path).convert("RGB")

        if method == "region_split_v0":
            regions = extract_regions_v0(image)
        elif method == "detr":
            regions = detector.detect(image)
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
                "confidence": region.confidence,
                "crop_path": crop_rel,
                "source_image_path": source_rel,
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
    invalid = outfit_ids - expected_ids
    if invalid:
        raise ValueError(
            f"garments contain outfit_ids not in records: {sorted(invalid)[:10]}"
        )

    if garments["garment_id"].duplicated().any():
        raise ValueError("garment_id must be unique")
