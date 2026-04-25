"""Bundle portability — validation and packing for local development.

A portable bundle is a self-contained directory that can be copied to any
machine and used without Raven access.  All file references in parquet files
must be relative to the bundle directory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a bundle for portability."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, bool] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


# Required for a minimal loadable bundle.
_REQUIRED_FILES = ("records.parquet", "embeddings.npy")
_REQUIRED_DIRS = ("thumbnails",)

# Optional artifacts that enrich the bundle.
_OPTIONAL_FILES = (
    "garments.parquet",
    "garment_embeddings.npy",
    "projection.parquet",
    "axis_scores.parquet",
    "manifest.json",
    "filter_summary.json",
    "contact_sheet.html",
    "garment_review.html",
    "retrieval_review.html",
    "eval_results.json",
)
_OPTIONAL_DIRS = (
    "detection_images",
    "garment_crops",
)

# Columns that hold relative file paths.
_PATH_COLUMNS = {
    "records.parquet": ["thumbnail_path", "detection_image_path"],
    "garments.parquet": ["crop_path", "source_image_path"],
}


def validate_portable_bundle(bundle_dir: str | Path) -> ValidationResult:
    """Validate that a bundle is self-contained and portable.

    Checks:
    - Required files and directories exist.
    - Path columns in parquet files are relative and resolve to existing files.
    - Embedding row counts match records/garments.
    - Optional artifacts are reported as present/missing.
    """
    bundle_dir = Path(bundle_dir)
    result = ValidationResult()

    if not bundle_dir.is_dir():
        result.error(f"Bundle directory not found: {bundle_dir}")
        return result

    # Required files
    for f in _REQUIRED_FILES:
        path = bundle_dir / f
        exists = path.exists()
        result.artifacts[f] = exists
        if not exists:
            # Check CSV fallback for parquet
            if f.endswith(".parquet"):
                csv = bundle_dir / f.replace(".parquet", ".csv")
                if csv.exists():
                    result.artifacts[f] = True
                    continue
            result.error(f"Missing required file: {f}")

    # Required dirs
    for d in _REQUIRED_DIRS:
        path = bundle_dir / d
        exists = path.is_dir()
        result.artifacts[d] = exists
        if not exists:
            result.error(f"Missing required directory: {d}")

    # Optional files
    for f in _OPTIONAL_FILES:
        result.artifacts[f] = (bundle_dir / f).exists()

    # Optional dirs
    for d in _OPTIONAL_DIRS:
        result.artifacts[d] = (bundle_dir / d).is_dir()

    # Warn about missing optional artifacts
    optional_missing = [
        k for k in list(_OPTIONAL_FILES) + list(_OPTIONAL_DIRS)
        if not result.artifacts.get(k, False)
    ]
    if optional_missing:
        result.warn(f"Missing optional artifacts: {', '.join(optional_missing)}")

    # Check path columns are relative and resolve
    for parquet_name, columns in _PATH_COLUMNS.items():
        parquet_path = bundle_dir / parquet_name
        csv_path = bundle_dir / parquet_name.replace(".parquet", ".csv")
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            continue

        for col in columns:
            if col not in df.columns:
                continue
            paths = df[col].dropna()
            for rel in paths:
                rel_str = str(rel)
                if rel_str.startswith("/"):
                    result.error(f"Absolute path in {parquet_name}.{col}: {rel_str}")
                    break
                full = bundle_dir / rel_str
                if not full.exists():
                    result.error(f"Missing file referenced by {parquet_name}.{col}: {rel_str}")
                    break

    # Embedding row counts
    records_path = bundle_dir / "records.parquet"
    records_csv = bundle_dir / "records.csv"
    if records_path.exists():
        records = pd.read_parquet(records_path)
    elif records_csv.exists():
        records = pd.read_csv(records_csv)
    else:
        records = None

    emb_path = bundle_dir / "embeddings.npy"
    if records is not None and emb_path.exists():
        emb = np.load(emb_path)
        if len(records) != emb.shape[0]:
            result.error(
                f"records ({len(records)}) vs embeddings ({emb.shape[0]}) row count mismatch"
            )

    garments_path = bundle_dir / "garments.parquet"
    garments_csv = bundle_dir / "garments.csv"
    garment_emb_path = bundle_dir / "garment_embeddings.npy"
    if garments_path.exists() or garments_csv.exists():
        if garments_path.exists():
            garments = pd.read_parquet(garments_path)
        else:
            garments = pd.read_csv(garments_csv)
        if garment_emb_path.exists():
            gemb = np.load(garment_emb_path)
            if len(garments) != gemb.shape[0]:
                result.error(
                    f"garments ({len(garments)}) vs garment_embeddings "
                    f"({gemb.shape[0]}) row count mismatch"
                )

    return result


def list_bundle_artifacts(bundle_dir: str | Path) -> dict[str, int]:
    """List bundle artifacts with their sizes in bytes."""
    bundle_dir = Path(bundle_dir)
    artifacts = {}
    for item in sorted(bundle_dir.rglob("*")):
        if item.is_file():
            rel = str(item.relative_to(bundle_dir))
            artifacts[rel] = item.stat().st_size
    return artifacts
