"""Load and query debug bundles exported by 01_build_debug_subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class DebugBundle:
    """A loaded debug bundle: records table, embeddings matrix, and bundle path.

    Optionally includes garment-level data (``garments`` DataFrame and
    ``garment_embeddings`` matrix) if the bundle has been processed by
    ``06_extract_garments.py``.
    """

    records: pd.DataFrame
    embeddings: np.ndarray
    bundle_dir: Path
    garments: pd.DataFrame | None = field(default=None, repr=False)
    garment_embeddings: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_images(self) -> int:
        return len(self.records)

    @property
    def has_garments(self) -> bool:
        return self.garments is not None and len(self.garments) > 0

    @property
    def n_garments(self) -> int:
        return len(self.garments) if self.garments is not None else 0

    def thumbnail_path(self, row_id: int) -> Path | None:
        """Return the absolute thumbnail path for a given row_id."""
        rel = self.records.loc[row_id, "thumbnail_path"]
        if pd.isna(rel):
            return None
        path = self.bundle_dir / rel
        return path if path.exists() else None

    def garment_crop_path(self, garment_id: int) -> Path | None:
        """Return the absolute crop path for a given garment_id."""
        if self.garments is None:
            return None
        row = self.garments.loc[self.garments["garment_id"] == garment_id]
        if row.empty:
            return None
        rel = row.iloc[0]["crop_path"]
        if pd.isna(rel):
            return None
        path = self.bundle_dir / rel
        return path if path.exists() else None

    def garments_for_outfit(self, outfit_id: int) -> pd.DataFrame:
        """Return garment rows for a given outfit_id."""
        if self.garments is None:
            return pd.DataFrame()
        return self.garments[self.garments["outfit_id"] == outfit_id]


def load_bundle(bundle_dir: str | Path) -> DebugBundle:
    """Load a debug bundle from *bundle_dir*.

    Expects the directory to contain:
    - ``records.parquet`` **or** ``records.csv``
    - ``embeddings.npy``
    - ``thumbnails/`` (referenced by the records table)
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    parquet = bundle_dir / "records.parquet"
    csv = bundle_dir / "records.csv"
    if parquet.exists():
        records = pd.read_parquet(parquet)
    elif csv.exists():
        records = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"No records.parquet or records.csv in {bundle_dir}"
        )

    emb_path = bundle_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings.npy in {bundle_dir}")
    embeddings = np.load(emb_path).astype(np.float32)

    if len(records) != embeddings.shape[0]:
        raise ValueError(
            f"Row count mismatch: {len(records)} records vs "
            f"{embeddings.shape[0]} embeddings"
        )

    # Optionally load garment data
    garments = None
    garment_embeddings = None
    garments_parquet = bundle_dir / "garments.parquet"
    garments_csv = bundle_dir / "garments.csv"
    if garments_parquet.exists():
        garments = pd.read_parquet(garments_parquet)
    elif garments_csv.exists():
        garments = pd.read_csv(garments_csv)

    garment_emb_path = bundle_dir / "garment_embeddings.npy"
    if garment_emb_path.exists():
        garment_embeddings = np.load(garment_emb_path).astype(np.float32)

    return DebugBundle(
        records=records,
        embeddings=embeddings,
        bundle_dir=bundle_dir,
        garments=garments,
        garment_embeddings=garment_embeddings,
    )


def nearest_neighbors(
    embeddings: np.ndarray,
    query_index: int,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Return the *k* nearest neighbors of ``embeddings[query_index]`` by cosine similarity.

    Returns a list of ``(index, similarity)`` pairs sorted by descending similarity.
    The query itself is excluded from the results.
    """
    query = embeddings[query_index]
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    # Guard against zero-norm vectors
    denom = np.maximum(norms * query_norm, 1e-12)
    similarities = embeddings @ query / denom
    # Exclude the query itself and clamp k
    similarities[query_index] = -np.inf
    k = min(k, len(embeddings) - 1)
    top_k = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_k]
