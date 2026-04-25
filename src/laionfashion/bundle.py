"""Load and query debug bundles exported by 01_build_debug_subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class DebugBundle:
    """A loaded debug bundle: records table, embeddings matrix, and bundle path."""

    records: pd.DataFrame
    embeddings: np.ndarray
    bundle_dir: Path

    @property
    def n_images(self) -> int:
        return len(self.records)

    def thumbnail_path(self, row_id: int) -> Path | None:
        """Return the absolute thumbnail path for a given row_id."""
        rel = self.records.loc[row_id, "thumbnail_path"]
        if pd.isna(rel):
            return None
        path = self.bundle_dir / rel
        return path if path.exists() else None


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

    return DebugBundle(records=records, embeddings=embeddings, bundle_dir=bundle_dir)


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
    # Exclude the query itself
    similarities[query_index] = -np.inf
    top_k = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_k]
