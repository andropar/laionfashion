"""Tests for laionfashion.bundle – loading and nearest-neighbor search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laionfashion.bundle import DebugBundle, load_bundle, nearest_neighbors


def _make_bundle(tmp_path: Path, n: int = 10, dim: int = 32, use_parquet: bool = True) -> Path:
    """Create a synthetic debug bundle on disk."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()

    records = []
    for i in range(n):
        thumb_name = f"{i:06d}_{i}.jpg"
        # Write a tiny placeholder file
        (thumb_dir / thumb_name).write_bytes(b"\xff\xd8dummy")
        records.append(
            {
                "row_id": i,
                "global_index": i * 100,
                "caption": f"person wearing outfit {i}",
                "thumbnail_path": f"thumbnails/{thumb_name}",
            }
        )

    df = pd.DataFrame.from_records(records)
    if use_parquet:
        df.to_parquet(tmp_path / "records.parquet", index=False)
    else:
        df.to_csv(tmp_path / "records.csv", index=False)

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalize so cosine similarity is just a dot product
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(tmp_path / "embeddings.npy", embeddings)

    return tmp_path


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------


def test_load_bundle_parquet(tmp_path: Path) -> None:
    bundle_dir = _make_bundle(tmp_path, n=5, use_parquet=True)
    bundle = load_bundle(bundle_dir)
    assert bundle.n_images == 5
    assert bundle.embeddings.shape == (5, 32)


def test_load_bundle_csv(tmp_path: Path) -> None:
    bundle_dir = _make_bundle(tmp_path, n=5, use_parquet=False)
    bundle = load_bundle(bundle_dir)
    assert bundle.n_images == 5


def test_load_bundle_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Bundle directory not found"):
        load_bundle(tmp_path / "nope")


def test_load_bundle_missing_records(tmp_path: Path) -> None:
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 4)))
    with pytest.raises(FileNotFoundError, match="No records"):
        load_bundle(tmp_path)


def test_load_bundle_missing_embeddings(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "records.parquet")
    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        load_bundle(tmp_path)


def test_load_bundle_row_count_mismatch(tmp_path: Path) -> None:
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "records.parquet")
    np.save(tmp_path / "embeddings.npy", np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Row count mismatch"):
        load_bundle(tmp_path)


# ---------------------------------------------------------------------------
# thumbnail_path
# ---------------------------------------------------------------------------


def test_thumbnail_path_returns_absolute(tmp_path: Path) -> None:
    bundle_dir = _make_bundle(tmp_path, n=3)
    bundle = load_bundle(bundle_dir)
    path = bundle.thumbnail_path(0)
    assert path is not None
    assert path.is_absolute()
    assert path.exists()


# ---------------------------------------------------------------------------
# nearest_neighbors
# ---------------------------------------------------------------------------


def test_nearest_neighbors_excludes_self() -> None:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    results = nearest_neighbors(emb, query_index=5, k=5)
    indices = [idx for idx, _ in results]
    assert 5 not in indices
    assert len(results) == 5


def test_nearest_neighbors_sorted_descending() -> None:
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((30, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    results = nearest_neighbors(emb, query_index=0, k=10)
    sims = [s for _, s in results]
    assert sims == sorted(sims, reverse=True)


def test_nearest_neighbors_identical_vectors() -> None:
    """When all vectors are the same, similarities should be ~1.0 (except self)."""
    emb = np.ones((5, 4), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    results = nearest_neighbors(emb, query_index=0, k=4)
    assert len(results) == 4
    for _, sim in results:
        assert abs(sim - 1.0) < 1e-5


def test_nearest_neighbors_k_larger_than_available() -> None:
    """When k > n_images - 1, clamp to available neighbors and never return -inf."""
    emb = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    results = nearest_neighbors(emb, query_index=0, k=10)
    assert len(results) == 2  # only 2 other images
    indices = [idx for idx, _ in results]
    assert 0 not in indices
    for _, sim in results:
        assert sim != float("-inf")
        assert -1.0 <= sim <= 1.0
