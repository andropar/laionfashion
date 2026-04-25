"""Tests for FAISS-based vector search."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from laionfashion.search import (
    FAISS_AVAILABLE,
    build_faiss_index,
    load_index,
    save_index,
    search_index,
)

pytestmark = pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_normalized(n: int, d: int = 64, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Return n random L2-normalized vectors of dimension d."""
    if rng is None:
        rng = np.random.RandomState(42)
    vecs = rng.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_basic(self) -> None:
        emb = _random_normalized(100)
        index = build_faiss_index(emb)
        assert index.ntotal == 100

    def test_dimension(self) -> None:
        emb = _random_normalized(50, d=128)
        index = build_faiss_index(emb)
        assert index.d == 128


class TestSearch:
    def test_returns_correct_k(self) -> None:
        emb = _random_normalized(100)
        index = build_faiss_index(emb)
        results = search_index(index, emb[0], k=10)
        assert len(results) == 10

    def test_k_larger_than_index(self) -> None:
        emb = _random_normalized(5)
        index = build_faiss_index(emb)
        results = search_index(index, emb[0], k=20)
        assert len(results) == 5

    def test_results_sorted_descending(self) -> None:
        emb = _random_normalized(100)
        index = build_faiss_index(emb)
        results = search_index(index, emb[0], k=10)
        similarities = [s for _, s in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_self_query_top_result(self) -> None:
        emb = _random_normalized(100)
        index = build_faiss_index(emb)
        for query_idx in [0, 42, 99]:
            results = search_index(index, emb[query_idx], k=5)
            top_idx, top_sim = results[0]
            assert top_idx == query_idx
            assert top_sim == pytest.approx(1.0, abs=1e-5)

    def test_2d_query(self) -> None:
        emb = _random_normalized(50)
        index = build_faiss_index(emb)
        results = search_index(index, emb[0].reshape(1, -1), k=3)
        assert len(results) == 3


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        emb = _random_normalized(100)
        index = build_faiss_index(emb)
        path = tmp_path / "test.faiss"
        save_index(index, path)
        assert path.exists()

        loaded = load_index(path)
        assert loaded.ntotal == index.ntotal

        # Search results should be identical.
        results_orig = search_index(index, emb[0], k=5)
        results_loaded = search_index(loaded, emb[0], k=5)
        assert results_orig == results_loaded


class TestMissingFaiss:
    def test_import_error_raised(self) -> None:
        with mock.patch.dict("sys.modules", {"faiss": None}):
            # Re-import to pick up the mocked faiss.
            import importlib
            import laionfashion.search as search_mod

            orig_available = search_mod.FAISS_AVAILABLE
            search_mod.FAISS_AVAILABLE = False
            try:
                with pytest.raises(ImportError, match="faiss is required"):
                    search_mod._require_faiss()
            finally:
                search_mod.FAISS_AVAILABLE = orig_available
