"""Tests for laionfashion.projection."""

from __future__ import annotations

import numpy as np
import pytest

from laionfashion.projection import (
    ProjectionMethod,
    project_embeddings,
)


def _random_embeddings(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


# ---------------------------------------------------------------------------
# Output shape and schema
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_returns_dataframe_with_correct_columns(self) -> None:
        df, _ = project_embeddings(_random_embeddings(10))
        assert list(df.columns) == ["row_id", "x", "y"]

    def test_row_count_matches_input(self) -> None:
        emb = _random_embeddings(25)
        df, _ = project_embeddings(emb)
        assert len(df) == 25

    def test_row_ids_are_sequential(self) -> None:
        df, _ = project_embeddings(_random_embeddings(8))
        assert list(df["row_id"]) == list(range(8))


# ---------------------------------------------------------------------------
# Trivial fallback (n <= 2)
# ---------------------------------------------------------------------------


class TestTrivialFallback:
    def test_single_image(self) -> None:
        df, method = project_embeddings(_random_embeddings(1))
        assert method == ProjectionMethod.TRIVIAL
        assert len(df) == 1

    def test_two_images(self) -> None:
        df, method = project_embeddings(_random_embeddings(2))
        assert method == ProjectionMethod.TRIVIAL
        assert len(df) == 2

    def test_forced_trivial(self) -> None:
        df, method = project_embeddings(_random_embeddings(20), method="trivial")
        assert method == ProjectionMethod.TRIVIAL


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


class TestPCA:
    def test_pca_on_small_bundle(self) -> None:
        df, method = project_embeddings(_random_embeddings(5))
        assert method == ProjectionMethod.PCA
        assert len(df) == 5

    def test_forced_pca(self) -> None:
        df, method = project_embeddings(_random_embeddings(50), method="pca")
        assert method == ProjectionMethod.PCA
        assert len(df) == 50

    def test_pca_is_deterministic(self) -> None:
        emb = _random_embeddings(30)
        df1, _ = project_embeddings(emb, method="pca", random_state=42)
        df2, _ = project_embeddings(emb, method="pca", random_state=42)
        np.testing.assert_array_equal(df1["x"].values, df2["x"].values)
        np.testing.assert_array_equal(df1["y"].values, df2["y"].values)

    def test_pca_different_seed_same_result(self) -> None:
        """PCA via SVD is deterministic regardless of random_state."""
        emb = _random_embeddings(20)
        df1, _ = project_embeddings(emb, method="pca", random_state=1)
        df2, _ = project_embeddings(emb, method="pca", random_state=99)
        np.testing.assert_array_almost_equal(df1["x"].values, df2["x"].values)


# ---------------------------------------------------------------------------
# UMAP (only if installed)
# ---------------------------------------------------------------------------


def _umap_installed() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _umap_installed(), reason="umap-learn not installed")
class TestUMAP:
    def test_auto_selects_umap_for_large_bundle(self) -> None:
        df, method = project_embeddings(_random_embeddings(50))
        assert method == ProjectionMethod.UMAP
        assert len(df) == 50

    def test_forced_umap(self) -> None:
        df, method = project_embeddings(_random_embeddings(20), method="umap")
        assert method == ProjectionMethod.UMAP

    def test_umap_output_finite(self) -> None:
        df, _ = project_embeddings(_random_embeddings(30), method="umap")
        assert np.all(np.isfinite(df["x"].values))
        assert np.all(np.isfinite(df["y"].values))


# ---------------------------------------------------------------------------
# Auto-selection logic
# ---------------------------------------------------------------------------


class TestAutoSelect:
    def test_tiny_gets_trivial(self) -> None:
        _, method = project_embeddings(_random_embeddings(2))
        assert method == ProjectionMethod.TRIVIAL

    def test_small_gets_pca(self) -> None:
        _, method = project_embeddings(_random_embeddings(10))
        assert method == ProjectionMethod.PCA

    @pytest.mark.skipif(not _umap_installed(), reason="umap-learn not installed")
    def test_large_gets_umap(self) -> None:
        _, method = project_embeddings(_random_embeddings(50))
        assert method == ProjectionMethod.UMAP


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_embeddings(self) -> None:
        """All-same vectors should not crash."""
        emb = np.ones((10, 8), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        df, _ = project_embeddings(emb, method="pca")
        assert len(df) == 10
        # All identical → PCA collapses to origin
        assert np.allclose(df["x"].values, 0.0, atol=1e-6)

    def test_three_images_pca(self) -> None:
        """Boundary: 3 images should use PCA, not trivial."""
        _, method = project_embeddings(_random_embeddings(3))
        assert method == ProjectionMethod.PCA
