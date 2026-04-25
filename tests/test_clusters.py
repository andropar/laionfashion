"""Tests for laionfashion.clusters — clustering, labelling, and exemplars."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laionfashion.clusters import cluster_embeddings, cluster_exemplars, label_clusters


def _random_embeddings(n: int, dim: int = 32, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


# ---------------------------------------------------------------------------
# cluster_embeddings
# ---------------------------------------------------------------------------


class TestClusterEmbeddings:
    def test_correct_number_of_clusters(self) -> None:
        emb = _random_embeddings(50)
        df = cluster_embeddings(emb, n_clusters=5)
        assert df["cluster_id"].nunique() == 5

    def test_all_row_ids_assigned(self) -> None:
        n = 30
        emb = _random_embeddings(n)
        df = cluster_embeddings(emb, n_clusters=4)
        assert set(df["row_id"]) == set(range(n))
        assert len(df) == n

    def test_returns_dataframe_with_expected_columns(self) -> None:
        emb = _random_embeddings(20)
        df = cluster_embeddings(emb, n_clusters=3)
        assert list(df.columns) == ["row_id", "cluster_id"]

    def test_small_bundle_fewer_than_n_clusters(self) -> None:
        """When n_samples < n_clusters, clamp and still produce valid output."""
        emb = _random_embeddings(3, dim=8)
        df = cluster_embeddings(emb, n_clusters=10)
        assert df["cluster_id"].nunique() <= 3
        assert len(df) == 3
        assert set(df["row_id"]) == {0, 1, 2}

    def test_single_sample(self) -> None:
        emb = _random_embeddings(1, dim=8)
        df = cluster_embeddings(emb, n_clusters=5)
        assert len(df) == 1
        assert df["cluster_id"].nunique() == 1

    def test_unsupported_method_raises(self) -> None:
        emb = _random_embeddings(10)
        with pytest.raises(ValueError, match="Unsupported"):
            cluster_embeddings(emb, method="dbscan")


# ---------------------------------------------------------------------------
# label_clusters
# ---------------------------------------------------------------------------


class TestLabelClusters:
    def test_without_axis_scores(self) -> None:
        emb = _random_embeddings(20)
        cluster_ids = np.array([0] * 10 + [1] * 10)
        labels = label_clusters(emb, cluster_ids, axis_scores=None)
        assert set(labels.keys()) == {0, 1}
        assert labels[0] == "Cluster 0"
        assert labels[1] == "Cluster 1"

    def test_with_synthetic_axis_scores(self) -> None:
        """Labels should reflect axis names when scores are provided."""
        n = 20
        emb = _random_embeddings(n)
        cluster_ids = np.array([0] * 10 + [1] * 10)

        # Cluster 0: high colorful, low formal
        # Cluster 1: low colorful, high formal
        axis_scores = pd.DataFrame({
            "row_id": np.arange(n),
            "colorful_vs_neutral": np.concatenate([
                np.full(10, 0.8), np.full(10, -0.8)
            ]),
            "formal_vs_casual": np.concatenate([
                np.full(10, -0.5), np.full(10, 0.9)
            ]),
        })

        labels = label_clusters(emb, cluster_ids, axis_scores)
        assert len(labels) == 2
        # Cluster 0 should mention colorful (highest abs score)
        assert "Colorful" in labels[0]
        # Cluster 1 should mention formal
        assert "Formal" in labels[1]

    def test_with_proxy_axis_names(self) -> None:
        """Proxy suffixes like _proxy should be stripped."""
        n = 10
        emb = _random_embeddings(n)
        cluster_ids = np.zeros(n, dtype=int)
        axis_scores = pd.DataFrame({
            "row_id": np.arange(n),
            "colorful_proxy": np.full(n, 0.9),
            "formal_proxy": np.full(n, 0.1),
        })
        labels = label_clusters(emb, cluster_ids, axis_scores)
        assert "Colorful" in labels[0]

    def test_empty_axis_columns(self) -> None:
        """If axis_scores has only row_id, fall back to numeric labels."""
        emb = _random_embeddings(5)
        cluster_ids = np.zeros(5, dtype=int)
        axis_scores = pd.DataFrame({"row_id": np.arange(5)})
        labels = label_clusters(emb, cluster_ids, axis_scores)
        assert labels[0] == "Cluster 0"


# ---------------------------------------------------------------------------
# cluster_exemplars
# ---------------------------------------------------------------------------


class TestClusterExemplars:
    def test_exemplars_are_within_cluster(self) -> None:
        n = 40
        emb = _random_embeddings(n)
        df = cluster_embeddings(emb, n_clusters=4)
        cluster_ids = df["cluster_id"].values
        exemplars = cluster_exemplars(emb, cluster_ids, n=5)

        for cid, indices in exemplars.items():
            for idx in indices:
                assert cluster_ids[idx] == cid, (
                    f"Exemplar {idx} assigned to cluster {cluster_ids[idx]}, "
                    f"expected {cid}"
                )

    def test_exemplar_count(self) -> None:
        emb = _random_embeddings(30)
        cluster_ids = np.array([0] * 15 + [1] * 15)
        exemplars = cluster_exemplars(emb, cluster_ids, n=5)
        for cid in [0, 1]:
            assert len(exemplars[cid]) == 5

    def test_small_cluster_returns_all_members(self) -> None:
        emb = _random_embeddings(5, dim=8)
        cluster_ids = np.array([0, 0, 1, 1, 1])
        exemplars = cluster_exemplars(emb, cluster_ids, n=10)
        assert len(exemplars[0]) == 2
        assert len(exemplars[1]) == 3

    def test_exemplars_sorted_by_distance(self) -> None:
        """First exemplar should be closer to centroid than last."""
        n = 50
        emb = _random_embeddings(n)
        cluster_ids = np.zeros(n, dtype=int)
        exemplars = cluster_exemplars(emb, cluster_ids, n=10)

        centroid = emb.mean(axis=0)
        dists = [np.linalg.norm(emb[i] - centroid) for i in exemplars[0]]
        assert dists == sorted(dists)
