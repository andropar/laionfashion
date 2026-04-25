"""Tests for laionfashion.evaluation — hold-out garment retrieval evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from laionfashion.evaluation import (
    EvalMetrics,
    EvalQuery,
    build_eval_queries,
    evaluate_retrieval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_garments(n_outfits: int = 6, categories: list[str] | None = None) -> pd.DataFrame:
    """Create synthetic garments with 2-3 garments per outfit."""
    if categories is None:
        categories = ["top", "bottom", "shoes"]
    rows = []
    gid = 0
    for oid in range(n_outfits):
        # Each outfit gets 2 or 3 garments
        n_garments = 2 + (oid % 2)
        for i in range(min(n_garments, len(categories))):
            rows.append({
                "outfit_id": oid,
                "garment_id": gid,
                "category": categories[i],
                "crop_path": f"garment_crops/{gid:06d}_{oid}_{categories[i]}.jpg",
            })
            gid += 1
    return pd.DataFrame(rows)


def _make_embeddings(garments: pd.DataFrame, dim: int = 32) -> np.ndarray:
    """Create synthetic normalized embeddings."""
    n = garments["garment_id"].max() + 1
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


# ---------------------------------------------------------------------------
# build_eval_queries
# ---------------------------------------------------------------------------


class TestBuildEvalQueries:
    def test_creates_queries_for_multi_garment_outfits(self) -> None:
        garments = _make_garments(n_outfits=4)
        queries = build_eval_queries(garments, min_outfit_garments=2)
        assert len(queries) > 0
        # All queries should come from outfits with >= 2 garments
        for q in queries:
            outfit_garments = garments[garments["outfit_id"] == q.query_outfit_id]
            assert len(outfit_garments) >= 2

    def test_skips_single_garment_outfits(self) -> None:
        # One outfit with 1 garment, others with 2+
        garments = pd.DataFrame([
            {"outfit_id": 0, "garment_id": 0, "category": "top", "crop_path": "a.jpg"},
            {"outfit_id": 1, "garment_id": 1, "category": "top", "crop_path": "b.jpg"},
            {"outfit_id": 1, "garment_id": 2, "category": "bottom", "crop_path": "c.jpg"},
        ])
        queries = build_eval_queries(garments, min_outfit_garments=2)
        outfit_ids = {q.query_outfit_id for q in queries}
        assert 0 not in outfit_ids
        assert 1 in outfit_ids

    def test_gt_outfit_garment_ids(self) -> None:
        garments = _make_garments(n_outfits=2)
        queries = build_eval_queries(garments)
        for q in queries:
            # GT should be other garments from same outfit, not the query itself
            assert q.query_garment_id not in q.gt_outfit_garment_ids
            for gt_gid in q.gt_outfit_garment_ids:
                gt_outfit = int(garments.loc[garments["garment_id"] == gt_gid, "outfit_id"].iloc[0])
                assert gt_outfit == q.query_outfit_id

    def test_target_categories_filter(self) -> None:
        garments = _make_garments(n_outfits=4)
        queries = build_eval_queries(garments, target_categories=["top"])
        assert all(q.query_category == "top" for q in queries)

    def test_empty_garments(self) -> None:
        garments = pd.DataFrame(columns=["outfit_id", "garment_id", "category", "crop_path"])
        queries = build_eval_queries(garments)
        assert queries == []


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------


class TestEvaluateRetrieval:
    def test_returns_metrics(self) -> None:
        garments = _make_garments(n_outfits=6)
        embeddings = _make_embeddings(garments)
        queries = build_eval_queries(garments)
        metrics = evaluate_retrieval(queries, garments, embeddings)
        assert metrics.n_queries == len(queries)
        assert 0 <= metrics.hit_at_1 <= 1
        assert 0 <= metrics.hit_at_5 <= 1
        assert 0 <= metrics.hit_at_10 <= 1
        assert 0 <= metrics.mrr <= 1

    def test_metrics_are_valid_ranges(self) -> None:
        """All metric values should be in [0, 1]."""
        garments = _make_garments(n_outfits=6)
        embeddings = _make_embeddings(garments)
        queries = build_eval_queries(garments)
        metrics = evaluate_retrieval(queries, garments, embeddings)
        assert 0 <= metrics.hit_at_1 <= 1
        assert 0 <= metrics.hit_at_5 <= 1
        assert 0 <= metrics.hit_at_10 <= 1
        assert 0 <= metrics.mrr <= 1

    def test_per_category_breakdown(self) -> None:
        garments = _make_garments(n_outfits=6)
        embeddings = _make_embeddings(garments)
        queries = build_eval_queries(garments)
        metrics = evaluate_retrieval(queries, garments, embeddings)
        assert len(metrics.per_category) > 0
        for cat, cm in metrics.per_category.items():
            assert "n_queries" in cm
            assert "mrr" in cm

    def test_to_dict(self) -> None:
        garments = _make_garments(n_outfits=4)
        embeddings = _make_embeddings(garments)
        queries = build_eval_queries(garments)
        metrics = evaluate_retrieval(queries, garments, embeddings)
        d = metrics.to_dict()
        assert "n_queries" in d
        assert "hit_at_1" in d
        assert "mrr" in d

    def test_empty_queries(self) -> None:
        garments = _make_garments(n_outfits=2)
        embeddings = _make_embeddings(garments)
        metrics = evaluate_retrieval([], garments, embeddings)
        assert metrics.n_queries == 0
        assert metrics.mrr == 0.0
