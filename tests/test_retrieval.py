"""Tests for garment embedding and cross-category retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laionfashion.garments import (
    GarmentRegion,
    MockDetector,
    MockEmbedder,
    embed_garment_crops,
    extract_garments_from_bundle,
    save_garments,
)
from laionfashion.retrieval import (
    RetrievalResult,
    retrieve_cross_category,
    retrieve_similar_garments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle_with_garments(tmp_path: Path, n: int = 6) -> tuple[pd.DataFrame, np.ndarray, Path]:
    """Create a bundle with garments and mock embeddings."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    records = []
    for i in range(n):
        name = f"{i:06d}_{i}.jpg"
        Image.new("RGB", (80, 120), color=(i * 40, 100, 200)).save(thumb_dir / name, quality=90)
        records.append({
            "row_id": i,
            "caption": f"person wearing outfit {i}",
            "thumbnail_path": f"thumbnails/{name}",
        })
    pd.DataFrame(records).to_parquet(tmp_path / "records.parquet", index=False)
    emb = np.random.default_rng(0).standard_normal((n, 16)).astype(np.float32)
    np.save(tmp_path / "embeddings.npy", emb)

    records_df = pd.read_parquet(tmp_path / "records.parquet")
    # Use a detector with 3 categories
    detector = MockDetector([
        GarmentRegion("top", 0, 0, 80, 50, confidence=0.9),
        GarmentRegion("bottom", 0, 50, 80, 50, confidence=0.85),
        GarmentRegion("shoes", 10, 100, 60, 20, confidence=0.7),
    ])
    garments = extract_garments_from_bundle(
        records_df, tmp_path, method="detr", detector=detector
    )
    save_garments(garments, tmp_path)

    # Embed with mock embedder
    embedder = MockEmbedder(dim=32)
    garment_embeddings = embed_garment_crops(garments, tmp_path, embedder)
    np.save(tmp_path / "garment_embeddings.npy", garment_embeddings)

    return garments, garment_embeddings, tmp_path


# ---------------------------------------------------------------------------
# Garment embedding
# ---------------------------------------------------------------------------


class TestEmbedGarmentCrops:
    def test_output_shape(self, tmp_path: Path) -> None:
        thumb_dir = tmp_path / "thumbnails"
        thumb_dir.mkdir()
        records = []
        for i in range(3):
            name = f"{i:06d}_{i}.jpg"
            Image.new("RGB", (80, 120)).save(thumb_dir / name, quality=90)
            records.append({"row_id": i, "thumbnail_path": f"thumbnails/{name}"})
        pd.DataFrame(records).to_parquet(tmp_path / "records.parquet", index=False)
        np.save(tmp_path / "embeddings.npy", np.zeros((3, 8), dtype=np.float32))

        records_df = pd.read_parquet(tmp_path / "records.parquet")
        garments = extract_garments_from_bundle(
            records_df, tmp_path, method="detr", detector=MockDetector()
        )
        embedder = MockEmbedder(dim=64)
        emb = embed_garment_crops(garments, tmp_path, embedder)
        assert emb.shape == (len(garments), 64)
        assert emb.dtype == np.float32

    def test_embeddings_are_normalized(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        norms = np.linalg.norm(emb, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_deterministic(self, tmp_path: Path) -> None:
        thumb_dir = tmp_path / "thumbnails"
        thumb_dir.mkdir()
        Image.new("RGB", (80, 120)).save(thumb_dir / "000000_0.jpg", quality=90)
        records = pd.DataFrame([{"row_id": 0, "thumbnail_path": "thumbnails/000000_0.jpg"}])
        records.to_parquet(tmp_path / "records.parquet", index=False)
        np.save(tmp_path / "embeddings.npy", np.zeros((1, 8), dtype=np.float32))

        garments = extract_garments_from_bundle(
            records, tmp_path, method="detr", detector=MockDetector()
        )
        e1 = embed_garment_crops(garments, tmp_path, MockEmbedder(dim=16, seed=42))
        e2 = embed_garment_crops(garments, tmp_path, MockEmbedder(dim=16, seed=42))
        np.testing.assert_array_equal(e1, e2)

    def test_bundle_loads_garment_embeddings(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        _, _, bundle_dir = _make_bundle_with_garments(tmp_path)
        bundle = load_bundle(bundle_dir)
        assert bundle.garment_embeddings is not None
        assert bundle.garment_embeddings.shape[0] == bundle.n_garments


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieveSimilarGarments:
    def test_returns_correct_category(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,  # a "top"
            garments=garments,
            embeddings=emb,
            target_category="bottom",
            k=5,
        )
        assert len(results) > 0
        assert all(r.category == "bottom" for r in results)

    def test_excludes_same_outfit(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        query_gid = 0
        query_outfit = int(garments.loc[garments["garment_id"] == query_gid, "outfit_id"].iloc[0])
        results = retrieve_similar_garments(
            query_garment_id=query_gid,
            garments=garments,
            embeddings=emb,
            target_category="bottom",
            exclude_same_outfit=True,
            k=10,
        )
        assert all(r.outfit_id != query_outfit for r in results)

    def test_includes_same_outfit_when_allowed(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,
            garments=garments,
            embeddings=emb,
            target_category="bottom",
            exclude_same_outfit=False,
            k=10,
        )
        outfit_ids = {r.outfit_id for r in results}
        # With 6 outfits, same-outfit bottom should be included
        assert int(garments.loc[garments["garment_id"] == 0, "outfit_id"].iloc[0]) in outfit_ids

    def test_excludes_query_garment(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,
            garments=garments,
            embeddings=emb,
            k=20,
        )
        assert all(r.garment_id != 0 for r in results)

    def test_sorted_by_similarity(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,
            garments=garments,
            embeddings=emb,
            k=10,
        )
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_no_target_retrieves_other_categories(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,  # a "top"
            garments=garments,
            embeddings=emb,
            target_category=None,
            k=20,
        )
        categories = {r.category for r in results}
        assert "top" not in categories
        assert len(categories) >= 1

    def test_invalid_garment_id_raises(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        with pytest.raises(KeyError):
            retrieve_similar_garments(
                query_garment_id=9999,
                garments=garments,
                embeddings=emb,
                k=5,
            )

    def test_empty_candidates_returns_empty(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_similar_garments(
            query_garment_id=0,
            garments=garments,
            embeddings=emb,
            target_category="nonexistent_category",
            k=5,
        )
        assert results == []


class TestRetrieveCrossCategory:
    def test_returns_dict_per_category(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_cross_category(
            query_garment_id=0,  # "top"
            garments=garments,
            embeddings=emb,
            k=3,
        )
        assert isinstance(results, dict)
        # Should have entries for "bottom" and "shoes" (not "top")
        assert "top" not in results
        assert "bottom" in results
        assert "shoes" in results

    def test_each_category_has_results(self, tmp_path: Path) -> None:
        garments, emb, _ = _make_bundle_with_garments(tmp_path)
        results = retrieve_cross_category(
            query_garment_id=0,
            garments=garments,
            embeddings=emb,
            k=3,
        )
        for cat, hits in results.items():
            assert len(hits) > 0
            assert all(r.category == cat for r in hits)
