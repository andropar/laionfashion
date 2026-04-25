"""Tests for laionfashion.image_scoring – scorer interface and mock scorers."""

from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laionfashion.image_scoring import ConstantScorer, ListScorer, ThresholdMockScorer


# ---------------------------------------------------------------------------
# Mock scorer tests
# ---------------------------------------------------------------------------


class TestConstantScorer:
    def test_returns_configured_score(self) -> None:
        scorer = ConstantScorer(score=0.42)
        img = Image.new("RGB", (8, 8))
        assert scorer.score_image(img) == 0.42

    def test_default_score(self) -> None:
        scorer = ConstantScorer()
        assert scorer.score_image(Image.new("RGB", (8, 8))) == 1.0


class TestThresholdMockScorer:
    def test_alternates_high_low(self) -> None:
        scorer = ThresholdMockScorer(high=0.5, low=-0.5)
        img = Image.new("RGB", (8, 8))
        assert scorer.score_image(img) == 0.5   # 1st call: high
        assert scorer.score_image(img) == -0.5   # 2nd call: low
        assert scorer.score_image(img) == 0.5   # 3rd call: high


# ---------------------------------------------------------------------------
# Integration with collect_caption_filtered_subset using mock scorer
# ---------------------------------------------------------------------------


def _write_test_tar(path: Path, n_images: int) -> None:
    with tarfile.open(path, "w") as tar:
        for index in range(n_images):
            image = Image.new("RGB", (8, 8), color=(index * 30, 0, 0))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_payload = image_bytes.getvalue()
            image_info = tarfile.TarInfo(f"{index:010d}.jpg")
            image_info.size = len(image_payload)
            tar.addfile(image_info, io.BytesIO(image_payload))

            metadata_payload = (
                f'{{"caption": "woman wearing outfit {index}", "url": "https://example.com/{index}"}}'
            ).encode()
            metadata_info = tarfile.TarInfo(f"{index:010d}.json")
            metadata_info.size = len(metadata_payload)
            tar.addfile(metadata_info, io.BytesIO(metadata_payload))


def _make_test_index(tmp_path: Path, n_images: int = 10):
    """Create a tiny NaturalSubsetIndex with synthetic data."""
    from laionfashion.data_access import NaturalSubsetIndex

    tar_path = tmp_path / "test.tar"
    _write_test_tar(tar_path, n_images)
    metadata_path = tmp_path / "_metadata.pkl"
    with metadata_path.open("wb") as f:
        pickle.dump(
            [{"tar_path": tar_path, "start_index": 0, "end_index": n_images, "failed_images": ()}],
            f,
        )
    return NaturalSubsetIndex.from_metadata(metadata_path)


class TestImageScorerIntegration:
    def test_without_scorer_accepts_all(self, tmp_path: Path) -> None:
        """Without image scorer, all caption-passing images are accepted."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=5)
        rng = np.random.default_rng(42)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=5, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
        )
        assert len(records) == 5
        assert diag.image_scored == 0

    def test_constant_scorer_accepts_above_threshold(self, tmp_path: Path) -> None:
        """Constant scorer above threshold accepts all."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=5)
        rng = np.random.default_rng(42)
        scorer = ConstantScorer(score=0.5)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=5, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
            image_scorer=scorer, min_image_score=0.0,
        )
        assert len(records) == 5
        assert diag.image_scored == 5
        assert diag.image_rejected == 0

    def test_constant_scorer_rejects_below_threshold(self, tmp_path: Path) -> None:
        """Constant scorer below threshold rejects all."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=5)
        rng = np.random.default_rng(42)
        scorer = ConstantScorer(score=-0.1)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=5, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
            image_scorer=scorer, min_image_score=0.0,
        )
        assert len(records) == 0
        assert diag.image_scored == 5
        assert diag.image_rejected == 5

    def test_alternating_scorer_accepts_half(self, tmp_path: Path) -> None:
        """Alternating scorer should accept ~half the images."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=6)
        rng = np.random.default_rng(42)
        scorer = ThresholdMockScorer(high=0.5, low=-0.5)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=10, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
            image_scorer=scorer, min_image_score=0.0,
        )
        # 3 accepted (odd calls), 3 rejected (even calls)
        assert len(records) == 3
        assert diag.image_scored == 6
        assert diag.image_rejected == 3

    def test_image_score_in_records(self, tmp_path: Path) -> None:
        """Accepted records should contain image_outfit_score column."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=3)
        rng = np.random.default_rng(42)
        scorer = ConstantScorer(score=0.75)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=3, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
            image_scorer=scorer, min_image_score=0.0,
        )
        assert "image_outfit_score" in records.columns
        assert all(records["image_outfit_score"] == 0.75)

    def test_diagnostics_image_scoring_stats(self, tmp_path: Path) -> None:
        """Diagnostics should include image scoring stats."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=4)
        rng = np.random.default_rng(42)
        scorer = ThresholdMockScorer(high=0.5, low=-0.5)
        records, diag = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=10, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
            image_scorer=scorer, min_image_score=0.0,
        )
        d = diag.to_dict()
        assert "image_scoring" in d
        assert d["image_scoring"]["scored"] == 4
        assert d["image_scoring"]["rejected"] == 2
        assert d["image_scoring"]["accepted"] == 2
        assert "score_distribution" in d["image_scoring"]

    def test_no_image_score_column_without_scorer(self, tmp_path: Path) -> None:
        """Without scorer, records should not have image_outfit_score column."""
        from laionfashion.debug_export import collect_caption_filtered_subset

        index = _make_test_index(tmp_path, n_images=3)
        rng = np.random.default_rng(42)
        records, _ = collect_caption_filtered_subset(
            index=index, rng=rng, n_images=3, candidate_scan=10,
            thumbnail_dir=tmp_path / "thumbs", thumbnail_size=64,
        )
        assert "image_outfit_score" not in records.columns


# ---------------------------------------------------------------------------
# ListScorer
# ---------------------------------------------------------------------------


class TestListScorer:
    def test_returns_scores_in_order(self) -> None:
        scorer = ListScorer([0.1, 0.5, 0.9])
        img = Image.new("RGB", (8, 8))
        assert scorer.score_image(img) == 0.1
        assert scorer.score_image(img) == 0.5
        assert scorer.score_image(img) == 0.9

    def test_cycles(self) -> None:
        scorer = ListScorer([0.2, 0.8])
        img = Image.new("RGB", (8, 8))
        assert scorer.score_image(img) == 0.2
        assert scorer.score_image(img) == 0.8
        assert scorer.score_image(img) == 0.2


# ---------------------------------------------------------------------------
# score_and_rank_candidates (post-ranking pipeline)
# ---------------------------------------------------------------------------


def _make_candidate_bundle(tmp_path: Path, n: int = 10) -> tuple[pd.DataFrame, Path]:
    """Create synthetic candidates with thumbnails on disk."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()

    records = []
    for i in range(n):
        thumb_name = f"{i:06d}_{i * 100}.jpg"
        img = Image.new("RGB", (8, 8), color=(i * 25, 0, 0))
        img.save(thumb_dir / thumb_name, quality=90)
        records.append({
            "row_id": i,
            "global_index": i * 100,
            "caption": f"woman wearing outfit {i}",
            "thumbnail_path": f"thumbnails/{thumb_name}",
        })

    return pd.DataFrame(records), tmp_path


class TestScoreAndRankCandidates:
    def test_top_n_selection(self, tmp_path: Path) -> None:
        """Top-N should select highest-scoring candidates."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=6)
        # Scores: 0.1, 0.5, 0.3, 0.9, 0.2, 0.7
        scorer = ListScorer([0.1, 0.5, 0.3, 0.9, 0.2, 0.7])
        ranked, diag = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=3,
        )
        assert len(ranked) == 3
        assert diag.n_candidates == 6
        assert diag.n_scored == 6
        assert diag.n_exported == 3
        # Top 3 scores should be 0.9, 0.7, 0.5 (descending)
        scores = ranked["image_outfit_score"].tolist()
        assert scores == [0.9, 0.7, 0.5]

    def test_row_ids_reindexed(self, tmp_path: Path) -> None:
        """Exported records should have sequential row_ids starting from 0."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=5)
        scorer = ListScorer([0.5, 0.1, 0.9, 0.3, 0.7])
        ranked, _ = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=3,
        )
        assert list(ranked["row_id"]) == [0, 1, 2]

    def test_n_export_larger_than_candidates(self, tmp_path: Path) -> None:
        """Requesting more than available should return all candidates."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=3)
        scorer = ConstantScorer(score=0.5)
        ranked, diag = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=100,
        )
        assert len(ranked) == 3
        assert diag.n_exported == 3

    def test_diagnostics_score_distribution(self, tmp_path: Path) -> None:
        """Diagnostics should have quantiles and stats."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=8)
        scorer = ListScorer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        _, diag = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=4,
        )
        d = diag.to_dict()
        dist = d["score_distribution"]
        assert dist["min"] == 0.1
        assert dist["max"] == 0.8
        assert dist["count"] == 8
        assert "median" in dist
        assert "p25" in dist
        assert "p75" in dist

    def test_preserves_original_columns(self, tmp_path: Path) -> None:
        """Ranked output should keep original columns plus image_outfit_score."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=4)
        scorer = ConstantScorer(score=0.5)
        ranked, _ = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=2,
        )
        assert "caption" in ranked.columns
        assert "global_index" in ranked.columns
        assert "image_outfit_score" in ranked.columns
        assert "thumbnail_path" in ranked.columns

    def test_missing_thumbnail_counted_as_error(self, tmp_path: Path) -> None:
        """Missing thumbnails should be counted as score errors."""
        from laionfashion.debug_export import score_and_rank_candidates

        candidates, bundle_dir = _make_candidate_bundle(tmp_path, n=3)
        # Delete one thumbnail
        thumbs = list((bundle_dir / "thumbnails").iterdir())
        thumbs[0].unlink()
        scorer = ConstantScorer(score=0.5)
        ranked, diag = score_and_rank_candidates(
            candidates=candidates, bundle_dir=bundle_dir,
            image_scorer=scorer, n_export=10,
        )
        assert diag.score_errors == 1
        assert diag.n_scored == 2
        assert len(ranked) == 2
