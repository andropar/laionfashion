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

from laionfashion.image_scoring import ConstantScorer, ThresholdMockScorer


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
