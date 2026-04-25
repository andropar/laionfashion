"""Tests for laionfashion.axes – load, save, validate, and demo axis builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laionfashion.axes import (
    axis_names,
    build_demo_axes,
    load_axis_scores,
    save_axis_scores,
    top_bottom_indices,
    validate_axis_scores,
)


def _sample_scores(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(n),
            "colorful_proxy": np.linspace(-1, 1, n).astype(np.float32),
            "formal_proxy": np.linspace(1, -1, n).astype(np.float32),
        }
    )


def _random_embeddings(n: int, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb


def _sample_records(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(n),
            "caption": [f"person wearing outfit {i}" for i in range(n)],
            "thumbnail_path": [f"thumbnails/{i:06d}.jpg" for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


class TestLoadSave:
    def test_save_and_load_parquet(self, tmp_path: Path) -> None:
        scores = _sample_scores()
        save_axis_scores(scores, tmp_path)
        loaded = load_axis_scores(tmp_path)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, scores)

    def test_save_and_load_csv_fallback(self, tmp_path: Path) -> None:
        scores = _sample_scores()
        # Write as CSV directly to test CSV loading
        scores.to_csv(tmp_path / "axis_scores.csv", index=False)
        loaded = load_axis_scores(tmp_path)
        assert loaded is not None
        assert list(loaded.columns) == ["row_id", "colorful_proxy", "formal_proxy"]

    def test_load_returns_none_if_missing(self, tmp_path: Path) -> None:
        assert load_axis_scores(tmp_path) is None


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_valid_scores(self) -> None:
        validate_axis_scores(_sample_scores(5), n_images=5)

    def test_missing_row_id(self) -> None:
        df = pd.DataFrame({"score": [0.5]})
        with pytest.raises(ValueError, match="row_id"):
            validate_axis_scores(df, n_images=1)

    def test_no_score_columns(self) -> None:
        df = pd.DataFrame({"row_id": [0, 1]})
        with pytest.raises(ValueError, match="at least one score column"):
            validate_axis_scores(df, n_images=2)

    def test_row_count_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Row count mismatch"):
            validate_axis_scores(_sample_scores(5), n_images=10)

    def test_row_id_mismatch(self) -> None:
        df = pd.DataFrame({"row_id": [0, 5], "score": [0.1, 0.2]})
        with pytest.raises(ValueError, match="row_id mismatch"):
            validate_axis_scores(df, n_images=2)


# ---------------------------------------------------------------------------
# axis_names / top_bottom_indices
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_axis_names(self) -> None:
        names = axis_names(_sample_scores())
        assert names == ["colorful_proxy", "formal_proxy"]

    def test_top_bottom_indices(self) -> None:
        scores = _sample_scores(10)
        top, bottom = top_bottom_indices(scores, "colorful_proxy", n=3)
        assert len(top) == 3
        assert len(bottom) == 3
        # Highest colorful_proxy scores are at the end (linspace -1..1)
        assert top[0] == 9
        # Lowest at the start
        assert bottom[0] == 0

    def test_top_bottom_clamps_to_available(self) -> None:
        scores = _sample_scores(3)
        top, bottom = top_bottom_indices(scores, "colorful_proxy", n=10)
        assert len(top) == 3
        assert len(bottom) == 3


# ---------------------------------------------------------------------------
# build_demo_axes
# ---------------------------------------------------------------------------


class TestBuildDemoAxes:
    def test_output_shape(self) -> None:
        emb = _random_embeddings(20)
        records = _sample_records(20)
        scores = build_demo_axes(emb, records)
        assert len(scores) == 20
        assert "row_id" in scores.columns
        axes = axis_names(scores)
        assert len(axes) == 4
        assert "colorful_proxy" in axes
        assert "formal_proxy" in axes
        assert "minimal_proxy" in axes
        assert "outdoor_proxy" in axes

    def test_scores_in_range(self) -> None:
        emb = _random_embeddings(30)
        records = _sample_records(30)
        scores = build_demo_axes(emb, records)
        for ax in axis_names(scores):
            assert scores[ax].min() >= -1.0 - 1e-6
            assert scores[ax].max() <= 1.0 + 1e-6

    def test_deterministic(self) -> None:
        emb = _random_embeddings(15)
        records = _sample_records(15)
        s1 = build_demo_axes(emb, records, random_state=42)
        s2 = build_demo_axes(emb, records, random_state=42)
        pd.testing.assert_frame_equal(s1, s2)

    def test_validates_against_bundle(self) -> None:
        emb = _random_embeddings(10)
        records = _sample_records(10)
        scores = build_demo_axes(emb, records)
        validate_axis_scores(scores, n_images=10)

    def test_works_with_tiny_bundle(self) -> None:
        emb = _random_embeddings(2, d=8)
        records = _sample_records(2)
        scores = build_demo_axes(emb, records)
        assert len(scores) == 2
        validate_axis_scores(scores, n_images=2)

    def test_caption_keywords_boost_score(self) -> None:
        """Captions with colorful keywords should score higher on colorful_proxy."""
        emb = np.ones((4, 8), dtype=np.float32)  # identical embeddings
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        records = pd.DataFrame(
            {
                "row_id": range(4),
                "caption": [
                    "a vibrant colorful outfit",
                    "plain neutral clothes",
                    "bright neon jacket",
                    "simple dark suit",
                ],
                "thumbnail_path": [""] * 4,
            }
        )
        scores = build_demo_axes(emb, records)
        # With identical embeddings, caption signal dominates
        colorful = scores["colorful_proxy"].values
        # Indices 0 and 2 have colorful keywords, should score higher than 1 and 3
        assert colorful[0] > colorful[1]
        assert colorful[2] > colorful[3]

    def test_no_caption_column(self) -> None:
        """Should not crash if records have no caption column."""
        emb = _random_embeddings(5)
        records = pd.DataFrame({"row_id": range(5), "thumbnail_path": [""] * 5})
        scores = build_demo_axes(emb, records)
        assert len(scores) == 5
