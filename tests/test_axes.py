"""Tests for laionfashion.axes – load, save, validate, demo axes, and CLIP axes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laionfashion.axes import (
    DEFAULT_PROMPT_AXES,
    PromptAxis,
    axis_names,
    build_clip_axes,
    build_demo_axes,
    compute_prompt_directions,
    load_axis_scores,
    save_axis_scores,
    score_embeddings_on_axes,
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


# ---------------------------------------------------------------------------
# Prompt-direction axes (CLIP axes with synthetic embeddings)
# ---------------------------------------------------------------------------


def _make_synthetic_text_embeddings(
    axes: list[PromptAxis], dim: int = 32
) -> dict[str, np.ndarray]:
    """Create synthetic text embeddings for testing (no CLIP needed)."""
    rng = np.random.default_rng(42)
    embeddings = {}
    for ax in axes:
        pos = rng.standard_normal(dim).astype(np.float32)
        pos /= np.linalg.norm(pos)
        neg = rng.standard_normal(dim).astype(np.float32)
        neg /= np.linalg.norm(neg)
        embeddings[ax.positive] = pos
        embeddings[ax.negative] = neg
    return embeddings


class TestPromptAxis:
    def test_dataclass_fields(self) -> None:
        ax = PromptAxis(name="test", positive="good", negative="bad")
        assert ax.name == "test"
        assert ax.positive == "good"
        assert ax.negative == "bad"

    def test_default_axes_exist(self) -> None:
        assert len(DEFAULT_PROMPT_AXES) >= 6
        names = [ax.name for ax in DEFAULT_PROMPT_AXES]
        assert "minimalist_vs_maximalist" in names
        assert "formal_vs_casual" in names
        assert "colorful_vs_neutral" in names


class TestComputePromptDirections:
    def test_direction_is_normalized(self) -> None:
        axes = [PromptAxis("test", "pos prompt", "neg prompt")]
        text_emb = {
            "pos prompt": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "neg prompt": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        directions = compute_prompt_directions(axes, text_emb)
        d = directions["test"]
        assert abs(np.linalg.norm(d) - 1.0) < 1e-5

    def test_direction_points_toward_positive(self) -> None:
        axes = [PromptAxis("test", "pos prompt", "neg prompt")]
        pos = np.array([1.0, 0.0], dtype=np.float32)
        neg = np.array([-1.0, 0.0], dtype=np.float32)
        text_emb = {"pos prompt": pos, "neg prompt": neg}
        directions = compute_prompt_directions(axes, text_emb)
        # Direction should point toward positive (x > 0)
        assert directions["test"][0] > 0

    def test_multiple_axes(self) -> None:
        axes = [
            PromptAxis("a", "pos_a", "neg_a"),
            PromptAxis("b", "pos_b", "neg_b"),
        ]
        text_emb = _make_synthetic_text_embeddings(axes, dim=16)
        directions = compute_prompt_directions(axes, text_emb)
        assert "a" in directions
        assert "b" in directions
        assert directions["a"].shape == (16,)


class TestScoreEmbeddingsOnAxes:
    def test_output_shape(self) -> None:
        emb = _random_embeddings(10, d=8)
        directions = {
            "axis_a": np.random.default_rng(0).standard_normal(8).astype(np.float32),
            "axis_b": np.random.default_rng(1).standard_normal(8).astype(np.float32),
        }
        scores = score_embeddings_on_axes(emb, directions)
        assert len(scores) == 10
        assert list(scores.columns) == ["row_id", "axis_a", "axis_b"]

    def test_aligned_image_scores_high(self) -> None:
        """An image aligned with the direction should score higher."""
        direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        emb = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],   # aligned with direction
                [-1.0, 0.0, 0.0, 0.0],  # opposite to direction
                [0.0, 1.0, 0.0, 0.0],   # orthogonal
            ],
            dtype=np.float32,
        )
        scores = score_embeddings_on_axes(emb, {"test_axis": direction})
        vals = scores["test_axis"].values
        assert vals[0] > vals[2] > vals[1]  # aligned > orthogonal > opposite


class TestBuildClipAxes:
    def test_full_pipeline_with_synthetic(self) -> None:
        """End-to-end with synthetic embeddings (no CLIP)."""
        axes = list(DEFAULT_PROMPT_AXES[:3])
        text_emb = _make_synthetic_text_embeddings(axes, dim=32)
        image_emb = _random_embeddings(20, d=32)

        scores = build_clip_axes(
            image_embeddings=image_emb,
            text_embeddings=text_emb,
            axes=axes,
        )
        assert len(scores) == 20
        assert "row_id" in scores.columns
        names = axis_names(scores)
        assert len(names) == 3
        assert names[0] == axes[0].name

    def test_validates_successfully(self) -> None:
        axes = list(DEFAULT_PROMPT_AXES[:2])
        text_emb = _make_synthetic_text_embeddings(axes, dim=16)
        image_emb = _random_embeddings(10, d=16)
        scores = build_clip_axes(
            image_embeddings=image_emb,
            text_embeddings=text_emb,
            axes=axes,
        )
        validate_axis_scores(scores, n_images=10)

    def test_default_axes_used_when_none(self) -> None:
        text_emb = _make_synthetic_text_embeddings(list(DEFAULT_PROMPT_AXES), dim=32)
        image_emb = _random_embeddings(5, d=32)
        scores = build_clip_axes(
            image_embeddings=image_emb,
            text_embeddings=text_emb,
        )
        assert len(axis_names(scores)) == len(DEFAULT_PROMPT_AXES)

    def test_scores_are_deterministic(self) -> None:
        axes = list(DEFAULT_PROMPT_AXES[:2])
        text_emb = _make_synthetic_text_embeddings(axes, dim=16)
        image_emb = _random_embeddings(8, d=16)
        s1 = build_clip_axes(image_embeddings=image_emb, text_embeddings=text_emb, axes=axes)
        s2 = build_clip_axes(image_embeddings=image_emb, text_embeddings=text_emb, axes=axes)
        pd.testing.assert_frame_equal(s1, s2)

    def test_save_and_reload(self, tmp_path: Path) -> None:
        axes = list(DEFAULT_PROMPT_AXES[:2])
        text_emb = _make_synthetic_text_embeddings(axes, dim=16)
        image_emb = _random_embeddings(5, d=16)
        scores = build_clip_axes(
            image_embeddings=image_emb,
            text_embeddings=text_emb,
            axes=axes,
        )
        save_axis_scores(scores, tmp_path)
        loaded = load_axis_scores(tmp_path)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, scores)
