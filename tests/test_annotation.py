"""Tests for laionfashion.annotation — pairwise annotation framework."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laionfashion.annotation import (
    AnnotationPair,
    evaluate_axis_agreement,
    generate_annotation_pairs,
    load_annotations,
    pairs_to_dataframe,
    save_annotation_tasks,
)


class TestGenerateAnnotationPairs:
    def test_correct_count(self) -> None:
        pairs = generate_annotation_pairs(n_images=50, axes=["formal", "casual"], pairs_per_axis=20)
        formal_pairs = [p for p in pairs if p.axis == "formal"]
        casual_pairs = [p for p in pairs if p.axis == "casual"]
        assert len(formal_pairs) == 20
        assert len(casual_pairs) == 20

    def test_no_self_pairs(self) -> None:
        pairs = generate_annotation_pairs(n_images=20, axes=["test"], pairs_per_axis=50)
        for p in pairs:
            assert p.image_a_row_id != p.image_b_row_id

    def test_unique_pairs(self) -> None:
        pairs = generate_annotation_pairs(n_images=30, axes=["test"], pairs_per_axis=30)
        keys = {(min(p.image_a_row_id, p.image_b_row_id),
                 max(p.image_a_row_id, p.image_b_row_id), p.axis) for p in pairs}
        assert len(keys) == len(pairs)

    def test_deterministic(self) -> None:
        p1 = generate_annotation_pairs(n_images=20, axes=["a"], pairs_per_axis=10, seed=42)
        p2 = generate_annotation_pairs(n_images=20, axes=["a"], pairs_per_axis=10, seed=42)
        assert [(p.image_a_row_id, p.image_b_row_id) for p in p1] == \
               [(p.image_a_row_id, p.image_b_row_id) for p in p2]

    def test_valid_row_ids(self) -> None:
        pairs = generate_annotation_pairs(n_images=10, axes=["test"], pairs_per_axis=20)
        for p in pairs:
            assert 0 <= p.image_a_row_id < 10
            assert 0 <= p.image_b_row_id < 10


class TestPairsToDataframe:
    def test_correct_columns(self) -> None:
        pairs = generate_annotation_pairs(n_images=10, axes=["test"], pairs_per_axis=5)
        df = pairs_to_dataframe(pairs)
        assert set(df.columns) == {"pair_id", "image_a_row_id", "image_b_row_id", "axis", "choice", "annotator"}

    def test_choice_empty(self) -> None:
        pairs = generate_annotation_pairs(n_images=10, axes=["test"], pairs_per_axis=5)
        df = pairs_to_dataframe(pairs)
        assert all(df["choice"] == "")


class TestSaveLoadAnnotations:
    def test_roundtrip(self, tmp_path: Path) -> None:
        pairs = generate_annotation_pairs(n_images=10, axes=["formal", "casual"], pairs_per_axis=5)
        df = pairs_to_dataframe(pairs)
        path = tmp_path / "tasks.csv"
        save_annotation_tasks(df, path)
        loaded = load_annotations(path)
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)


class TestEvaluateAxisAgreement:
    def test_perfect_agreement(self) -> None:
        """When human choices match model scores, accuracy should be 1.0."""
        annotations = pd.DataFrame([
            {"pair_id": 0, "image_a_row_id": 0, "image_b_row_id": 1, "axis": "formal", "choice": "a"},
            {"pair_id": 1, "image_a_row_id": 2, "image_b_row_id": 3, "axis": "formal", "choice": "b"},
        ])
        axis_scores = pd.DataFrame({
            "row_id": [0, 1, 2, 3],
            "formal": [0.8, 0.2, 0.1, 0.9],  # 0 > 1, 3 > 2
        })
        result = evaluate_axis_agreement(annotations, axis_scores, "formal")
        assert result["accuracy"] == 1.0
        assert result["n_evaluated"] == 2

    def test_zero_agreement(self) -> None:
        """When human choices contradict model scores, accuracy should be 0.0."""
        annotations = pd.DataFrame([
            {"pair_id": 0, "image_a_row_id": 0, "image_b_row_id": 1, "axis": "formal", "choice": "b"},
        ])
        axis_scores = pd.DataFrame({
            "row_id": [0, 1],
            "formal": [0.9, 0.1],  # model says a > b, human says b
        })
        result = evaluate_axis_agreement(annotations, axis_scores, "formal")
        assert result["accuracy"] == 0.0

    def test_ignores_ties(self) -> None:
        annotations = pd.DataFrame([
            {"pair_id": 0, "image_a_row_id": 0, "image_b_row_id": 1, "axis": "formal", "choice": "tie"},
        ])
        axis_scores = pd.DataFrame({"row_id": [0, 1], "formal": [0.5, 0.5]})
        result = evaluate_axis_agreement(annotations, axis_scores, "formal")
        assert result["n_evaluated"] == 0

    def test_no_annotations(self) -> None:
        annotations = pd.DataFrame(columns=["pair_id", "image_a_row_id", "image_b_row_id", "axis", "choice"])
        axis_scores = pd.DataFrame({"row_id": [0, 1], "formal": [0.5, 0.5]})
        result = evaluate_axis_agreement(annotations, axis_scores, "formal")
        assert result["n_evaluated"] == 0
