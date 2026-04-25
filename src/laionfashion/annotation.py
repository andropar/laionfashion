"""Pairwise annotation framework for style axis calibration.

Creates annotation tasks where a human compares two outfits on a style
dimension (e.g. "which is more formal?").  Results can be used to evaluate
and calibrate prompt-direction axes.

The annotation format is a simple CSV/parquet with:
    pair_id, image_a_row_id, image_b_row_id, axis, choice (a/b/tie), annotator
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnnotationPair:
    """A pair of images to compare on a given axis."""

    pair_id: int
    image_a_row_id: int
    image_b_row_id: int
    axis: str


def generate_annotation_pairs(
    n_images: int,
    axes: list[str],
    pairs_per_axis: int = 100,
    seed: int = 42,
) -> list[AnnotationPair]:
    """Generate random pairs for pairwise annotation.

    Parameters
    ----------
    n_images:
        Total number of images in the bundle.
    axes:
        Style axis names to generate pairs for.
    pairs_per_axis:
        Number of pairs per axis.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    pair_id = 0

    for axis in axes:
        seen = set()
        attempts = 0
        while len([p for p in pairs if p.axis == axis]) < pairs_per_axis and attempts < pairs_per_axis * 10:
            a = rng.integers(0, n_images)
            b = rng.integers(0, n_images)
            if a == b:
                attempts += 1
                continue
            key = (min(a, b), max(a, b), axis)
            if key in seen:
                attempts += 1
                continue
            seen.add(key)
            pairs.append(AnnotationPair(
                pair_id=pair_id,
                image_a_row_id=int(a),
                image_b_row_id=int(b),
                axis=axis,
            ))
            pair_id += 1
            attempts += 1

    return pairs


def pairs_to_dataframe(pairs: list[AnnotationPair]) -> pd.DataFrame:
    """Convert annotation pairs to a DataFrame."""
    return pd.DataFrame([
        {
            "pair_id": p.pair_id,
            "image_a_row_id": p.image_a_row_id,
            "image_b_row_id": p.image_b_row_id,
            "axis": p.axis,
            "choice": "",  # to be filled by annotator
            "annotator": "",
        }
        for p in pairs
    ])


def save_annotation_tasks(pairs: pd.DataFrame, path: Path) -> None:
    """Save annotation pairs to CSV (human-editable)."""
    pairs.to_csv(path, index=False)


def load_annotations(path: Path) -> pd.DataFrame:
    """Load completed annotations from CSV."""
    return pd.read_csv(path)


def evaluate_axis_agreement(
    annotations: pd.DataFrame,
    axis_scores: pd.DataFrame,
    axis: str,
) -> dict[str, float]:
    """Evaluate how well model axis scores agree with human pairwise judgments.

    For each annotated pair where choice is 'a' or 'b', check whether the
    model's axis score agrees (i.e., the chosen image has a higher score on
    that axis).

    Returns dict with accuracy, n_evaluated, n_agree, n_disagree.
    """
    ax_annotations = annotations[
        (annotations["axis"] == axis) &
        (annotations["choice"].isin(["a", "b"]))
    ]

    if ax_annotations.empty:
        return {"accuracy": 0.0, "n_evaluated": 0, "n_agree": 0, "n_disagree": 0}

    n_agree = 0
    n_disagree = 0

    for _, row in ax_annotations.iterrows():
        a_id = int(row["image_a_row_id"])
        b_id = int(row["image_b_row_id"])
        choice = row["choice"]

        a_score = axis_scores.loc[axis_scores["row_id"] == a_id, axis]
        b_score = axis_scores.loc[axis_scores["row_id"] == b_id, axis]

        if a_score.empty or b_score.empty:
            continue

        a_val = float(a_score.iloc[0])
        b_val = float(b_score.iloc[0])

        if choice == "a" and a_val > b_val:
            n_agree += 1
        elif choice == "b" and b_val > a_val:
            n_agree += 1
        else:
            n_disagree += 1

    n_total = n_agree + n_disagree
    return {
        "accuracy": n_agree / n_total if n_total > 0 else 0.0,
        "n_evaluated": n_total,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
    }
