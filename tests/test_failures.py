"""Tests for laionfashion.failures — failure case detection and reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laionfashion.bundle import DebugBundle
from laionfashion.failures import (
    FailureCase,
    detect_common_failures,
    format_failure_report,
)


def _make_bundle(
    n: int = 20,
    dim: int = 32,
    *,
    garments: pd.DataFrame | None = None,
    extra_record_cols: dict | None = None,
    seed: int = 42,
) -> DebugBundle:
    """Create a synthetic in-memory DebugBundle for testing."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    records_data: dict = {
        "row_id": np.arange(n),
        "caption": [f"outfit {i}" for i in range(n)],
        "thumbnail_path": [f"thumbnails/{i:06d}.jpg" for i in range(n)],
    }
    if extra_record_cols:
        records_data.update(extra_record_cols)

    records = pd.DataFrame(records_data)

    return DebugBundle(
        records=records,
        embeddings=emb,
        bundle_dir=Path("/tmp/fake_bundle"),
        garments=garments,
    )


# ---------------------------------------------------------------------------
# No garments detection
# ---------------------------------------------------------------------------


class TestNoGarments:
    def test_detects_missing_garments(self) -> None:
        """Images with no matching garment rows should be flagged."""
        n = 10
        # Only provide garments for images 0-6, leaving 7-9 unmatched
        garments = pd.DataFrame({
            "outfit_id": list(range(7)),
            "garment_id": list(range(7)),
            "category": ["top"] * 7,
        })
        bundle = _make_bundle(n=n, garments=garments)
        failures = detect_common_failures(bundle)

        no_garment_failures = [f for f in failures if f.category == "no_garments"]
        assert len(no_garment_failures) == 1
        fc = no_garment_failures[0]
        assert 7 in fc.example_row_ids
        assert 8 in fc.example_row_ids
        assert 9 in fc.example_row_ids

    def test_no_garments_data_skips_check(self) -> None:
        """When no garments parquet exists, the check is skipped."""
        bundle = _make_bundle(n=10, garments=None)
        failures = detect_common_failures(bundle)
        no_garment_failures = [f for f in failures if f.category == "no_garments"]
        assert len(no_garment_failures) == 0

    def test_all_images_have_garments(self) -> None:
        """No failure when every image has at least one garment."""
        n = 5
        garments = pd.DataFrame({
            "outfit_id": list(range(n)),
            "garment_id": list(range(n)),
            "category": ["top"] * n,
        })
        bundle = _make_bundle(n=n, garments=garments)
        failures = detect_common_failures(bundle)
        no_garment_failures = [f for f in failures if f.category == "no_garments"]
        assert len(no_garment_failures) == 0


# ---------------------------------------------------------------------------
# Low outfit scores
# ---------------------------------------------------------------------------


class TestLowOutfitScores:
    def test_detects_low_scores(self) -> None:
        n = 20
        scores = np.full(n, 0.8)
        # Make one clearly low
        scores[3] = 0.001
        bundle = _make_bundle(
            n=n,
            extra_record_cols={"image_outfit_score": scores},
        )
        failures = detect_common_failures(bundle)
        low_score_failures = [f for f in failures if f.category == "low_outfit_score"]
        assert len(low_score_failures) == 1
        fc = low_score_failures[0]
        assert 3 in fc.example_row_ids

    def test_no_score_column_skips_check(self) -> None:
        bundle = _make_bundle(n=10)
        failures = detect_common_failures(bundle)
        low_score_failures = [f for f in failures if f.category == "low_outfit_score"]
        assert len(low_score_failures) == 0


# ---------------------------------------------------------------------------
# format_failure_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_empty_failures(self) -> None:
        report = format_failure_report([])
        assert "No issues detected" in report

    def test_nonempty_report(self) -> None:
        failures = [
            FailureCase(
                category="test_issue",
                description="Something went wrong.",
                example_row_ids=[1, 2, 3],
                severity="warning",
            )
        ]
        report = format_failure_report(failures)
        assert "test_issue" in report
        assert "[WARNING]" in report
        assert "1, 2, 3" in report

    def test_severity_ordering(self) -> None:
        failures = [
            FailureCase(category="info_thing", description="Info.", severity="info"),
            FailureCase(category="error_thing", description="Error.", severity="error"),
            FailureCase(category="warn_thing", description="Warn.", severity="warning"),
        ]
        report = format_failure_report(failures)
        # Error should come before warning, which comes before info
        error_pos = report.index("[ERROR]")
        warn_pos = report.index("[WARNING]")
        info_pos = report.index("[INFO]")
        assert error_pos < warn_pos < info_pos
