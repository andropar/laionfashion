"""Tests for FilterDiagnostics and review artifact output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from laionfashion.debug_export import FilterDiagnostics
from laionfashion.filtering import RejectReason


def _populated_diagnostics() -> FilterDiagnostics:
    diag = FilterDiagnostics()
    diag.scanned = 100
    diag.record_accept("woman wearing a dress", "wearing")
    diag.record_accept("street style outfit", "outfit")
    diag.record_reject(RejectReason.EXCLUDED, "conformal coating jacket", "conformal")
    diag.record_reject(RejectReason.EXCLUDED, "hospital gown patient", "hospital")
    diag.record_reject(RejectReason.NO_FASHION_SIGNAL, "a sunset over the lake")
    diag.record_reject(RejectReason.NO_FASHION_SIGNAL, "abstract pattern on wall")
    diag.record_reject(RejectReason.EMPTY, "")
    return diag


class TestFilterDiagnostics:
    def test_to_dict(self) -> None:
        diag = _populated_diagnostics()
        d = diag.to_dict()
        assert d["scanned"] == 100
        assert d["accepted"] == 2
        assert d["reject_counts"]["excluded_term"] == 2
        assert d["reject_counts"]["no_fashion_signal"] == 2
        assert d["reject_counts"]["empty_caption"] == 1
        assert 0 < d["accept_rate"] < 1

    def test_sample_cap(self) -> None:
        diag = FilterDiagnostics()
        for i in range(200):
            diag.record_accept(f"caption {i}")
            diag.record_reject(RejectReason.NO_FASHION_SIGNAL, f"rejected {i}")
        assert len(diag.accepted_samples) == 100
        assert len(diag.rejected_samples["no_fashion_signal"]) == 100


class TestReviewArtifacts:
    def test_writes_all_files(self, tmp_path: Path) -> None:
        diag = _populated_diagnostics()
        artifacts = diag.write_review_artifacts(tmp_path)
        assert "accepted_captions" in artifacts
        assert "rejected_captions" in artifacts
        assert "filter_summary" in artifacts
        assert (tmp_path / artifacts["accepted_captions"]).exists()
        assert (tmp_path / artifacts["rejected_captions"]).exists()
        assert (tmp_path / artifacts["filter_summary"]).exists()

    def test_accepted_csv_content(self, tmp_path: Path) -> None:
        diag = _populated_diagnostics()
        diag.write_review_artifacts(tmp_path)
        with (tmp_path / "accepted_captions.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["caption"] == "woman wearing a dress"
        assert rows[0]["matched_term"] == "wearing"

    def test_rejected_csv_content(self, tmp_path: Path) -> None:
        diag = _populated_diagnostics()
        diag.write_review_artifacts(tmp_path)
        with (tmp_path / "rejected_captions.csv").open() as f:
            rows = list(csv.DictReader(f))
        # 2 excluded + 2 no_signal + 1 empty = 5
        assert len(rows) == 5
        reasons = {r["reason"] for r in rows}
        assert "excluded_term" in reasons
        assert "no_fashion_signal" in reasons

    def test_filter_summary_json(self, tmp_path: Path) -> None:
        diag = _populated_diagnostics()
        diag.write_review_artifacts(tmp_path)
        with (tmp_path / "filter_summary.json").open() as f:
            summary = json.load(f)
        assert summary["scanned"] == 100
        assert summary["accepted"] == 2
        assert summary["accepted_sample_count"] == 2
        assert summary["rejected_sample_count"] == 5

    def test_empty_diagnostics(self, tmp_path: Path) -> None:
        diag = FilterDiagnostics()
        artifacts = diag.write_review_artifacts(tmp_path)
        # No accepted/rejected samples → those files not created
        assert "accepted_captions" not in artifacts
        assert "rejected_captions" not in artifacts
        # Summary always written
        assert "filter_summary" in artifacts
