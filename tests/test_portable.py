"""Tests for bundle portability — validation, packing, and cross-machine loading."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laionfashion.portable import (
    ValidationResult,
    list_bundle_artifacts,
    validate_portable_bundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_full_bundle(tmp_path: Path, n: int = 5) -> Path:
    """Create a complete portable bundle with all artifact types."""
    bundle = tmp_path / "test_bundle"
    bundle.mkdir(parents=True)

    thumb_dir = bundle / "thumbnails"
    thumb_dir.mkdir()
    det_dir = bundle / "detection_images"
    det_dir.mkdir()
    crop_dir = bundle / "garment_crops"
    crop_dir.mkdir()

    # Records with thumbnails and detection images
    records = []
    for i in range(n):
        name = f"{i:06d}_{i}.jpg"
        Image.new("RGB", (80, 120)).save(thumb_dir / name, quality=90)
        Image.new("RGB", (200, 300)).save(det_dir / name, quality=90)
        records.append({
            "row_id": i,
            "caption": f"person wearing outfit {i}",
            "thumbnail_path": f"thumbnails/{name}",
            "detection_image_path": f"detection_images/{name}",
        })
    pd.DataFrame(records).to_parquet(bundle / "records.parquet", index=False)

    # Embeddings
    emb = np.random.default_rng(0).standard_normal((n, 16)).astype(np.float32)
    np.save(bundle / "embeddings.npy", emb)

    # Garments
    garment_rows = []
    for i in range(n):
        for cat in ["top", "bottom"]:
            gid = i * 2 + (0 if cat == "top" else 1)
            crop_name = f"{gid:06d}_{i}_{cat}.jpg"
            Image.new("RGB", (60, 40)).save(crop_dir / crop_name, quality=90)
            garment_rows.append({
                "outfit_id": i,
                "garment_id": gid,
                "category": cat,
                "crop_path": f"garment_crops/{crop_name}",
                "source_image_path": f"detection_images/{i:06d}_{i}.jpg",
            })
    pd.DataFrame(garment_rows).to_parquet(bundle / "garments.parquet", index=False)

    # Garment embeddings
    gemb = np.random.default_rng(1).standard_normal((n * 2, 16)).astype(np.float32)
    np.save(bundle / "garment_embeddings.npy", gemb)

    # Projection
    proj = pd.DataFrame({"row_id": range(n), "x": np.zeros(n), "y": np.zeros(n)})
    proj.to_parquet(bundle / "projection.parquet", index=False)

    # Manifest (informational)
    import json
    with (bundle / "manifest.json").open("w") as f:
        json.dump({"n_exported": n}, f)

    return bundle


def _make_minimal_bundle(tmp_path: Path, n: int = 3) -> Path:
    """Create the minimum viable bundle (records + embeddings + thumbnails)."""
    bundle = tmp_path / "minimal"
    bundle.mkdir()
    thumb_dir = bundle / "thumbnails"
    thumb_dir.mkdir()

    records = []
    for i in range(n):
        name = f"{i:06d}_{i}.jpg"
        Image.new("RGB", (80, 120)).save(thumb_dir / name, quality=90)
        records.append({
            "row_id": i,
            "caption": f"outfit {i}",
            "thumbnail_path": f"thumbnails/{name}",
        })
    pd.DataFrame(records).to_parquet(bundle / "records.parquet", index=False)
    np.save(bundle / "embeddings.npy", np.zeros((n, 8), dtype=np.float32))
    return bundle


# ---------------------------------------------------------------------------
# validate_portable_bundle
# ---------------------------------------------------------------------------


class TestValidatePortableBundle:
    def test_full_bundle_passes(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = validate_portable_bundle(bundle)
        assert result.ok, f"Errors: {result.errors}"

    def test_minimal_bundle_passes(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        result = validate_portable_bundle(bundle)
        assert result.ok, f"Errors: {result.errors}"
        # Should have warnings about missing optional artifacts
        assert len(result.warnings) > 0

    def test_missing_records_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "records.parquet").unlink()
        result = validate_portable_bundle(bundle)
        assert not result.ok
        assert any("records" in e for e in result.errors)

    def test_missing_embeddings_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        (bundle / "embeddings.npy").unlink()
        result = validate_portable_bundle(bundle)
        assert not result.ok

    def test_missing_thumbnails_dir_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        shutil.rmtree(bundle / "thumbnails")
        result = validate_portable_bundle(bundle)
        assert not result.ok

    def test_row_count_mismatch_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path, n=3)
        np.save(bundle / "embeddings.npy", np.zeros((5, 8), dtype=np.float32))
        result = validate_portable_bundle(bundle)
        assert not result.ok
        assert any("mismatch" in e for e in result.errors)

    def test_absolute_path_in_parquet_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        records = pd.read_parquet(bundle / "records.parquet")
        records.at[0, "thumbnail_path"] = "/absolute/path/to/thumb.jpg"
        records.to_parquet(bundle / "records.parquet", index=False)
        result = validate_portable_bundle(bundle)
        assert not result.ok
        assert any("Absolute path" in e for e in result.errors)

    def test_missing_referenced_file_fails(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        # Delete a thumbnail that's referenced in records
        thumbs = list((bundle / "thumbnails").iterdir())
        thumbs[0].unlink()
        result = validate_portable_bundle(bundle)
        assert not result.ok
        assert any("Missing file" in e for e in result.errors)

    def test_garment_embedding_mismatch_fails(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        # Overwrite with wrong row count
        np.save(bundle / "garment_embeddings.npy", np.zeros((999, 16), dtype=np.float32))
        result = validate_portable_bundle(bundle)
        assert not result.ok

    def test_artifacts_dict_populated(self, tmp_path: Path) -> None:
        bundle = _make_full_bundle(tmp_path)
        result = validate_portable_bundle(bundle)
        assert result.artifacts["records.parquet"] is True
        assert result.artifacts["embeddings.npy"] is True
        assert result.artifacts["garments.parquet"] is True

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        result = validate_portable_bundle(tmp_path / "nope")
        assert not result.ok


# ---------------------------------------------------------------------------
# Portability: copy bundle to a new location
# ---------------------------------------------------------------------------


class TestBundlePortability:
    def test_bundle_works_after_copy(self, tmp_path: Path) -> None:
        """Bundle should be loadable after copying to a different directory."""
        from laionfashion.bundle import load_bundle

        src = _make_full_bundle(tmp_path / "src")
        dst = tmp_path / "dst" / "copied_bundle"
        shutil.copytree(src, dst)

        bundle = load_bundle(dst)
        assert bundle.n_images == 5
        assert bundle.has_garments
        assert bundle.garment_embeddings is not None

        # Thumbnails resolve
        thumb = bundle.thumbnail_path(0)
        assert thumb is not None
        assert thumb.exists()

        # Garment crops resolve
        crop = bundle.garment_crop_path(0)
        assert crop is not None
        assert crop.exists()

    def test_validation_works_after_copy(self, tmp_path: Path) -> None:
        src = _make_full_bundle(tmp_path / "src")
        dst = tmp_path / "dst" / "moved"
        shutil.copytree(src, dst)

        result = validate_portable_bundle(dst)
        assert result.ok, f"Errors: {result.errors}"

    def test_retrieval_works_after_copy(self, tmp_path: Path) -> None:
        """Cross-category retrieval should work on a copied bundle."""
        from laionfashion.bundle import load_bundle
        from laionfashion.retrieval import retrieve_similar_garments

        src = _make_full_bundle(tmp_path / "src")
        dst = tmp_path / "copied"
        shutil.copytree(src, dst)

        bundle = load_bundle(dst)
        results = retrieve_similar_garments(
            query_garment_id=0,
            garments=bundle.garments,
            embeddings=bundle.garment_embeddings,
            target_category="bottom",
            k=3,
        )
        assert len(results) > 0

    def test_evaluation_works_after_copy(self, tmp_path: Path) -> None:
        """Evaluation harness should work on a copied bundle."""
        from laionfashion.bundle import load_bundle
        from laionfashion.evaluation import build_eval_queries, evaluate_retrieval

        src = _make_full_bundle(tmp_path / "src")
        dst = tmp_path / "eval_copy"
        shutil.copytree(src, dst)

        bundle = load_bundle(dst)
        queries = build_eval_queries(bundle.garments)
        assert len(queries) > 0
        metrics = evaluate_retrieval(queries, bundle.garments, bundle.garment_embeddings)
        assert metrics.n_queries > 0


# ---------------------------------------------------------------------------
# list_bundle_artifacts
# ---------------------------------------------------------------------------


class TestListBundleArtifacts:
    def test_lists_all_files(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        artifacts = list_bundle_artifacts(bundle)
        assert "records.parquet" in artifacts
        assert "embeddings.npy" in artifacts
        assert any("thumbnails/" in k for k in artifacts)

    def test_sizes_are_positive(self, tmp_path: Path) -> None:
        bundle = _make_minimal_bundle(tmp_path)
        artifacts = list_bundle_artifacts(bundle)
        assert all(size > 0 for size in artifacts.values())
