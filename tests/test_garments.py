"""Tests for laionfashion.garments — region extraction, schema, bundle integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from laionfashion.garments import (
    CATEGORY_LOWER,
    CATEGORY_UPPER,
    GarmentRegion,
    extract_garments_from_bundle,
    extract_regions_v0,
    load_garments,
    save_garments,
    validate_garments,
    write_garment_crop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(tmp_path: Path, n: int = 5, img_size: tuple[int, int] = (80, 120)) -> Path:
    """Create a synthetic bundle with colored thumbnails."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()
    records = []
    for i in range(n):
        thumb_name = f"{i:06d}_{i}.jpg"
        img = Image.new("RGB", img_size, color=(i * 50, 100, 200))
        img.save(thumb_dir / thumb_name, quality=90)
        records.append({
            "row_id": i,
            "global_index": i * 100,
            "caption": f"person wearing outfit {i}",
            "thumbnail_path": f"thumbnails/{thumb_name}",
        })
    pd.DataFrame(records).to_parquet(tmp_path / "records.parquet", index=False)
    emb = np.random.default_rng(0).standard_normal((n, 16)).astype(np.float32)
    np.save(tmp_path / "embeddings.npy", emb)
    return tmp_path


# ---------------------------------------------------------------------------
# extract_regions_v0
# ---------------------------------------------------------------------------


class TestExtractRegionsV0:
    def test_produces_two_regions(self) -> None:
        img = Image.new("RGB", (80, 120))
        regions = extract_regions_v0(img)
        assert len(regions) == 2
        categories = {r.category for r in regions}
        assert categories == {CATEGORY_UPPER, CATEGORY_LOWER}

    def test_upper_region_covers_top(self) -> None:
        img = Image.new("RGB", (80, 120))
        regions = extract_regions_v0(img)
        upper = [r for r in regions if r.category == CATEGORY_UPPER][0]
        assert upper.bbox_y == 0
        assert upper.bbox_x == 0
        assert upper.bbox_w == 80
        assert upper.bbox_h > 0
        assert upper.bbox_h <= 120

    def test_lower_region_covers_bottom(self) -> None:
        img = Image.new("RGB", (80, 120))
        regions = extract_regions_v0(img)
        lower = [r for r in regions if r.category == CATEGORY_LOWER][0]
        assert lower.bbox_y > 0
        assert lower.bbox_y + lower.bbox_h == 120

    def test_regions_overlap_at_waist(self) -> None:
        """Upper and lower regions should overlap for waist coverage."""
        img = Image.new("RGB", (80, 120))
        regions = extract_regions_v0(img)
        upper = [r for r in regions if r.category == CATEGORY_UPPER][0]
        lower = [r for r in regions if r.category == CATEGORY_LOWER][0]
        assert upper.bbox_h > lower.bbox_y  # overlap exists

    def test_tiny_image_returns_empty(self) -> None:
        img = Image.new("RGB", (5, 5))
        assert extract_regions_v0(img) == []

    def test_wide_image(self) -> None:
        img = Image.new("RGB", (200, 50))
        regions = extract_regions_v0(img)
        assert len(regions) == 2


# ---------------------------------------------------------------------------
# write_garment_crop
# ---------------------------------------------------------------------------


class TestWriteGarmentCrop:
    def test_writes_crop_file(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        region = GarmentRegion("upper_body", 10, 20, 50, 30)
        out = tmp_path / "crop.jpg"
        write_garment_crop(img, region, out)
        assert out.exists()
        crop = Image.open(out)
        assert crop.size == (50, 30)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (50, 50))
        region = GarmentRegion("lower_body", 0, 0, 50, 50)
        out = tmp_path / "sub" / "dir" / "crop.jpg"
        write_garment_crop(img, region, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# extract_garments_from_bundle
# ---------------------------------------------------------------------------


class TestExtractGarmentsFromBundle:
    def test_produces_correct_schema(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        assert "outfit_id" in garments.columns
        assert "garment_id" in garments.columns
        assert "category" in garments.columns
        assert "bbox_x" in garments.columns
        assert "crop_path" in garments.columns
        assert "method" in garments.columns

    def test_two_garments_per_image(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=4)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        assert len(garments) == 8  # 4 images * 2 regions

    def test_garment_ids_unique(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=5)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        assert not garments["garment_id"].duplicated().any()

    def test_crop_files_exist(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        for _, row in garments.iterrows():
            crop = bundle_dir / row["crop_path"]
            assert crop.exists(), f"Missing crop: {crop}"

    def test_method_column(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=1)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        assert all(garments["method"] == "region_split_v0")

    def test_skips_missing_thumbnails(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        # Delete one thumbnail
        thumbs = list((bundle_dir / "thumbnails").iterdir())
        thumbs[0].unlink()
        garments = extract_garments_from_bundle(records, bundle_dir)
        assert len(garments) == 4  # 2 remaining images * 2 regions


# ---------------------------------------------------------------------------
# save / load / validate
# ---------------------------------------------------------------------------


class TestSaveLoadValidate:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        save_garments(garments, bundle_dir)
        loaded = load_garments(bundle_dir)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, garments)

    def test_load_returns_none_if_missing(self, tmp_path: Path) -> None:
        assert load_garments(tmp_path) is None

    def test_validate_passes_for_valid(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        validate_garments(garments, n_outfits=3)

    def test_validate_missing_columns(self) -> None:
        df = pd.DataFrame({"outfit_id": [0], "garment_id": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_garments(df, n_outfits=1)

    def test_validate_invalid_outfit_id(self) -> None:
        df = pd.DataFrame({
            "outfit_id": [99],
            "garment_id": [0],
            "category": ["upper_body"],
            "crop_path": ["x.jpg"],
        })
        with pytest.raises(ValueError, match="outfit_ids not in records"):
            validate_garments(df, n_outfits=5)

    def test_validate_duplicate_garment_id(self) -> None:
        df = pd.DataFrame({
            "outfit_id": [0, 0],
            "garment_id": [0, 0],
            "category": ["upper_body", "lower_body"],
            "crop_path": ["a.jpg", "b.jpg"],
        })
        with pytest.raises(ValueError, match="garment_id must be unique"):
            validate_garments(df, n_outfits=1)


# ---------------------------------------------------------------------------
# Bundle loader integration
# ---------------------------------------------------------------------------


class TestBundleWithGarments:
    def test_bundle_without_garments(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path)
        bundle = load_bundle(bundle_dir)
        assert not bundle.has_garments
        assert bundle.n_garments == 0

    def test_bundle_with_garments(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        assert bundle.has_garments
        assert bundle.n_garments == 6

    def test_garment_crop_path(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        path = bundle.garment_crop_path(0)
        assert path is not None
        assert path.exists()

    def test_garments_for_outfit(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(records, bundle_dir)
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        outfit_garments = bundle.garments_for_outfit(1)
        assert len(outfit_garments) == 2
        assert set(outfit_garments["category"]) == {"upper_body", "lower_body"}
