"""Tests for laionfashion.garments — detection, schema, bundle integration."""

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
    MockDetector,
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
# GarmentRegion
# ---------------------------------------------------------------------------


class TestGarmentRegion:
    def test_fields(self) -> None:
        r = GarmentRegion("top", 10, 20, 50, 60, confidence=0.95)
        assert r.category == "top"
        assert r.bbox_x == 10
        assert r.confidence == 0.95

    def test_default_confidence_is_nan(self) -> None:
        r = GarmentRegion("bottom", 0, 0, 10, 10)
        assert np.isnan(r.confidence)


# ---------------------------------------------------------------------------
# extract_regions_v0 (fallback)
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
        assert upper.bbox_w == 80
        assert upper.bbox_h > 0

    def test_tiny_image_returns_empty(self) -> None:
        img = Image.new("RGB", (5, 5))
        assert extract_regions_v0(img) == []


# ---------------------------------------------------------------------------
# MockDetector
# ---------------------------------------------------------------------------


class TestMockDetector:
    def test_default_regions(self) -> None:
        detector = MockDetector()
        img = Image.new("RGB", (80, 120))
        regions = detector.detect(img)
        assert len(regions) == 2
        assert regions[0].category == "top"
        assert regions[1].category == "bottom"

    def test_custom_regions(self) -> None:
        custom = [GarmentRegion("shoes", 10, 90, 30, 20, confidence=0.8)]
        detector = MockDetector(regions=custom)
        regions = detector.detect(Image.new("RGB", (80, 120)))
        assert len(regions) == 1
        assert regions[0].category == "shoes"


# ---------------------------------------------------------------------------
# write_garment_crop
# ---------------------------------------------------------------------------


class TestWriteGarmentCrop:
    def test_writes_crop_file(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        region = GarmentRegion("top", 10, 20, 50, 30)
        out = tmp_path / "crop.jpg"
        write_garment_crop(img, region, out)
        assert out.exists()
        crop = Image.open(out)
        assert crop.size == (50, 30)

    def test_clamps_to_image_bounds(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (50, 50))
        region = GarmentRegion("top", -5, -5, 60, 60)
        out = tmp_path / "crop.jpg"
        write_garment_crop(img, region, out)
        crop = Image.open(out)
        assert crop.size == (50, 50)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (50, 50))
        region = GarmentRegion("bottom", 0, 0, 50, 50)
        out = tmp_path / "sub" / "dir" / "crop.jpg"
        write_garment_crop(img, region, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# extract_garments_from_bundle (with MockDetector)
# ---------------------------------------------------------------------------


class TestExtractGarmentsFromBundle:
    def test_correct_schema_with_mock(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        required = {"outfit_id", "garment_id", "category", "bbox_x", "bbox_y",
                     "bbox_w", "bbox_h", "confidence", "crop_path", "method"}
        assert required.issubset(set(garments.columns))

    def test_two_garments_per_image_with_mock(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=4)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        assert len(garments) == 8  # 4 images * 2 regions

    def test_garment_ids_unique(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=5)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        assert not garments["garment_id"].duplicated().any()

    def test_crop_files_exist(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        for _, row in garments.iterrows():
            crop = bundle_dir / row["crop_path"]
            assert crop.exists(), f"Missing crop: {crop}"

    def test_confidence_stored(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        assert "confidence" in garments.columns
        assert all(garments["confidence"].notna())
        assert all(garments["confidence"] > 0)

    def test_method_column_detr(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=1)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        assert all(garments["method"] == "detr")

    def test_method_column_v0(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=1)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="region_split_v0"
        )
        assert all(garments["method"] == "region_split_v0")

    def test_skips_missing_thumbnails(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        thumbs = list((bundle_dir / "thumbnails").iterdir())
        thumbs[0].unlink()
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        assert len(garments) == 4  # 2 remaining images * 2 regions

    def test_custom_categories(self, tmp_path: Path) -> None:
        """MockDetector with shoes-only detection."""
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        detector = MockDetector([
            GarmentRegion("shoes", 10, 90, 60, 30, confidence=0.85),
            GarmentRegion("outer", 0, 0, 80, 80, confidence=0.70),
            GarmentRegion("hat", 20, 0, 40, 20, confidence=0.60),
        ])
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=detector
        )
        assert len(garments) == 6  # 2 images * 3 regions
        assert set(garments["category"]) == {"shoes", "outer", "hat"}


# ---------------------------------------------------------------------------
# save / load / validate
# ---------------------------------------------------------------------------


class TestSaveLoadValidate:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        save_garments(garments, bundle_dir)
        loaded = load_garments(bundle_dir)
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, garments)

    def test_load_returns_none_if_missing(self, tmp_path: Path) -> None:
        assert load_garments(tmp_path) is None

    def test_validate_passes_for_valid(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        validate_garments(garments, n_outfits=3)

    def test_validate_missing_columns(self) -> None:
        df = pd.DataFrame({"outfit_id": [0], "garment_id": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_garments(df, n_outfits=1)

    def test_validate_invalid_outfit_id(self) -> None:
        df = pd.DataFrame({
            "outfit_id": [99],
            "garment_id": [0],
            "category": ["top"],
            "crop_path": ["x.jpg"],
        })
        with pytest.raises(ValueError, match="outfit_ids not in records"):
            validate_garments(df, n_outfits=5)

    def test_validate_duplicate_garment_id(self) -> None:
        df = pd.DataFrame({
            "outfit_id": [0, 0],
            "garment_id": [0, 0],
            "category": ["top", "bottom"],
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
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        assert bundle.has_garments
        assert bundle.n_garments == 6

    def test_garment_crop_path(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path, n=2)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        path = bundle.garment_crop_path(0)
        assert path is not None
        assert path.exists()

    def test_garments_for_outfit(self, tmp_path: Path) -> None:
        from laionfashion.bundle import load_bundle

        bundle_dir = _make_bundle(tmp_path, n=3)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        garments = extract_garments_from_bundle(
            records, bundle_dir, method="detr", detector=MockDetector()
        )
        save_garments(garments, bundle_dir)

        bundle = load_bundle(bundle_dir)
        outfit_garments = bundle.garments_for_outfit(1)
        assert len(outfit_garments) == 2
        assert set(outfit_garments["category"]) == {"top", "bottom"}
