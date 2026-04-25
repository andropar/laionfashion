"""Tests for laionfashion.debug_export — image writing and resolution behavior."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from laionfashion.debug_export import write_resized


class TestWriteResized:
    def test_downscales_large_image(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (1000, 600))
        out = tmp_path / "out.jpg"
        write_resized(img, out, max_size=200)
        result = Image.open(out)
        assert max(result.size) == 200

    def test_does_not_upscale_small_image(self, tmp_path: Path) -> None:
        """Source images smaller than max_size are saved at original size."""
        img = Image.new("RGB", (256, 384))
        out = tmp_path / "out.jpg"
        write_resized(img, out, max_size=768)
        result = Image.open(out)
        # Should NOT be upscaled to 768
        assert result.size == (256, 384)

    def test_preserves_aspect_ratio(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (800, 400))
        out = tmp_path / "out.jpg"
        write_resized(img, out, max_size=200)
        result = Image.open(out)
        assert result.size == (200, 100)

    def test_detection_and_thumbnail_independent(self, tmp_path: Path) -> None:
        """Two calls with different max_size produce different outputs."""
        img = Image.new("RGB", (600, 400))
        thumb = tmp_path / "thumb.jpg"
        det = tmp_path / "det.jpg"
        write_resized(img, thumb, max_size=160)
        write_resized(img, det, max_size=512)
        t = Image.open(thumb)
        d = Image.open(det)
        assert max(t.size) == 160
        assert max(d.size) == 512
        assert max(d.size) > max(t.size)
