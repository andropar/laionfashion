"""Tests for laionfashion.review – contact sheet generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from laionfashion.bundle import load_bundle
from laionfashion.review import render_contact_sheet_html


def _make_bundle(tmp_path: Path, n: int = 5) -> Path:
    """Create a minimal synthetic bundle."""
    thumb_dir = tmp_path / "thumbnails"
    thumb_dir.mkdir()

    records = []
    for i in range(n):
        thumb_name = f"{i:06d}_{i}.jpg"
        # Write a tiny valid JPEG-ish file
        (thumb_dir / thumb_name).write_bytes(b"\xff\xd8\xff\xe0dummy")
        records.append(
            {
                "row_id": i,
                "global_index": i * 100,
                "caption": f"Person wearing outfit {i}",
                "thumbnail_path": f"thumbnails/{thumb_name}",
            }
        )

    pd.DataFrame(records).to_parquet(tmp_path / "records.parquet", index=False)
    emb = np.random.default_rng(0).standard_normal((n, 16)).astype(np.float32)
    np.save(tmp_path / "embeddings.npy", emb)
    return tmp_path


class TestContactSheet:
    def test_renders_html(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path))
        html = render_contact_sheet_html(bundle)
        assert "<!DOCTYPE html>" in html
        assert "Contact sheet" in html
        assert "Person wearing outfit 0" in html

    def test_includes_all_images(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path, n=8))
        html = render_contact_sheet_html(bundle)
        for i in range(8):
            assert f"#{i}" in html

    def test_respects_n_images(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path, n=10))
        html = render_contact_sheet_html(bundle, n_images=3)
        assert "3 of 10 images" in html
        assert "#0" in html
        assert "#2" in html
        # row #9 should not appear as a row ID label
        assert '>#9<' not in html

    def test_embeds_thumbnails_as_base64(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path))
        html = render_contact_sheet_html(bundle)
        assert "data:image/jpeg;base64," in html

    def test_escapes_html_in_captions(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=1)
        records = pd.read_parquet(bundle_dir / "records.parquet")
        records.at[0, "caption"] = 'Caption with <script>alert("xss")</script>'
        records.to_parquet(bundle_dir / "records.parquet", index=False)
        bundle = load_bundle(bundle_dir)
        html = render_contact_sheet_html(bundle)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_handles_missing_thumbnail(self, tmp_path: Path) -> None:
        bundle_dir = _make_bundle(tmp_path, n=1)
        # Delete the thumbnail
        for f in (bundle_dir / "thumbnails").iterdir():
            f.unlink()
        bundle = load_bundle(bundle_dir)
        html = render_contact_sheet_html(bundle)
        assert "no thumbnail" in html

    def test_grid_columns(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path))
        html = render_contact_sheet_html(bundle, n_cols=3)
        assert "repeat(3, 1fr)" in html

    def test_caveat_present(self, tmp_path: Path) -> None:
        bundle = load_bundle(_make_bundle(tmp_path))
        html = render_contact_sheet_html(bundle)
        assert "local/private" in html
