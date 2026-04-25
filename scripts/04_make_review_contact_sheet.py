#!/usr/bin/env python3
"""Generate an HTML contact sheet for visual review of a debug bundle.

Usage:
    python scripts/04_make_review_contact_sheet.py <bundle_dir> [--cols 5] [--max-images 200]

Writes contact_sheet.html into the bundle directory.  Open it in a browser
to visually inspect thumbnails, captions, and row IDs.

This is debug review tooling — keep outputs local/private.
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.review import render_contact_sheet_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an HTML contact sheet for a debug bundle."
    )
    parser.add_argument("bundle_dir", type=Path, help="Path to a debug bundle directory.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns in the grid.")
    parser.add_argument("--max-images", type=int, default=200, help="Maximum images to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Loaded bundle: {bundle.n_images} images")

    n = min(args.max_images, bundle.n_images)
    html_content = render_contact_sheet_html(bundle, n_images=n, n_cols=args.cols)

    out_path = args.bundle_dir / "contact_sheet.html"
    out_path.write_text(html_content)
    print(f"Wrote {out_path} ({n} images, {args.cols} columns)")


if __name__ == "__main__":
    main()
