#!/usr/bin/env python3
"""Pack a debug bundle into a portable archive for local development.

Usage:
    python scripts/10_pack_bundle.py <bundle_dir> --output <path.tar.gz>
    python scripts/10_pack_bundle.py <bundle_dir>  # auto-names the archive

Creates a .tar.gz of the bundle including metadata, thumbnails, garment crops,
embeddings, evaluation outputs, and review HTML.  Detection images are excluded
by default (--include-detection-images to include them).
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.portable import list_bundle_artifacts, validate_portable_bundle

# Files/dirs to always exclude from the archive.
_EXCLUDE_PATTERNS = {
    "__pycache__",
    ".DS_Store",
    "*.pyc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack a bundle into a portable archive.")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output archive path. Defaults to <bundle_name>.tar.gz in the current directory.",
    )
    parser.add_argument(
        "--include-detection-images", action="store_true", default=False,
        help="Include detection_images/ (can be large).",
    )
    return parser.parse_args()


def _should_exclude(rel_path: str, include_detection: bool) -> bool:
    parts = Path(rel_path).parts
    for part in parts:
        if part in _EXCLUDE_PATTERNS:
            return True
        for pattern in _EXCLUDE_PATTERNS:
            if pattern.startswith("*") and part.endswith(pattern[1:]):
                return True
    if not include_detection and parts and parts[0] == "detection_images":
        return True
    return False


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    bundle_name = bundle_dir.name

    # Validate first
    result = validate_portable_bundle(bundle_dir)
    if not result.ok:
        print("Bundle validation failed:")
        for e in result.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    if result.warnings:
        for w in result.warnings:
            print(f"  WARNING: {w}")

    # Determine output path
    output = args.output or Path(f"{bundle_name}.tar.gz")

    # List and filter artifacts
    artifacts = list_bundle_artifacts(bundle_dir)
    included = {
        rel: size for rel, size in artifacts.items()
        if not _should_exclude(rel, args.include_detection_images)
    }
    excluded_count = len(artifacts) - len(included)
    total_size = sum(included.values())

    print(f"Bundle: {bundle_name}")
    print(f"Files:  {len(included)} included, {excluded_count} excluded")
    print(f"Size:   {total_size / 1024 / 1024:.1f} MB (uncompressed)")

    # Show artifact summary
    categories: dict[str, tuple[int, int]] = {}
    for rel, size in included.items():
        top = Path(rel).parts[0] if "/" in rel else rel
        count, total = categories.get(top, (0, 0))
        categories[top] = (count + 1, total + size)
    print("\nArtifact summary:")
    for cat, (count, size) in sorted(categories.items()):
        print(f"  {cat}: {count} files, {size / 1024:.0f} KB")

    # Create archive
    with tarfile.open(output, "w:gz") as tar:
        for rel in sorted(included):
            full = bundle_dir / rel
            arcname = f"{bundle_name}/{rel}"
            tar.add(str(full), arcname=arcname)

    archive_size = output.stat().st_size
    print(f"\nWrote {output} ({archive_size / 1024 / 1024:.1f} MB compressed)")


if __name__ == "__main__":
    main()
