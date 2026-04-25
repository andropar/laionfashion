#!/usr/bin/env python3
"""Validate that a debug bundle is portable and self-contained.

Usage:
    python scripts/11_validate_portable_bundle.py <bundle_dir>

Checks:
- Required files exist (records.parquet, embeddings.npy, thumbnails/).
- Path columns in parquet files are relative and resolve to existing files.
- Embedding row counts match records/garments.
- Reports optional artifacts as present/missing.

Exits with code 0 on success, 1 on errors.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.portable import validate_portable_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a portable bundle.")
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_portable_bundle(args.bundle_dir)

    # Report artifacts
    print(f"Bundle: {args.bundle_dir}")
    print(f"\nArtifacts:")
    for name, present in sorted(result.artifacts.items()):
        status = "OK" if present else "missing"
        print(f"  {name}: {status}")

    # Warnings
    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  {w}")

    # Errors
    if result.errors:
        print(f"\nErrors:")
        for e in result.errors:
            print(f"  {e}")
        print(f"\nValidation FAILED")
        sys.exit(1)

    print(f"\nValidation OK")


if __name__ == "__main__":
    main()
