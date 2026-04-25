#!/usr/bin/env python3
"""Generate a failure case report for a bundle.

Usage:
    python scripts/18_failure_report.py <bundle_dir>

Detects common failure cases and writes failure_report.md to the bundle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle
from laionfashion.failures import detect_common_failures, format_failure_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate failure case report.")
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    print(f"Bundle: {bundle.n_images} images, {bundle.n_garments} garments")

    failures = detect_common_failures(bundle)
    report = format_failure_report(failures)

    out_path = args.bundle_dir / "failure_report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")

    if failures:
        print(f"\nDetected {len(failures)} failure categories:")
        for f in failures:
            print(f"  [{f.severity}] {f.category}: {f.description}")
    else:
        print("\nNo failures detected.")


if __name__ == "__main__":
    main()
