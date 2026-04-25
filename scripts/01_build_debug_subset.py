#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.config import load_data_paths
from laionfashion.data_access import FEATURE_REGISTRY, NaturalSubsetIndex
from laionfashion.debug_export import collect_caption_filtered_subset, export_embeddings
from laionfashion.outputs import make_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small local-debug subset from LAION-natural fashion-like captions."
    )
    parser.add_argument("--n-images", type=int, default=1000)
    parser.add_argument("--candidate-scan", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-key",
        choices=sorted(FEATURE_REGISTRY),
        default="openclip_vit_l_14_quickgelu_metaclip_fullcc.ln_post",
    )
    parser.add_argument("--thumbnail-size", type=int, default=256)
    parser.add_argument(
        "--require-person-context",
        action="store_true",
        default=False,
        help="Require a person hint even for context-term matches (stricter filtering).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_data_paths()
    out_dir = make_output_dir(__file__, paths.output_root)

    index = NaturalSubsetIndex.from_paths(paths)
    rng = np.random.default_rng(args.seed)
    records, filter_diag = collect_caption_filtered_subset(
        index=index,
        rng=rng,
        n_images=args.n_images,
        candidate_scan=args.candidate_scan,
        thumbnail_dir=out_dir / "thumbnails",
        thumbnail_size=args.thumbnail_size,
        require_person_context=args.require_person_context,
    )
    if records.empty:
        raise RuntimeError(
            "No records matched the debug filter. Increase --candidate-scan or adjust filters."
        )

    records_path = out_dir / "records.parquet"
    try:
        records.to_parquet(records_path, index=False)
    except Exception:
        records_path = out_dir / "records.csv"
        records.to_csv(records_path, index=False)

    embedding_info = export_embeddings(
        records=records,
        feature_key=args.feature_key,
        output_path=out_dir / "embeddings.npy",
        index=index,
    )

    review_artifacts = filter_diag.write_review_artifacts(out_dir)

    manifest = {
        "n_requested": int(args.n_images),
        "n_exported": int(len(records)),
        "candidate_scan": int(args.candidate_scan),
        "seed": int(args.seed),
        "feature": embedding_info,
        "filter_diagnostics": filter_diag.to_dict(),
        "review_artifacts": review_artifacts,
        "records_path": str(records_path),
        "thumbnail_dir": str(out_dir / "thumbnails"),
        "data_paths": {
            "subset_root": str(paths.subset_root),
            "memmap_root": str(paths.memmap_root),
        },
        "caveats": [
            "Caption keyword filtering is only a debug bootstrap, not a safety or quality filter.",
            "Real LAION images should stay local/private unless licensing and hosting are handled.",
            "No person detector, NSFW detector, or minor filter has been applied yet.",
        ],
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote debug subset to {out_dir}")


if __name__ == "__main__":
    main()

