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
from laionfashion.debug_export import (
    collect_caption_filtered_subset,
    export_embeddings,
    rewrite_thumbnails_for_ranked,
    score_and_rank_candidates,
)
from laionfashion.filtering import SELECTION_MODES
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
        "--detection-image-size",
        type=int,
        default=None,
        help="Long-edge size for higher-res detection images (e.g. 512, 768). "
             "Written to detection_images/ for garment parsing. Off by default.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=sorted(SELECTION_MODES),
        default=None,
        help=(
            "Caption selection mode: broad (score>=0.5), strict (score>=1.5), "
            "outfit (score>=2.5). Default: tier-based filtering."
        ),
    )
    parser.add_argument(
        "--min-filter-score",
        type=float,
        default=None,
        help="Explicit minimum caption score. Overrides --selection-mode.",
    )
    parser.add_argument(
        "--require-person-context",
        action="store_true",
        default=False,
        help="Legacy: require a person hint even for context-term matches.",
    )

    # CLIP post-ranking (preferred workflow)
    parser.add_argument(
        "--clip-rerank",
        action="store_true",
        default=False,
        help=(
            "Collect a larger candidate pool (--n-candidates), score all with "
            "CLIP, and export the top --n-images. Preferred over --use-image-clip-filter."
        ),
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Number of caption-matched candidates to collect before CLIP ranking. "
             "Defaults to 5 * --n-images.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="ViT-B-32",
        help="Open CLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion400m_e31",
        help="Open CLIP pretrained weights (default: laion400m_e31, falls back to openai).",
    )

    # Legacy inline CLIP gate (kept for backward compat)
    parser.add_argument(
        "--use-image-clip-filter",
        action="store_true",
        default=False,
        help="Legacy: inline CLIP gate during collection. Prefer --clip-rerank.",
    )
    parser.add_argument(
        "--min-image-outfit-score",
        type=float,
        default=0.0,
        help="Minimum CLIP outfit score for inline gate (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_data_paths()
    out_dir = make_output_dir(__file__, paths.output_root)

    # Resolve candidate pool size
    if args.clip_rerank:
        n_collect = args.n_candidates or (5 * args.n_images)
    else:
        n_collect = args.n_images

    # Legacy inline scorer
    inline_scorer = None
    if args.use_image_clip_filter and not args.clip_rerank:
        from laionfashion.image_scoring import CLIPOutfitScorer
        print(f"Loading CLIP model (inline gate): {args.clip_model_name}/{args.clip_pretrained}")
        inline_scorer = CLIPOutfitScorer(
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
        )

    index = NaturalSubsetIndex.from_paths(paths)
    rng = np.random.default_rng(args.seed)

    print(f"Collecting up to {n_collect} caption-matched candidates...")
    det_dir = out_dir / "detection_images" if args.detection_image_size else None
    records, filter_diag = collect_caption_filtered_subset(
        index=index,
        rng=rng,
        n_images=n_collect,
        candidate_scan=args.candidate_scan,
        thumbnail_dir=out_dir / "thumbnails",
        thumbnail_size=args.thumbnail_size,
        require_person_context=args.require_person_context,
        selection_mode=args.selection_mode,
        min_score=args.min_filter_score,
        image_scorer=inline_scorer,
        min_image_score=args.min_image_outfit_score if inline_scorer else 0.0,
        detection_image_dir=det_dir,
        detection_image_size=args.detection_image_size,
    )
    if records.empty:
        raise RuntimeError(
            "No records matched the caption filter. Increase --candidate-scan or adjust filters."
        )
    print(f"Caption filtering: {len(records)} candidates from {filter_diag.scanned} scanned")

    # CLIP post-ranking
    ranking_diag = None
    if args.clip_rerank:
        from laionfashion.image_scoring import CLIPOutfitScorer
        print(f"Loading CLIP model (reranking): {args.clip_model_name}/{args.clip_pretrained}")
        rerank_scorer = CLIPOutfitScorer(
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
        )
        print(f"Scoring {len(records)} candidates, exporting top {args.n_images}...")
        records, ranking_diag = score_and_rank_candidates(
            candidates=records,
            bundle_dir=out_dir,
            image_scorer=rerank_scorer,
            n_export=args.n_images,
        )
        records = rewrite_thumbnails_for_ranked(records, out_dir)
        print(
            f"CLIP reranking: {ranking_diag.n_scored} scored, "
            f"{ranking_diag.n_exported} exported"
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
    if args.clip_rerank and ranking_diag is not None:
        manifest["clip_reranking"] = {
            "model": args.clip_model_name,
            "pretrained": args.clip_pretrained,
            "n_candidates": ranking_diag.n_candidates,
            "n_exported": ranking_diag.n_exported,
            **ranking_diag.to_dict(),
        }
    if args.use_image_clip_filter and not args.clip_rerank:
        manifest["image_clip_filter"] = {
            "model": args.clip_model_name,
            "pretrained": args.clip_pretrained,
            "min_image_outfit_score": args.min_image_outfit_score,
        }

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote debug subset to {out_dir}")


if __name__ == "__main__":
    main()
