#!/usr/bin/env python3
"""Export bundle data as JSON for the standalone HTML dashboard.

Usage:
    python scripts/19_export_dashboard_data.py <bundle_dir>

Creates app/dashboard_data/ with JSON files consumed by app/explorer.html.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.bundle import load_bundle, nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dashboard data from a bundle.")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--garment-neighbors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)
    out = REPO_ROOT / "app" / "dashboard_data"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Bundle: {bundle.n_images} images, {bundle.n_garments} garments")

    # 1. Projection + clusters + captions
    proj = pd.read_parquet(args.bundle_dir / "projection.parquet")
    clusters = pd.read_parquet(args.bundle_dir / "clusters.parquet")
    merged = proj.merge(clusters, on="row_id", how="left")
    merged["caption"] = bundle.records["caption"].values
    if "image_outfit_score" in bundle.records.columns:
        merged["outfit_score"] = bundle.records["image_outfit_score"].values

    projection_data = merged.to_dict(orient="records")
    with (out / "projection.json").open("w") as f:
        json.dump(projection_data, f)
    print(f"  projection.json: {len(projection_data)} points")

    # 2. Axis scores
    axes_df = pd.read_parquet(args.bundle_dir / "axis_scores.parquet")
    axis_data = {}
    axis_sorted = {}
    for col in axes_df.columns:
        if col == "row_id":
            continue
        vals = axes_df[col].values
        axis_data[col] = [round(float(v), 4) for v in vals]
        # Pre-sorted indices for axis strip
        sorted_idx = np.argsort(vals).tolist()
        axis_sorted[col] = sorted_idx
    with (out / "axis_scores.json").open("w") as f:
        json.dump(axis_data, f)
    with (out / "axis_sorted.json").open("w") as f:
        json.dump(axis_sorted, f)
    print(f"  axis_scores.json: {len(axis_data)} axes")
    print(f"  axis_sorted.json: sorted indices for strip view")

    # 3. Outfit neighbors
    print("  Computing outfit neighbors...")
    neighbors_data = {}
    for i in range(bundle.n_images):
        nn = nearest_neighbors(bundle.embeddings, i, k=args.max_neighbors)
        neighbors_data[str(i)] = [{"row_id": idx, "similarity": round(sim, 4)} for idx, sim in nn]
    with (out / "neighbors.json").open("w") as f:
        json.dump(neighbors_data, f)
    print(f"  neighbors.json: {len(neighbors_data)} entries")

    # 4. Garments + garment cross-category retrieval
    if bundle.has_garments:
        garment_data = []
        for _, row in bundle.garments.iterrows():
            garment_data.append({
                "garment_id": int(row["garment_id"]),
                "outfit_id": int(row["outfit_id"]),
                "category": str(row["category"]),
                "confidence": round(float(row["confidence"]), 3) if pd.notna(row.get("confidence")) else None,
                "crop_path": str(row["crop_path"]),
            })
        with (out / "garments.json").open("w") as f:
            json.dump(garment_data, f)
        print(f"  garments.json: {len(garment_data)} garments")

        # Garment-level cross-category neighbors
        if bundle.garment_embeddings is not None:
            print("  Computing garment cross-category neighbors...")
            from laionfashion.retrieval import retrieve_similar_garments
            garment_neighbors = {}
            categories = bundle.garments["category"].unique().tolist()
            for _, row in bundle.garments.iterrows():
                gid = int(row["garment_id"])
                gcat = row["category"]
                cross = {}
                for target_cat in categories:
                    if target_cat == gcat:
                        continue
                    results = retrieve_similar_garments(
                        query_garment_id=gid,
                        garments=bundle.garments,
                        embeddings=bundle.garment_embeddings,
                        target_category=target_cat,
                        exclude_same_outfit=True,
                        k=args.garment_neighbors,
                    )
                    if results:
                        cross[target_cat] = [
                            {"garment_id": r.garment_id, "outfit_id": r.outfit_id,
                             "similarity": round(r.similarity, 4), "crop_path": r.crop_path}
                            for r in results
                        ]
                if cross:
                    garment_neighbors[str(gid)] = cross
            # Write per-outfit neighbor files for lazy loading (avoids 96MB monolith)
            gid_to_outfit = {g["garment_id"]: g["outfit_id"] for g in garment_data}
            per_outfit: dict[int, dict] = {}
            for gid_str, cats in garment_neighbors.items():
                oid = gid_to_outfit.get(int(gid_str))
                if oid is not None:
                    per_outfit.setdefault(oid, {})[gid_str] = cats
            gn_dir = out / "gn"
            if gn_dir.exists():
                shutil.rmtree(gn_dir)
            gn_dir.mkdir()
            for oid, data in per_outfit.items():
                with (gn_dir / f"{oid}.json").open("w") as f:
                    json.dump(data, f, separators=(",", ":"))
            print(f"  gn/: {len(per_outfit)} per-outfit neighbor files")

    # 5. Config
    manifest = json.load((args.bundle_dir / "manifest.json").open())
    config = {
        "n_images": bundle.n_images,
        "n_garments": bundle.n_garments,
        "embedding_dim": int(bundle.embeddings.shape[1]),
        "axes": [c for c in axes_df.columns if c != "row_id"],
        "n_clusters": int(clusters["cluster_id"].nunique()),
        "cluster_labels": dict(sorted({
            str(cid): label
            for cid, label in zip(clusters["cluster_id"], clusters["cluster_label"])
        }.items())),
        "categories": sorted(bundle.garments["category"].unique().tolist()) if bundle.has_garments else [],
    }
    if "clip_reranking" in manifest:
        config["clip_reranking"] = manifest["clip_reranking"]
    if "filter_diagnostics" in manifest:
        config["filter_diagnostics"] = manifest["filter_diagnostics"]
    if "axis_scores" in manifest:
        config["axis_info"] = manifest["axis_scores"]
    with (out / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json")

    # 6. Copy thumbnails and garment crops
    thumb_dst = out / "thumbnails"
    if thumb_dst.exists():
        shutil.rmtree(thumb_dst)
    shutil.copytree(args.bundle_dir / "thumbnails", thumb_dst)
    print(f"  thumbnails/: {len(list(thumb_dst.iterdir()))} files")

    if (args.bundle_dir / "garment_crops").exists():
        crops_dst = out / "garment_crops"
        if crops_dst.exists():
            shutil.rmtree(crops_dst)
        shutil.copytree(args.bundle_dir / "garment_crops", crops_dst)
        print(f"  garment_crops/: {len(list(crops_dst.iterdir()))} files")

    # 7. Thumbnail index
    thumb_index = {}
    for _, row in bundle.records.iterrows():
        rid = int(row["row_id"])
        thumb_rel = row["thumbnail_path"]
        thumb_index[str(rid)] = Path(thumb_rel).name
    with (out / "thumb_index.json").open("w") as f:
        json.dump(thumb_index, f)

    print(f"\nDone. Serve with: python -m http.server 8080 -d app")


if __name__ == "__main__":
    main()
