#!/usr/bin/env python3
"""Generate garment review HTML sheets for visual inspection.

Usage:
    python scripts/09_garment_review.py <bundle_dir> [--retrieval-examples 5]

Produces:
- garment_review.html: crops grouped by outfit with categories and confidence
- retrieval_review.html: for selected query garments, shows cross-category
  retrieval results (query crop → retrieved crops per category)
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
from laionfashion.retrieval import retrieve_cross_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate garment review HTML sheets.")
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--max-outfits", type=int, default=50)
    parser.add_argument("--retrieval-examples", type=int, default=8,
                        help="Number of query garments for retrieval review.")
    parser.add_argument("--retrieval-k", type=int, default=5)
    return parser.parse_args()


def _img_tag(path: Path, width: int = 120) -> str:
    if path.exists():
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{b64}" style="width:{width}px;height:auto;">'
    return f'<span style="color:#94a3b8;font-size:11px;">missing</span>'


_CSS = """
body { font-family: -apple-system, sans-serif; background: #f8fafc; margin: 0; padding: 16px; }
h1 { font-size: 18px; color: #334155; }
h2 { font-size: 15px; color: #475569; margin-top: 24px; }
.outfit { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; padding: 12px;
  background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; }
.garment { text-align: center; }
.garment img { border-radius: 4px; }
.label { font-size: 11px; color: #64748b; margin-top: 2px; }
.query-row { display: flex; gap: 16px; margin-bottom: 20px; padding: 12px;
  background: #fff; border: 1px solid #e2e8f0; border-radius: 6px; }
.query-img { flex-shrink: 0; }
.results { display: flex; flex-wrap: wrap; gap: 8px; }
.result { text-align: center; }
.result img { border-radius: 4px; }
.caveat { font-size: 12px; color: #94a3b8; margin-top: 16px; padding-top: 12px;
  border-top: 1px solid #e2e8f0; }
"""


def build_garment_review(bundle, max_outfits: int) -> str:
    garments = bundle.garments
    parts = [f"<html><head><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>Garment review</h1>")
    parts.append(f"<p>{bundle.n_garments} garments from {bundle.n_images} outfits</p>")

    outfit_ids = sorted(garments["outfit_id"].unique())[:max_outfits]
    for oid in outfit_ids:
        og = garments[garments["outfit_id"] == oid]
        thumb = bundle.thumbnail_path(oid)
        parts.append(f'<h2>Outfit #{oid}</h2>')
        parts.append('<div class="outfit">')
        if thumb:
            parts.append(f'<div class="garment">{_img_tag(thumb, 100)}<div class="label">source</div></div>')
        for _, row in og.iterrows():
            crop = bundle.bundle_dir / row["crop_path"]
            conf = f"{row['confidence']:.2f}" if 'confidence' in row and not (row.get('confidence') != row.get('confidence')) else ""
            parts.append(
                f'<div class="garment">{_img_tag(crop, 90)}'
                f'<div class="label">{html.escape(row["category"])}</div>'
                f'<div class="label">{conf}</div></div>'
            )
        parts.append('</div>')

    parts.append('<div class="caveat">Debug review — images are local/private.</div>')
    parts.append("</body></html>")
    return "\n".join(parts)


def build_retrieval_review(bundle, n_examples: int, k: int) -> str:
    garments = bundle.garments
    embeddings = bundle.garment_embeddings

    # Pick query garments: one per category, spread across outfits
    queries = []
    for cat in sorted(garments["category"].unique()):
        cat_garments = garments[garments["category"] == cat]
        step = max(1, len(cat_garments) // max(1, n_examples // len(garments["category"].unique())))
        for i in range(0, len(cat_garments), step):
            if len(queries) >= n_examples:
                break
            queries.append(int(cat_garments.iloc[i]["garment_id"]))
        if len(queries) >= n_examples:
            break

    parts = [f"<html><head><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>Retrieval review (CLIP baseline)</h1>")
    parts.append(f"<p>{len(queries)} queries, k={k}</p>")

    for qid in queries:
        qrow = garments.loc[garments["garment_id"] == qid].iloc[0]
        query_crop = bundle.bundle_dir / qrow["crop_path"]

        results = retrieve_cross_category(
            query_garment_id=qid,
            garments=garments,
            embeddings=embeddings,
            k=k,
        )

        parts.append(f'<div class="query-row">')
        parts.append(f'<div class="query-img">{_img_tag(query_crop, 120)}'
                     f'<div class="label">query: {html.escape(qrow["category"])} (outfit #{qrow["outfit_id"]})</div></div>')

        for cat, hits in sorted(results.items()):
            parts.append(f'<div><div class="label" style="font-weight:bold;margin-bottom:4px;">{html.escape(cat)}</div>')
            parts.append('<div class="results">')
            for r in hits:
                crop = bundle.bundle_dir / r.crop_path
                parts.append(
                    f'<div class="result">{_img_tag(crop, 80)}'
                    f'<div class="label">{r.similarity:.3f}</div>'
                    f'<div class="label">outfit #{r.outfit_id}</div></div>'
                )
            parts.append('</div></div>')
        parts.append('</div>')

    parts.append('<div class="caveat">CLIP baseline retrieval — expected to be mediocre. '
                 'This is the baseline to beat with learned embeddings.</div>')
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)

    if not bundle.has_garments:
        raise RuntimeError("No garments found. Run 06_extract_garments.py first.")

    # Garment review
    garment_html = build_garment_review(bundle, args.max_outfits)
    garment_path = args.bundle_dir / "garment_review.html"
    garment_path.write_text(garment_html)
    print(f"Wrote {garment_path}")

    # Retrieval review (only if embeddings exist)
    if bundle.garment_embeddings is not None:
        retrieval_html = build_retrieval_review(
            bundle, args.retrieval_examples, args.retrieval_k
        )
        retrieval_path = args.bundle_dir / "retrieval_review.html"
        retrieval_path.write_text(retrieval_html)
        print(f"Wrote {retrieval_path}")
    else:
        print("Skipping retrieval review — no garment_embeddings.npy. Run 07_embed_garments.py first.")


if __name__ == "__main__":
    main()
