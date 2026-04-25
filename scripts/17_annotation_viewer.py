#!/usr/bin/env python3
"""Generate an interactive HTML annotation viewer for pairwise comparison.

Usage:
    python scripts/17_annotation_viewer.py <bundle_dir>

Reads annotation_tasks.csv from the bundle and produces annotation_viewer.html,
a self-contained HTML page where a human can compare image pairs side by side,
select a choice (A / B / Tie), and download the results as CSV.

Thumbnails are embedded as base64 data URIs so the page works offline.
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from laionfashion.annotation import load_annotations
from laionfashion.bundle import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotation viewer HTML.")
    parser.add_argument("bundle_dir", type=Path)
    return parser.parse_args()


def _thumbnail_b64(bundle, row_id: int) -> str:
    """Return a base64 data URI for a thumbnail, or a placeholder."""
    thumb_path = bundle.thumbnail_path(row_id)
    if thumb_path and thumb_path.exists():
        data = thumb_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    return ""


def render_annotation_viewer(bundle, tasks) -> str:
    """Render the annotation viewer HTML page."""
    bundle_name = html.escape(bundle.bundle_dir.name)
    n_pairs = len(tasks)

    # Collect all unique thumbnails needed and build a lookup
    needed_ids = set(tasks["image_a_row_id"].tolist() + tasks["image_b_row_id"].tolist())
    thumb_map: dict[int, str] = {}
    for rid in sorted(needed_ids):
        thumb_map[rid] = _thumbnail_b64(bundle, rid)

    # Build pair cards
    pair_cards: list[str] = []
    for _, row in tasks.iterrows():
        pair_id = int(row["pair_id"])
        a_id = int(row["image_a_row_id"])
        b_id = int(row["image_b_row_id"])
        axis = html.escape(str(row["axis"]))
        existing_choice = str(row.get("choice", "")).strip()

        a_src = thumb_map.get(a_id, "")
        b_src = thumb_map.get(b_id, "")

        a_img = (
            f'<img src="{a_src}" alt="Image A (#{a_id})">'
            if a_src
            else '<div class="no-thumb">no thumbnail</div>'
        )
        b_img = (
            f'<img src="{b_src}" alt="Image B (#{b_id})">'
            if b_src
            else '<div class="no-thumb">no thumbnail</div>'
        )

        # Pre-check radio if there is an existing choice
        check_a = "checked" if existing_choice == "a" else ""
        check_b = "checked" if existing_choice == "b" else ""
        check_tie = "checked" if existing_choice == "tie" else ""

        pair_cards.append(f"""
<div class="pair-card" data-pair-id="{pair_id}" data-a-id="{a_id}" data-b-id="{b_id}" data-axis="{axis}">
  <div class="pair-header">
    <span class="pair-num">Pair #{pair_id}</span>
    <span class="axis-badge">{axis}</span>
  </div>
  <div class="pair-images">
    <div class="img-col">
      <div class="img-label">A</div>
      {a_img}
      <div class="rid">#{a_id}</div>
    </div>
    <div class="img-col">
      <div class="img-label">B</div>
      {b_img}
      <div class="rid">#{b_id}</div>
    </div>
  </div>
  <div class="pair-controls">
    <label><input type="radio" name="pair_{pair_id}" value="a" {check_a}> A</label>
    <label><input type="radio" name="pair_{pair_id}" value="b" {check_b}> B</label>
    <label><input type="radio" name="pair_{pair_id}" value="tie" {check_tie}> Tie</label>
  </div>
</div>""")

    cards_html = "\n".join(pair_cards)

    # Build axis filter buttons
    axes = sorted(tasks["axis"].unique())
    axis_buttons = ['<button class="axis-filter active" data-axis="all">All</button>']
    for ax in axes:
        axis_buttons.append(
            f'<button class="axis-filter" data-axis="{html.escape(ax)}">'
            f'{html.escape(ax)}</button>'
        )
    axis_buttons_html = "\n".join(axis_buttons)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Annotation viewer — {bundle_name}</title>
<style>
* {{
    box-sizing: border-box;
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8fafc;
    margin: 0;
    padding: 16px;
    color: #334155;
}}
h1 {{
    font-size: 18px;
    color: #334155;
    margin: 0 0 4px;
}}
.meta {{
    font-size: 13px;
    color: #64748b;
    margin-bottom: 16px;
}}
.toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding: 12px;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
}}
.toolbar label {{
    font-size: 13px;
    color: #475569;
}}
.toolbar input[type="text"] {{
    padding: 4px 8px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    font-size: 13px;
    width: 180px;
}}
.toolbar button {{
    padding: 6px 14px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #fff;
    font-size: 13px;
    cursor: pointer;
    color: #334155;
}}
.toolbar button:hover {{
    background: #f1f5f9;
}}
.toolbar button.primary {{
    background: #3b82f6;
    color: #fff;
    border-color: #3b82f6;
}}
.toolbar button.primary:hover {{
    background: #2563eb;
}}
.progress {{
    font-size: 13px;
    color: #64748b;
    margin-bottom: 12px;
}}
.axis-filters {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 16px;
}}
.axis-filter {{
    padding: 4px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #fff;
    font-size: 12px;
    cursor: pointer;
    color: #475569;
}}
.axis-filter:hover {{
    background: #f1f5f9;
}}
.axis-filter.active {{
    background: #3b82f6;
    color: #fff;
    border-color: #3b82f6;
}}
.pairs-container {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.pair-card {{
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
    transition: border-color 0.15s;
}}
.pair-card.answered {{
    border-color: #86efac;
}}
.pair-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}
.pair-num {{
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
}}
.axis-badge {{
    font-size: 11px;
    padding: 2px 8px;
    background: #f1f5f9;
    border-radius: 10px;
    color: #475569;
}}
.pair-images {{
    display: flex;
    gap: 16px;
    justify-content: center;
    margin-bottom: 10px;
}}
.img-col {{
    text-align: center;
    flex: 0 0 auto;
}}
.img-label {{
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 4px;
}}
.img-col img {{
    width: 160px;
    height: auto;
    border-radius: 4px;
    display: block;
    border: 1px solid #e2e8f0;
}}
.no-thumb {{
    width: 160px;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f1f5f9;
    color: #94a3b8;
    font-size: 12px;
    border-radius: 4px;
}}
.rid {{
    font-size: 11px;
    color: #94a3b8;
    margin-top: 2px;
}}
.pair-controls {{
    display: flex;
    justify-content: center;
    gap: 20px;
}}
.pair-controls label {{
    font-size: 13px;
    cursor: pointer;
    color: #475569;
}}
.pair-controls input[type="radio"] {{
    margin-right: 4px;
}}
.caveat {{
    font-size: 12px;
    color: #94a3b8;
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
}}
</style>
</head>
<body>
<h1>Annotation viewer</h1>
<div class="meta">{bundle_name} · {n_pairs} pairs · {len(axes)} axes</div>

<div class="toolbar">
  <label>Annotator: <input type="text" id="annotator-name" placeholder="your name"></label>
  <button class="primary" id="download-btn">Download CSV</button>
  <span class="progress" id="progress">0 / {n_pairs} answered</span>
</div>

<div class="axis-filters">
{axis_buttons_html}
</div>

<div class="pairs-container" id="pairs-container">
{cards_html}
</div>

<div class="caveat">
Debug review artifact — images are local/private. Not a safety-reviewed dataset.
</div>

<script>
(function() {{
  const totalPairs = {n_pairs};

  // Progress tracking
  function updateProgress() {{
    let answered = 0;
    document.querySelectorAll('.pair-card').forEach(card => {{
      const pairId = card.dataset.pairId;
      const checked = document.querySelector('input[name="pair_' + pairId + '"]:checked');
      if (checked) {{
        card.classList.add('answered');
        answered++;
      }} else {{
        card.classList.remove('answered');
      }}
    }});
    document.getElementById('progress').textContent = answered + ' / ' + totalPairs + ' answered';
  }}

  // Listen for radio changes
  document.getElementById('pairs-container').addEventListener('change', updateProgress);

  // Initial progress count (for pre-filled choices)
  updateProgress();

  // Axis filtering
  document.querySelectorAll('.axis-filter').forEach(btn => {{
    btn.addEventListener('click', function() {{
      document.querySelectorAll('.axis-filter').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      const axis = this.dataset.axis;
      document.querySelectorAll('.pair-card').forEach(card => {{
        if (axis === 'all' || card.dataset.axis === axis) {{
          card.style.display = '';
        }} else {{
          card.style.display = 'none';
        }}
      }});
    }});
  }});

  // Download CSV
  document.getElementById('download-btn').addEventListener('click', function() {{
    const annotator = document.getElementById('annotator-name').value.trim() || 'anonymous';
    let csv = 'pair_id,image_a_row_id,image_b_row_id,axis,choice,annotator\\n';

    document.querySelectorAll('.pair-card').forEach(card => {{
      const pairId = card.dataset.pairId;
      const aId = card.dataset.aId;
      const bId = card.dataset.bId;
      const axis = card.dataset.axis;
      const checked = document.querySelector('input[name="pair_' + pairId + '"]:checked');
      const choice = checked ? checked.value : '';
      csv += pairId + ',' + aId + ',' + bId + ',' + axis + ',' + choice + ',' + annotator + '\\n';
    }});

    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotations_' + annotator + '.csv';
    a.click();
    URL.revokeObjectURL(url);
  }});
}})();
</script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle_dir)

    tasks_path = args.bundle_dir / "annotation_tasks.csv"
    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found.")
        print("Run scripts/15_generate_annotation_tasks.py first.")
        sys.exit(1)

    tasks = load_annotations(tasks_path)
    print(f"Bundle: {bundle.n_images} images")
    print(f"Annotation tasks: {len(tasks)} pairs")

    html_content = render_annotation_viewer(bundle, tasks)

    out_path = args.bundle_dir / "annotation_viewer.html"
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html_content):,} bytes)")


if __name__ == "__main__":
    main()
