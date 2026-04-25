"""Review tooling for debug bundles — contact sheets and visual inspection."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from laionfashion.bundle import DebugBundle


def render_contact_sheet_html(
    bundle: DebugBundle,
    *,
    n_images: int | None = None,
    n_cols: int = 5,
) -> str:
    """Render an HTML contact sheet for visual review.

    Thumbnails are embedded as base64 data URIs so the HTML file is
    self-contained and works offline.

    Parameters
    ----------
    bundle:
        A loaded debug bundle.
    n_images:
        Number of images to include.  Defaults to all.
    n_cols:
        Number of columns in the grid.
    """
    n = min(n_images or bundle.n_images, bundle.n_images)
    records = bundle.records.head(n)

    cells: list[str] = []
    for _, row in records.iterrows():
        row_id = int(row.get("row_id", 0))
        caption = str(row.get("caption", ""))
        caption_short = html.escape(caption[:120])
        caption_full = html.escape(caption)

        thumb_path = bundle.thumbnail_path(row_id)
        if thumb_path and thumb_path.exists():
            data = thumb_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="row {row_id}">'
        else:
            img_tag = '<div class="no-thumb">no thumbnail</div>'

        cells.append(
            f'<div class="cell">'
            f'{img_tag}'
            f'<div class="rid">#{row_id}</div>'
            f'<div class="cap" title="{caption_full}">{caption_short}</div>'
            f'</div>'
        )

    grid_html = "\n".join(cells)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Contact sheet — {html.escape(bundle.bundle_dir.name)}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8fafc;
    margin: 0;
    padding: 16px;
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
.grid {{
    display: grid;
    grid-template-columns: repeat({n_cols}, 1fr);
    gap: 12px;
}}
.cell {{
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
    text-align: center;
}}
.cell img {{
    width: 100%;
    height: auto;
    display: block;
}}
.no-thumb {{
    width: 100%;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f1f5f9;
    color: #94a3b8;
    font-size: 12px;
}}
.rid {{
    font-size: 11px;
    color: #94a3b8;
    padding: 4px 8px 0;
}}
.cap {{
    font-size: 12px;
    color: #475569;
    padding: 2px 8px 8px;
    line-height: 1.3;
    word-break: break-word;
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
<h1>Contact sheet</h1>
<div class="meta">{html.escape(bundle.bundle_dir.name)} · {n} of {bundle.n_images} images</div>
<div class="grid">
{grid_html}
</div>
<div class="caveat">
Debug review artifact — images are local/private. Not a safety-reviewed dataset.
</div>
</body>
</html>"""
