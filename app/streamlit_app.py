"""Fashion Embedding Explorer – Streamlit app for browsing debug bundles."""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from laionfashion.bundle import DebugBundle, load_bundle, nearest_neighbors

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fashion Embedding Explorer", layout="wide")
st.title("Fashion Embedding Explorer")

# ---------------------------------------------------------------------------
# Bundle selection
# ---------------------------------------------------------------------------

DEFAULT_BUNDLE_ROOT = Path("scripts/outputs/01_build_debug_subset")

bundle_path = st.sidebar.text_input(
    "Bundle path",
    value=str(DEFAULT_BUNDLE_ROOT),
    help="Path to a debug bundle directory containing records.parquet, embeddings.npy, and thumbnails/",
)


@st.cache_resource
def _load_bundle(path: str) -> DebugBundle:
    return load_bundle(path)


# Auto-detect: if the user points at the parent outputs dir, list available bundles
bundle_dir = Path(bundle_path)
if bundle_dir.is_dir() and not (bundle_dir / "embeddings.npy").exists():
    # Look for sub-directories that contain a bundle
    candidates = sorted(
        [
            d
            for d in bundle_dir.iterdir()
            if d.is_dir() and (d / "embeddings.npy").exists()
        ],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        st.info(f"No bundles found under `{bundle_dir}`. Enter a path to a bundle directory in the sidebar.")
        st.stop()
    chosen = st.sidebar.selectbox(
        "Available bundles",
        candidates,
        format_func=lambda p: p.name,
    )
    bundle_dir = chosen

try:
    bundle = _load_bundle(str(bundle_dir))
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.success(f"Loaded {bundle.n_images} images from `{bundle_dir.name}`")

# ---------------------------------------------------------------------------
# Image grid
# ---------------------------------------------------------------------------

N_COLS = st.sidebar.slider("Grid columns", 2, 8, 4)
MAX_GRID = st.sidebar.slider("Images shown", 10, min(200, bundle.n_images), min(50, bundle.n_images))

st.header("Image Grid")

records = bundle.records.head(MAX_GRID)
cols = st.columns(N_COLS)

for i, (_, row) in enumerate(records.iterrows()):
    col = cols[i % N_COLS]
    thumb = bundle.thumbnail_path(row["row_id"] if "row_id" in row else i)
    with col:
        if thumb and thumb.exists():
            st.image(str(thumb), use_container_width=True)
        else:
            st.write("*(no thumbnail)*")
        caption = row.get("caption", "")
        if caption:
            st.caption(caption[:120])

# ---------------------------------------------------------------------------
# Single-image selection and nearest neighbors
# ---------------------------------------------------------------------------

st.header("Nearest Neighbors")

selected_idx = st.number_input(
    "Select image index (row_id)",
    min_value=0,
    max_value=bundle.n_images - 1,
    value=0,
    step=1,
)

k = st.sidebar.slider("Number of neighbors", 1, 30, 8)

neighbors = nearest_neighbors(bundle.embeddings, selected_idx, k=k)

# Show selected image
st.subheader("Query image")
query_thumb = bundle.thumbnail_path(selected_idx)
query_row = bundle.records.iloc[selected_idx]
qcol1, qcol2 = st.columns([1, 3])
with qcol1:
    if query_thumb and query_thumb.exists():
        st.image(str(query_thumb), use_container_width=True)
    else:
        st.write("*(no thumbnail)*")
with qcol2:
    st.write(f"**Caption:** {query_row.get('caption', 'N/A')}")
    st.write(f"**Global index:** {query_row.get('global_index', 'N/A')}")

# Show neighbors
st.subheader(f"Top {k} neighbors")
nn_cols = st.columns(min(k, N_COLS))
for j, (idx, sim) in enumerate(neighbors):
    col = nn_cols[j % len(nn_cols)]
    nn_row = bundle.records.iloc[idx]
    nn_thumb = bundle.thumbnail_path(idx)
    with col:
        if nn_thumb and nn_thumb.exists():
            st.image(str(nn_thumb), use_container_width=True)
        else:
            st.write("*(no thumbnail)*")
        st.caption(f"sim={sim:.3f}")
        cap = nn_row.get("caption", "")
        if cap:
            st.caption(cap[:100])
