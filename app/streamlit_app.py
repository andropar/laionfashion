"""Fashion Embedding Explorer – Streamlit app for browsing debug bundles."""

from __future__ import annotations

import streamlit as st
import pandas as pd
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


@st.cache_data
def _load_projection(bundle_dir: str) -> pd.DataFrame | None:
    """Load projection.parquet/.csv from a bundle dir, or return None."""
    d = Path(bundle_dir)
    parquet = d / "projection.parquet"
    csv = d / "projection.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


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

projection = _load_projection(str(bundle_dir))

st.sidebar.success(f"Loaded {bundle.n_images} images from `{bundle_dir.name}`")
if projection is not None:
    st.sidebar.info(f"Projection loaded ({len(projection)} points)")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

N_COLS = st.sidebar.slider("Grid columns", 2, 8, 4)
MAX_GRID = st.sidebar.slider("Images shown", 10, min(200, bundle.n_images), min(50, bundle.n_images))
k = st.sidebar.slider("Number of neighbors", 1, 30, 8)

# ---------------------------------------------------------------------------
# Shared state: selected image index
# ---------------------------------------------------------------------------

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

selected_idx = st.session_state.selected_idx

# ---------------------------------------------------------------------------
# Embedding map (style-space projection)
# ---------------------------------------------------------------------------

if projection is not None:
    import plotly.graph_objects as go

    st.header("Embedding map")

    proj = projection.copy()
    # Attach captions for hover
    if "caption" in bundle.records.columns:
        proj["caption"] = bundle.records["caption"].values[: len(proj)]
    else:
        proj["caption"] = ""
    proj["caption_short"] = proj["caption"].str[:80]

    # Highlight the selected point
    is_selected = proj["row_id"] == selected_idx
    nn_indices = {idx for idx, _ in nearest_neighbors(bundle.embeddings, selected_idx, k=k)}
    is_neighbor = proj["row_id"].isin(nn_indices)

    fig = go.Figure()

    # Background points
    mask_bg = ~is_selected & ~is_neighbor
    fig.add_trace(
        go.Scatter(
            x=proj.loc[mask_bg, "x"],
            y=proj.loc[mask_bg, "y"],
            mode="markers",
            marker=dict(size=6, color="#94a3b8", opacity=0.5),
            text=proj.loc[mask_bg, "caption_short"],
            customdata=proj.loc[mask_bg, "row_id"],
            hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
            name="Other",
        )
    )

    # Neighbor points
    fig.add_trace(
        go.Scatter(
            x=proj.loc[is_neighbor, "x"],
            y=proj.loc[is_neighbor, "y"],
            mode="markers",
            marker=dict(size=9, color="#3b82f6", opacity=0.85),
            text=proj.loc[is_neighbor, "caption_short"],
            customdata=proj.loc[is_neighbor, "row_id"],
            hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
            name="Neighbors",
        )
    )

    # Selected point
    fig.add_trace(
        go.Scatter(
            x=proj.loc[is_selected, "x"],
            y=proj.loc[is_selected, "y"],
            mode="markers",
            marker=dict(size=14, color="#ef4444", symbol="star"),
            text=proj.loc[is_selected, "caption_short"],
            customdata=proj.loc[is_selected, "row_id"],
            hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
            name="Selected",
        )
    )

    fig.update_layout(
        xaxis_title="Projection x",
        yaxis_title="Projection y",
        height=500,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        dragmode="pan",
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Style-space projection of image embeddings. "
        "Proximity reflects embedding similarity, not ground-truth style categories."
    )
else:
    st.info(
        "No projection found. Run `python scripts/02_build_projection.py <bundle_dir>` "
        "to generate one."
    )

# ---------------------------------------------------------------------------
# Image grid
# ---------------------------------------------------------------------------

st.header("Image grid")

records = bundle.records.head(MAX_GRID)
cols = st.columns(N_COLS)

for i, (_, row) in enumerate(records.iterrows()):
    col = cols[i % N_COLS]
    row_id = row["row_id"] if "row_id" in row else i
    thumb = bundle.thumbnail_path(row_id)
    with col:
        if thumb and thumb.exists():
            st.image(str(thumb), use_container_width=True)
        else:
            st.write("*(no thumbnail)*")
        caption = row.get("caption", "")
        if caption:
            st.caption(caption[:120])
        if st.button(f"Select #{row_id}", key=f"sel_{row_id}"):
            st.session_state.selected_idx = row_id
            st.rerun()

# ---------------------------------------------------------------------------
# Single-image selection and nearest neighbors
# ---------------------------------------------------------------------------

st.header("Nearest neighbors")

selected_idx = st.number_input(
    "Select image index (row_id)",
    min_value=0,
    max_value=bundle.n_images - 1,
    value=st.session_state.selected_idx,
    step=1,
    key="nn_selector",
)
if selected_idx != st.session_state.selected_idx:
    st.session_state.selected_idx = selected_idx
    st.rerun()

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
