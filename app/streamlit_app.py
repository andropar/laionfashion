"""Fashion Embedding Explorer — interactive analysis of outfit embedding spaces."""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from laionfashion.bundle import DebugBundle, load_bundle, nearest_neighbors
from laionfashion.axes import axis_names, load_axis_scores, top_bottom_indices
from laionfashion.retrieval import retrieve_cross_category

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fashion Embedding Explorer",
    page_icon=":shirt:",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Tighten spacing for a cleaner look */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.75rem !important; margin-bottom: 0.25rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    /* Smaller captions */
    .stCaption { font-size: 0.78rem !important; }
    /* Footer styling */
    .footer-text { text-align: center; color: #94a3b8; font-size: 0.75rem;
                   padding: 1.5rem 0 0.5rem 0; border-top: 1px solid #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Fashion Embedding Explorer")
st.caption("Interactive exploration of outfit embedding spaces, style axes, and garment retrieval.")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

DEFAULT_BUNDLE_ROOT = Path("scripts/outputs/01_build_debug_subset")


@st.cache_resource
def _load_bundle(path: str) -> DebugBundle:
    return load_bundle(path)


@st.cache_data
def _load_projection(bundle_dir: str) -> pd.DataFrame | None:
    d = Path(bundle_dir)
    for name in ("projection.parquet", "projection.csv"):
        p = d / name
        if p.exists():
            return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
    return None


@st.cache_data
def _load_axes(bundle_dir: str) -> pd.DataFrame | None:
    return load_axis_scores(bundle_dir)


@st.cache_data
def _load_clusters(bundle_dir: str) -> pd.DataFrame | None:
    p = Path(bundle_dir) / "clusters.parquet"
    if p.exists():
        return pd.read_parquet(p)
    p = Path(bundle_dir) / "clusters.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def _load_manifest(bundle_dir: str) -> dict | None:
    p = Path(bundle_dir) / "manifest.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_axis_name(name: str) -> str:
    """'formal_vs_casual' -> 'Formal vs casual'."""
    return name.replace("_", " ").capitalize()


def _axis_prompt_info(manifest: dict | None, axis_name: str) -> tuple[str | None, str | None]:
    if manifest is None:
        return None, None
    prompts = manifest.get("axis_scores", {}).get("prompts", {})
    ax = prompts.get(axis_name, {})
    return ax.get("positive"), ax.get("negative")


def _is_clip_axes(manifest: dict | None) -> bool:
    if manifest is None:
        return False
    return manifest.get("axis_scores", {}).get("method") == "clip_prompt_direction"


_SHOWCASE_AXES = [
    "colorful_vs_neutral",
    "streetwear_vs_classic",
    "sporty_vs_dressy",
    "minimalist_vs_maximalist",
    "polished_vs_rough",
    "formal_vs_casual",
]


def _order_axes(axes: list[str]) -> list[str]:
    ordered = [a for a in _SHOWCASE_AXES if a in axes]
    ordered += [a for a in axes if a not in ordered]
    return ordered


def _show_thumbnail(bundle: DebugBundle, row_id: int, use_container_width: bool = True) -> None:
    """Render a thumbnail image or a placeholder."""
    thumb = bundle.thumbnail_path(row_id)
    if thumb and thumb.exists():
        st.image(str(thumb), use_container_width=use_container_width)
    else:
        st.markdown("*(no image)*")


def _show_garment_crop(bundle: DebugBundle, garment_id: int, use_container_width: bool = True) -> None:
    """Render a garment crop image or a placeholder."""
    crop = bundle.garment_crop_path(garment_id)
    if crop and crop.exists():
        st.image(str(crop), use_container_width=use_container_width)
    else:
        st.markdown("*(no crop)*")


# ---------------------------------------------------------------------------
# Bundle selection (sidebar)
# ---------------------------------------------------------------------------

bundle_path = st.sidebar.text_input(
    "Bundle path",
    value=str(DEFAULT_BUNDLE_ROOT),
    help="Path to a debug bundle directory.",
)

bundle_dir = Path(bundle_path)
if bundle_dir.is_dir() and not (bundle_dir / "embeddings.npy").exists():
    candidates = sorted(
        [d for d in bundle_dir.iterdir() if d.is_dir() and (d / "embeddings.npy").exists()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        st.info(f"No bundles found under `{bundle_dir}`.")
        st.stop()
    chosen = st.sidebar.selectbox("Available bundles", candidates, format_func=lambda p: p.name)
    bundle_dir = chosen

try:
    bundle = _load_bundle(str(bundle_dir))
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

projection = _load_projection(str(bundle_dir))
axis_scores = _load_axes(str(bundle_dir))
clusters = _load_clusters(str(bundle_dir))
manifest = _load_manifest(str(bundle_dir))
clip_axes = _is_clip_axes(manifest)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown(f"**{bundle_dir.name}** — {bundle.n_images} images")
if bundle.has_garments:
    st.sidebar.markdown(f"{bundle.n_garments} garments detected")

k = st.sidebar.slider("Neighbor count", 1, 30, 8)

available_axes = _order_axes(axis_names(axis_scores)) if axis_scores is not None else []
selected_axis: str | None = None
if available_axes:
    st.sidebar.markdown("---")
    label = "Style axis" if clip_axes else "Style axis (proxy)"
    st.sidebar.subheader(label)
    selected_axis = st.sidebar.selectbox(
        "Color embedding map by axis",
        ["(none)"] + available_axes,
        format_func=lambda x: "(none)" if x == "(none)" else _format_axis_name(x),
    )
    if selected_axis == "(none)":
        selected_axis = None
    if selected_axis:
        pos_prompt, neg_prompt = _axis_prompt_info(manifest, selected_axis)
        if pos_prompt and neg_prompt:
            st.sidebar.caption(f"**+** {pos_prompt}")
            st.sidebar.caption(f"**\u2212** {neg_prompt}")
        elif not clip_axes:
            st.sidebar.caption("Proxy axis — derived from embedding PCA and caption keywords.")

# Cluster color toggle
color_by_cluster = False
if clusters is not None and projection is not None:
    st.sidebar.markdown("---")
    color_by_cluster = st.sidebar.checkbox("Color map by cluster", value=False)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0
selected_idx = st.session_state.selected_idx


# ===================================================================
# TABS
# ===================================================================

tab_names = ["Embedding map", "Style axes", "Nearest neighbors"]
if bundle.has_garments:
    tab_names.append("Garment view")
tab_names.append("Dataset info")

tabs = st.tabs(tab_names)
tab_iter = iter(tabs)

# ---------------------------------------------------------------------------
# TAB 1: Embedding map
# ---------------------------------------------------------------------------

with next(tab_iter):
    if projection is not None:
        import plotly.graph_objects as go

        proj = projection.copy()
        if "caption" in bundle.records.columns:
            proj["caption"] = bundle.records["caption"].values[: len(proj)]
        else:
            proj["caption"] = ""
        proj["caption_short"] = proj["caption"].str[:80]

        is_selected = proj["row_id"] == selected_idx
        nn_indices = {idx for idx, _ in nearest_neighbors(bundle.embeddings, selected_idx, k=k)}
        is_neighbor = proj["row_id"].isin(nn_indices)

        # Merge optional data
        use_axis_color = (
            selected_axis is not None
            and axis_scores is not None
            and not color_by_cluster
        )
        has_clusters = clusters is not None

        if use_axis_color:
            proj = proj.merge(axis_scores[["row_id", selected_axis]], on="row_id", how="left")
            axis_vals = proj[selected_axis].dropna()
            cmin, cmax = float(axis_vals.min()), float(axis_vals.max())
            if cmin < 0 and cmax > 0:
                bound = max(abs(cmin), abs(cmax))
                cmin, cmax = -bound, bound

        if has_clusters:
            proj = proj.merge(clusters, on="row_id", how="left")
            # Build cluster label column
            if "cluster_label" not in proj.columns:
                proj["cluster_label"] = proj["cluster_id"].apply(
                    lambda x: f"Cluster {int(x)}" if pd.notna(x) else ""
                )

        # --- Build hover text ---
        def _hover(row: pd.Series) -> str:
            parts = [str(row["caption_short"])]
            if use_axis_color and selected_axis in row.index and pd.notna(row.get(selected_axis)):
                parts.append(f"{_format_axis_name(selected_axis)}: {row[selected_axis]:.3f}")
            if has_clusters and "cluster_label" in row.index:
                parts.append(str(row["cluster_label"]))
            return "<br>".join(parts)

        fig = go.Figure()
        mask_bg = ~is_selected & ~is_neighbor

        if color_by_cluster and has_clusters:
            # Discrete cluster coloring
            fig.add_trace(
                go.Scatter(
                    x=proj.loc[mask_bg, "x"],
                    y=proj.loc[mask_bg, "y"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=proj.loc[mask_bg, "cluster_id"],
                        colorscale="Turbo",
                        opacity=0.7,
                    ),
                    text=[_hover(r) for _, r in proj.loc[mask_bg].iterrows()],
                    customdata=proj.loc[mask_bg, "row_id"],
                    hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
                    name="Outfits",
                )
            )
        elif use_axis_color:
            fig.add_trace(
                go.Scatter(
                    x=proj.loc[mask_bg, "x"],
                    y=proj.loc[mask_bg, "y"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=proj.loc[mask_bg, selected_axis],
                        colorscale="RdYlBu_r",
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(title=_format_axis_name(selected_axis)),
                        opacity=0.7,
                    ),
                    text=[_hover(r) for _, r in proj.loc[mask_bg].iterrows()],
                    customdata=proj.loc[mask_bg, "row_id"],
                    hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
                    name="Outfits",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=proj.loc[mask_bg, "x"],
                    y=proj.loc[mask_bg, "y"],
                    mode="markers",
                    marker=dict(size=5, color="#94a3b8", opacity=0.45),
                    text=[_hover(r) for _, r in proj.loc[mask_bg].iterrows()],
                    customdata=proj.loc[mask_bg, "row_id"],
                    hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
                    name="Outfits",
                )
            )

        # Neighbor and selected overlays
        fig.add_trace(
            go.Scatter(
                x=proj.loc[is_neighbor, "x"],
                y=proj.loc[is_neighbor, "y"],
                mode="markers",
                marker=dict(size=9, color="#3b82f6", opacity=0.85,
                            line=dict(width=1, color="white")),
                text=[_hover(r) for _, r in proj.loc[is_neighbor].iterrows()],
                customdata=proj.loc[is_neighbor, "row_id"],
                hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
                name="Neighbors",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=proj.loc[is_selected, "x"],
                y=proj.loc[is_selected, "y"],
                mode="markers",
                marker=dict(size=14, color="#ef4444", symbol="star",
                            line=dict(width=1, color="white")),
                text=[_hover(r) for _, r in proj.loc[is_selected].iterrows()],
                customdata=proj.loc[is_selected, "row_id"],
                hovertemplate="%{text}<br>row_id=%{customdata}<extra></extra>",
                name="Selected",
            )
        )

        fig.update_layout(
            xaxis=dict(title="Projection x", showgrid=False, zeroline=False),
            yaxis=dict(title="Projection y", showgrid=False, zeroline=False),
            height=600,
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            dragmode="pan",
            plot_bgcolor="#fafafa",
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Style-space projection of image embeddings. "
            "Proximity reflects cosine similarity in the embedding space."
        )

        # Quick-select below the map
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            new_idx = st.number_input(
                "Selected outfit (row_id)",
                min_value=0,
                max_value=bundle.n_images - 1,
                value=selected_idx,
                step=1,
                key="map_selector",
            )
            if new_idx != st.session_state.selected_idx:
                st.session_state.selected_idx = new_idx
                st.rerun()
        with col_info:
            query_row = bundle.records.iloc[selected_idx]
            cap = query_row.get("caption", "")
            if cap:
                st.markdown(f"**Caption:** {cap[:200]}")
            if "image_outfit_score" in query_row.index:
                st.markdown(f"**Outfit score:** {query_row['image_outfit_score']:.4f}")
    else:
        st.info(
            "No projection found. Run `python scripts/02_build_projection.py <bundle_dir>` "
            "to generate one."
        )


# ---------------------------------------------------------------------------
# TAB 2: Style axes
# ---------------------------------------------------------------------------

with next(tab_iter):
    if not available_axes:
        st.info(
            "No axis scores found. Run `python scripts/05_build_clip_axes.py <bundle_dir>` "
            "to compute style axes."
        )
    elif selected_axis is None:
        st.info("Select a style axis from the sidebar to explore top and bottom examples.")
    else:
        st.header(_format_axis_name(selected_axis))

        pos_prompt, neg_prompt = _axis_prompt_info(manifest, selected_axis)
        if pos_prompt and neg_prompt:
            col_p, col_n = st.columns(2)
            with col_p:
                st.markdown(f"**High end:** {pos_prompt}")
            with col_n:
                st.markdown(f"**Low end:** {neg_prompt}")
            st.caption(
                "CLIP prompt-direction axis — scores reflect cosine similarity "
                "with the positive vs negative prompt direction."
            )
        elif not clip_axes:
            st.caption("Proxy axis — derived from embedding PCA and caption keywords.")

        n_examples = min(5, bundle.n_images // 2) if bundle.n_images >= 4 else bundle.n_images
        top_ids, bottom_ids = top_bottom_indices(axis_scores, selected_axis, n=n_examples)

        parts = selected_axis.split("_vs_")
        high_label = parts[0].replace("_", " ").capitalize() if len(parts) == 2 else "Highest"
        low_label = parts[1].replace("_", " ").capitalize() if len(parts) == 2 else "Lowest"

        col_top, col_bottom = st.columns(2)

        with col_top:
            st.subheader(f"\u2191 {high_label}")
            cols = st.columns(n_examples)
            for j, rid in enumerate(top_ids):
                score_val = float(axis_scores.loc[axis_scores["row_id"] == rid, selected_axis].iloc[0])
                with cols[j]:
                    _show_thumbnail(bundle, rid)
                    st.caption(f"{score_val:+.3f}")

        with col_bottom:
            st.subheader(f"\u2193 {low_label}")
            cols = st.columns(n_examples)
            for j, rid in enumerate(bottom_ids):
                score_val = float(axis_scores.loc[axis_scores["row_id"] == rid, selected_axis].iloc[0])
                with cols[j]:
                    _show_thumbnail(bundle, rid)
                    st.caption(f"{score_val:+.3f}")


# ---------------------------------------------------------------------------
# TAB 3: Nearest neighbors
# ---------------------------------------------------------------------------

with next(tab_iter):
    nn_col_left, nn_col_right = st.columns([1, 3])

    with nn_col_left:
        st.subheader("Query")
        new_idx = st.number_input(
            "Outfit index (row_id)",
            min_value=0,
            max_value=bundle.n_images - 1,
            value=selected_idx,
            step=1,
            key="nn_selector",
        )
        if new_idx != st.session_state.selected_idx:
            st.session_state.selected_idx = new_idx
            st.rerun()

        _show_thumbnail(bundle, selected_idx)

        query_row = bundle.records.iloc[selected_idx]
        cap = query_row.get("caption", "")
        if cap:
            st.caption(cap[:150])

        # Show axis scores compactly
        if axis_scores is not None and available_axes:
            scores_row = axis_scores.loc[axis_scores["row_id"] == selected_idx]
            if not scores_row.empty:
                score_data = {
                    _format_axis_name(a): f"{float(scores_row[a].iloc[0]):+.3f}"
                    for a in available_axes[:6]
                }
                st.markdown("**Axis scores**")
                for name, val in score_data.items():
                    st.caption(f"{name}: {val}")

    with nn_col_right:
        st.subheader(f"Top {k} neighbors")
        neighbors = nearest_neighbors(bundle.embeddings, selected_idx, k=k)
        n_cols = min(k, 4)
        nn_cols = st.columns(n_cols)
        for j, (idx, sim) in enumerate(neighbors):
            with nn_cols[j % n_cols]:
                _show_thumbnail(bundle, idx)
                st.caption(f"sim {sim:.3f}")


# ---------------------------------------------------------------------------
# TAB 4: Garment view (conditional)
# ---------------------------------------------------------------------------

if bundle.has_garments:
    with next(tab_iter):
        st.subheader("Garment decomposition")

        gar_col_left, gar_col_right = st.columns([1, 3])

        with gar_col_left:
            st.markdown("**Selected outfit**")
            _show_thumbnail(bundle, selected_idx)

        with gar_col_right:
            outfit_garments = bundle.garments_for_outfit(selected_idx)
            if outfit_garments.empty:
                st.info("No garments detected for this outfit.")
            else:
                categories = sorted(outfit_garments["category"].unique())
                st.markdown(f"**Detected garments:** {', '.join(categories)}")
                g_cols = st.columns(min(len(outfit_garments), 5))
                for j, (_, g_row) in enumerate(outfit_garments.iterrows()):
                    with g_cols[j % len(g_cols)]:
                        _show_garment_crop(bundle, int(g_row["garment_id"]))
                        conf = g_row.get("confidence", float("nan"))
                        label = g_row["category"]
                        if not np.isnan(conf):
                            st.caption(f"{label} ({conf:.0%})")
                        else:
                            st.caption(label)

        # Cross-category retrieval
        st.markdown("---")
        st.subheader("Cross-category retrieval")

        if bundle.garment_embeddings is None:
            st.info("No garment embeddings found. Run `07_embed_garments.py` to enable retrieval.")
        elif outfit_garments.empty:
            st.info("Select an outfit with detected garments to run retrieval.")
        else:
            query_gid = st.selectbox(
                "Query garment",
                outfit_garments["garment_id"].tolist(),
                format_func=lambda gid: (
                    f"{outfit_garments.loc[outfit_garments['garment_id'] == gid, 'category'].iloc[0]} "
                    f"(id {gid})"
                ),
                key="retrieval_query",
            )

            if st.button("Find compatible items", key="btn_retrieve"):
                results = retrieve_cross_category(
                    query_garment_id=query_gid,
                    garments=bundle.garments,
                    embeddings=bundle.garment_embeddings,
                    k=5,
                )
                if not results:
                    st.warning("No compatible items found.")
                else:
                    for cat, hits in results.items():
                        st.markdown(f"**{cat.capitalize()}**")
                        r_cols = st.columns(min(len(hits), 5))
                        for j, hit in enumerate(hits):
                            with r_cols[j % len(r_cols)]:
                                _show_garment_crop(bundle, hit.garment_id)
                                st.caption(f"sim {hit.similarity:.3f}")


# ---------------------------------------------------------------------------
# TAB 5: Dataset info
# ---------------------------------------------------------------------------

with next(tab_iter):
    info_left, info_right = st.columns(2)

    with info_left:
        st.subheader("Bundle summary")
        stats = {
            "Images": f"{bundle.n_images:,}",
            "Embedding dim": str(bundle.embeddings.shape[1]),
        }
        if projection is not None:
            stats["Projection points"] = f"{len(projection):,}"
        if axis_scores is not None:
            stats["Style axes"] = str(len(available_axes))
        if clusters is not None:
            n_clusters = clusters["cluster_id"].nunique()
            stats["Clusters"] = str(n_clusters)
        if bundle.has_garments:
            stats["Garments"] = f"{bundle.n_garments:,}"
            stats["Garment categories"] = ", ".join(
                sorted(bundle.garments["category"].unique())
            )

        for label, value in stats.items():
            st.markdown(f"**{label}:** {value}")

    with info_right:
        st.subheader("Pipeline details")
        if manifest:
            if "clip_reranking" in manifest:
                cr = manifest["clip_reranking"]
                st.markdown(
                    f"**CLIP reranking:** top {cr.get('n_exported', '?')} "
                    f"of {cr.get('n_candidates', '?')} candidates"
                )
                sd = cr.get("score_distribution", {})
                if sd:
                    st.markdown(
                        f"**Score range:** {sd.get('min', 0):.4f} \u2013 {sd.get('max', 0):.4f} "
                        f"(cutoff: {sd.get('p75', 0):.4f})"
                    )
            fd = manifest.get("filter_diagnostics", {})
            if fd:
                st.markdown(f"**Captions scanned:** {fd.get('scanned', '?'):,}")
                st.markdown(f"**Accept rate:** {fd.get('accept_rate', 0):.1%}")
            if "axis_scores" in manifest:
                ax_info = manifest["axis_scores"]
                method = ax_info.get("method", "unknown")
                method_label = "CLIP prompt direction" if method == "clip_prompt_direction" else method
                st.markdown(f"**Axis method:** {method_label}")
                if ax_info.get("clip_model"):
                    st.markdown(
                        f"**Axis model:** {ax_info['clip_model']}"
                        f"/{ax_info.get('clip_pretrained', '?')}"
                    )
        else:
            st.caption("No manifest.json found in this bundle.")

    st.markdown("---")
    st.caption(
        "Debug subset from LAION-natural. Caption keyword + CLIP image scoring "
        "used for selection. Axes are exploratory prompt directions, not "
        "ground-truth labels. Images are local/private."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="footer-text">'
    "Exploratory prototype \u2014 scores are model-derived, not ground truth."
    "</p>",
    unsafe_allow_html=True,
)
