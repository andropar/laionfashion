"""Style-axis scores for debug bundles.

Axis scores are stored as a DataFrame with ``row_id`` plus one or more numeric
score columns.  Each column represents a style axis — either a proxy derived
from embeddings/captions, or a real prompt-direction score from a contrastive
text encoder.

The API is designed so that proxy axes and real prompt-direction axes share the
same load/save/validate interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_axis_scores(bundle_dir: str | Path) -> pd.DataFrame | None:
    """Load ``axis_scores.parquet`` or ``.csv`` from *bundle_dir*, or return *None*."""
    d = Path(bundle_dir)
    parquet = d / "axis_scores.parquet"
    csv = d / "axis_scores.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def save_axis_scores(
    scores: pd.DataFrame,
    bundle_dir: str | Path,
) -> Path:
    """Write *scores* to ``axis_scores.parquet`` (falling back to CSV) in *bundle_dir*."""
    d = Path(bundle_dir)
    out = d / "axis_scores.parquet"
    try:
        scores.to_parquet(out, index=False)
    except Exception:
        out = d / "axis_scores.csv"
        scores.to_csv(out, index=False)
    return out


def validate_axis_scores(scores: pd.DataFrame, n_images: int) -> None:
    """Raise ``ValueError`` if *scores* is malformed or misaligned with the bundle."""
    if "row_id" not in scores.columns:
        raise ValueError("axis_scores must contain a 'row_id' column")
    axes = axis_names(scores)
    if not axes:
        raise ValueError("axis_scores must contain at least one score column besides 'row_id'")
    if len(scores) != n_images:
        raise ValueError(
            f"Row count mismatch: {len(scores)} axis rows vs {n_images} bundle images"
        )
    expected_ids = set(range(n_images))
    actual_ids = set(scores["row_id"].values)
    if actual_ids != expected_ids:
        raise ValueError(
            f"row_id mismatch: expected {{0..{n_images - 1}}}, "
            f"got {len(actual_ids)} distinct values"
        )


def axis_names(scores: pd.DataFrame) -> list[str]:
    """Return the list of axis column names (everything except ``row_id``)."""
    return [c for c in scores.columns if c != "row_id"]


def top_bottom_indices(
    scores: pd.DataFrame,
    axis: str,
    n: int = 5,
) -> tuple[list[int], list[int]]:
    """Return ``(top_row_ids, bottom_row_ids)`` for *axis*, sorted by score.

    *top* = highest scores (descending), *bottom* = lowest scores (ascending).
    """
    sorted_df = scores.sort_values(axis, ascending=False)
    n = min(n, len(sorted_df))
    top = sorted_df.head(n)["row_id"].tolist()
    bottom = sorted_df.tail(n)["row_id"].tolist()[::-1]  # lowest first
    return top, bottom


# ---------------------------------------------------------------------------
# Demo / proxy axis builders
# ---------------------------------------------------------------------------

# Caption keyword lists for proxy axes.  These are rough heuristics, not
# prompt-direction scores — they are meant to bootstrap the UI until a
# matching contrastive text encoder is available on the server.

_PROXY_KEYWORDS: dict[str, tuple[list[str], float]] = {
    "colorful_proxy": (
        ["colorful", "bright", "vibrant", "neon", "multicolor", "rainbow", "bold color"],
        1.0,
    ),
    "formal_proxy": (
        ["formal", "suit", "blazer", "tie", "business", "office", "elegant", "tuxedo"],
        1.0,
    ),
    "minimal_proxy": (
        ["minimal", "simple", "clean", "basic", "understated", "plain", "monochrome"],
        1.0,
    ),
    "outdoor_proxy": (
        ["outdoor", "hiking", "nature", "park", "garden", "beach", "mountain", "trail"],
        1.0,
    ),
}


def build_demo_axes(
    embeddings: np.ndarray,
    records: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build deterministic proxy axis scores from embeddings and caption metadata.

    Returns a DataFrame with ``row_id`` plus one column per proxy axis.  Scores
    are in [-1, 1] (caption keyword match + embedding PCA component).

    These are **demo/proxy axes** — not real prompt-direction scores.  They exist
    to bootstrap the explorer UI.
    """
    n = len(records)
    rng = np.random.default_rng(random_state)
    result = pd.DataFrame({"row_id": np.arange(n, dtype=int)})

    # PCA components as embedding-based signal
    X = embeddings.astype(np.float64)
    X = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    # Use up to 4 components, cycling if fewer dimensions
    n_components = min(U.shape[1], len(_PROXY_KEYWORDS))
    pca_scores = U[:, :n_components] * S[:n_components]

    captions = records["caption"].fillna("").str.lower() if "caption" in records.columns else pd.Series([""] * n)

    for i, (axis_name, (keywords, _weight)) in enumerate(_PROXY_KEYWORDS.items()):
        # Caption keyword signal: 1.0 if any keyword matches, 0.0 otherwise
        caption_signal = captions.apply(
            lambda c, kw=keywords: 1.0 if any(k in c for k in kw) else 0.0
        ).values

        # Embedding PCA signal (use component i mod n_components)
        comp_idx = i % n_components
        emb_signal = pca_scores[:, comp_idx].copy()
        # Normalize to [-1, 1]
        emb_range = emb_signal.max() - emb_signal.min()
        if emb_range > 1e-12:
            emb_signal = 2 * (emb_signal - emb_signal.min()) / emb_range - 1
        else:
            emb_signal = np.zeros(n)

        # Blend: 0.3 caption + 0.7 embedding (embedding dominates for structure)
        blended = 0.3 * caption_signal + 0.7 * emb_signal
        # Normalize final score to [-1, 1]
        bl_min, bl_max = blended.min(), blended.max()
        bl_range = bl_max - bl_min
        if bl_range > 1e-12:
            blended = 2 * (blended - bl_min) / bl_range - 1
        else:
            blended = np.zeros(n)

        result[axis_name] = blended.astype(np.float32)

    return result
