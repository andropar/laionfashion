"""2D projection of embedding matrices for style-space visualization."""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# UMAP needs at least this many samples to be meaningful.
# Below this threshold we fall back to PCA (or trivial layout for n<=2).
MIN_SAMPLES_FOR_UMAP = 15


class ProjectionMethod(str, Enum):
    UMAP = "umap"
    PCA = "pca"
    TRIVIAL = "trivial"


def project_embeddings(
    embeddings: np.ndarray,
    *,
    method: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, ProjectionMethod]:
    """Project *embeddings* (n, d) to 2D and return a DataFrame with columns ``row_id, x, y``.

    Parameters
    ----------
    embeddings:
        Float array of shape ``(n, d)``.
    method:
        Force ``"umap"``, ``"pca"``, or ``"trivial"``.  When *None* (default),
        the function picks automatically: UMAP if available and ``n >= MIN_SAMPLES_FOR_UMAP``,
        PCA if ``n >= 3``, else a deterministic trivial layout.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    (df, method_used):
        *df* has columns ``row_id`` (int), ``x`` (float), ``y`` (float).
        *method_used* indicates which algorithm ran.
    """
    n = embeddings.shape[0]

    if method is not None:
        chosen = ProjectionMethod(method)
    else:
        chosen = _auto_select(n)

    if chosen == ProjectionMethod.UMAP:
        coords = _project_umap(embeddings, random_state=random_state)
    elif chosen == ProjectionMethod.PCA:
        coords = _project_pca(embeddings, random_state=random_state)
    else:
        coords = _project_trivial(n, random_state=random_state)

    df = pd.DataFrame(
        {"row_id": np.arange(n, dtype=int), "x": coords[:, 0], "y": coords[:, 1]}
    )
    return df, chosen


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_select(n: int) -> ProjectionMethod:
    if n < 3:
        return ProjectionMethod.TRIVIAL
    if n >= MIN_SAMPLES_FOR_UMAP and _umap_available():
        return ProjectionMethod.UMAP
    return ProjectionMethod.PCA


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except ImportError:
        return False


def _project_umap(embeddings: np.ndarray, *, random_state: int) -> np.ndarray:
    import umap

    n = embeddings.shape[0]
    n_neighbors = min(15, n - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings.astype(np.float32))
    logger.info("UMAP projection: %d points, n_neighbors=%d", n, n_neighbors)
    return coords


def _project_pca(embeddings: np.ndarray, *, random_state: int) -> np.ndarray:
    """PCA via SVD — deterministic, no extra dependencies."""
    X = embeddings.astype(np.float64)
    X = X - X.mean(axis=0)
    # Use full SVD for small matrices, truncated for larger ones
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    return coords.astype(np.float32)


def _project_trivial(n: int, *, random_state: int) -> np.ndarray:
    """Deterministic layout for very small bundles (1–2 images)."""
    rng = np.random.default_rng(random_state)
    return rng.standard_normal((n, 2)).astype(np.float32)
