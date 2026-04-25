"""Embedding clustering and labelling for debug bundles.

Provides KMeans-based clustering, human-readable label generation from axis
scores, and centroid-based exemplar selection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 10,
    method: str = "kmeans",
) -> pd.DataFrame:
    """Cluster *embeddings* and return a DataFrame with ``row_id`` and ``cluster_id``.

    Parameters
    ----------
    embeddings:
        (n, d) float array of image embeddings.
    n_clusters:
        Desired number of clusters.  Automatically clamped to the number of
        samples if fewer samples are available.
    method:
        Clustering algorithm.  Currently only ``"kmeans"`` is supported.

    Returns
    -------
    DataFrame with columns ``row_id`` (int) and ``cluster_id`` (int).
    """
    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method: {method!r}")

    n_samples = embeddings.shape[0]
    effective_k = min(n_clusters, n_samples)
    if effective_k < n_clusters:
        logger.warning(
            "Clamped n_clusters from %d to %d (only %d samples)",
            n_clusters,
            effective_k,
            n_samples,
        )

    km = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings.astype(np.float64))

    return pd.DataFrame({
        "row_id": np.arange(n_samples, dtype=int),
        "cluster_id": labels.astype(int),
    })


def label_clusters(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    axis_scores: pd.DataFrame | None = None,
) -> dict[int, str]:
    """Generate human-readable labels for each cluster.

    When *axis_scores* is provided (a DataFrame with ``row_id`` plus numeric
    axis columns), labels are derived from the axes with the highest mean
    scores within each cluster.  Otherwise, clusters are labelled with their
    numeric id.

    Returns
    -------
    Dict mapping cluster_id to a descriptive label string.
    """
    unique_ids = np.unique(cluster_ids)

    if axis_scores is None:
        return {int(cid): f"Cluster {cid}" for cid in unique_ids}

    # Identify axis columns (everything except row_id)
    axis_cols = [c for c in axis_scores.columns if c != "row_id"]
    if not axis_cols:
        return {int(cid): f"Cluster {cid}" for cid in unique_ids}

    # Build a lookup: row_id -> row index in axis_scores
    scores_arr = axis_scores.set_index("row_id")[axis_cols]

    labels: dict[int, str] = {}
    for cid in unique_ids:
        mask = cluster_ids == cid
        member_ids = np.where(mask)[0]
        # Get mean axis scores for this cluster
        cluster_scores = scores_arr.loc[
            scores_arr.index.isin(member_ids)
        ].mean()
        # Pick top-2 axes by absolute mean score
        top_axes = cluster_scores.abs().nlargest(2).index.tolist()
        # Build a readable label from the axis names
        parts = []
        for ax in top_axes:
            val = cluster_scores[ax]
            # Clean up axis name: remove _proxy, _vs_ -> pick the dominant side
            name = _friendly_axis_name(ax, val)
            parts.append(name)
        labels[int(cid)] = " ".join(parts) if parts else f"Cluster {cid}"

    return labels


def _friendly_axis_name(axis: str, score: float) -> str:
    """Turn an axis name into a short human-readable word.

    Handles patterns like ``colorful_vs_neutral`` (picks side based on sign)
    and ``colorful_proxy`` (strips ``_proxy``).
    """
    name = axis.lower()
    # Handle "X_vs_Y" axes
    if "_vs_" in name:
        parts = name.split("_vs_")
        chosen = parts[0] if score >= 0 else parts[1]
        return chosen.replace("_", " ").capitalize()
    # Strip common suffixes
    for suffix in ("_proxy", "_score"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").capitalize()


def cluster_exemplars(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    n: int = 5,
) -> dict[int, list[int]]:
    """For each cluster, find the *n* images closest to the cluster centroid.

    Parameters
    ----------
    embeddings:
        (n_samples, d) float array.
    cluster_ids:
        Integer cluster assignment per sample (length n_samples).
    n:
        Number of exemplars per cluster.

    Returns
    -------
    Dict mapping cluster_id to a list of row indices (sorted by distance
    to centroid, nearest first).
    """
    unique_ids = np.unique(cluster_ids)
    exemplars: dict[int, list[int]] = {}

    for cid in unique_ids:
        mask = cluster_ids == cid
        member_indices = np.where(mask)[0]
        cluster_emb = embeddings[member_indices].astype(np.float64)
        centroid = cluster_emb.mean(axis=0)

        # Euclidean distance to centroid
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        k = min(n, len(member_indices))
        nearest = np.argsort(dists)[:k]
        exemplars[int(cid)] = [int(member_indices[i]) for i in nearest]

    return exemplars
