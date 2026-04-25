"""Cross-category garment retrieval by cosine similarity.

Given a query garment (e.g. a top), retrieve the nearest garments from a
different category (e.g. bottoms, shoes) using precomputed CLIP embeddings.

This is the baseline compatibility retrieval — it uses frozen CLIP features
and cosine similarity.  The expectation is that this will be mediocre
(CLIP wasn't trained for cross-category compatibility), providing a baseline
to compare against learned embeddings later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval hit."""

    garment_id: int
    outfit_id: int
    category: str
    similarity: float
    crop_path: str


def retrieve_similar_garments(
    *,
    query_garment_id: int,
    garments: pd.DataFrame,
    embeddings: np.ndarray,
    target_category: str | None = None,
    exclude_same_outfit: bool = True,
    k: int = 10,
) -> list[RetrievalResult]:
    """Retrieve garments most similar to a query garment by cosine similarity.

    Parameters
    ----------
    query_garment_id:
        The garment_id of the query crop.
    garments:
        The garments DataFrame (must have garment_id, outfit_id, category, crop_path).
    embeddings:
        (n_garments, dim) L2-normalized embedding matrix.  Row *i* corresponds
        to garment_id *i*.
    target_category:
        If set, only retrieve garments from this category (e.g. "bottom",
        "shoes").  If *None*, retrieve from all categories except the query's.
    exclude_same_outfit:
        If *True* (default), exclude garments from the same outfit as the query.
    k:
        Number of results to return.

    Returns
    -------
    List of :class:`RetrievalResult` sorted by descending similarity.
    """
    query_row = garments.loc[garments["garment_id"] == query_garment_id]
    if query_row.empty:
        raise KeyError(f"garment_id {query_garment_id} not found")
    query_row = query_row.iloc[0]
    query_category = query_row["category"]
    query_outfit_id = int(query_row["outfit_id"])

    query_emb = embeddings[query_garment_id]

    # Build candidate mask
    mask = pd.Series(True, index=garments.index)
    # Exclude the query itself
    mask &= garments["garment_id"] != query_garment_id
    # Category filter
    if target_category is not None:
        mask &= garments["category"] == target_category
    else:
        mask &= garments["category"] != query_category
    # Exclude same outfit
    if exclude_same_outfit:
        mask &= garments["outfit_id"] != query_outfit_id

    candidates = garments[mask]
    if candidates.empty:
        return []

    candidate_ids = candidates["garment_id"].values
    candidate_embs = embeddings[candidate_ids]

    # Cosine similarity (embeddings are L2-normalized)
    sims = candidate_embs @ query_emb
    top_k = min(k, len(sims))
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = candidates.iloc[idx]
        results.append(
            RetrievalResult(
                garment_id=int(row["garment_id"]),
                outfit_id=int(row["outfit_id"]),
                category=str(row["category"]),
                similarity=float(sims[idx]),
                crop_path=str(row["crop_path"]),
            )
        )
    return results


def retrieve_cross_category(
    *,
    query_garment_id: int,
    garments: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 5,
    exclude_same_outfit: bool = True,
) -> dict[str, list[RetrievalResult]]:
    """Retrieve nearest garments from each other category.

    Returns a dict mapping category name to list of results.
    """
    query_row = garments.loc[garments["garment_id"] == query_garment_id]
    if query_row.empty:
        raise KeyError(f"garment_id {query_garment_id} not found")
    query_category = query_row.iloc[0]["category"]

    other_categories = sorted(
        c for c in garments["category"].unique() if c != query_category
    )

    results = {}
    for cat in other_categories:
        hits = retrieve_similar_garments(
            query_garment_id=query_garment_id,
            garments=garments,
            embeddings=embeddings,
            target_category=cat,
            exclude_same_outfit=exclude_same_outfit,
            k=k,
        )
        if hits:
            results[cat] = hits
    return results
