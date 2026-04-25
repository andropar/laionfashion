"""Hold-out garment retrieval evaluation.

The core task: for each outfit with >= 2 garments, hold out one garment and
rank candidates from the same category (from other outfits) by embedding
similarity.  The held-out garment's co-occurring items provide ground truth
— do the retrieved candidates come from outfits that "look compatible"?

Since we don't have human compatibility labels, we use a proxy evaluation:
the held-out garment is treated as the ground-truth target, and we measure
whether it ranks highly among all same-category candidates.  This tests
whether the embedding space encodes *co-occurrence structure*, not
necessarily aesthetic compatibility.

Metrics:
- **Recall@K**: fraction of queries where the ground-truth garment is in
  the top K retrieved candidates (not applicable here since the GT garment
  is held out — we instead use a co-occurrence-based proxy, see below).
- **MRR** (mean reciprocal rank): mean of 1/rank for the ground truth.
- **Hit@K**: fraction of queries where any garment from the ground-truth
  outfit appears in the top K.

The evaluation supports plugging in different embedding matrices to compare
CLIP baseline vs. learned embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EvalQuery:
    """A single evaluation query: one garment held out from an outfit."""

    query_garment_id: int
    query_outfit_id: int
    query_category: str
    # Ground truth: other garment_ids from the same outfit (any category)
    gt_outfit_garment_ids: list[int]
    # For same-category evaluation: garment_ids from same outfit, same category
    gt_same_category_ids: list[int]


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    n_queries: int = 0
    # Co-occurrence Hit@K: does any garment from the GT outfit appear in top K?
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    # MRR: mean reciprocal rank of the first GT-outfit garment in results
    mrr: float = 0.0
    # Per-category breakdown
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "n_queries": self.n_queries,
            "hit_at_1": round(self.hit_at_1, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "hit_at_10": round(self.hit_at_10, 4),
            "mrr": round(self.mrr, 4),
        }
        if self.per_category:
            d["per_category"] = {
                cat: {k: round(v, 4) for k, v in metrics.items()}
                for cat, metrics in self.per_category.items()
            }
        return d


def build_eval_queries(
    garments: pd.DataFrame,
    *,
    min_outfit_garments: int = 2,
    target_categories: list[str] | None = None,
) -> list[EvalQuery]:
    """Build evaluation queries by holding out each garment from multi-garment outfits.

    Parameters
    ----------
    garments:
        The garments DataFrame.
    min_outfit_garments:
        Minimum garments in an outfit to include in evaluation.
    target_categories:
        If set, only create queries for these categories (e.g. ["top", "bottom"]).
    """
    queries = []
    for outfit_id, group in garments.groupby("outfit_id"):
        if len(group) < min_outfit_garments:
            continue
        for _, row in group.iterrows():
            category = row["category"]
            if target_categories and category not in target_categories:
                continue
            gid = int(row["garment_id"])
            # Other garments from same outfit (any category)
            others = group[group["garment_id"] != gid]
            gt_outfit_ids = others["garment_id"].tolist()
            # Same category from same outfit
            gt_same_cat = others[others["category"] == category]["garment_id"].tolist()
            queries.append(
                EvalQuery(
                    query_garment_id=gid,
                    query_outfit_id=int(outfit_id),
                    query_category=category,
                    gt_outfit_garment_ids=gt_outfit_ids,
                    gt_same_category_ids=gt_same_cat,
                )
            )
    return queries


def evaluate_retrieval(
    queries: list[EvalQuery],
    garments: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    target_categories: list[str] | None = None,
) -> EvalMetrics:
    """Run hold-out evaluation: for each query, retrieve from other outfits and
    check whether garments from the ground-truth outfit rank highly.

    For each query garment (held out), we retrieve the most similar garments
    from other outfits.  The target is either a specific category or all
    categories except the query's.  A "hit" occurs when any garment from the
    query's original outfit appears in the top K results.

    Parameters
    ----------
    queries:
        List of EvalQuery objects from ``build_eval_queries``.
    garments:
        Full garments DataFrame.
    embeddings:
        (n_garments, dim) L2-normalized embedding matrix.
    k_values:
        K values for hit@K computation.
    target_categories:
        If set, only retrieve from these categories.
    """
    if not queries:
        return EvalMetrics()

    max_k = max(k_values)
    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []

    # Per-category tracking
    cat_hits: dict[str, dict[int, int]] = {}
    cat_rr: dict[str, list[float]] = {}
    cat_counts: dict[str, int] = {}

    for query in queries:
        qid = query.query_garment_id
        query_emb = embeddings[qid]
        gt_set = set(query.gt_outfit_garment_ids)

        # Build candidate pool: other outfits, filtered by category
        mask = (garments["outfit_id"] != query.query_outfit_id)
        if target_categories:
            mask &= garments["category"].isin(target_categories)
        else:
            # Retrieve from categories other than the query's
            mask &= garments["category"] != query.query_category

        candidates = garments[mask]
        if candidates.empty:
            continue

        cand_ids = candidates["garment_id"].values
        cand_embs = embeddings[cand_ids]
        sims = cand_embs @ query_emb
        ranked_indices = np.argsort(sims)[::-1]
        ranked_gids = cand_ids[ranked_indices]

        # Find rank of first GT-outfit garment
        first_rank = None
        for rank, gid in enumerate(ranked_gids, start=1):
            if gid in gt_set:
                first_rank = rank
                break

        rr = 1.0 / first_rank if first_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        for k in k_values:
            top_k_gids = set(ranked_gids[:k])
            if top_k_gids & gt_set:
                hit_counts[k] += 1

        # Per-category tracking
        cat = query.query_category
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_rr.setdefault(cat, []).append(rr)
        for k in k_values:
            cat_hits.setdefault(cat, {}).setdefault(k, 0)
            top_k_gids = set(ranked_gids[:k])
            if top_k_gids & gt_set:
                cat_hits[cat][k] += 1

    n = len(queries)
    metrics = EvalMetrics(
        n_queries=n,
        mrr=float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    )
    for k in k_values:
        rate = hit_counts[k] / n if n > 0 else 0.0
        if k == 1:
            metrics.hit_at_1 = rate
        elif k == 5:
            metrics.hit_at_5 = rate
        elif k == 10:
            metrics.hit_at_10 = rate

    # Per-category breakdown
    for cat in sorted(cat_counts):
        cn = cat_counts[cat]
        cat_metrics = {
            "n_queries": cn,
            "mrr": float(np.mean(cat_rr[cat])) if cat_rr.get(cat) else 0.0,
        }
        for k in k_values:
            cat_metrics[f"hit_at_{k}"] = cat_hits.get(cat, {}).get(k, 0) / cn if cn > 0 else 0.0
        metrics.per_category[cat] = cat_metrics

    return metrics
