"""Failure case detection and reporting for debug bundles.

Provides a framework for identifying common quality issues: missing garments,
low outfit scores, outlier embeddings, and small clusters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from laionfashion.bundle import DebugBundle

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """A detected quality issue in a bundle."""

    category: str
    description: str
    example_row_ids: list[int] = field(default_factory=list)
    severity: str = "warning"  # "info", "warning", "error"


def detect_common_failures(bundle: "DebugBundle") -> list[FailureCase]:
    """Analyse a loaded bundle and return a list of detected failure cases.

    Checks performed (each is skipped if the required data is absent):

    1. Images with no detected garments (requires ``garments.parquet``).
    2. Images with very low CLIP outfit scores (requires ``image_outfit_score``
       column in ``records``).
    3. Outlier embeddings far from all cluster centroids.
    4. Very small clusters (fewer than 3 images).
    """
    failures: list[FailureCase] = []

    # 1. No detected garments
    failures.extend(_check_no_garments(bundle))

    # 2. Low outfit scores
    failures.extend(_check_low_outfit_scores(bundle))

    # 3. Outlier embeddings
    failures.extend(_check_outlier_embeddings(bundle))

    # 4. Small clusters
    failures.extend(_check_small_clusters(bundle))

    return failures


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_no_garments(bundle: "DebugBundle") -> list[FailureCase]:
    """Detect images that have no garment detections."""
    if not bundle.has_garments:
        return []

    garments = bundle.garments
    assert garments is not None
    outfit_ids_with_garments = set(garments["outfit_id"].unique())
    all_row_ids = set(bundle.records["row_id"].values)
    missing = sorted(all_row_ids - outfit_ids_with_garments)

    if not missing:
        return []

    return [
        FailureCase(
            category="no_garments",
            description=(
                f"{len(missing)} image(s) have no detected garments. "
                "These may be non-fashion images or detection failures."
            ),
            example_row_ids=missing[:20],
            severity="warning",
        )
    ]


def _check_low_outfit_scores(bundle: "DebugBundle") -> list[FailureCase]:
    """Detect images with very low CLIP outfit scores."""
    if "image_outfit_score" not in bundle.records.columns:
        return []

    scores = bundle.records[["row_id", "image_outfit_score"]].dropna(
        subset=["image_outfit_score"]
    )
    if scores.empty:
        return []

    # Flag images below the 5th percentile *or* below the absolute threshold
    # of 0.15, whichever catches more — we want to surface clearly bad images.
    p5 = scores["image_outfit_score"].quantile(0.05)
    threshold = max(p5, 0.15)
    low = scores[scores["image_outfit_score"] <= threshold]

    if low.empty:
        return []

    row_ids = low["row_id"].tolist()
    return [
        FailureCase(
            category="low_outfit_score",
            description=(
                f"{len(row_ids)} image(s) have very low outfit scores "
                f"(<= {threshold:.3f}). They may not depict fashion content."
            ),
            example_row_ids=row_ids[:20],
            severity="warning",
        )
    ]


def _check_outlier_embeddings(bundle: "DebugBundle") -> list[FailureCase]:
    """Detect embeddings that are far from all cluster centroids.

    Uses a simple KMeans with k=min(8, n) and flags points whose distance to
    their nearest centroid exceeds mean + 2*std of all distances.
    """
    from sklearn.cluster import KMeans

    n = bundle.n_images
    if n < 4:
        return []

    k = min(8, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(bundle.embeddings.astype(np.float64))
    centroids = km.cluster_centers_

    # Distance of each point to its assigned centroid
    dists = np.linalg.norm(
        bundle.embeddings - centroids[labels], axis=1
    )
    mean_dist = dists.mean()
    std_dist = dists.std()
    threshold = mean_dist + 2 * std_dist

    outlier_mask = dists > threshold
    outlier_ids = np.where(outlier_mask)[0].tolist()

    if not outlier_ids:
        return []

    return [
        FailureCase(
            category="outlier_embedding",
            description=(
                f"{len(outlier_ids)} image(s) have embeddings far from any "
                f"cluster centroid (distance > {threshold:.4f}). They may be "
                "unusual or corrupted images."
            ),
            example_row_ids=outlier_ids[:20],
            severity="info",
        )
    ]


def _check_small_clusters(bundle: "DebugBundle") -> list[FailureCase]:
    """Detect very small clusters (fewer than 3 members)."""
    from sklearn.cluster import KMeans

    n = bundle.n_images
    if n < 4:
        return []

    k = min(8, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(bundle.embeddings.astype(np.float64))

    unique, counts = np.unique(labels, return_counts=True)
    small = [(int(uid), int(cnt)) for uid, cnt in zip(unique, counts) if cnt < 3]

    if not small:
        return []

    # Collect member row_ids from small clusters
    member_ids: list[int] = []
    for cid, _ in small:
        member_ids.extend(np.where(labels == cid)[0].tolist())

    return [
        FailureCase(
            category="small_cluster",
            description=(
                f"{len(small)} cluster(s) have fewer than 3 images. "
                "These may represent niche styles or data issues."
            ),
            example_row_ids=member_ids[:20],
            severity="info",
        )
    ]


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_failure_report(failures: list[FailureCase]) -> str:
    """Format a list of failure cases as a Markdown report."""
    if not failures:
        return "# Failure Report\n\nNo issues detected.\n"

    lines = ["# Failure Report", ""]
    severity_order = {"error": 0, "warning": 1, "info": 2}
    sorted_failures = sorted(
        failures, key=lambda f: severity_order.get(f.severity, 99)
    )

    for fc in sorted_failures:
        icon = {"error": "[ERROR]", "warning": "[WARNING]", "info": "[INFO]"}.get(
            fc.severity, "[?]"
        )
        lines.append(f"## {icon} {fc.category}")
        lines.append("")
        lines.append(fc.description)
        lines.append("")
        if fc.example_row_ids:
            ids_str = ", ".join(str(r) for r in fc.example_row_ids[:10])
            lines.append(f"Example row IDs: {ids_str}")
            if len(fc.example_row_ids) > 10:
                lines.append(f"  ... and {len(fc.example_row_ids) - 10} more")
            lines.append("")

    return "\n".join(lines)
