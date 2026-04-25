"""FAISS-based vector search for fashion embeddings.

Provides cosine similarity search via inner-product on L2-normalized vectors.
Falls back gracefully when faiss is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False


def _require_faiss() -> None:
    if not FAISS_AVAILABLE:
        raise ImportError(
            "faiss is required for vector search but is not installed. "
            "Install it with: pip install faiss-cpu"
        )


def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """Build an inner-product FAISS index from L2-normalized embeddings.

    Since the embeddings are L2-normalized, inner-product search is
    equivalent to cosine similarity.

    Parameters
    ----------
    embeddings:
        2-D float32 array of shape (n, d), assumed to be L2-normalized.

    Returns
    -------
    faiss.IndexFlatIP
        A trained, populated FAISS index.
    """
    _require_faiss()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def search_index(
    index: "faiss.IndexFlatIP",
    query_embedding: np.ndarray,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Search the index for the k nearest neighbours.

    Parameters
    ----------
    index:
        A FAISS inner-product index.
    query_embedding:
        1-D or 2-D float32 array (single query vector).
    k:
        Number of results to return.

    Returns
    -------
    list[tuple[int, float]]
        (index, similarity) pairs sorted by descending similarity.
    """
    _require_faiss()
    query = np.ascontiguousarray(query_embedding, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    # Clamp k to the number of vectors in the index.
    k = min(k, index.ntotal)
    distances, indices = index.search(query, k)
    return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]


def save_index(index: "faiss.IndexFlatIP", path: str | Path) -> None:
    """Serialize a FAISS index to disk."""
    _require_faiss()
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: str | Path) -> "faiss.IndexFlatIP":
    """Deserialize a FAISS index from disk."""
    _require_faiss()
    index = faiss.read_index(str(path))
    logger.info("Loaded FAISS index from %s (%d vectors)", path, index.ntotal)
    return index
