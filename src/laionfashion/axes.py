"""Style-axis scores for debug bundles.

Axis scores are stored as a DataFrame with ``row_id`` plus one or more numeric
score columns.  Each column represents a style axis — either a proxy derived
from embeddings/captions, or a real prompt-direction score from a contrastive
text encoder.

The API is designed so that proxy axes and real prompt-direction axes share the
same load/save/validate interface.

Prompt-direction axes:
    1. Encode the positive and negative prompts with a CLIP text encoder.
    2. Compute the direction vector: ``normalize(pos - neg)``.
    3. Score each image embedding by dot product with the direction.
    Higher scores → closer to the positive prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Prompt-direction axes (real CLIP axes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptAxis:
    """A style axis defined by positive and negative text prompts."""

    name: str
    positive: str
    negative: str


# Default axes — treat as exploratory, not ground truth.
DEFAULT_PROMPT_AXES: tuple[PromptAxis, ...] = (
    PromptAxis(
        name="minimalist_vs_maximalist",
        positive="a minimalist clean simple outfit, plain colors, few accessories",
        negative="a maximalist layered busy outfit with bold prints and many accessories",
    ),
    PromptAxis(
        name="formal_vs_casual",
        positive="a person in formal professional attire, tailored suit, dress shirt, office wear",
        negative="a person in casual relaxed clothes, t-shirt, sweatpants, loungewear",
    ),
    PromptAxis(
        name="streetwear_vs_classic",
        positive="a person in urban streetwear, oversized hoodie, sneakers, graphic tee, skate style",
        negative="a person in classic traditional clothing, blazer, oxford shoes, preppy style",
    ),
    PromptAxis(
        name="colorful_vs_neutral",
        positive="a colorful vibrant outfit with bright bold saturated colors, red yellow blue green",
        negative="a neutral muted outfit in black white grey beige, monochrome, earth tones",
    ),
    PromptAxis(
        name="polished_vs_rough",
        positive="a polished well-coordinated put-together look, pressed clean fabrics, neat",
        negative="a rough rugged distressed look, worn denim, faded fabrics, grunge style",
    ),
    PromptAxis(
        name="sporty_vs_dressy",
        positive="a person in sporty athletic wear, running shoes, leggings, track jacket, gym clothes",
        negative="a person dressed up for an occasion, cocktail dress, heels, evening wear",
    ),
)


def compute_prompt_directions(
    axes: Sequence[PromptAxis],
    text_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute normalized direction vectors from pre-encoded text embeddings.

    Parameters
    ----------
    axes:
        Prompt axis definitions.
    text_embeddings:
        Dict mapping prompt text to its embedding vector.  Must contain
        entries for every ``axis.positive`` and ``axis.negative``.

    Returns
    -------
    Dict mapping axis name to normalized direction vector.
    """
    directions = {}
    for axis in axes:
        pos = text_embeddings[axis.positive].astype(np.float64)
        neg = text_embeddings[axis.negative].astype(np.float64)
        direction = pos - neg
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm
        directions[axis.name] = direction.astype(np.float32)
    return directions


def score_embeddings_on_axes(
    image_embeddings: np.ndarray,
    directions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Score image embeddings against prompt-direction axes.

    Parameters
    ----------
    image_embeddings:
        (n, d) float array of L2-normalized image embeddings.
    directions:
        Dict mapping axis name to (d,) direction vector from
        ``compute_prompt_directions``.

    Returns
    -------
    DataFrame with ``row_id`` plus one column per axis (dot product scores).
    """
    n = image_embeddings.shape[0]
    result = pd.DataFrame({"row_id": np.arange(n, dtype=int)})
    emb = image_embeddings.astype(np.float32)
    for axis_name, direction in directions.items():
        scores = emb @ direction.astype(np.float32)
        result[axis_name] = scores
    return result


def encode_texts_with_clip(
    texts: list[str],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion400m_e31",
) -> dict[str, np.ndarray]:
    """Encode a list of texts with a CLIP model.

    Returns a dict mapping each text to its L2-normalized embedding.
    Requires ``open_clip`` and ``torch``.
    """
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    except Exception:
        logger.warning("Failed to load %s/%s, falling back to openai", model_name, pretrained)
        pretrained = "openai"
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=device
        )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        features_np = features.cpu().numpy()
        for text, vec in zip(texts, features_np):
            embeddings[text] = vec

    return embeddings


def encode_images_with_clip(
    image_paths: list[Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion400m_e31",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode images with a CLIP model.

    Returns an (n, d) L2-normalized embedding matrix.
    Requires ``open_clip`` and ``torch``.
    """
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    except Exception:
        logger.warning("Failed to load %s/%s, falling back to openai", model_name, pretrained)
        pretrained = "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=device
        )
    model.eval()

    all_features = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


def build_clip_axes(
    *,
    image_embeddings: np.ndarray,
    text_embeddings: dict[str, np.ndarray],
    axes: Sequence[PromptAxis] | None = None,
) -> pd.DataFrame:
    """Build prompt-direction axis scores from pre-encoded CLIP embeddings.

    This is the main entry point for computing real style axes.  It expects
    that both image and text embeddings come from the same CLIP model.

    Parameters
    ----------
    image_embeddings:
        (n, d) L2-normalized CLIP image embeddings.
    text_embeddings:
        Dict mapping prompt texts to L2-normalized CLIP text embeddings.
    axes:
        Prompt axis definitions.  Defaults to :data:`DEFAULT_PROMPT_AXES`.

    Returns
    -------
    DataFrame with ``row_id`` plus one column per axis.
    """
    if axes is None:
        axes = DEFAULT_PROMPT_AXES
    directions = compute_prompt_directions(axes, text_embeddings)
    return score_embeddings_on_axes(image_embeddings, directions)
