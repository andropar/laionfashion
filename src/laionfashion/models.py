"""Multi-model embedding support for comparing CLIP variants.

Provides a unified interface for loading and using different CLIP-family
models (e.g., OpenCLIP ViT-B/32, Marqo FashionCLIP) so that embeddings
from multiple models can be compared side-by-side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a CLIP-family model."""

    name: str
    model_name: str
    pretrained: str
    description: str


# Registry of available models.
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "clip-vit-b-32": ModelConfig(
        name="clip-vit-b-32",
        model_name="ViT-B-32",
        pretrained="laion400m_e31",
        description="OpenCLIP ViT-B/32",
    ),
    "fashionclip": ModelConfig(
        name="fashionclip",
        model_name="hf-hub:Marqo/marqo-fashionCLIP",
        pretrained="",
        description="Marqo FashionCLIP",
    ),
}

# Type alias for the loaded model tuple.
ModelTuple = Tuple  # (model, preprocess, tokenizer, device)


def load_clip_model(
    config_or_name: Union[ModelConfig, str],
) -> tuple:
    """Load a CLIP model and return ``(model, preprocess, tokenizer, device)``.

    Parameters
    ----------
    config_or_name:
        Either a :class:`ModelConfig` instance or a string key into
        :data:`AVAILABLE_MODELS`.

    Returns
    -------
    tuple of (model, preprocess, tokenizer, device)
    """
    if isinstance(config_or_name, str):
        config = AVAILABLE_MODELS[config_or_name]
    else:
        config = config_or_name

    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.model_name
    pretrained = config.pretrained

    # For hf-hub models the pretrained arg is ignored by open_clip
    # (the hub path already encodes the weights).
    if model_name.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
        except Exception:
            logger.warning(
                "Failed to load %s/%s, falling back to openai",
                model_name,
                pretrained,
            )
            pretrained = "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", device=device
            )
        tokenizer = open_clip.get_tokenizer(model_name)

    model.eval()
    logger.info(
        "Loaded %s (%s) on %s", config.name, config.description, device
    )
    return (model, preprocess, tokenizer, device)


def encode_images(
    model_tuple: tuple,
    images: Sequence[Image.Image],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode PIL images and return an (n, d) L2-normalized embedding matrix.

    Parameters
    ----------
    model_tuple:
        The tuple returned by :func:`load_clip_model`.
    images:
        Sequence of PIL images to encode.
    batch_size:
        Number of images per forward pass.

    Returns
    -------
    np.ndarray of shape ``(len(images), embedding_dim)`` with dtype float32.
    """
    import torch

    model, preprocess, _tokenizer, device = model_tuple

    all_features: List[np.ndarray] = []
    for start in tqdm(
        range(0, len(images), batch_size), desc="Encoding images"
    ):
        batch_images = images[start : start + batch_size]
        tensors = [preprocess(img) for img in batch_images]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


def encode_texts(
    model_tuple: tuple,
    texts: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Encode texts and return a dict mapping each text to its L2-normalized embedding.

    Parameters
    ----------
    model_tuple:
        The tuple returned by :func:`load_clip_model`.
    texts:
        List of prompt strings to encode.

    Returns
    -------
    Dict mapping each input text to its (d,) float32 embedding vector.
    """
    import torch

    model, _preprocess, tokenizer, device = model_tuple

    embeddings: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        tokens = tokenizer(list(texts)).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        features_np = features.cpu().numpy()
        for text, vec in zip(texts, features_np):
            embeddings[text] = vec.astype(np.float32)

    return embeddings
