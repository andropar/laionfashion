"""Image-side outfit/person scoring using CLIP or a compatible model.

This module provides a protocol-based interface so that:
- The real CLIP scorer loads a model and scores PIL images against prompts.
- Tests can inject a mock scorer without any model dependency.
- The pipeline (debug_export) only needs to call ``scorer.score_image(image)``.

The default scorer uses CLIP ViT-B/32 via open_clip and scores each image
against a set of positive (person-in-outfit) and negative (product-only,
landscape) prompts.  The returned score is the mean cosine similarity with
positive prompts minus the mean cosine similarity with negative prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — any scorer must implement this
# ---------------------------------------------------------------------------


class ImageScorer(Protocol):
    """Protocol for image-side scoring.  Implementations must provide
    ``score_image`` returning a float score."""

    def score_image(self, image: Image.Image) -> float: ...


# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------

POSITIVE_PROMPTS = (
    "a photo of a person wearing an outfit",
    "a full body photo of a person wearing clothes",
    "a street style fashion photo of a person",
    "a person posing in casual clothing",
    "a woman or man wearing a stylish outfit",
)

NEGATIVE_PROMPTS = (
    "a product photo of clothing on a white background",
    "a photo of shoes only without a person",
    "a catalog image of a garment without a person",
    "a landscape or object photo without people",
    "a close-up of fabric or textile texture",
)


# ---------------------------------------------------------------------------
# CLIP outfit scorer
# ---------------------------------------------------------------------------


@dataclass
class CLIPOutfitScorer:
    """Score images for person-in-outfit relevance using CLIP.

    The score is: mean(positive similarities) - mean(negative similarities),
    roughly in [-1, 1].  Higher = more likely to contain a person wearing
    visible clothing.
    """

    model_name: str = "ViT-B-32"
    pretrained: str = "laion400m_e31"

    def __post_init__(self) -> None:
        import open_clip
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, device=self._device
            )
        except Exception:
            logger.warning(
                "Failed to load %s/%s, falling back to %s/openai",
                self.model_name, self.pretrained, self.model_name,
            )
            self.pretrained = "openai"  # type: ignore[misc]
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai", device=self._device
            )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

        # Pre-encode text prompts
        with torch.no_grad():
            pos_tokens = self._tokenizer(list(POSITIVE_PROMPTS)).to(self._device)
            neg_tokens = self._tokenizer(list(NEGATIVE_PROMPTS)).to(self._device)
            self._pos_features = self._model.encode_text(pos_tokens)
            self._neg_features = self._model.encode_text(neg_tokens)
            self._pos_features = self._pos_features / self._pos_features.norm(dim=-1, keepdim=True)
            self._neg_features = self._neg_features / self._neg_features.norm(dim=-1, keepdim=True)

        logger.info(
            "CLIPOutfitScorer ready: %s/%s on %s, %d positive / %d negative prompts",
            self.model_name, self.pretrained, self._device,
            len(POSITIVE_PROMPTS), len(NEGATIVE_PROMPTS),
        )

    def score_image(self, image: Image.Image) -> float:
        """Score a single PIL image.  Returns positive - negative similarity."""
        import torch

        img_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            img_features = self._model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            pos_sim = (img_features @ self._pos_features.T).mean().item()
            neg_sim = (img_features @ self._neg_features.T).mean().item()

        return float(pos_sim - neg_sim)


# ---------------------------------------------------------------------------
# Constant scorer (for testing)
# ---------------------------------------------------------------------------


class ConstantScorer:
    """Always returns the same score.  Useful for testing."""

    def __init__(self, score: float = 1.0) -> None:
        self._score = score

    def score_image(self, image: Image.Image) -> float:
        return self._score


class ThresholdMockScorer:
    """Returns alternating high/low scores.  Useful for testing filtering."""

    def __init__(self, high: float = 0.5, low: float = -0.5) -> None:
        self._high = high
        self._low = low
        self._count = 0

    def score_image(self, image: Image.Image) -> float:
        self._count += 1
        return self._high if self._count % 2 == 1 else self._low


class ListScorer:
    """Returns scores from a pre-defined list, cycling if needed."""

    def __init__(self, scores: list[float]) -> None:
        if not scores:
            raise ValueError("scores must be non-empty")
        self._scores = scores
        self._idx = 0

    def score_image(self, image: Image.Image) -> float:
        score = self._scores[self._idx % len(self._scores)]
        self._idx += 1
        return score
