"""Tests for laionfashion.models — multi-model embedding support."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from laionfashion.models import (
    AVAILABLE_MODELS,
    ModelConfig,
    encode_images,
    encode_texts,
    load_clip_model,
)


def _mock_open_clip_and_torch():
    """Create mock open_clip and torch modules and inject into sys.modules."""
    mock_open_clip = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    return mock_open_clip, mock_torch


# ---------------------------------------------------------------------------
# ModelConfig dataclass
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_fields(self):
        cfg = ModelConfig(
            name="test-model",
            model_name="ViT-Test",
            pretrained="test_weights",
            description="A test model",
        )
        assert cfg.name == "test-model"
        assert cfg.model_name == "ViT-Test"
        assert cfg.pretrained == "test_weights"
        assert cfg.description == "A test model"

    def test_frozen(self):
        cfg = ModelConfig("a", "b", "c", "d")
        with pytest.raises(AttributeError):
            cfg.name = "other"  # type: ignore[misc]

    def test_equality(self):
        a = ModelConfig("x", "y", "z", "w")
        b = ModelConfig("x", "y", "z", "w")
        assert a == b


# ---------------------------------------------------------------------------
# AVAILABLE_MODELS registry
# ---------------------------------------------------------------------------


class TestAvailableModels:
    def test_has_clip_vit_b_32(self):
        assert "clip-vit-b-32" in AVAILABLE_MODELS
        cfg = AVAILABLE_MODELS["clip-vit-b-32"]
        assert cfg.model_name == "ViT-B-32"
        assert cfg.pretrained == "laion400m_e31"

    def test_has_fashionclip(self):
        assert "fashionclip" in AVAILABLE_MODELS
        cfg = AVAILABLE_MODELS["fashionclip"]
        assert "hf-hub:" in cfg.model_name
        assert cfg.pretrained == ""

    def test_all_configs_are_model_config(self):
        for key, cfg in AVAILABLE_MODELS.items():
            assert isinstance(cfg, ModelConfig), f"{key} is not a ModelConfig"
            assert cfg.name == key, f"Config name '{cfg.name}' != dict key '{key}'"


# ---------------------------------------------------------------------------
# load_clip_model (mocked — no real weights)
# ---------------------------------------------------------------------------


class TestLoadClipModel:
    def test_load_by_name(self):
        mock_oc, mock_torch = _mock_open_clip_and_torch()
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_tokenizer = MagicMock()
        mock_oc.get_tokenizer.return_value = mock_tokenizer

        with patch.dict(sys.modules, {"open_clip": mock_oc, "torch": mock_torch}):
            result = load_clip_model("clip-vit-b-32")

        assert result == (mock_model, mock_preprocess, mock_tokenizer, "cpu")
        mock_model.eval.assert_called_once()

    def test_load_by_config(self):
        mock_oc, mock_torch = _mock_open_clip_and_torch()
        mock_model = MagicMock()
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, MagicMock())
        mock_oc.get_tokenizer.return_value = MagicMock()

        cfg = ModelConfig("custom", "ViT-L-14", "openai", "Custom model")
        with patch.dict(sys.modules, {"open_clip": mock_oc, "torch": mock_torch}):
            result = load_clip_model(cfg)

        assert result[3] == "cpu"
        mock_oc.create_model_and_transforms.assert_called_once_with(
            "ViT-L-14", pretrained="openai", device="cpu"
        )

    def test_load_hf_hub_model(self):
        mock_oc, mock_torch = _mock_open_clip_and_torch()
        mock_model = MagicMock()
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, MagicMock())
        mock_oc.get_tokenizer.return_value = MagicMock()

        with patch.dict(sys.modules, {"open_clip": mock_oc, "torch": mock_torch}):
            result = load_clip_model("fashionclip")

        # hf-hub models should not pass pretrained kwarg
        mock_oc.create_model_and_transforms.assert_called_once_with(
            "hf-hub:Marqo/marqo-fashionCLIP", device="cpu"
        )

    def test_load_unknown_name_raises(self):
        with pytest.raises(KeyError):
            load_clip_model("nonexistent-model")


# ---------------------------------------------------------------------------
# encode_images / encode_texts (mocked)
# ---------------------------------------------------------------------------


class TestEncoding:
    def _make_model_tuple(self):
        """Build a mock model tuple that returns deterministic embeddings."""
        import torch

        dim = 8
        model = MagicMock()

        def fake_encode_image(batch):
            n = batch.shape[0]
            out = torch.randn(n, dim)
            out = out / out.norm(dim=-1, keepdim=True)
            return out

        model.encode_image = MagicMock(side_effect=fake_encode_image)

        def fake_encode_text(tokens):
            n = tokens.shape[0]
            out = torch.randn(n, dim)
            out = out / out.norm(dim=-1, keepdim=True)
            return out

        model.encode_text = MagicMock(side_effect=fake_encode_text)

        def fake_preprocess(img):
            return torch.randn(3, 32, 32)

        preprocess = MagicMock(side_effect=fake_preprocess)

        def fake_tokenizer(texts):
            return torch.zeros(len(texts), 10, dtype=torch.long)

        tokenizer = MagicMock(side_effect=fake_tokenizer)

        return (model, preprocess, tokenizer, "cpu")

    def test_encode_images_shape(self):
        from PIL import Image

        model_tuple = self._make_model_tuple()
        imgs = [Image.new("RGB", (32, 32)) for _ in range(5)]
        result = encode_images(model_tuple, imgs, batch_size=2)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5
        assert result.dtype == np.float32

    def test_encode_images_normalized(self):
        from PIL import Image

        model_tuple = self._make_model_tuple()
        imgs = [Image.new("RGB", (32, 32)) for _ in range(3)]
        result = encode_images(model_tuple, imgs)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_texts_returns_dict(self):
        model_tuple = self._make_model_tuple()
        texts = ["a cat", "a dog"]
        result = encode_texts(model_tuple, texts)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"a cat", "a dog"}
        for vec in result.values():
            assert isinstance(vec, np.ndarray)
            assert vec.dtype == np.float32
