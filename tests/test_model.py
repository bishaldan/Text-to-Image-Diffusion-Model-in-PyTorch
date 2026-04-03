from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from src.model.text_encoder import FrozenTextEncoder, TextEncoderConfig
    from src.model.unet import TinyConditionalUNet


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for model tests")
class ModelTests(unittest.TestCase):
    def test_text_encoder_wrapper_returns_expected_shape(self) -> None:
        class DummyTokenizer:
            def __call__(self, captions, **_: object):
                batch_size = len(captions)
                return {
                    "input_ids": torch.ones((batch_size, 5), dtype=torch.long),
                    "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
                }

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(hidden_size=16)

            def forward(self, **kwargs: object) -> SimpleNamespace:
                input_ids = kwargs["input_ids"]
                batch_size, seq_len = input_ids.shape
                hidden = torch.randn((batch_size, seq_len, 16))
                return SimpleNamespace(last_hidden_state=hidden)

        with patch("src.model.text_encoder.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()):
            with patch("src.model.text_encoder.AutoModel.from_pretrained", return_value=DummyModel()):
                encoder = FrozenTextEncoder(
                    TextEncoderConfig(model_name="dummy-model", max_length=12)
                )
                embeddings = encoder(["red circle", "blue square"], device=torch.device("cpu"))

        self.assertEqual(embeddings.shape, (2, 16))

    def test_unet_forward_preserves_image_shape(self) -> None:
        model = TinyConditionalUNet(
            in_channels=3,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            time_embed_dim=64,
            cond_embed_dim=32,
            text_embed_dim=24,
        )
        images = torch.randn((2, 3, 64, 64))
        timesteps = torch.randint(0, 100, (2,))
        cond = torch.randn((2, 24))
        output = model(images, timesteps, cond)
        self.assertEqual(output.shape, images.shape)

    def test_conditioning_path_accepts_batched_inputs(self) -> None:
        model = TinyConditionalUNet(
            in_channels=3,
            base_channels=16,
            channel_multipliers=[1, 2],
            time_embed_dim=32,
            cond_embed_dim=16,
            text_embed_dim=8,
        )
        images = torch.randn((3, 3, 64, 64))
        timesteps = torch.tensor([1, 5, 10], dtype=torch.long)
        first = model(images, timesteps, torch.zeros((3, 8)))
        second = model(images, timesteps, torch.ones((3, 8)))
        self.assertEqual(first.shape, second.shape)
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()

