from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import optim

    from src.model.diffusion import DiffusionConfig, GaussianDiffusion
    from src.model.unet import TinyConditionalUNet
    from src.train import build_scheduler
    from src.utils.checkpointing import load_checkpoint, save_checkpoint


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for smoke tests")
class SmokeTests(unittest.TestCase):
    def test_training_step_checkpoint_and_sampling(self) -> None:
        model = TinyConditionalUNet(
            in_channels=3,
            base_channels=16,
            channel_multipliers=[1, 2],
            time_embed_dim=32,
            cond_embed_dim=16,
            text_embed_dim=8,
        )
        diffusion = GaussianDiffusion(
            model=model,
            config=DiffusionConfig(
                timesteps=16,
                beta_start=0.0001,
                beta_end=0.02,
                cond_drop_prob=0.0,
            ),
        )
        optimizer = optim.AdamW(diffusion.parameters(), lr=1e-3)
        scheduler = build_scheduler(
            optimizer,
            training_config={"lr_scheduler": "cosine"},
            total_updates=2,
        )

        for _ in range(2):
            images = torch.randn((2, 3, 64, 64))
            cond = torch.randn((2, 8))
            loss = diffusion.training_loss(images, cond)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "smoke.pt"
            save_checkpoint(
                checkpoint_path,
                model_state=diffusion.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                step=2,
                epoch=0,
                config={"seed": 1},
                best_val_loss=float(loss.item()),
            )
            self.assertTrue(checkpoint_path.exists())
            payload = load_checkpoint(checkpoint_path)
            self.assertIn("model_state", payload)

            samples = diffusion.sample(
                batch_size=2,
                image_size=64,
                channels=3,
                text_embeddings=torch.randn((2, 8)),
                guidance_scale=1.0,
                device=torch.device("cpu"),
            )
            self.assertEqual(samples.shape, (2, 3, 64, 64))
            self.assertEqual(samples.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
