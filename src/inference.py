from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.model.diffusion import DiffusionConfig, GaussianDiffusion
from src.model.text_encoder import FrozenTextEncoder, TextEncoderConfig
from src.model.unet import TinyConditionalUNet
from src.utils.checkpointing import load_checkpoint
from src.utils.image_utils import tensor_to_pil


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class CheckpointSummary:
    run_id: str
    checkpoint_path: str
    config_path: str | None
    training_prompts: list[str]
    guidance_scale: float
    image_size: int
    timesteps: int


class InferenceService:
    def __init__(self, checkpoints_root: str | Path = "checkpoints") -> None:
        self.checkpoints_root = Path(checkpoints_root)
        self.device = choose_device()
        self._loaded_checkpoint_path: Path | None = None
        self._payload: dict[str, Any] | None = None
        self._diffusion: GaussianDiffusion | None = None
        self._text_encoder: FrozenTextEncoder | None = None

    def list_checkpoints(self) -> list[CheckpointSummary]:
        checkpoint_paths = sorted(self.checkpoints_root.glob("*/best.pt"), reverse=True)
        summaries: list[CheckpointSummary] = []
        for path in checkpoint_paths:
            try:
                payload = load_checkpoint(path, map_location="cpu")
                config = payload["config"]
                run_id = path.parent.name
                run_config = Path(config["outputs"]["runs_dir"]) / run_id / "config.snapshot.yaml"
                summaries.append(
                    CheckpointSummary(
                        run_id=run_id,
                        checkpoint_path=str(path),
                        config_path=str(run_config) if run_config.exists() else None,
                        training_prompts=list(config["training"].get("sample_prompts", [])),
                        guidance_scale=float(config["training"]["guidance_scale"]),
                        image_size=int(config["dataset"]["image_size"]),
                        timesteps=int(config["model"]["timesteps"]),
                    )
                )
            except Exception:
                continue
        return summaries

    def _ensure_loaded(self, checkpoint_path: str | Path) -> dict[str, Any]:
        checkpoint = Path(checkpoint_path)
        if self._loaded_checkpoint_path == checkpoint and self._payload is not None:
            return self._payload

        payload = load_checkpoint(checkpoint, map_location=self.device)
        config = payload["config"]

        text_encoder = FrozenTextEncoder(
            TextEncoderConfig(
                model_name=config["model"]["text_encoder_name"],
                max_length=config["model"]["text_max_length"],
            )
        ).to(self.device)
        unet = TinyConditionalUNet(
            in_channels=config["model"]["in_channels"],
            base_channels=config["model"]["base_channels"],
            channel_multipliers=config["model"]["channel_multipliers"],
            time_embed_dim=config["model"]["time_embed_dim"],
            cond_embed_dim=config["model"]["cond_embed_dim"],
            text_embed_dim=text_encoder.output_dim,
        )
        diffusion = GaussianDiffusion(
            model=unet,
            config=DiffusionConfig(
                timesteps=config["model"]["timesteps"],
                beta_start=config["model"]["beta_start"],
                beta_end=config["model"]["beta_end"],
                cond_drop_prob=config["model"].get("cond_drop_prob", 0.0),
            ),
        ).to(self.device)
        diffusion.load_state_dict(payload["model_state"])
        diffusion.eval()

        self._loaded_checkpoint_path = checkpoint
        self._payload = payload
        self._text_encoder = text_encoder
        self._diffusion = diffusion
        return payload

    @torch.no_grad()
    def generate(
        self,
        *,
        checkpoint_path: str | Path,
        prompt: str,
        guidance_scale: float | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        payload = self._ensure_loaded(checkpoint_path)
        assert self._text_encoder is not None
        assert self._diffusion is not None

        config = payload["config"]
        effective_guidance = guidance_scale or float(config["training"]["guidance_scale"])
        text_embeddings = self._text_encoder([prompt], device=self.device)
        images = self._diffusion.sample(
            batch_size=1,
            image_size=config["dataset"]["image_size"],
            channels=config["model"]["in_channels"],
            text_embeddings=text_embeddings,
            guidance_scale=effective_guidance,
            device=self.device,
        )
        image = tensor_to_pil(images[0])
        meta = {
            "checkpoint_path": str(checkpoint_path),
            "guidance_scale": effective_guidance,
            "run_id": Path(checkpoint_path).parent.name,
            "image_size": int(config["dataset"]["image_size"]),
            "timesteps": int(config["model"]["timesteps"]),
        }
        return image, meta


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
