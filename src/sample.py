from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.model.diffusion import DiffusionConfig, GaussianDiffusion
from src.model.text_encoder import FrozenTextEncoder, TextEncoderConfig
from src.model.unet import TinyConditionalUNet
from src.utils.checkpointing import load_checkpoint
from src.utils.image_utils import save_image_grid


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--prompts", nargs="+", required=True, help="Prompt strings to sample.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Optional classifier-free guidance scale override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path for the sample grid.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = choose_device()
    payload = load_checkpoint(args.checkpoint, map_location=device)
    config = payload["config"]

    text_encoder = FrozenTextEncoder(
        TextEncoderConfig(
            model_name=config["model"]["text_encoder_name"],
            max_length=config["model"]["text_max_length"],
        )
    ).to(device)
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
    ).to(device)
    diffusion.load_state_dict(payload["model_state"])
    diffusion.eval()

    text_embeddings = text_encoder(args.prompts, device=device)
    images = diffusion.sample(
        batch_size=len(args.prompts),
        image_size=config["dataset"]["image_size"],
        channels=config["model"]["in_channels"],
        text_embeddings=text_embeddings,
        guidance_scale=args.guidance_scale or config["training"]["guidance_scale"],
        device=device,
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path("outputs") / "samples" / f"inference_{timestamp}.png"
    else:
        output_path = args.output
    save_image_grid(images, output_path, columns=min(4, len(args.prompts)))
    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    main()
