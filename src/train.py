from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, save_config_snapshot
from src.data.synthetic_dataset import SyntheticCaptionDataset, collate_batch
from src.model.diffusion import DiffusionConfig, GaussianDiffusion
from src.model.text_encoder import FrozenTextEncoder, TextEncoderConfig
from src.model.unet import TinyConditionalUNet
from src.utils.checkpointing import save_checkpoint
from src.utils.image_utils import save_image_grid
from src.utils.seed import seed_everything


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny text-to-image diffusion model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    return parser


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict[str, Any], text_embed_dim: int) -> GaussianDiffusion:
    model_config = config["model"]
    unet = TinyConditionalUNet(
        in_channels=model_config["in_channels"],
        base_channels=model_config["base_channels"],
        channel_multipliers=model_config["channel_multipliers"],
        time_embed_dim=model_config["time_embed_dim"],
        cond_embed_dim=model_config["cond_embed_dim"],
        text_embed_dim=text_embed_dim,
    )
    diffusion = GaussianDiffusion(
        model=unet,
        config=DiffusionConfig(
            timesteps=model_config["timesteps"],
            beta_start=model_config["beta_start"],
            beta_end=model_config["beta_end"],
            cond_drop_prob=model_config.get("cond_drop_prob", 0.0),
        ),
    )
    return diffusion


def build_scheduler(
    optimizer: optim.Optimizer,
    *,
    training_config: dict[str, Any],
    total_updates: int,
):
    scheduler_name = training_config.get("lr_scheduler", "cosine")
    if scheduler_name == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(total_updates, 1))
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(total_updates, 1))
    raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")


def make_run_dirs(config: dict[str, Any]) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outputs_config = config["outputs"]
    checkpoints_dir = Path(outputs_config["checkpoints_dir"]) / timestamp
    runs_dir = Path(outputs_config["runs_dir"]) / timestamp
    samples_dir = Path(outputs_config["samples_dir"]) / timestamp
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    return {
        "timestamp": Path(timestamp),
        "checkpoints": checkpoints_dir,
        "runs": runs_dir,
        "samples": samples_dir,
    }


@torch.no_grad()
def evaluate_loss(
    diffusion: GaussianDiffusion,
    text_encoder: FrozenTextEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    diffusion.eval()
    losses = []
    for batch in dataloader:
        images = batch["images"].to(device)
        text_embeddings = text_encoder(batch["captions"], device=device)
        losses.append(diffusion.training_loss(images, text_embeddings).item())
    diffusion.train()
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def write_validation_samples(
    diffusion: GaussianDiffusion,
    text_encoder: FrozenTextEncoder,
    prompts: list[str],
    config: dict[str, Any],
    destination: Path,
    device: torch.device,
) -> None:
    model_config = config["model"]
    training_config = config["training"]
    text_embeddings = text_encoder(prompts, device=device)
    images = diffusion.sample(
        batch_size=len(prompts),
        image_size=config["dataset"]["image_size"],
        channels=model_config["in_channels"],
        text_embeddings=text_embeddings,
        guidance_scale=training_config["guidance_scale"],
        device=device,
    )
    save_image_grid(images, destination, columns=min(4, len(prompts)))


def train() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = choose_device()

    train_dataset = SyntheticCaptionDataset(
        root=config["dataset"]["root"],
        metadata_file=config["dataset"]["train_metadata"],
    )
    val_dataset = SyntheticCaptionDataset(
        root=config["dataset"]["root"],
        metadata_file=config["dataset"]["val_metadata"],
    )

    training_config = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config["num_workers"],
        collate_fn=collate_batch,
    )

    text_encoder = FrozenTextEncoder(
        TextEncoderConfig(
            model_name=config["model"]["text_encoder_name"],
            max_length=config["model"]["text_max_length"],
        )
    )
    diffusion = build_model(config, text_embed_dim=text_encoder.output_dim).to(device)
    text_encoder.to(device)

    optimizer = optim.AdamW(
        diffusion.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    total_updates = max(
        1,
        (len(train_loader) * training_config["epochs"]) // max(training_config["grad_accum_steps"], 1),
    )
    scheduler = build_scheduler(
        optimizer,
        training_config=training_config,
        total_updates=total_updates,
    )

    run_dirs = make_run_dirs(config)
    save_config_snapshot(config, run_dirs["runs"] / "config.snapshot.yaml")

    best_val_loss = math.inf
    global_step = 0
    scaler = None
    if training_config.get("mixed_precision") and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(training_config["epochs"]):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{training_config['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["images"].to(device)
            text_embeddings = text_encoder(batch["captions"], device=device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = diffusion.training_loss(images, text_embeddings) / training_config["grad_accum_steps"]
                scaler.scale(loss).backward()
            else:
                loss = diffusion.training_loss(images, text_embeddings) / training_config["grad_accum_steps"]
                loss.backward()

            if batch_index % training_config["grad_accum_steps"] == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(diffusion.parameters(), training_config["max_grad_norm"])
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix(loss=f"{loss.item() * training_config['grad_accum_steps']:.4f}")

            if global_step % training_config["validate_every_steps"] == 0:
                val_loss = evaluate_loss(diffusion, text_encoder, val_loader, device)
                sample_prompts = training_config["sample_prompts"][: training_config["num_val_samples"]]
                write_validation_samples(
                    diffusion,
                    text_encoder,
                    sample_prompts,
                    config,
                    run_dirs["samples"] / f"step_{global_step:06d}.png",
                    device,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        run_dirs["checkpoints"] / "best.pt",
                        model_state=diffusion.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        scheduler_state=scheduler.state_dict(),
                        step=global_step,
                        epoch=epoch,
                        config=config,
                        best_val_loss=best_val_loss,
                    )

            if global_step % training_config["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    run_dirs["checkpoints"] / f"step_{global_step:06d}.pt",
                    model_state=diffusion.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    step=global_step,
                    epoch=epoch,
                    config=config,
                    best_val_loss=best_val_loss,
                )

    if not math.isfinite(best_val_loss):
        best_val_loss = evaluate_loss(diffusion, text_encoder, val_loader, device)
        save_checkpoint(
            run_dirs["checkpoints"] / "best.pt",
            model_state=diffusion.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            step=global_step,
            epoch=training_config["epochs"],
            config=config,
            best_val_loss=best_val_loss,
        )
    save_checkpoint(
        run_dirs["checkpoints"] / "last.pt",
        model_state=diffusion.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=global_step,
        epoch=training_config["epochs"],
        config=config,
        best_val_loss=best_val_loss,
    )


if __name__ == "__main__":
    train()
