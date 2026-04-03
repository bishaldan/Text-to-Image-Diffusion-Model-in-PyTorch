from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    destination: str | Path,
    *,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any] | None,
    scheduler_state: dict[str, Any] | None,
    step: int,
    epoch: int,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "step": step,
            "epoch": epoch,
            "config": config,
            "best_val_loss": best_val_loss,
        },
        destination_path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
