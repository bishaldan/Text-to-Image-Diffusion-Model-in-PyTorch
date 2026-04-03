from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).to(torch.uint8)
    array = image.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def save_image_grid(images: Sequence[torch.Tensor], destination: str | Path, columns: int = 4) -> None:
    pil_images = [tensor_to_pil(image) for image in images]
    if not pil_images:
        raise ValueError("Expected at least one image to save.")

    width, height = pil_images[0].size
    rows = (len(pil_images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), color=(18, 18, 18))
    for index, image in enumerate(pil_images):
        x = (index % columns) * width
        y = (index // columns) * height
        grid.paste(image, (x, y))

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(destination_path)

