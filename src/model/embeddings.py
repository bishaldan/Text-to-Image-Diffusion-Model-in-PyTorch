from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * exponent)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=1)
        return embeddings

