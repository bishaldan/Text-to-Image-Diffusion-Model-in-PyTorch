from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from src.model.embeddings import SinusoidalTimeEmbedding


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        cond_embedding: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.time_proj(time_embedding) + self.cond_proj(cond_embedding)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.layer(x)


class TinyConditionalUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        channel_multipliers: Iterable[int],
        time_embed_dim: int,
        cond_embed_dim: int,
        text_embed_dim: int,
    ) -> None:
        super().__init__()
        multipliers = list(channel_multipliers)
        if not multipliers:
            raise ValueError("Expected at least one channel multiplier.")

        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(text_embed_dim, cond_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_embed_dim * 2, cond_embed_dim),
        )

        channels = [base_channels * multiplier for multiplier in multipliers]
        self.input_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channels[0]
        for index, next_channels in enumerate(channels):
            block = ResidualBlock(current_channels, next_channels, time_embed_dim, cond_embed_dim)
            self.down_blocks.append(block)
            current_channels = next_channels
            if index < len(channels) - 1:
                self.downsamples.append(Downsample(current_channels))

        self.mid_block1 = ResidualBlock(current_channels, current_channels, time_embed_dim, cond_embed_dim)
        self.mid_block2 = ResidualBlock(current_channels, current_channels, time_embed_dim, cond_embed_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(channels))):
            skip_channels = channels[index]
            in_block_channels = current_channels + skip_channels
            out_channels = skip_channels
            self.up_blocks.append(
                ResidualBlock(in_block_channels, out_channels, time_embed_dim, cond_embed_dim)
            )
            current_channels = out_channels
            if index > 0:
                self.upsamples.append(Upsample(current_channels))

        self.out_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.out_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        cond_embedding = self.cond_proj(text_embeddings)

        x = self.input_conv(x)
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            x = block(x, time_embedding, cond_embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.mid_block1(x, time_embedding, cond_embedding)
        x = self.mid_block2(x, time_embedding, cond_embedding)

        upsample_index = 0
        for index, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_embedding, cond_embedding)
            if index < len(self.upsamples):
                x = self.upsamples[upsample_index](x)
                upsample_index += 1

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

