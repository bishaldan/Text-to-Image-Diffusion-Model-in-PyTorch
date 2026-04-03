from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int
    beta_start: float
    beta_end: float
    cond_drop_prob: float


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, config: DiffusionConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def training_loss(
        self,
        x_start: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_start.shape[0]
        device = x_start.device
        timesteps = torch.randint(0, self.config.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)

        if self.config.cond_drop_prob > 0:
            keep_mask = (
                torch.rand(batch_size, device=device) > self.config.cond_drop_prob
            ).float().unsqueeze(1)
            conditioned = text_embeddings * keep_mask
        else:
            conditioned = text_embeddings

        predicted_noise = self.model(x_noisy, timesteps, conditioned)
        return torch.mean((noise - predicted_noise) ** 2)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        timestep: int,
        text_embeddings: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        t = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)
        cond_pred = self.model(x, t, text_embeddings)
        if guidance_scale > 1.0:
            uncond_pred = self.model(x, t, torch.zeros_like(text_embeddings))
            pred_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            pred_noise = cond_pred

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)

        mean = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
        if timestep == 0:
            return mean

        noise = torch.randn_like(x)
        variance = torch.sqrt(self.posterior_variance[t].view(-1, 1, 1, 1))
        return mean + variance * noise

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        image_size: int,
        channels: int,
        text_embeddings: torch.Tensor,
        guidance_scale: float,
        device: torch.device,
    ) -> torch.Tensor:
        x = torch.randn((batch_size, channels, image_size, image_size), device=device)
        for timestep in reversed(range(self.config.timesteps)):
            x = self.p_sample(x, timestep, text_embeddings, guidance_scale)
        return x.clamp(-1.0, 1.0)
