from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class TextEncoderConfig:
    model_name: str
    max_length: int


class FrozenTextEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        # The slower tokenizer path keeps setup simpler in lightweight containers
        # where optional fast-tokenizer conversion deps may be absent.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("The selected encoder does not expose a hidden_size.")
        self.output_dim = int(hidden_size)

    def forward(self, captions: list[str], device: torch.device) -> torch.Tensor:
        batch = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch = {name: tensor.to(device) for name, tensor in batch.items()}

        with torch.no_grad():
            encoded = self.encoder(**batch)

        hidden_state = encoded.last_hidden_state
        attention_mask = batch["attention_mask"].unsqueeze(-1)
        masked_state = hidden_state * attention_mask
        pooled = masked_state.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        return pooled
