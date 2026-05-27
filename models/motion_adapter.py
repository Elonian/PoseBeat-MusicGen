from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MotionAdapterConfig:
    motion_dim: int
    cross_attention_dim: int = 370


@dataclass
class MotionConditioning:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None = None


class MotionAdapter(nn.Module):
    """Motion conditioning adapter for precomputed motion/genre tensors."""

    def __init__(self, config: MotionAdapterConfig):
        super().__init__()
        self.config = config
        if config.motion_dim == config.cross_attention_dim:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(config.motion_dim, config.cross_attention_dim)

    def forward(
        self,
        motion: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MotionConditioning:
        if motion.ndim != 3:
            raise ValueError(f"motion must be [batch, frames, dim], got {tuple(motion.shape)}")
        if motion.shape[-1] != self.config.motion_dim:
            raise ValueError(
                f"motion dim mismatch: expected {self.config.motion_dim}, got {motion.shape[-1]}"
            )
        return MotionConditioning(hidden_states=self.proj(motion), attention_mask=attention_mask)
