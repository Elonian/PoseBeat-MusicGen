from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MotionAdapterConfig:
    motion_dim: int
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_motion_frames: int = 256
    primary_cross_attention_dim: int = 768
    secondary_cross_attention_dim: int = 1024
    use_learned_positional_embedding: bool = True


@dataclass
class MotionConditioning:
    primary: torch.Tensor
    secondary: torch.Tensor
    attention_mask: torch.Tensor | None = None


class MotionAdapter(nn.Module):
    """Maps DMD-style SMPL motion features to audio-generator attention tokens."""

    def __init__(self, config: MotionAdapterConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.motion_dim)
        self.input_proj = nn.Linear(config.motion_dim, config.hidden_dim)

        if config.use_learned_positional_embedding:
            self.position = nn.Parameter(
                torch.zeros(1, config.max_motion_frames, config.hidden_dim)
            )
            nn.init.normal_(self.position, std=0.02)
        else:
            self.register_buffer(
                "position",
                self._sinusoidal_position(config.max_motion_frames, config.hidden_dim),
                persistent=False,
            )

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.primary_proj = nn.Linear(config.hidden_dim, config.primary_cross_attention_dim)
        self.secondary_proj = nn.Linear(config.hidden_dim, config.secondary_cross_attention_dim)

    @staticmethod
    def _sinusoidal_position(length: int, dim: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        return pe

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
        if motion.shape[1] > self.config.max_motion_frames:
            raise ValueError(
                f"motion has {motion.shape[1]} frames but max_motion_frames is "
                f"{self.config.max_motion_frames}"
            )

        hidden = self.input_proj(self.input_norm(motion))
        hidden = hidden + self.position[:, : hidden.shape[1], :].to(hidden.dtype)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        hidden = self.temporal_encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.output_norm(hidden)
        return MotionConditioning(
            primary=self.primary_proj(hidden),
            secondary=self.secondary_proj(hidden),
            attention_mask=attention_mask,
        )
