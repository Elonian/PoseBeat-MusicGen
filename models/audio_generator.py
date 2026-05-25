from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.audioldm2.modeling_audioldm2 import (
    AudioLDM2UNet2DConditionModel,
)
from torch import nn
from transformers import SpeechT5HifiGan

from .motion_adapter import MotionAdapter, MotionConditioning


@dataclass
class AudioGeneratorComponents:
    unet: AudioLDM2UNet2DConditionModel
    scheduler: DDIMScheduler
    vae: AutoencoderKL | None = None
    vocoder: SpeechT5HifiGan | None = None


def freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def load_audio_generator_components(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    load_vae: bool = True,
    load_vocoder: bool = False,
) -> AudioGeneratorComponents:
    """Load the frozen audio generator components used by adapter training."""

    root = Path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio generator checkpoint folder not found: {root}")

    unet = AudioLDM2UNet2DConditionModel.from_pretrained(
        root,
        subfolder="unet",
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(
        root,
        subfolder="scheduler",
        local_files_only=True,
    )

    vae = None
    if load_vae:
        vae = AutoencoderKL.from_pretrained(
            root,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)

    vocoder = None
    if load_vocoder:
        vocoder = SpeechT5HifiGan.from_pretrained(
            root,
            subfolder="vocoder",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)

    return AudioGeneratorComponents(unet=unet, scheduler=scheduler, vae=vae, vocoder=vocoder)


class MotionConditionedAudioGenerator(nn.Module):
    """Frozen audio denoiser controlled by a trainable motion adapter."""

    def __init__(self, unet: AudioLDM2UNet2DConditionModel, motion_adapter: MotionAdapter):
        super().__init__()
        self.unet = unet
        self.motion_adapter = motion_adapter

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        motion: torch.Tensor,
        motion_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioning: MotionConditioning = self.motion_adapter(
            motion,
            attention_mask=motion_attention_mask,
        )
        return self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=conditioning.primary,
            encoder_hidden_states_1=conditioning.secondary,
            encoder_attention_mask_1=conditioning.attention_mask,
            return_dict=False,
        )[0]
