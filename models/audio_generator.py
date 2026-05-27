from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torch import nn


@dataclass
class AudioPipelineComponents:
    unet: UNet2DConditionModel | None = None
    scheduler: DDIMScheduler | DDPMScheduler | None = None
    vae: AutoencoderKL | None = None


def freeze_module(module: nn.Module | None) -> None:
    if module is None:
        return
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _first_existing_subfolder(root: Path, names: tuple[str, ...]) -> str:
    for name in names:
        if (root / name).is_dir():
            return name
    raise FileNotFoundError(f"none of these subfolders exist under {root}: {names}")


def load_audio_pipeline_components(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    load_unet: bool = True,
    load_vae: bool = True,
    load_scheduler: bool = True,
) -> AudioPipelineComponents:
    """Load the local mel-image latent diffusion pipeline components."""

    root = Path(checkpoint_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"audio pipeline checkpoint folder not found: {root}")

    unet = None
    if load_unet:
        unet = UNet2DConditionModel.from_pretrained(
            root,
            subfolder="unet",
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)

    vae = None
    if load_vae:
        vae_subfolder = _first_existing_subfolder(root, ("vqvae", "vae"))
        vae = AutoencoderKL.from_pretrained(
            root,
            subfolder=vae_subfolder,
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)

    scheduler = None
    if load_scheduler:
        scheduler = DDIMScheduler.from_pretrained(
            root,
            subfolder="scheduler",
            local_files_only=True,
        )

    return AudioPipelineComponents(unet=unet, scheduler=scheduler, vae=vae)


def create_conditioned_unet(
    *,
    sample_size: tuple[int, int],
    cross_attention_dim: int,
    in_channels: int = 1,
    out_channels: int = 1,
) -> UNet2DConditionModel:
    """Create the conditional latent UNet architecture used for motion conditioning."""

    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=cross_attention_dim,
    )


def create_noise_scheduler(kind: str, num_train_steps: int) -> DDIMScheduler | DDPMScheduler:
    kind = kind.lower()
    if kind == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_steps)
    if kind == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_steps)
    raise ValueError(f"unsupported scheduler: {kind}")


class MotionConditionedUNet(nn.Module):
    """Thin wrapper around a conditional UNet."""

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if conditioning.ndim != 3:
            raise ValueError(
                f"conditioning must be [batch, frames, dim], got {tuple(conditioning.shape)}"
            )
        output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=conditioning,
            return_dict=False,
        )
        return output[0]


def save_audio_pipeline(
    output_dir: str | Path,
    *,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler | DDPMScheduler,
    mel_config: dict[str, Any],
) -> None:
    """Save a compatible diffusers audio pipeline folder."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(output / "unet")
    vae.save_pretrained(output / "vqvae")
    scheduler.save_pretrained(output / "scheduler")

    mel_dir = output / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    rendered_mel_config = {
        "_class_name": "Mel",
        "_diffusers_version": "0.18.1",
        "hop_length": int(mel_config["hop_length"]),
        "n_fft": int(mel_config["n_fft"]),
        "n_iter": int(mel_config.get("n_iter", 32)),
        "sample_rate": int(mel_config["sample_rate"]),
        "top_db": int(mel_config.get("top_db", 80)),
        "x_res": int(mel_config["x_res"]),
        "y_res": int(mel_config["y_res"]),
    }
    (mel_dir / "mel_config.json").write_text(
        json.dumps(rendered_mel_config, indent=2) + "\n",
        encoding="utf-8",
    )

    model_index = {
        "_class_name": "AudioDiffusionPipeline",
        "_diffusers_version": "0.18.1",
        "mel": ["audio_diffusion", "Mel"],
        "scheduler": ["diffusers", scheduler.__class__.__name__],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vqvae": ["diffusers", "AutoencoderKL"],
    }
    (output / "model_index.json").write_text(
        json.dumps(model_index, indent=2) + "\n",
        encoding="utf-8",
    )


load_audio_generator_components = load_audio_pipeline_components
