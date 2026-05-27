from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MelImageConfig:
    x_res: int = 256
    y_res: int = 256
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    top_db: int = 80
    n_iter: int = 32
    vae_scale_factor: int = 8

    @property
    def latent_size(self) -> tuple[int, int]:
        return (self.y_res // self.vae_scale_factor, self.x_res // self.vae_scale_factor)


def mel_config_from_checkpoint(checkpoint_dir: str | Path) -> MelImageConfig:
    root = Path(checkpoint_dir)
    mel_path = root / "mel" / "mel_config.json"
    if not mel_path.exists():
        raise FileNotFoundError(f"mel_config.json not found: {mel_path}")
    with mel_path.open("r", encoding="utf-8") as handle:
        mel_config = json.load(handle)

    vae_subfolder = "vqvae" if (root / "vqvae").is_dir() else "vae"
    with (root / vae_subfolder / "config.json").open("r", encoding="utf-8") as handle:
        vae_config = json.load(handle)
    scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)

    return MelImageConfig(
        x_res=int(mel_config["x_res"]),
        y_res=int(mel_config["y_res"]),
        sample_rate=int(mel_config["sample_rate"]),
        n_fft=int(mel_config["n_fft"]),
        hop_length=int(mel_config["hop_length"]),
        top_db=int(mel_config.get("top_db", 80)),
        n_iter=int(mel_config.get("n_iter", 32)),
        vae_scale_factor=scale_factor,
    )


@torch.no_grad()
def encode_images_to_latents(
    vae: torch.nn.Module,
    images: torch.Tensor,
    *,
    scaling_factor: float | None = None,
) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.sample()
    if scaling_factor is None:
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    return latents * scaling_factor
