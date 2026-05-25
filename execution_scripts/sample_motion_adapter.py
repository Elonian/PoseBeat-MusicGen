#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MotionAdapter, MotionAdapterConfig
from models.audio_generator import freeze_module, load_audio_generator_components
from utils.audio_latents import mel_config_from_checkpoint
from utils.checkpoints import load_adapter_state
from utils.config import as_path, load_config, optional, require
from utils.logging import setup_logging
from utils.motion_dataset import infer_motion_dim, load_motion_encodings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio from one DMD motion encoding.")
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    output_dir = as_path(require(cfg, "paths.output_dir"))
    logger = setup_logging(output_dir / "logs", name="posebeat.sample", filename="sample_motion_adapter.log")
    checkpoint_dir = Path(require(cfg, "paths.audio_generator_checkpoint_dir")).expanduser()
    logger.info("device=%s checkpoint=%s", device, args.checkpoint)

    motion_pickle = as_path(require(cfg, "paths.test_motion_pickle"))
    encodings = load_motion_encodings(motion_pickle)
    key = args.motion_key or sorted(encodings)[0]
    logger.info("motion key=%s motion pickle=%s", key, motion_pickle)
    motion = torch.as_tensor(np.asarray(encodings[key]), dtype=dtype, device=device).unsqueeze(0)

    motion_dim = int(optional(cfg, "model.motion_dim", 0)) or infer_motion_dim(encodings)
    adapter = MotionAdapter(
        MotionAdapterConfig(
            motion_dim=motion_dim,
            hidden_dim=int(optional(cfg, "model.hidden_dim", 512)),
            num_layers=int(optional(cfg, "model.num_layers", 4)),
            num_heads=int(optional(cfg, "model.num_heads", 8)),
            dropout=0.0,
            max_motion_frames=int(optional(cfg, "model.max_motion_frames", 256)),
            primary_cross_attention_dim=int(optional(cfg, "model.primary_cross_attention_dim", 768)),
            secondary_cross_attention_dim=int(optional(cfg, "model.secondary_cross_attention_dim", 1024)),
        )
    ).to(device)
    state = load_adapter_state(args.checkpoint, map_location=device)
    adapter.load_state_dict(state["adapter"])
    adapter.eval()

    components = load_audio_generator_components(
        checkpoint_dir,
        device=device,
        dtype=dtype,
        load_vae=True,
        load_vocoder=True,
    )
    freeze_module(components.unet)
    freeze_module(components.vae)
    freeze_module(components.vocoder)
    assert components.vae is not None
    assert components.vocoder is not None

    mel_cfg = mel_config_from_checkpoint(
        checkpoint_dir,
        float(optional(cfg, "audio.audio_length_seconds", 5.0)),
    )
    latent_h = mel_cfg.target_frames // mel_cfg.vae_scale_factor
    latent_w = mel_cfg.n_mels // mel_cfg.vae_scale_factor
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (1, components.unet.config.in_channels, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        conditioning = adapter(motion)
        components.scheduler.set_timesteps(args.steps, device=device)
        for timestep in components.scheduler.timesteps:
            latent_input = components.scheduler.scale_model_input(latents, timestep)
            noise_pred = components.unet(
                latent_input,
                timestep,
                encoder_hidden_states=conditioning.primary,
                encoder_hidden_states_1=conditioning.secondary,
                return_dict=False,
            )[0]
            latents = components.scheduler.step(noise_pred, timestep, latents).prev_sample

        mel = components.vae.decode(latents / components.vae.config.scaling_factor).sample
        waveform = components.vocoder(mel.squeeze(1)).detach().cpu().float().numpy()[0]

    output = Path(args.output) if args.output else output_dir / f"{key}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.clip(waveform, -1.0, 1.0)
    scipy.io.wavfile.write(output, components.vocoder.config.sampling_rate, waveform)
    logger.info("wrote sample: %s", output)
    print(output)


if __name__ == "__main__":
    main()
