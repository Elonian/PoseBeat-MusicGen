#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audio_latents import encode_images_to_latents
from utils.config import as_path, load_config, optional, require
from utils.logging import log_config, setup_logging
from utils.motion_dataset import MelConditionDataset, collate_mel_conditions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optionally cache VAE latents from the mel-image Arrow dataset."
    )
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    output_dir = as_path(require(cfg, "paths.output_dir"))
    latent_dir = as_path(require(cfg, "paths.train_latent_dir"))
    latent_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "logs", name="posebeat.cache", filename="cache_audio_latents.log")
    log_config(logger, cfg)
    logger.info("device=%s dtype=%s latent_dir=%s", device, dtype, latent_dir)

    dataset = MelConditionDataset(
        require(cfg, "paths.train_mel_dataset_dir"),
        require(cfg, "paths.train_motion_pickle"),
        split=str(optional(cfg, "data.dataset_split", "train")),
        image_channels=int(optional(cfg, "model.image_channels", 1)),
        strict=bool(optional(cfg, "data.strict_condition_match", True)),
        limit=args.limit,
    )
    from models import freeze_module, load_audio_pipeline_components

    components = load_audio_pipeline_components(
        require(cfg, "paths.vae_checkpoint_dir"),
        device=device,
        dtype=dtype,
        load_unet=False,
        load_vae=True,
        load_scheduler=False,
    )
    assert components.vae is not None
    freeze_module(components.vae)

    dataloader = DataLoader(
        dataset,
        batch_size=int(optional(cfg, "training.batch_size", 8)),
        shuffle=False,
        num_workers=int(optional(cfg, "training.num_workers", 2)),
        collate_fn=collate_mel_conditions,
        pin_memory=device.type == "cuda",
    )

    cached = 0
    skipped = 0
    for batch in tqdm(dataloader, desc="cache-audio-latents"):
        missing_indices = [
            i for i, key in enumerate(batch["keys"]) if not (latent_dir / f"{key}.pt").exists()
        ]
        if not missing_indices:
            skipped += len(batch["keys"])
            continue

        images = batch["image"][missing_indices].to(device=device, dtype=dtype)
        latents = encode_images_to_latents(components.vae, images).cpu()
        for local_index, latent in zip(missing_indices, latents):
            key = batch["keys"][local_index]
            torch.save(
                {
                    "key": key,
                    "latents": latent,
                    "conditioning": batch["conditioning"][local_index],
                },
                latent_dir / f"{key}.pt",
            )
            cached += 1

    logger.info("cache complete: cached=%d skipped_existing=%d", cached, skipped)


if __name__ == "__main__":
    main()
