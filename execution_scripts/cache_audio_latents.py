#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.audio_generator import freeze_module, load_audio_generator_components
from utils.audio_latents import AudioToLatents, mel_config_from_checkpoint
from utils.config import as_path, load_config, optional, require
from utils.logging import log_config, setup_logging
from utils.motion_dataset import load_motion_encodings, resolve_audio_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache clean audio VAE latents for sliced wavs.")
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    output_dir = as_path(require(cfg, "paths.output_dir"))
    logger = setup_logging(
        output_dir / "logs",
        name="posebeat.cache",
        filename=f"cache_audio_latents_{args.split}.log",
    )
    log_config(logger, cfg)

    checkpoint_dir = Path(require(cfg, "paths.audio_generator_checkpoint_dir")).expanduser()
    motion_pickle = as_path(require(cfg, f"paths.{args.split}_motion_pickle"))
    audio_dir = as_path(require(cfg, f"paths.{args.split}_audio_dir"))
    latent_dir = as_path(require(cfg, f"paths.{args.split}_latent_dir"))
    latent_dir.mkdir(parents=True, exist_ok=True)
    logger.info("split=%s device=%s dtype=%s", args.split, device, dtype)
    logger.info("audio generator checkpoint: %s", checkpoint_dir)
    logger.info("motion pickle: %s", motion_pickle)
    logger.info("audio dir: %s", audio_dir)
    logger.info("latent dir: %s", latent_dir)

    audio_seconds = float(optional(cfg, "audio.audio_length_seconds", 5.0))
    components = load_audio_generator_components(
        checkpoint_dir,
        device=device,
        dtype=dtype,
        load_vae=True,
        load_vocoder=False,
    )
    freeze_module(components.vae)
    assert components.vae is not None

    mel_config = mel_config_from_checkpoint(checkpoint_dir, audio_seconds)
    encoder = AudioToLatents(components.vae, mel_config, device=device, dtype=dtype)
    encodings = load_motion_encodings(motion_pickle)
    keys = sorted(encodings)
    if args.limit is not None:
        keys = keys[: args.limit]
    logger.info("motion items selected: %d", len(keys))

    failures: list[str] = []
    cached = 0
    skipped_existing = 0
    for key in tqdm(keys, desc=f"cache-{args.split}-latents"):
        output_path = latent_dir / f"{key}.pt"
        if output_path.exists():
            skipped_existing += 1
            continue
        try:
            audio_file = resolve_audio_file(audio_dir, key)
            if audio_file.stat().st_size == 0:
                raise ValueError("audio file is empty")
            latents = encoder.encode_file(audio_file).cpu()
        except Exception as exc:
            failures.append(f"{key}\t{exc}")
            continue
        torch.save(
            {
                "key": key,
                "latents": latents,
                "audio_file": str(audio_file),
                "mel_config": vars(mel_config),
            },
            output_path,
        )
        cached += 1

    if failures:
        failure_path = latent_dir / f"cache_failures_{args.split}.tsv"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        logger.warning("failed=%d; report=%s", len(failures), failure_path)
    logger.info(
        "cache complete: cached=%d skipped_existing=%d failed=%d",
        cached,
        skipped_existing,
        len(failures),
    )


if __name__ == "__main__":
    main()
