#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MotionAdapter, MotionAdapterConfig, MotionConditionedAudioGenerator
from models.audio_generator import freeze_module, load_audio_generator_components
from utils.checkpoints import save_training_checkpoint
from utils.config import as_path, load_config, optional, require
from utils.logging import (
    JsonlMetricLogger,
    RunningAverage,
    count_parameters,
    format_count,
    log_config,
    setup_logging,
)
from utils.motion_dataset import (
    MotionLatentDataset,
    collate_motion_latents,
    infer_motion_dim,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the motion adapter for the music generator.")
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32
    output_dir = as_path(require(cfg, "paths.output_dir"))
    logger = setup_logging(
        output_dir / "logs",
        name="posebeat.train",
        filename="train_motion_adapter.log",
    )
    log_config(logger, cfg)
    logger.info("device=%s dtype=%s", device, dtype)

    dataset = MotionLatentDataset(
        require(cfg, "paths.train_motion_pickle"),
        require(cfg, "paths.train_latent_dir"),
        strict=bool(optional(cfg, "training.strict_latent_cache", True)),
    )
    motion_dim = int(optional(cfg, "model.motion_dim", 0)) or infer_motion_dim(dataset.encodings)
    adapter_config = MotionAdapterConfig(
        motion_dim=motion_dim,
        hidden_dim=int(optional(cfg, "model.hidden_dim", 512)),
        num_layers=int(optional(cfg, "model.num_layers", 4)),
        num_heads=int(optional(cfg, "model.num_heads", 8)),
        dropout=float(optional(cfg, "model.dropout", 0.1)),
        max_motion_frames=int(optional(cfg, "model.max_motion_frames", 256)),
        primary_cross_attention_dim=int(optional(cfg, "model.primary_cross_attention_dim", 768)),
        secondary_cross_attention_dim=int(optional(cfg, "model.secondary_cross_attention_dim", 1024)),
    )
    logger.info("training pairs=%d motion_dim=%d", len(dataset), motion_dim)

    components = load_audio_generator_components(
        require(cfg, "paths.audio_generator_checkpoint_dir"),
        device=device,
        dtype=dtype,
        load_vae=False,
        load_vocoder=False,
    )
    freeze_module(components.unet)

    adapter = MotionAdapter(adapter_config).to(device=device, dtype=dtype)
    model = MotionConditionedAudioGenerator(components.unet, adapter)
    logger.info(
        "unet params=%s frozen_trainable=%s",
        format_count(count_parameters(components.unet)),
        format_count(count_parameters(components.unet, trainable_only=True)),
    )
    logger.info(
        "adapter params=%s trainable=%s",
        format_count(count_parameters(adapter)),
        format_count(count_parameters(adapter, trainable_only=True)),
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(optional(cfg, "training.learning_rate", 1e-4)),
        weight_decay=float(optional(cfg, "training.weight_decay", 1e-2)),
    )

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        adapter.load_state_dict(state["adapter"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        logger.info("resumed checkpoint: %s", args.resume)

    dataloader = DataLoader(
        dataset,
        batch_size=int(optional(cfg, "training.batch_size", 4)),
        shuffle=True,
        num_workers=int(optional(cfg, "training.num_workers", 2)),
        collate_fn=collate_motion_latents,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = JsonlMetricLogger(output_dir / "logs" / "train_metrics.jsonl")
    writer = None
    if bool(optional(cfg, "logging.tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    global_step = 0
    epochs = int(optional(cfg, "training.epochs", 20))
    save_every = int(optional(cfg, "training.save_every_steps", 1000))
    log_every = int(optional(cfg, "logging.log_every_steps", 25))
    grad_clip = float(optional(cfg, "training.gradient_clip", 1.0))
    loss_meter = RunningAverage()

    model.train()
    for epoch in range(epochs):
        logger.info("starting epoch=%d/%d", epoch + 1, epochs)
        progress = tqdm(dataloader, desc=f"epoch-{epoch}")
        for batch in progress:
            clean_latents = batch["latents"].to(device=device, dtype=dtype)
            motion = batch["motion"].to(device=device, dtype=dtype)
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(
                0,
                components.scheduler.config.num_train_timesteps,
                (clean_latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = components.scheduler.add_noise(clean_latents, noise, timesteps)

            noise_pred = model(noisy_latents, timesteps, motion)
            loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            loss_meter.update(loss.item(), clean_latents.shape[0])
            progress.set_postfix(loss=f"{loss.item():.5f}", step=global_step)

            if global_step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d epoch=%d loss=%.6f avg_loss=%.6f lr=%.3e",
                    global_step,
                    epoch,
                    loss.item(),
                    loss_meter.average,
                    lr,
                )
                metric_logger.log(
                    step=global_step,
                    epoch=epoch,
                    loss=loss.item(),
                    avg_loss=loss_meter.average,
                    learning_rate=lr,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/avg_loss", loss_meter.average, global_step)
                    writer.add_scalar("train/learning_rate", lr, global_step)
                loss_meter.reset()

            if global_step % save_every == 0:
                checkpoint_path = checkpoint_dir / f"adapter_step_{global_step}.pt"
                save_training_checkpoint(
                    checkpoint_path,
                    adapter=adapter,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    config=cfg,
                )
                logger.info("saved checkpoint: %s", checkpoint_path)

    final_path = checkpoint_dir / "adapter_final.pt"
    save_training_checkpoint(
        final_path,
        adapter=adapter,
        optimizer=optimizer,
        step=global_step,
        epoch=epochs,
        config=cfg,
    )
    logger.info("saved final checkpoint: %s", final_path)
    metric_logger.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
