#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.audio_latents import encode_images_to_latents
from utils.config import as_path, load_config, optional, require
from utils.logging import (
    JsonlMetricLogger,
    RunningAverage,
    count_parameters,
    format_count,
    log_config,
    setup_logging,
)
from utils.motion_dataset import MelConditionDataset, collate_mel_conditions


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the motion-conditioned latent audio UNet."
    )
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--from-pretrained", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per process/GPU.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per process.")
    parser.add_argument("--epochs", type=int, default=None)
    return parser


def _distributed_state(requested_device: str | None) -> tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if not distributed:
        return (
            torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu")),
            rank,
            local_rank,
            world_size,
            False,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return device, rank, local_rank, world_size, True


def _cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def _unwrap_unet(unet: torch.nn.Module) -> torch.nn.Module:
    if isinstance(unet, DistributedDataParallel):
        return unet.module
    return unet


def _mean_loss_for_logging(loss: torch.Tensor, distributed: bool, world_size: int) -> float:
    logged_loss = loss.detach()
    if distributed:
        dist.all_reduce(logged_loss, op=dist.ReduceOp.SUM)
        logged_loss = logged_loss / world_size
    return float(logged_loss.item())


def _make_ema_model(unet: torch.nn.Module, cfg: dict):
    if not bool(optional(cfg, "training.use_ema", True)):
        return None
    from diffusers.training_utils import EMAModel

    return EMAModel(
        unet.parameters(),
        decay=float(optional(cfg, "training.ema_max_decay", 0.9999)),
        use_ema_warmup=True,
        inv_gamma=float(optional(cfg, "training.ema_inv_gamma", 1.0)),
        power=float(optional(cfg, "training.ema_power", 0.75)),
    )


def _ema_step(ema_model, unet: torch.nn.Module) -> None:
    if ema_model is None:
        return
    try:
        ema_model.step(unet.parameters())
    except TypeError:
        ema_model.step(unet)


def _save_pipeline_with_optional_ema(
    path: Path,
    *,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    scheduler,
    mel_config: dict,
    ema_model,
) -> None:
    from modules import save_audio_pipeline

    if ema_model is None:
        save_audio_pipeline(path, unet=unet, vae=vae, scheduler=scheduler, mel_config=mel_config)
        return

    stored = False
    if hasattr(ema_model, "store"):
        ema_model.store(unet.parameters())
        stored = True
    ema_model.copy_to(unet.parameters())
    save_audio_pipeline(path, unet=unet, vae=vae, scheduler=scheduler, mel_config=mel_config)
    if stored and hasattr(ema_model, "restore"):
        ema_model.restore(unet.parameters())


def _load_training_components(
    cfg: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
    condition_dim: int,
    image_shape: tuple[int, int, int],
    from_pretrained: str | None,
):
    from modules import (
        create_motion_conditioned_unet,
        create_noise_scheduler,
        load_audio_pipeline_components,
    )

    if from_pretrained:
        components = load_audio_pipeline_components(
            from_pretrained,
            device=device,
            dtype=dtype,
            load_unet=True,
            load_vae=True,
            load_scheduler=True,
        )
        assert components.unet is not None
        assert components.vae is not None
        assert components.scheduler is not None
        return components

    vae_dir = as_path(require(cfg, "paths.vae_checkpoint_dir"))
    components = load_audio_pipeline_components(
        vae_dir,
        device=device,
        dtype=dtype,
        load_unet=False,
        load_vae=True,
        load_scheduler=False,
    )
    assert components.vae is not None

    channels, height, width = image_shape
    with torch.no_grad():
        latent = components.vae.encode(
            torch.zeros((1, channels, height, width), device=device, dtype=dtype)
        ).latent_dist.sample()

    components.unet = create_motion_conditioned_unet(
        sample_size=tuple(latent.shape[-2:]),
        cross_attention_dim=condition_dim,
        in_channels=int(components.vae.config.latent_channels),
        out_channels=int(components.vae.config.latent_channels),
        variant=str(optional(cfg, "model.unet_variant", "base")),
    ).to(device=device, dtype=dtype)
    components.scheduler = create_noise_scheduler(
        str(optional(cfg, "training.scheduler", "ddim")),
        int(optional(cfg, "training.num_train_steps", 1000)),
    )
    return components


def train_from_args(args: argparse.Namespace) -> None:
    from modules import freeze_module

    cfg = load_config(args.config)
    device, rank, local_rank, world_size, distributed = _distributed_state(args.device)
    is_main_process = rank == 0
    dtype = torch.float32
    output_dir = as_path(require(cfg, "paths.output_dir"))
    log_dir = as_path(optional(cfg, "paths.log_dir", output_dir / "logs"))
    log_name = "train_motion_adapter.log" if is_main_process else f"train_motion_adapter_rank{rank}.log"
    logger = setup_logging(
        log_dir,
        name=f"posebeat.train.rank{rank}",
        filename=log_name,
    )
    if is_main_process:
        log_config(logger, cfg)
    logger.info(
        "rank=%d local_rank=%d world_size=%d distributed=%s device=%s dtype=%s",
        rank,
        local_rank,
        world_size,
        distributed,
        device,
        dtype,
    )

    dataset = MelConditionDataset(
        require(cfg, "paths.train_mel_dataset_dir"),
        require(cfg, "paths.train_motion_pickle"),
        split=str(optional(cfg, "data.dataset_split", "train")),
        image_channels=int(optional(cfg, "model.image_channels", 1)),
        image_size=optional(cfg, "data.image_size", None),
        strict=bool(optional(cfg, "data.strict_condition_match", True)),
        limit=args.limit or optional(cfg, "training.max_train_samples", None),
    )
    first = dataset[0]
    image_shape = tuple(first["image"].shape)
    condition_shape = tuple(first["conditioning"].shape)
    condition_dim = int(first["conditioning"].shape[-1])
    logger.info(
        "training pairs=%d image_shape=%s condition_shape=%s",
        len(dataset),
        image_shape,
        condition_shape,
    )

    pretrained = args.from_pretrained or optional(cfg, "paths.from_pretrained_model_dir", None)
    components = _load_training_components(
        cfg,
        device=device,
        dtype=dtype,
        condition_dim=condition_dim,
        image_shape=image_shape,
        from_pretrained=pretrained,
    )
    assert components.unet is not None
    assert components.vae is not None
    assert components.scheduler is not None
    freeze_module(components.vae)

    logger.info("unet params=%s", format_count(count_parameters(components.unet)))
    logger.info("vae params=%s frozen", format_count(count_parameters(components.vae)))
    logger.info("unet_variant=%s sample_size=%s", optional(cfg, "model.unet_variant", "base"), components.unet.config.sample_size)
    logger.info("cross_attention_dim=%d", components.unet.config.cross_attention_dim)

    train_unet: torch.nn.Module = components.unet
    if distributed:
        if device.type == "cuda":
            train_unet = DistributedDataParallel(
                components.unet,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            train_unet = DistributedDataParallel(components.unet)
        logger.info("enabled DistributedDataParallel")

    optimizer = torch.optim.AdamW(
        train_unet.parameters(),
        lr=float(optional(cfg, "training.learning_rate", 1e-4)),
        betas=(
            float(optional(cfg, "training.adam_beta1", 0.95)),
            float(optional(cfg, "training.adam_beta2", 0.999)),
        ),
        weight_decay=float(optional(cfg, "training.weight_decay", 1e-6)),
        eps=float(optional(cfg, "training.adam_epsilon", 1e-8)),
    )
    ema_model = _make_ema_model(components.unet, cfg)
    if ema_model is not None:
        logger.info("EMA enabled")

    global_step = 0
    start_epoch = 0
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location=device)
        components.unet.load_state_dict(resume_state["unet"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if ema_model is not None and resume_state.get("ema") is not None:
            ema_model.load_state_dict(resume_state["ema"])
        global_step = int(resume_state.get("step", 0))
        start_epoch = int(resume_state.get("epoch", 0))
        logger.info("resumed checkpoint: %s", args.resume)

    batch_size = int(args.batch_size or optional(cfg, "training.batch_size", 8))
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else optional(cfg, "training.num_workers", 2)
    )
    train_sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    logger.info(
        "batch_size_per_process=%d effective_batch_size=%d num_workers_per_process=%d total_workers=%d",
        batch_size,
        batch_size * world_size,
        num_workers,
        num_workers * world_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_mel_conditions,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    total_epochs = int(args.epochs or optional(cfg, "training.epochs", 100))
    total_steps = max(1, len(dataloader) * total_epochs)
    lr_scheduler = None
    scheduler_name = optional(cfg, "training.lr_scheduler", None)
    if scheduler_name:
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            str(scheduler_name),
            optimizer=optimizer,
            num_warmup_steps=int(optional(cfg, "training.lr_warmup_steps", 500)),
            num_training_steps=total_steps,
            last_epoch=global_step - 1 if global_step > 0 else -1,
        )
        if resume_state is not None and resume_state.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(resume_state["lr_scheduler"])

    checkpoint_dir = output_dir / "checkpoints"
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = (
        JsonlMetricLogger(log_dir / "train_metrics.jsonl")
        if is_main_process
        else None
    )
    writer = None
    if is_main_process and bool(optional(cfg, "logging.tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    save_every_steps = int(optional(cfg, "training.save_every_steps", 1000))
    save_every_epochs = int(optional(cfg, "training.save_every_epochs", 10))
    log_every = int(optional(cfg, "logging.log_every_steps", 25))
    grad_clip = float(optional(cfg, "training.gradient_clip", 1.0))
    loss_meter = RunningAverage()
    mel_config = {
        "x_res": int(optional(cfg, "audio.x_res", 256)),
        "y_res": int(optional(cfg, "audio.y_res", 256)),
        "sample_rate": int(optional(cfg, "audio.sample_rate", 22050)),
        "hop_length": int(optional(cfg, "audio.hop_length", 512)),
        "n_fft": int(optional(cfg, "audio.n_fft", 2048)),
        "top_db": int(optional(cfg, "audio.top_db", 80)),
        "n_iter": int(optional(cfg, "audio.n_iter", 32)),
    }

    train_unet.train()
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        logger.info("starting epoch=%d/%d", epoch + 1, total_epochs)
        progress = tqdm(dataloader, desc=f"train-epoch-{epoch}", disable=not is_main_process)
        for batch in progress:
            clean_images = batch["image"].to(device=device, dtype=dtype)
            conditioning = batch["conditioning"].to(device=device, dtype=dtype)

            with torch.no_grad():
                clean_latents = encode_images_to_latents(components.vae, clean_images)

            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(
                0,
                components.scheduler.config.num_train_timesteps,
                (clean_latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = components.scheduler.add_noise(clean_latents, noise, timesteps)
            noise_pred = train_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=conditioning,
            ).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_unet.parameters(), grad_clip)
            optimizer.step()
            _ema_step(ema_model, components.unet)
            if lr_scheduler is not None:
                lr_scheduler.step()

            global_step += 1
            loss_value = _mean_loss_for_logging(loss, distributed, world_size)
            loss_meter.update(loss_value, clean_latents.shape[0] * world_size)
            if is_main_process:
                progress.set_postfix(loss=f"{loss_value:.5f}", step=global_step)

            if is_main_process and global_step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "step=%d epoch=%d loss=%.6f avg_loss=%.6f lr=%.3e",
                    global_step,
                    epoch,
                    loss_value,
                    loss_meter.average,
                    lr,
                )
                assert metric_logger is not None
                metric_logger.log(
                    step=global_step,
                    epoch=epoch,
                    loss=loss_value,
                    avg_loss=loss_meter.average,
                    learning_rate=lr,
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, global_step)
                    writer.add_scalar("train/avg_loss", loss_meter.average, global_step)
                    writer.add_scalar("train/learning_rate", lr, global_step)
                loss_meter.reset()

            if is_main_process and save_every_steps > 0 and global_step % save_every_steps == 0:
                checkpoint_path = checkpoint_dir / f"unet_step_{global_step}.pt"
                torch.save(
                    {
                        "unet": components.unet.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        "ema": ema_model.state_dict() if ema_model is not None else None,
                        "step": global_step,
                        "epoch": epoch,
                        "config": cfg,
                    },
                    checkpoint_path,
                )
                logger.info("saved training checkpoint: %s", checkpoint_path)

        if is_main_process and save_every_epochs > 0 and (epoch + 1) % save_every_epochs == 0:
            pipeline_dir = output_dir / f"pipeline_epoch_{epoch + 1:04d}"
            _save_pipeline_with_optional_ema(
                pipeline_dir,
                unet=components.unet,
                vae=components.vae,
                scheduler=components.scheduler,
                mel_config=mel_config,
                ema_model=ema_model,
            )
            logger.info("saved audio pipeline: %s", pipeline_dir)
        _barrier(distributed)

    if is_main_process:
        final_pipeline_dir = output_dir / "pipeline_final"
        _save_pipeline_with_optional_ema(
            final_pipeline_dir,
            unet=components.unet,
            vae=components.vae,
            scheduler=components.scheduler,
            mel_config=mel_config,
            ema_model=ema_model,
        )
        logger.info("saved final audio pipeline: %s", final_pipeline_dir)
    _barrier(distributed)
    if metric_logger is not None:
        metric_logger.close()
    if writer is not None:
        writer.close()
    _cleanup_distributed(distributed)


def main(argv: list[str] | None = None) -> None:
    args = make_arg_parser().parse_args(argv)
    try:
        train_from_args(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
