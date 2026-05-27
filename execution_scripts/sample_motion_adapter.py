#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import as_path, load_config, optional, require
from utils.logging import setup_logging
from utils.motion_dataset import load_motion_encodings


def _load_audio_diffusion_pipeline(model_dir: Path, device: torch.device):
    try:
        from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline

        return AudioDiffusionPipeline.from_pretrained(model_dir, local_files_only=True).to(device)
    except Exception:
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(model_dir, local_files_only=True).to(device)


def _audio_from_output(output, pipe) -> tuple[int, np.ndarray]:
    mel = getattr(pipe, "mel", None)
    sample_rate = mel.get_sample_rate() if mel is not None else None
    audio = output.audios
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    while audio.ndim > 1:
        audio = audio[0]
    if sample_rate is None:
        sample_rate = 22050
    return int(sample_rate), np.asarray(audio, dtype=np.float32)


def generate_one(
    *,
    pipe,
    key: str,
    conditioning: np.ndarray,
    output_path: Path,
    device: torch.device,
    seed: int,
    eta: float,
) -> None:
    generator = torch.Generator(device=device).manual_seed(seed)
    encoding = torch.as_tensor(conditioning, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = pipe(generator=generator, eta=eta, encoding=encoding)
    sample_rate, audio = _audio_from_output(output, pipe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import scipy.io.wavfile

    scipy.io.wavfile.write(output_path, sample_rate, np.clip(audio, -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with the motion-conditioned pipeline.")
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--motion-key", default=None)
    parser.add_argument("--all-cdcd", action="store_true")
    parser.add_argument("--cdcd-list", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = as_path(args.output_dir or require(cfg, "paths.generated_audio_dir"))
    logger = setup_logging(output_dir / "logs", name="posebeat.sample", filename="sample_motion_adapter.log")

    model_dir = as_path(args.model_dir or require(cfg, "paths.pretrained_model_dir"))
    motion_pickle = as_path(require(cfg, "paths.test_motion_pickle"))
    encodings = load_motion_encodings(motion_pickle)
    pipe = _load_audio_diffusion_pipeline(model_dir, device)
    seed = int(args.seed if args.seed is not None else optional(cfg, "inference.seed", 2391504374279719))
    eta = float(args.eta if args.eta is not None else optional(cfg, "inference.eta", 0.0))
    logger.info("model=%s motion_pickle=%s device=%s seed=%d eta=%s", model_dir, motion_pickle, device, seed, eta)

    if args.all_cdcd:
        list_path = as_path(args.cdcd_list or require(cfg, "paths.cdcd_list"))
        names = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = [Path(name).stem for name in names]
    else:
        keys = [args.motion_key or sorted(encodings)[0]]

    for key in tqdm(keys, desc="generate-audio"):
        if key not in encodings:
            raise KeyError(f"motion key not found in {motion_pickle}: {key}")
        output_path = Path(args.output) if args.output and len(keys) == 1 else output_dir / f"{key}.wav"
        generate_one(
            pipe=pipe,
            key=key,
            conditioning=np.asarray(encodings[key], dtype=np.float32),
            output_path=output_path,
            device=device,
            seed=seed,
            eta=eta,
        )
        logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
