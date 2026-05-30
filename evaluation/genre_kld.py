#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import as_path, load_config, require


def normalize_data(data: np.ndarray) -> np.ndarray:
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)


def read_cdcd_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_genre_model(model_path: Path, device):
    import torch

    from evaluation.genre_model import MS_SincResNet

    if not model_path.exists():
        raise FileNotFoundError(
            f"MS-SincResNet weights not found: {model_path}. "
            "Download the official genre classifier weights and pass --model-path."
        )
    state_dict = torch.load(model_path, map_location=device)
    model = MS_SincResNet().to(device)
    model.load_state_dict(state_dict["state_dict"])
    model.eval()
    return model


def genre_distribution(model, wav_path: Path, device: torch.device) -> np.ndarray:
    import scipy.io.wavfile
    from scipy import signal

    _, data = scipy.io.wavfile.read(wav_path)
    data = signal.resample(data, 16000 * 30)
    data = data[24000:72000]
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction, _, _, _ = model(tensor)
    return normalize_data(prediction.detach().cpu().numpy()[0])


def evaluate_genre_kld(
    *,
    input_dir: Path,
    output_dir: Path,
    cdcd_list: Path,
    model_path: Path,
    device: torch.device,
) -> float:
    model = load_genre_model(model_path, device)
    from scipy.special import kl_div

    values: list[float] = []
    for name in tqdm(read_cdcd_names(cdcd_list), desc="genre-kld"):
        reference = genre_distribution(model, input_dir / name, device)
        generated = genre_distribution(model, output_dir / name, device)
        values.append(float(np.sum(kl_div(reference, generated))))
    return float(np.mean(ma.masked_invalid(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute genre KLD using MS-SincResNet.")
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cdcd-list", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    score = evaluate_genre_kld(
        input_dir=as_path(args.input_dir or require(cfg, "paths.test_audio_dir")),
        output_dir=as_path(args.output_dir or require(cfg, "paths.normalized_audio_dir")),
        cdcd_list=as_path(args.cdcd_list or require(cfg, "paths.cdcd_list")),
        model_path=as_path(args.model_path or require(cfg, "paths.genre_model_path")),
        device=device,
    )
    print(f"genre_kld: {score}")


if __name__ == "__main__":
    main()
