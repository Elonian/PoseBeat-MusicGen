#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import as_path, load_config, optional, require


def beat_detect(audio: np.ndarray, sr: int = 22050) -> list[int]:
    """Reference beat coverage/hit binning: one onset bin per second."""

    import librosa

    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        wait=1,
        delta=0.2,
        pre_avg=1,
        post_avg=1,
        post_max=1,
        units="time",
    )
    seconds = int(np.ceil(len(audio) / sr))
    beats = [0] * seconds
    for onset_time in onsets:
        index = int(np.trunc(onset_time))
        if 0 <= index < len(beats):
            beats[index] = 1
    return beats


def beat_scores(reference: list[int], generated: list[int]) -> tuple[float, float]:
    if len(reference) != len(generated):
        length = min(len(reference), len(generated))
        reference = reference[:length]
        generated = generated[:length]
    total_beats = sum(reference)
    if total_beats == 0:
        return 0.0, 0.0
    cover_beats = sum(generated)
    hit_beats = sum(1 for ref, gen in zip(reference, generated) if ref == 1 and gen == 1)
    return cover_beats / total_beats, hit_beats / total_beats


def read_cdcd_keys(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_cdcd(
    *,
    input_dir: Path,
    output_dir: Path,
    cdcd_list: Path,
    compute_fad: bool,
) -> dict[str, float]:
    keys = read_cdcd_keys(cdcd_list)
    cover_scores: list[float] = []
    hit_scores: list[float] = []
    missing: list[str] = []

    for key in keys:
        reference_path = input_dir / f"{key}.wav"
        generated_path = output_dir / f"{key}.wav"
        if not reference_path.exists() or not generated_path.exists():
            missing.append(key)
            continue

        import librosa

        reference_audio, sr = librosa.load(reference_path, sr=22050)
        generated_audio, _ = librosa.load(generated_path, sr=sr)
        cover, hit = beat_scores(beat_detect(reference_audio, sr), beat_detect(generated_audio, sr))
        cover_scores.append(cover)
        hit_scores.append(hit)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} CDCD wav pairs are missing. First missing keys: {preview}")
    if not cover_scores:
        raise ValueError(f"No CDCD wav pairs found for {cdcd_list}")

    result = {
        "beat_coverage": float(np.mean(cover_scores)),
        "beat_hit": float(np.mean(hit_scores)),
    }
    if compute_fad:
        from frechet_audio_distance import FrechetAudioDistance

        frechet = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        result["fad"] = float(frechet.score(str(input_dir), str(output_dir)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CDCD beat coverage/hit and optional FAD.")
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cdcd-list", default=None)
    parser.add_argument("--skip-fad", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = evaluate_cdcd(
        input_dir=as_path(args.input_dir or require(cfg, "paths.test_audio_dir")),
        output_dir=as_path(args.output_dir or require(cfg, "paths.normalized_audio_dir")),
        cdcd_list=as_path(args.cdcd_list or require(cfg, "paths.cdcd_list")),
        compute_fad=not args.skip_fad and bool(optional(cfg, "evaluation.compute_fad", True)),
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
