#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import as_path, load_config, optional, require


def match_target_amplitude(sound, target_dbfs: float):
    return sound.apply_gain(target_dbfs - sound.dBFS)


def _normalize_with_soundfile(
    audio_path: Path,
    output_path: Path,
    *,
    target_dbfs: float,
    gain_db: float,
    first_seconds: float,
) -> bool:
    try:
        import soundfile as sf
    except Exception:
        return False

    try:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        return False

    import numpy as np

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame_count = int(round(first_seconds * sample_rate))
    if frame_count > 0:
        audio = audio[:frame_count]
    if audio.size == 0:
        return False

    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms > 0:
        current_dbfs = 20.0 * np.log10(rms)
        audio = audio * float(10.0 ** ((target_dbfs - current_dbfs + gain_db) / 20.0))
    elif gain_db:
        audio = audio * float(10.0 ** (gain_db / 20.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return True


def normalize_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    target_dbfs: float = -20.0,
    gain_db: float = 8.0,
    first_seconds: float = 5.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    from pydub import AudioSegment

    for audio_path in tqdm(sorted(input_dir.glob("*.wav")), desc="normalize-loudness"):
        output_path = output_dir / audio_path.name
        if _normalize_with_soundfile(
            audio_path,
            output_path,
            target_dbfs=target_dbfs,
            gain_db=gain_db,
            first_seconds=first_seconds,
        ):
            continue
        sound = AudioSegment.from_file(audio_path, format="wav")
        window = sound[: int(first_seconds * 1000)]
        normalized = match_target_amplitude(window, target_dbfs)
        if gain_db:
            normalized = normalized.apply_gain(gain_db)
        normalized.export(output_path, format="wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize generated wav loudness for evaluation.")
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-dbfs", type=float, default=None)
    parser.add_argument("--gain-db", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_folder(
        as_path(args.input_dir or require(cfg, "paths.generated_audio_dir")),
        as_path(args.output_dir or require(cfg, "paths.normalized_audio_dir")),
        target_dbfs=float(
            args.target_dbfs
            if args.target_dbfs is not None
            else optional(cfg, "evaluation.normalize_target_dbfs", -20.0)
        ),
        gain_db=float(
            args.gain_db
            if args.gain_db is not None
            else optional(cfg, "evaluation.normalize_gain_db", 8.0)
        ),
    )


if __name__ == "__main__":
    main()
