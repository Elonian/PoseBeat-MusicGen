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
        sound = AudioSegment.from_file(audio_path, format="wav")
        window = sound[: int(first_seconds * 1000)]
        normalized = match_target_amplitude(window, target_dbfs)
        if gain_db:
            normalized = normalized.apply_gain(gain_db)
        normalized.export(output_dir / audio_path.name, format="wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize generated wav loudness like official DMD post_process.py.")
    parser.add_argument("--config", default="configs/motion_audio_adapter.yaml")
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
