#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import as_path, load_config, optional, require
from visualiser.visualise_aistpp_motion_audio import (
    forward_kinematics,
    read_wav_mono,
    split_slice_key,
)


def read_cdcd_keys(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sliced_motion_positions(motion_dir: Path, key: str) -> np.ndarray | None:
    path = motion_dir / f"{key}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "pos" not in payload or "q" not in payload:
        return None
    root_pos = np.asarray(payload["pos"], dtype=np.float32).reshape(-1, 3)
    rotations = np.asarray(payload["q"], dtype=np.float32).reshape(root_pos.shape[0], 24, 3)
    return forward_kinematics(rotations, root_pos)


def load_raw_motion_positions(
    data_root: Path,
    key: str,
    *,
    data_fps: int,
    raw_fps: int = 60,
    seconds: float = 5.0,
) -> np.ndarray:
    base_key, slice_index = split_slice_key(key)
    path = data_root / "aistplusplus_raw" / "motions" / f"{base_key}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"raw AIST++ motion not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    poses = np.asarray(payload["smpl_poses"], dtype=np.float32).reshape(-1, 24, 3)
    trans = np.asarray(payload["smpl_trans"], dtype=np.float32).reshape(-1, 3)
    step = max(1, raw_fps // data_fps)
    target_frames = int(round(seconds * data_fps))
    start = int(slice_index * seconds * raw_fps)
    end = start + target_frames * step
    if end > len(poses):
        raise ValueError(f"slice {key} exceeds raw motion length in {path}")
    return forward_kinematics(poses[start:end:step][:target_frames], trans[start:end:step][:target_frames])


def motion_beats(positions: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import argrelextrema

    kinetic_velocity = np.mean(
        np.sqrt(np.sum((positions[1:] - positions[:-1]) ** 2, axis=2)),
        axis=1,
    )
    kinetic_velocity = gaussian_filter(kinetic_velocity, 5)
    return argrelextrema(kinetic_velocity, np.less)[0]


def music_beats(audio: np.ndarray, sample_rate: int, motion_fps: int) -> np.ndarray:
    import librosa

    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    return beat_times * motion_fps


def beat_align_score(music_beat_frames: np.ndarray, motion_beat_frames: np.ndarray) -> float:
    if len(music_beat_frames) == 0 or len(motion_beat_frames) == 0:
        return 0.0
    total = 0.0
    for beat in music_beat_frames:
        total += np.exp(-np.min((motion_beat_frames - beat) ** 2) / 2 / 9)
    return float(total / len(music_beat_frames))


def evaluate_bas(
    *,
    data_root: Path,
    music_dir: Path,
    cdcd_list: Path,
    motion_dir: Path | None,
    motion_fps: int,
) -> float:
    scores: list[float] = []
    for key in read_cdcd_keys(cdcd_list):
        audio_path = music_dir / f"{key}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"generated wav not found: {audio_path}")
        audio, sample_rate = read_wav_mono(audio_path)

        positions = load_sliced_motion_positions(motion_dir, key) if motion_dir else None
        if positions is None:
            positions = load_raw_motion_positions(
                data_root,
                key,
                data_fps=motion_fps,
                seconds=min(5.0, len(audio) / sample_rate),
            )

        scores.append(beat_align_score(music_beats(audio, sample_rate, motion_fps), motion_beats(positions)))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute beat alignment score.")
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--motion-dir", default=None)
    parser.add_argument("--music-dir", default=None)
    parser.add_argument("--cdcd-list", default=None)
    parser.add_argument("--motion-fps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    motion_dir = as_path(args.motion_dir) if args.motion_dir else None
    score = evaluate_bas(
        data_root=as_path(args.data_root or require(cfg, "paths.data_root")),
        music_dir=as_path(args.music_dir or require(cfg, "paths.normalized_audio_dir")),
        cdcd_list=as_path(args.cdcd_list or require(cfg, "paths.cdcd_list")),
        motion_dir=motion_dir,
        motion_fps=int(args.motion_fps or optional(cfg, "evaluation.beat_motion_fps", 30)),
    )
    print(f"beat_align_score: {score}")


if __name__ == "__main__":
    main()
