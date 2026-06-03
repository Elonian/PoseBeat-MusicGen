#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_cdcd import beat_detect, beat_scores
from utils.config import as_path, load_config, optional, require


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def _read_cdcd_keys(path: Path) -> list[str]:
    return [Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_reference_keys(reference_dir: Path) -> list[str]:
    return sorted(path.stem for path in reference_dir.glob("*.wav"))


def _limited(keys: list[str], limit: int | None) -> list[str]:
    return keys if limit is None else keys[:limit]


def _resolve_keys(args: argparse.Namespace, cfg: dict[str, Any], reference_dir: Path) -> list[str]:
    if args.key_source == "cdcd":
        return _limited(_read_cdcd_keys(as_path(args.cdcd_list or require(cfg, "paths.cdcd_list"))), args.limit)
    return _limited(_read_reference_keys(reference_dir), args.limit)


def _pair_inventory(keys: list[str], reference_dir: Path, generated_dir: Path) -> dict[str, Any]:
    missing_reference = [key for key in keys if not (reference_dir / f"{key}.wav").exists()]
    missing_generated = [key for key in keys if not (generated_dir / f"{key}.wav").exists()]
    paired = [
        key
        for key in keys
        if (reference_dir / f"{key}.wav").exists() and (generated_dir / f"{key}.wav").exists()
    ]
    return {
        "requested": len(keys),
        "paired": len(paired),
        "missing_reference": missing_reference,
        "missing_generated": missing_generated,
        "paired_keys": paired,
    }


def _metric_error(message: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": message}


def _need_modules(*names: str) -> str | None:
    missing = [name for name in names if not _module_available(name)]
    if missing:
        return "missing Python dependencies: " + ", ".join(missing)
    return None


def compute_beat_coverage_hit(
    keys: list[str],
    reference_dir: Path,
    generated_dir: Path,
) -> dict[str, Any]:
    missing = _need_modules("librosa")
    if missing:
        return _metric_error(missing)

    import librosa

    cover_scores: list[float] = []
    hit_scores: list[float] = []
    for key in keys:
        reference_audio, sr = librosa.load(reference_dir / f"{key}.wav", sr=22050)
        generated_audio, _ = librosa.load(generated_dir / f"{key}.wav", sr=sr)
        cover, hit = beat_scores(beat_detect(reference_audio, sr), beat_detect(generated_audio, sr))
        cover_scores.append(cover)
        hit_scores.append(hit)

    return {
        "status": "ok",
        "beat_coverage": float(np.mean(cover_scores)),
        "beat_hit": float(np.mean(hit_scores)),
        "items": len(keys),
    }


def compute_beat_alignment(
    keys: list[str],
    *,
    data_root: Path,
    generated_dir: Path,
    motion_dir: Path | None,
    motion_fps: int,
) -> dict[str, Any]:
    missing = _need_modules("librosa", "scipy")
    if missing:
        return _metric_error(missing)

    from evaluation.beat_align_score import (
        beat_align_score,
        load_raw_motion_positions,
        load_sliced_motion_positions,
        motion_beats,
        music_beats,
    )
    from visualiser.visualise_aistpp_motion_audio import read_wav_mono

    scores: list[float] = []
    for key in keys:
        audio, sample_rate = read_wav_mono(generated_dir / f"{key}.wav")
        positions = load_sliced_motion_positions(motion_dir, key) if motion_dir else None
        if positions is None:
            positions = load_raw_motion_positions(
                data_root,
                key,
                data_fps=motion_fps,
                seconds=min(5.0, len(audio) / sample_rate),
            )
        scores.append(beat_align_score(music_beats(audio, sample_rate, motion_fps), motion_beats(positions)))

    return {"status": "ok", "beat_align_score": float(np.mean(scores)), "items": len(keys)}


def _stage_audio_pairs(keys: list[str], reference_dir: Path, generated_dir: Path) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    temp = tempfile.TemporaryDirectory(prefix="posebeat_eval_")
    root = Path(temp.name)
    ref_stage = root / "reference"
    gen_stage = root / "generated"
    ref_stage.mkdir()
    gen_stage.mkdir()
    for key in keys:
        for source_dir, target_dir in ((reference_dir, ref_stage), (generated_dir, gen_stage)):
            source = source_dir / f"{key}.wav"
            target = target_dir / source.name
            try:
                os.symlink(source, target)
            except OSError:
                shutil.copy2(source, target)
    return temp, ref_stage, gen_stage


def compute_fad(
    keys: list[str],
    reference_dir: Path,
    generated_dir: Path,
    *,
    reference_mode: str,
) -> dict[str, Any]:
    missing = _need_modules("frechet_audio_distance")
    if missing:
        return _metric_error(missing)

    from frechet_audio_distance import FrechetAudioDistance

    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    if reference_mode == "all":
        return {
            "status": "ok",
            "fad": float(frechet.score(str(reference_dir), str(generated_dir))),
            "reference_mode": "all_reference_wavs",
            "reference_items": len(list(reference_dir.glob("*.wav"))),
            "generated_items": len(list(generated_dir.glob("*.wav"))),
        }

    temp, ref_stage, gen_stage = _stage_audio_pairs(keys, reference_dir, generated_dir)
    try:
        return {
            "status": "ok",
            "fad": float(frechet.score(str(ref_stage), str(gen_stage))),
            "reference_mode": "paired_cdcd_wavs",
            "items": len(keys),
        }
    finally:
        temp.cleanup()


def _normalize_data(values: np.ndarray) -> np.ndarray:
    data_min = np.min(values)
    data_max = np.max(values)
    if data_max == data_min:
        return np.zeros_like(values)
    return (values - data_min) / (data_max - data_min)


def _genre_distribution(model, wav_path: Path, device) -> np.ndarray:
    import scipy.io.wavfile
    import torch
    from scipy import signal

    _, data = scipy.io.wavfile.read(wav_path)
    data = signal.resample(data, 16000 * 30)
    data = data[24000:72000]
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction, _, _, _ = model(tensor)
    return _normalize_data(prediction.detach().cpu().numpy()[0])


def compute_genre_kld(
    keys: list[str],
    *,
    reference_dir: Path,
    generated_dir: Path,
    model_path: Path,
    device_name: str | None,
) -> dict[str, Any]:
    missing = _need_modules("torch", "scipy")
    if missing:
        return _metric_error(missing)
    if not model_path.exists():
        return _metric_error(f"MS-SincResNet weights not found: {model_path}")

    import torch
    from scipy.special import kl_div

    from evaluation.genre_model import MS_SincResNet

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(model_path, map_location=device)
    model = MS_SincResNet().to(device)
    model.load_state_dict(state_dict["state_dict"])
    model.eval()

    values: list[float] = []
    for key in keys:
        reference = _genre_distribution(model, reference_dir / f"{key}.wav", device)
        generated = _genre_distribution(model, generated_dir / f"{key}.wav", device)
        values.append(float(np.sum(kl_div(reference, generated))))

    return {"status": "ok", "genre_kld": float(np.mean(np.ma.masked_invalid(values))), "items": len(keys)}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    reference_dir = as_path(args.reference_dir or require(cfg, "paths.test_audio_dir"))
    generated_dir = as_path(args.generated_dir or require(cfg, "paths.normalized_audio_dir"))
    data_root = as_path(args.data_root or require(cfg, "paths.data_root"))
    genre_model_path = as_path(args.genre_model_path or require(cfg, "paths.genre_model_path"))
    motion_dir = as_path(args.motion_dir) if args.motion_dir else None

    keys = _resolve_keys(args, cfg, reference_dir)
    inventory = _pair_inventory(keys, reference_dir, generated_dir)
    paired_keys = inventory["paired_keys"]
    result: dict[str, Any] = {
        "config": str(as_path(args.config)),
        "key_source": args.key_source,
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "data_root": str(data_root),
        "inventory": {
            key: value
            for key, value in inventory.items()
            if key != "paired_keys"
        },
        "metrics": {},
    }

    if inventory["missing_reference"] or inventory["missing_generated"]:
        result["ready"] = False
        result["reason"] = "missing reference/generated wav pairs"
        if args.strict:
            preview = ", ".join((inventory["missing_reference"] + inventory["missing_generated"])[:10])
            raise FileNotFoundError(f"Evaluation pairs are incomplete. First missing keys: {preview}")
    else:
        result["ready"] = True

    if not paired_keys:
        result["reason"] = "no paired wav files available"
        return result

    metrics = set(args.metrics)
    if "beat" in metrics:
        result["metrics"]["beat"] = compute_beat_coverage_hit(paired_keys, reference_dir, generated_dir)
    if "bas" in metrics:
        result["metrics"]["bas"] = compute_beat_alignment(
            paired_keys,
            data_root=data_root,
            generated_dir=generated_dir,
            motion_dir=motion_dir,
            motion_fps=int(args.motion_fps or optional(cfg, "evaluation.beat_motion_fps", 30)),
        )
    if "fad" in metrics:
        result["metrics"]["fad"] = compute_fad(
            paired_keys,
            reference_dir,
            generated_dir,
            reference_mode=args.fad_reference_mode,
        )
    if "genre" in metrics:
        result["metrics"]["genre"] = compute_genre_kld(
            paired_keys,
            reference_dir=reference_dir,
            generated_dir=generated_dir,
            model_path=genre_model_path,
            device_name=args.device,
        )

    unavailable = {
        name: payload
        for name, payload in result["metrics"].items()
        if isinstance(payload, dict) and payload.get("status") != "ok"
    }
    if unavailable and args.strict:
        raise RuntimeError(f"Some metrics were unavailable: {unavailable}")
    return result


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DMD-style PoseBeat evaluation with preflight checks."
    )
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--reference-dir", default=None, help="Ground-truth sliced wav directory.")
    parser.add_argument("--generated-dir", default=None, help="Generated/normalized wav directory.")
    parser.add_argument("--data-root", default=None, help="AIST data root for raw motion lookup.")
    parser.add_argument("--motion-dir", default=None, help="Optional sliced motion pickle directory.")
    parser.add_argument("--cdcd-list", default=None)
    parser.add_argument("--genre-model-path", default=None)
    parser.add_argument("--key-source", choices=("cdcd", "test"), default="cdcd")
    parser.add_argument("--metrics", nargs="+", choices=("beat", "bas", "fad", "genre"), default=["beat", "bas", "fad", "genre"])
    parser.add_argument(
        "--fad-reference-mode",
        choices=("all", "paired"),
        default="all",
        help="Use all reference wavs like the original DMD eval script, or only paired CDCD wavs.",
    )
    parser.add_argument("--motion-fps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail if pairs/dependencies/weights are missing.")
    parser.add_argument("--json-output", default=None)
    return parser


def main() -> None:
    args = make_arg_parser().parse_args()
    result = evaluate(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json_output:
        output_path = as_path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
