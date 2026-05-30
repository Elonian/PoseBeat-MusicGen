#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value
from diffusers.pipelines.audio_diffusion import Mel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, optional, require

logger = logging.getLogger("posebeat.render_mel_dataset")


def _parse_resolution(value: str | None, cfg: dict) -> tuple[int, int]:
    if value is None:
        return int(optional(cfg, "audio.x_res", 256)), int(optional(cfg, "audio.y_res", 256))
    try:
        size = int(value)
        return size, size
    except ValueError:
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError("resolution must be a single integer or width,height") from None
        return parts[0], parts[1]


def _is_silent_image(image) -> bool:
    return bool(np.all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255))


def _resolve_paths(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path]:
    input_dir = Path(args.input_dir or require(cfg, "paths.train_audio_dir")).expanduser()
    output_dir = Path(args.output_dir or require(cfg, "paths.train_mel_dataset_dir")).expanduser()
    return input_dir, output_dir


def _iter_mel_examples(
    *,
    audio_files: list[str],
    x_res: int,
    y_res: int,
    hop_length: int,
    sample_rate: int,
    n_fft: int,
):
    mel = Mel(
        x_res=x_res,
        y_res=y_res,
        hop_length=hop_length,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )

    for audio_file_text in tqdm(audio_files, desc="render-mel"):
        audio_file = Path(audio_file_text)
        try:
            mel.load_audio(str(audio_file))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.warning("skipping unreadable audio %s: %s", audio_file, exc)
            continue

        for slice_index in range(mel.get_number_of_slices()):
            image = mel.audio_slice_to_image(slice_index)
            if image.width != x_res or image.height != y_res:
                raise RuntimeError(
                    f"unexpected image size for {audio_file}: {(image.width, image.height)}"
                )
            if _is_silent_image(image):
                logger.warning("skipping silent slice %s slice=%d", audio_file, slice_index)
                continue
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                image_bytes = output.getvalue()
            yield {
                "image": {"bytes": image_bytes},
                "audio_file": str(audio_file),
                "slice": slice_index,
            }


def render_dataset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    input_dir, output_dir = _resolve_paths(args, cfg)
    x_res, y_res = _parse_resolution(args.resolution, cfg)
    hop_length = int(args.hop_length or optional(cfg, "audio.hop_length", 512))
    sample_rate = int(args.sample_rate or optional(cfg, "audio.sample_rate", 22050))
    n_fft = int(args.n_fft or optional(cfg, "audio.n_fft", 2048))

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output dataset already exists: {output_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"input audio directory does not exist: {input_dir}")

    source_audio_files = sorted(input_dir.glob("*.wav"))
    empty_audio_files = [path for path in source_audio_files if path.stat().st_size == 0]
    if empty_audio_files:
        preview = ", ".join(path.name for path in empty_audio_files[:5])
        logger.warning(
            "skipping %d empty audio files in %s; first empty files: %s",
            len(empty_audio_files),
            input_dir,
            preview,
        )
    audio_files = [str(path) for path in source_audio_files if path.stat().st_size > 0]
    if args.limit is not None:
        audio_files = audio_files[: args.limit]
    if not audio_files:
        raise ValueError(f"no .wav files found in {input_dir}")

    features = Features(
        {
            "image": Image(),
            "audio_file": Value(dtype="string"),
            "slice": Value(dtype="int16"),
        }
    )
    dataset = Dataset.from_generator(
        _iter_mel_examples,
        features=features,
        gen_kwargs={
            "audio_files": audio_files,
            "x_res": x_res,
            "y_res": y_res,
            "hop_length": hop_length,
            "sample_rate": sample_rate,
            "n_fft": n_fft,
        },
        num_proc=args.num_proc,
    )
    if len(dataset) == 0:
        raise ValueError("no valid mel images were rendered")

    dataset_dict = DatasetDict({args.dataset_split: dataset})
    dataset_dict.save_to_disk(str(output_dir))
    if args.push_to_hub:
        dataset_dict.push_to_hub(args.push_to_hub)

    logger.info(
        "rendered rows=%d rendered_input_files=%d source_input_files=%d skipped_empty_files=%d output=%s",
        len(dataset),
        len(audio_files),
        len(source_audio_files),
        len(empty_audio_files),
        output_dir,
    )


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a mel-image Arrow dataset from sliced WAV files."
    )
    parser.add_argument("--config", default="configs/motion_to_music.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resolution", default=None, help="Single integer or width,height.")
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    render_dataset(make_arg_parser().parse_args())


if __name__ == "__main__":
    main()
