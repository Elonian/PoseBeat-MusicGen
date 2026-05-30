#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk
from diffusers.pipelines.audio_diffusion import Mel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, optional, require
from utils.motion_dataset import motion_key_from_audio_file

logger = logging.getLogger("posebeat.build_full_hires_mel_dataset")


def _image_bytes(image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def _dataset_split(path: str | Path, split: str):
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        return dataset[split]
    return dataset


def _real_index_by_key(real_dataset) -> dict[str, int]:
    return {
        motion_key_from_audio_file(audio_file): index
        for index, audio_file in enumerate(real_dataset["audio_file"])
    }


def _balanced_index_shards(
    reference_keys: list[str],
    real_keys: set[str],
    num_shards: int,
    *,
    original_indices: list[int] | None = None,
) -> list[list[int]]:
    if original_indices is None:
        original_indices = list(range(len(reference_keys)))

    real_indices = []
    reconstructed_indices = []
    for local_index, key in enumerate(reference_keys):
        index = original_indices[local_index]
        if key in real_keys:
            real_indices.append(index)
        else:
            reconstructed_indices.append(index)

    shards = [[] for _ in range(num_shards)]
    for offset, index in enumerate(reconstructed_indices):
        shards[offset % num_shards].append(index)
    for offset, index in enumerate(real_indices):
        shards[offset % num_shards].append(index)
    return shards


def _iter_rows(
    *,
    indices: list[int],
    reference_dataset_dir: str,
    real_dataset_dir: str,
    split: str,
    old_x_res: int,
    old_y_res: int,
    old_hop_length: int,
    new_x_res: int,
    new_y_res: int,
    new_hop_length: int,
    sample_rate: int,
    n_fft: int,
    top_db: int,
    reconstruction_n_iter: int,
):
    reference = _dataset_split(reference_dataset_dir, split)
    real = _dataset_split(real_dataset_dir, split)
    real_by_key = _real_index_by_key(real)
    old_mel = Mel(
        x_res=old_x_res,
        y_res=old_y_res,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=old_hop_length,
        top_db=top_db,
        n_iter=reconstruction_n_iter,
    )
    new_mel = Mel(
        x_res=new_x_res,
        y_res=new_y_res,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=new_hop_length,
        top_db=top_db,
    )

    flat_indices = []
    for item in indices:
        if isinstance(item, list):
            flat_indices.extend(item)
        else:
            flat_indices.append(item)

    for index in tqdm(flat_indices, desc="build-full-hires"):
        row = reference[index]
        key = motion_key_from_audio_file(row["audio_file"])
        if key in real_by_key:
            image = real[real_by_key[key]]["image"]
        else:
            audio = old_mel.image_to_audio(row["image"])
            new_mel.load_audio(raw_audio=audio)
            image = new_mel.audio_slice_to_image(0)

        yield {
            "image": {"bytes": _image_bytes(image)},
            "audio_file": row["audio_file"],
            "slice": row["slice"],
        }


def build_dataset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    split = str(optional(cfg, "data.dataset_split", "train"))
    reference_dir = Path(args.reference_dataset_dir or require(cfg, "paths.reference_mel_dataset_dir"))
    real_dir = Path(args.real_dataset_dir or require(cfg, "paths.real_mel_dataset_dir"))
    output_dir = Path(args.output_dir or require(cfg, "paths.train_mel_dataset_dir"))
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output dataset already exists: {output_dir}")

    reference = _dataset_split(reference_dir, split)
    real = _dataset_split(real_dir, split)
    reference_keys = [motion_key_from_audio_file(audio_file) for audio_file in reference["audio_file"]]
    real_keys = set(_real_index_by_key(real))
    reconstructed = sum(1 for key in reference_keys if key not in real_keys)
    reused = len(reference_keys) - reconstructed
    logger.info(
        "building full hires dataset: reference_rows=%d real_rows=%d reused_real=%d reconstructed_from_reference=%d",
        len(reference),
        len(real),
        reused,
        reconstructed,
    )

    features = Features(
        {
            "image": Image(),
            "audio_file": Value(dtype="string"),
            "slice": Value(dtype="int16"),
        }
    )
    indices = _balanced_index_shards(reference_keys, real_keys, max(1, int(args.num_proc or 1)))
    if args.limit is not None:
        limited = list(range(len(reference)))[: args.limit]
        indices = _balanced_index_shards(
            [reference_keys[index] for index in limited],
            real_keys,
            max(1, int(args.num_proc or 1)),
            original_indices=limited,
        )
    if args.num_proc is None or args.num_proc <= 1:
        indices_for_generator = [index for shard in indices for index in shard]
    else:
        indices_for_generator = indices

    dataset = Dataset.from_generator(
        _iter_rows,
        features=features,
        gen_kwargs={
            "indices": indices_for_generator,
            "reference_dataset_dir": str(reference_dir),
            "real_dataset_dir": str(real_dir),
            "split": split,
            "old_x_res": int(optional(cfg, "full_build.reference_x_res", 256)),
            "old_y_res": int(optional(cfg, "full_build.reference_y_res", 256)),
            "old_hop_length": int(optional(cfg, "full_build.reference_hop_length", 512)),
            "new_x_res": int(optional(cfg, "audio.x_res", 512)),
            "new_y_res": int(optional(cfg, "audio.y_res", 512)),
            "new_hop_length": int(optional(cfg, "audio.hop_length", 256)),
            "sample_rate": int(optional(cfg, "audio.sample_rate", 22050)),
            "n_fft": int(optional(cfg, "audio.n_fft", 2048)),
            "top_db": int(optional(cfg, "audio.top_db", 80)),
            "reconstruction_n_iter": int(args.reconstruction_n_iter),
        },
        num_proc=args.num_proc,
    )
    DatasetDict({split: dataset}).save_to_disk(str(output_dir))

    manifest = {
        "reference_dataset_dir": str(reference_dir),
        "real_dataset_dir": str(real_dir),
        "output_dir": str(output_dir),
        "rows": len(dataset),
        "limited_rows": args.limit,
        "reused_real_rows": reused,
        "reconstructed_rows": reconstructed,
        "reconstruction_n_iter": int(args.reconstruction_n_iter),
    }
    with (output_dir / "build_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logger.info("saved full hires dataset: %s", output_dir)


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the full high-resolution mel dataset using real rows plus reconstructed missing rows."
    )
    parser.add_argument("--config", default="configs/motion_to_music_hires.yaml")
    parser.add_argument("--reference-dataset-dir", default=None)
    parser.add_argument("--real-dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reconstruction-n-iter", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(make_arg_parser().parse_args())


if __name__ == "__main__":
    main()
