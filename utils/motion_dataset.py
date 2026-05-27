from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_motion_encodings(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        encodings = pickle.load(handle)
    if not isinstance(encodings, dict):
        raise ValueError(f"Motion encoding file must contain a dict: {path}")
    return encodings


def infer_motion_dim(encodings: dict[str, Any]) -> int:
    if not encodings:
        raise ValueError("Cannot infer motion dimension from an empty encoding dict")
    first = next(iter(encodings.values()))
    array = np.asarray(first)
    if array.ndim != 2:
        raise ValueError(f"Expected motion encoding [frames, dim], got {array.shape}")
    return int(array.shape[-1])


def infer_motion_frames(encodings: dict[str, Any]) -> int:
    if not encodings:
        raise ValueError("Cannot infer motion frame count from an empty encoding dict")
    first = next(iter(encodings.values()))
    array = np.asarray(first)
    if array.ndim != 2:
        raise ValueError(f"Expected motion encoding [frames, dim], got {array.shape}")
    return int(array.shape[0])


def motion_key_from_audio_file(audio_file: str | Path) -> str:
    return Path(str(audio_file)).stem


def image_to_mel_tensor(image: Any, *, channels: int = 1) -> torch.Tensor:
    """Match the reference ToTensor()+Normalize([0.5], [0.5]) preprocessing."""

    if channels == 3:
        image = image.convert("RGB")
    elif channels == 1:
        image = image.convert("L")
    else:
        raise ValueError(f"mel images must be 1 or 3 channels, got {channels}")

    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, :, :]
    else:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy((array - 0.5) / 0.5)


class MelConditionDataset(Dataset):
    """Pairs mel-spectrogram images with motion/genre condition tensors."""

    def __init__(
        self,
        mel_dataset_dir: str | Path,
        motion_pickle: str | Path,
        *,
        split: str = "train",
        image_channels: int = 1,
        strict: bool = True,
        limit: int | None = None,
    ):
        from datasets import DatasetDict, load_from_disk

        self.mel_dataset_dir = Path(mel_dataset_dir)
        self.motion_pickle = Path(motion_pickle)
        self.encodings = load_motion_encodings(self.motion_pickle)
        dataset = load_from_disk(str(self.mel_dataset_dir))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        self.dataset = dataset
        self.image_channels = image_channels

        audio_files = list(self.dataset["audio_file"])
        pairs: list[tuple[int, str]] = []
        missing: list[str] = []
        for index, audio_file in enumerate(audio_files):
            key = motion_key_from_audio_file(audio_file)
            if key in self.encodings:
                pairs.append((index, key))
            else:
                missing.append(key)

        if strict and missing:
            preview = ", ".join(missing[:5])
            raise KeyError(
                f"{len(missing)} mel rows have no motion encoding in {self.motion_pickle}. "
                f"First missing keys: {preview}"
            )
        if limit is not None:
            pairs = pairs[:limit]
        if not pairs:
            raise ValueError("No matching mel-image and motion-condition pairs found")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, key = self.pairs[index]
        row = self.dataset[row_index]
        image = image_to_mel_tensor(row["image"], channels=self.image_channels)
        conditioning = torch.as_tensor(np.asarray(self.encodings[key]), dtype=torch.float32)
        return {"key": key, "image": image, "conditioning": conditioning}


def collate_mel_conditions(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "keys": [item["key"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "conditioning": torch.stack([item["conditioning"] for item in batch], dim=0),
    }


def resolve_audio_file(audio_dir: str | Path, key: str) -> Path:
    audio_dir = Path(audio_dir)
    for suffix in (".wav", ".flac", ".mp3"):
        candidate = audio_dir / f"{key}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No audio file found for key {key} in {audio_dir}")


class MotionLatentDataset(Dataset):
    """Legacy cached-latent dataset retained for old experiments."""

    def __init__(
        self,
        motion_pickle: str | Path,
        latent_dir: str | Path,
        *,
        strict: bool = True,
    ):
        self.motion_pickle = Path(motion_pickle)
        self.latent_dir = Path(latent_dir)
        self.encodings = load_motion_encodings(self.motion_pickle)

        keys: list[str] = []
        missing: list[str] = []
        for key in sorted(self.encodings):
            latent_path = self.latent_dir / f"{key}.pt"
            if latent_path.exists():
                keys.append(key)
            else:
                missing.append(key)

        if strict and missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} cached latent files are missing in {self.latent_dir}. "
                f"First missing keys: {preview}"
            )
        if not keys:
            raise ValueError(f"No matching motion/latent pairs found in {self.latent_dir}")
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self.keys[index]
        motion = torch.as_tensor(np.asarray(self.encodings[key]), dtype=torch.float32)
        latent_payload = torch.load(self.latent_dir / f"{key}.pt", map_location="cpu")
        if isinstance(latent_payload, dict):
            latents = latent_payload["latents"]
        else:
            latents = latent_payload
        latents = latents.squeeze(0).to(torch.float32)
        return {"key": key, "motion": motion, "latents": latents}


def collate_motion_latents(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "keys": [item["key"] for item in batch],
        "motion": torch.stack([item["motion"] for item in batch], dim=0),
        "latents": torch.stack([item["latents"] for item in batch], dim=0),
    }
