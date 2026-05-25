from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


@dataclass
class AudioGeneratorMelConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 320
    win_length: int = 1024
    n_mels: int = 64
    f_min: float = 0.0
    f_max: float | None = 8000.0
    audio_length_seconds: float = 5.0
    vae_scale_factor: int = 4

    @property
    def target_frames(self) -> int:
        frames = math.ceil(self.audio_length_seconds * self.sample_rate / self.hop_length)
        return int(math.ceil(frames / self.vae_scale_factor) * self.vae_scale_factor)


def mel_config_from_checkpoint(checkpoint_dir: str | Path, audio_length_seconds: float) -> AudioGeneratorMelConfig:
    root = Path(checkpoint_dir)
    with open(root / "vocoder" / "config.json", "r", encoding="utf-8") as handle:
        vocoder_config = json.load(handle)
    with open(root / "vae" / "config.json", "r", encoding="utf-8") as handle:
        vae_config = json.load(handle)

    sample_rate = int(vocoder_config["sampling_rate"])
    hop_length = int(math.prod(vocoder_config["upsample_rates"]))
    n_mels = int(vocoder_config["model_in_dim"])
    scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1)
    return AudioGeneratorMelConfig(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=n_mels,
        f_max=sample_rate / 2,
        audio_length_seconds=audio_length_seconds,
        vae_scale_factor=scale_factor,
    )


class AudioToLatents:
    """Converts waveform files into frozen audio-generator VAE latents."""

    def __init__(
        self,
        vae: torch.nn.Module,
        config: AudioGeneratorMelConfig,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.vae = vae
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        ).to(self.device)

    def load_audio(self, path: str | Path) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(path)
        except ImportError:
            import soundfile as sf

            audio, sample_rate = sf.read(path, always_2d=True)
            waveform = torch.from_numpy(np.asarray(audio).T).to(torch.float32)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate,
            )

        target_samples = int(round(self.config.audio_length_seconds * self.config.sample_rate))
        if waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_samples]

        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        waveform = waveform / waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return waveform * 0.5

    def wav_to_log_mel(self, path: str | Path) -> torch.Tensor:
        waveform = self.load_audio(path).to(self.device, dtype=self.dtype)
        mel = self.mel(waveform)
        log_mel = torch.log(mel.clamp_min(1e-5))
        log_mel = log_mel.transpose(1, 2)

        target_frames = self.config.target_frames
        if log_mel.shape[1] < target_frames:
            log_mel = F.pad(log_mel, (0, 0, 0, target_frames - log_mel.shape[1]))
        else:
            log_mel = log_mel[:, :target_frames, :]
        return log_mel.unsqueeze(1)

    @torch.no_grad()
    def encode_file(self, path: str | Path) -> torch.Tensor:
        mel = self.wav_to_log_mel(path)
        latents = self.vae.encode(mel).latent_dist.sample()
        return latents * self.vae.config.scaling_factor
