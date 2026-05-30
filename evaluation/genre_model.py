from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SincConvFast(nn.Module):
    """MS-SincResNet Sinc convolution layer used by the official DMD genre metric."""

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel, got {in_channels}")
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class GenreResNet(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.model = models.resnet18(weights=None)
        except TypeError:
            self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        return self.model(x)


class MS_SincResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerNorm = nn.LayerNorm([1, 48000])
        self.sincNet1 = nn.Sequential(
            SincConvFast(out_channels=160, kernel_size=251),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024),
        )
        self.sincNet2 = nn.Sequential(
            SincConvFast(out_channels=160, kernel_size=501),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024),
        )
        self.sincNet3 = nn.Sequential(
            SincConvFast(out_channels=160, kernel_size=1001),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024),
        )
        self.resnet = GenreResNet()

    def forward(self, x):
        x = self.layerNorm(x)
        feat1 = self.sincNet1(x)
        feat2 = self.sincNet2(x)
        feat3 = self.sincNet3(x)
        x = torch.cat((feat1.unsqueeze(dim=1), feat2.unsqueeze(dim=1), feat3.unsqueeze(dim=1)), dim=1)
        x = self.resnet(x)
        return x, feat1, feat2, feat3
