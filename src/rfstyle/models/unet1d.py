from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=t.device))
        angles = t.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        groups = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(time)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet1D(nn.Module):
    """Small frequency-domain U-Net: (X_t, Y, t) -> predicted noise."""

    def __init__(self, base_channels: int = 32, channel_mults=(1, 2, 4)):
        super().__init__()
        channels = [base_channels * m for m in channel_mults]
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv1d(4, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, channel in enumerate(channels):
            self.down_blocks.append(ResidualBlock(channel, channel, time_dim))
            if index < len(channels) - 1:
                self.downsamples.append(nn.Conv1d(channel, channels[index + 1], 4, stride=2, padding=1))
        self.middle = ResidualBlock(channels[-1], channels[-1], time_dim)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in reversed(range(len(channels))):
            channel = channels[index]
            self.up_blocks.append(ResidualBlock(channel * 2, channel, time_dim))
            if index > 0:
                self.upsamples.append(nn.ConvTranspose1d(channel, channels[index - 1], 4, stride=2, padding=1))
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv1d(channels[0], 2, 3, padding=1),
        )

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        time = self.time_embedding(timestep)
        x = self.input(torch.cat((noisy_target, condition), dim=1))
        skips = []
        for index, block in enumerate(self.down_blocks):
            x = block(x, time)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
        x = self.middle(x, time)
        for index, block in enumerate(self.up_blocks):
            x = block(torch.cat((x, skips.pop()), dim=1), time)
            if index < len(self.upsamples):
                x = self.upsamples[index](x)
        return self.output(x)

