from __future__ import annotations

import math

import torch
from torch import nn


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least 4")
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=timestep.device) / (half - 1))
        angles = timestep.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.activation(self.norm1(x)))
        hidden = hidden + self.time_projection(self.activation(time))[:, :, None]
        return self.conv2(self.activation(self.norm2(hidden))) + self.skip(x)


class ConditionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditionalUNet1D(nn.Module):
    """RefDiff 1.1 time-domain conditional U-Net for IQ waveforms."""

    def __init__(self, base_channels: int = 32, channel_multipliers: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers cannot be empty")
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        time_channels = base_channels * 4
        self.required_multiple = 2 ** (len(channels) - 1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels), nn.Linear(base_channels, time_channels),
            nn.SiLU(), nn.Linear(time_channels, time_channels)
        )
        self.condition_input = nn.Conv1d(2, channels[0], 3, padding=1)
        self.condition_blocks = nn.ModuleList(ConditionBlock(c) for c in channels)
        self.condition_downsamples = nn.ModuleList(
            nn.Conv1d(channels[i], channels[i + 1], 4, stride=2, padding=1)
            for i in range(len(channels) - 1)
        )
        self.noisy_input = nn.Conv1d(2, channels[0], 3, padding=1)
        self.condition_projections = nn.ModuleList(nn.Conv1d(c, c, 1) for c in channels)
        self.down_blocks = nn.ModuleList(ResidualBlock(c, c, time_channels) for c in channels)
        self.downsamples = nn.ModuleList(
            nn.Conv1d(channels[i], channels[i + 1], 4, stride=2, padding=1)
            for i in range(len(channels) - 1)
        )
        self.middle = ResidualBlock(channels[-1], channels[-1], time_channels)
        self.up_blocks = nn.ModuleList(ResidualBlock(c * 2, c, time_channels) for c in reversed(channels))
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose1d(channels[i], channels[i - 1], 4, stride=2, padding=1)
            for i in reversed(range(1, len(channels)))
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(channels[0]), channels[0]), nn.SiLU(),
            nn.Conv1d(channels[0], 2, 3, padding=1)
        )

    def forward(self, noisy_target: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if noisy_target.shape != condition.shape or noisy_target.ndim != 3 or noisy_target.shape[1] != 2:
            raise ValueError("inputs must share shape [batch, 2, length]")
        if noisy_target.shape[-1] % self.required_multiple:
            raise ValueError(f"signal length must be divisible by {self.required_multiple}")
        time = self.time_embedding(timestep)
        condition_features, encoded = [], self.condition_input(condition)
        for level, block in enumerate(self.condition_blocks):
            encoded = block(encoded)
            condition_features.append(encoded)
            if level < len(self.condition_downsamples):
                encoded = self.condition_downsamples[level](encoded)
        hidden, skips = self.noisy_input(noisy_target), []
        for level, block in enumerate(self.down_blocks):
            hidden = block(hidden + self.condition_projections[level](condition_features[level]), time)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        hidden = self.middle(hidden, time)
        for level, block in enumerate(self.up_blocks):
            hidden = block(torch.cat((hidden, skips.pop()), dim=1), time)
            if level < len(self.upsamples):
                hidden = self.upsamples[level](hidden)
        return self.output(hidden)

