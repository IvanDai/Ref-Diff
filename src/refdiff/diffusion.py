from __future__ import annotations

import math
from collections.abc import Callable

import torch


def _cosine_betas(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((points / timesteps + offset) / (1 + offset)) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


class GaussianDiffusion(torch.nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if schedule == "cosine":
            betas = _cosine_betas(timesteps)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise ValueError(f"unknown noise schedule: {schedule!r}")
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bar", torch.cumprod(1 - betas, dim=0))

    @staticmethod
    def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
        selected = values.gather(0, timestep)
        return selected.reshape(timestep.shape[0], *((1,) * (ndim - 1)))

    def add_noise(
        self,
        clean: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alpha_bar, timestep, clean.ndim)
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    def predict_clean(
        self,
        noisy: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bar, timestep, noisy.ndim)
        return (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        *,
        steps: int = 50,
        initial_noise: torch.Tensor | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> torch.Tensor:
        if not 1 <= steps <= self.timesteps:
            raise ValueError("DDIM steps must be between 1 and the training timesteps")
        sample = torch.randn_like(condition) if initial_noise is None else initial_noise.clone()
        schedule = torch.linspace(
            self.timesteps - 1, 0, steps, device=condition.device
        ).round().long()
        schedule = torch.unique_consecutive(schedule)
        for index, scalar_timestep in enumerate(schedule):
            timestep = torch.full(
                (condition.shape[0],),
                int(scalar_timestep.item()),
                dtype=torch.long,
                device=condition.device,
            )
            predicted_noise = model(sample, condition, timestep)
            clean = self.predict_clean(sample, predicted_noise, timestep)
            if index == len(schedule) - 1:
                sample = clean
                if progress_callback is not None:
                    progress_callback(index + 1, len(schedule))
                continue
            next_timestep = torch.full_like(timestep, int(schedule[index + 1].item()))
            alpha_next = self._extract(self.alpha_bar, next_timestep, sample.ndim)
            sample = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * predicted_noise
            if progress_callback is not None:
                progress_callback(index + 1, len(schedule))
        return sample
