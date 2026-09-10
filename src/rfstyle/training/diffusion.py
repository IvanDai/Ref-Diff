from __future__ import annotations

import torch


class GaussianDiffusion:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (ndim - 1)))

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alpha_bar, t, clean.ndim)
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def ddim_sample(self, model, condition: torch.Tensor, steps: int = 50) -> torch.Tensor:
        x = torch.randn_like(condition)
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).long()
        for index, scalar_t in enumerate(schedule):
            t = torch.full((condition.shape[0],), scalar_t, device=condition.device, dtype=torch.long)
            alpha_t = self._extract(self.alpha_bar, t, x.ndim)
            eps = model(x, condition, t)
            clean = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt().clamp_min(1e-8)
            if index == len(schedule) - 1:
                x = clean
            else:
                next_t = torch.full_like(t, schedule[index + 1])
                alpha_next = self._extract(self.alpha_bar, next_t, x.ndim)
                x = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * eps
        return x

