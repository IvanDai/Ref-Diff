from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .diffusion import GaussianDiffusion
from .metrics import aggregate_nmse


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.lerp_(incoming, 1 - self.decay)
            else:
                value.copy_(incoming)


@dataclass
class ValidationResult:
    loss: float
    low_t_loss: float
    middle_t_loss: float
    high_t_loss: float


@dataclass
class EarlyStopping:
    """Stop after restoration NMSE fails to improve meaningfully."""

    patience: int
    min_epochs: int = 0
    min_relative_improvement: float = 0.0
    best: float = math.inf
    bad_evaluations: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("early-stopping patience must be positive")
        if self.min_epochs < 0:
            raise ValueError("early-stopping min_epochs cannot be negative")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError(
                "early-stopping min_relative_improvement must be in [0, 1)"
            )

    def update(self, value: float, epoch: int) -> bool:
        if not math.isfinite(value):
            raise ValueError("early-stopping metric must be finite")
        threshold = self.best * (1 - self.min_relative_improvement)
        if value < threshold:
            self.best = value
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
        return epoch >= self.min_epochs and self.bad_evaluations >= self.patience


def _limited(loader: Iterable, max_batches: int | None):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def train_epoch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ema: ExponentialMovingAverage,
    *,
    gradient_clip: float = 1.0,
    max_batches: int | None = None,
    log_interval: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in _limited(loader, max_batches):
        target = batch["target"].to(device)
        condition = batch["condition"].to(device)
        timestep = torch.randint(diffusion.timesteps, (target.shape[0],), device=device)
        noisy, noise = diffusion.add_noise(target, timestep)
        loss = F.mse_loss(model(noisy, condition, timestep), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        ema.update(model)
        total_loss += loss.item()
        batches += 1
        if log_interval and batches % log_interval == 0:
            print(f"train_step={batches} mean_loss={total_loss / batches:.6f}")
    if batches == 0:
        raise ValueError("training loader produced no batches")
    return total_loss / batches


@torch.no_grad()
def validate_noise(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: Iterable,
    device: torch.device,
    *,
    max_batches: int | None = None,
    seed: int = 0,
) -> ValidationResult:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    counts = torch.zeros(3, dtype=torch.int64)
    total_squared_error = 0.0
    total_elements = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in _limited(loader, max_batches):
        target = batch["target"].to(device)
        condition = batch["condition"].to(device)
        timestep = torch.linspace(
            0,
            diffusion.timesteps - 1,
            target.shape[0],
            device=device,
        ).round().long()
        noise = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=device,
            generator=generator,
        )
        noisy, noise = diffusion.add_noise(target, timestep, noise)
        prediction = model(noisy, condition, timestep)
        sample_mse = (prediction - noise).square().flatten(1).mean(1)
        bins = torch.clamp(timestep * 3 // diffusion.timesteps, max=2)
        for bin_index in range(3):
            selected = sample_mse[bins == bin_index]
            if selected.numel():
                totals[bin_index] += selected.sum().cpu().double()
                counts[bin_index] += selected.numel()
        total_squared_error += (prediction - noise).square().sum().item()
        total_elements += noise.numel()
    if total_elements == 0:
        raise ValueError("validation loader produced no batches")
    values = [
        float(totals[i] / counts[i]) if counts[i] else math.nan for i in range(3)
    ]
    return ValidationResult(total_squared_error / total_elements, *values)


@torch.no_grad()
def validate_restoration(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: Iterable,
    device: torch.device,
    *,
    steps: int,
    sample_count: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    restored_parts = []
    target_parts = []
    raw_parts = []
    seen = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        remaining = sample_count - seen
        if remaining <= 0:
            break
        condition = batch["condition"][:remaining].to(device)
        target = batch["target"][:remaining].to(device)
        initial_noise = torch.randn(
            condition.shape,
            dtype=condition.dtype,
            device=device,
            generator=generator,
        )
        restored = diffusion.ddim_sample(
            model, condition, steps=steps, initial_noise=initial_noise
        )
        restored_parts.append(restored.cpu())
        target_parts.append(target.cpu())
        raw_parts.append(condition.cpu())
        seen += condition.shape[0]
    if seen == 0:
        raise ValueError("restoration loader produced no samples")
    restored = torch.cat(restored_parts)
    target = torch.cat(target_parts)
    raw = torch.cat(raw_parts)
    restored_nmse = aggregate_nmse(restored, target).item()
    raw_nmse = aggregate_nmse(raw, target).item()
    return {
        "sample_count": float(seen),
        "restoration_nmse": restored_nmse,
        "restoration_nmse_db": 10 * math.log10(max(restored_nmse, 1e-12)),
        "raw_nmse": raw_nmse,
        "raw_nmse_db": 10 * math.log10(max(raw_nmse, 1e-12)),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
