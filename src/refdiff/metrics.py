from __future__ import annotations

import torch


def nmse(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return strict per-example NMSE without target-derived alignment."""
    if estimate.shape != reference.shape or estimate.ndim < 2:
        raise ValueError("estimate and reference must have the same batched shape")
    dimensions = tuple(range(1, estimate.ndim))
    error = (estimate - reference).square().sum(dim=dimensions)
    power = reference.square().sum(dim=dimensions).clamp_min(1e-12)
    return error / power


def aggregate_nmse(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape")
    return (estimate - reference).square().sum() / reference.square().sum().clamp_min(1e-12)

