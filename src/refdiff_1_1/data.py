from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def iq_to_time(iq: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return unit-power time-domain IQ without Fourier transforms."""
    values = torch.as_tensor(iq, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"expected IQ shape [2, length], got {tuple(values.shape)}")
    power = values.square().sum(dim=0).mean().sqrt().clamp_min(1e-8)
    return values / power


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class RID2026TimeDomainPairs(Dataset):
    """Read paired time-domain `y_rx` and `x_ref` IQ records."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        split_counts: dict[str, int],
        *,
        modulations: list[str] | None = None,
        esn0_values: list[float] | None = None,
    ) -> None:
        self.path = Path(path)
        self._handle: h5py.File | None = None
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split: {split!r}")
        if set(split_counts) != {"train", "valid", "test"}:
            raise ValueError("split_counts must contain train, valid, and test")
        with h5py.File(self.path, "r") as handle:
            required = {"modulation", "esn0_db", "y_rx", "x_ref"}
            missing = required - set(handle)
            if missing:
                raise ValueError(f"RID2026 file is missing datasets: {sorted(missing)}")
            self.available_modulations = tuple(_decode(x) for x in handle["modulation"][:])
            self.available_esn0 = tuple(float(x) for x in handle["esn0_db"][:])
            shape = handle["y_rx"].shape
            if len(shape) != 5 or shape[-2:] != (2, shape[-1]) or handle["x_ref"].shape != shape:
                raise ValueError("y_rx and x_ref must share shape [M, S, E, 2, L]")
            if shape[:2] != (len(self.available_modulations), len(self.available_esn0)):
                raise ValueError("RID2026 coordinate arrays do not match signal arrays")
            self.frame_length = int(shape[-1])
            examples_per_condition = int(shape[2])

        counts = {name: int(value) for name, value in split_counts.items()}
        if any(value < 0 for value in counts.values()) or sum(counts.values()) != examples_per_condition:
            raise ValueError(f"split counts must be non-negative and sum to {examples_per_condition}")
        self.example_start = {
            "train": 0,
            "valid": counts["train"],
            "test": counts["train"] + counts["valid"],
        }[split]
        self.examples_in_split = counts[split]
        selected_modulations = self.available_modulations if modulations is None else tuple(modulations)
        selected_esn0 = self.available_esn0 if esn0_values is None else tuple(float(x) for x in esn0_values)
        unknown_modulations = set(selected_modulations) - set(self.available_modulations)
        unknown_esn0 = set(selected_esn0) - set(self.available_esn0)
        if unknown_modulations or unknown_esn0:
            raise ValueError(f"unknown filters: modulations={sorted(unknown_modulations)}, esn0={sorted(unknown_esn0)}")
        self.conditions = tuple(
            (self.available_modulations.index(modulation), self.available_esn0.index(esn0))
            for modulation in selected_modulations
            for esn0 in selected_esn0
        )
        if not self.conditions or self.examples_in_split == 0:
            raise ValueError("selected split contains no examples")

    def __len__(self) -> int:
        return len(self.conditions) * self.examples_in_split

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        condition_index, example_offset = divmod(index, self.examples_in_split)
        modulation_index, esn0_index = self.conditions[condition_index]
        location = (modulation_index, esn0_index, self.example_start + example_offset)
        handle = self._file()
        return {
            "condition": iq_to_time(handle["y_rx"][location]),
            "target": iq_to_time(handle["x_ref"][location]),
            "modulation": self.available_modulations[modulation_index],
            "esn0_db": self.available_esn0[esn0_index],
            "coordinate": torch.tensor(location, dtype=torch.int64),
        }


class ConditionBatchSampler(Sampler[list[int]]):
    """Shuffle contiguous condition-local batches for HDF5 locality."""

    def __init__(self, dataset: RID2026TimeDomainPairs, batch_size: int, *, seed: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0

    def __len__(self) -> int:
        return len(self.dataset.conditions) * ((self.dataset.examples_in_split + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        batches = []
        examples = self.dataset.examples_in_split
        for condition_index in range(len(self.dataset.conditions)):
            start = condition_index * examples
            batches.extend(
                [list(range(start + offset, start + min(offset + self.batch_size, examples)))
                 for offset in range(0, examples, self.batch_size)]
            )
        order = torch.randperm(len(batches), generator=torch.Generator().manual_seed(self.seed + self.epoch))
        self.epoch += 1
        for index in order.tolist():
            yield batches[index]

