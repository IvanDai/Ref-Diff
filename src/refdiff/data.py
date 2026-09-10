from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def iq_to_spectrum(iq: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert [2, length] IQ to a unit-power centered complex spectrum."""
    values = torch.as_tensor(iq, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"expected IQ shape [2, length], got {tuple(values.shape)}")
    signal = torch.complex(values[0], values[1])
    rms = signal.abs().square().mean().sqrt().clamp_min(1e-8)
    spectrum = torch.fft.fftshift(torch.fft.fft(signal / rms, norm="ortho"))
    return torch.stack((spectrum.real, spectrum.imag)).float()


class RID2026Pairs(Dataset):
    """Read paired `y_rx` and `x_ref` records from one RID2026 HDF5 file."""

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
            if len(shape) != 5 or shape[-2] != 2 or handle["x_ref"].shape != shape:
                raise ValueError("y_rx and x_ref must share shape [M, S, E, 2, L]")
            if shape[:2] != (len(self.available_modulations), len(self.available_esn0)):
                raise ValueError("RID2026 coordinate arrays do not match signal arrays")
            self.frame_length = int(shape[-1])
            examples_per_condition = int(shape[2])

        counts = {name: int(value) for name, value in split_counts.items()}
        if any(value < 0 for value in counts.values()):
            raise ValueError("split counts cannot be negative")
        if sum(counts.values()) != examples_per_condition:
            raise ValueError(
                f"split counts sum to {sum(counts.values())}, expected {examples_per_condition}"
            )
        offsets = {
            "train": 0,
            "valid": counts["train"],
            "test": counts["train"] + counts["valid"],
        }
        self.example_start = offsets[split]
        self.examples_in_split = counts[split]

        selected_modulations = (
            self.available_modulations if modulations is None else tuple(modulations)
        )
        unknown_modulations = set(selected_modulations) - set(self.available_modulations)
        if unknown_modulations:
            raise ValueError(f"unknown modulations: {sorted(unknown_modulations)}")
        selected_esn0 = (
            self.available_esn0
            if esn0_values is None
            else tuple(float(value) for value in esn0_values)
        )
        unknown_esn0 = set(selected_esn0) - set(self.available_esn0)
        if unknown_esn0:
            raise ValueError(f"unknown Es/N0 values: {sorted(unknown_esn0)}")

        modulation_indices = [self.available_modulations.index(x) for x in selected_modulations]
        esn0_indices = [self.available_esn0.index(x) for x in selected_esn0]
        self.conditions = tuple(
            (modulation_index, esn0_index)
            for modulation_index in modulation_indices
            for esn0_index in esn0_indices
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
        example_index = self.example_start + example_offset
        location = (modulation_index, esn0_index, example_index)
        handle = self._file()
        return {
            "condition": iq_to_spectrum(handle["y_rx"][location]),
            "target": iq_to_spectrum(handle["x_ref"][location]),
            "modulation": self.available_modulations[modulation_index],
            "esn0_db": self.available_esn0[esn0_index],
            "coordinate": torch.tensor(location, dtype=torch.int64),
        }


class ConditionBatchSampler(Sampler[list[int]]):
    """Shuffle contiguous condition-local batches while preserving HDF5 locality."""

    def __init__(
        self,
        dataset: RID2026Pairs,
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        full, remainder = divmod(self.dataset.examples_in_split, self.batch_size)
        per_condition = full + int(bool(remainder) and not self.drop_last)
        return len(self.dataset.conditions) * per_condition

    def __iter__(self):
        batches = []
        examples = self.dataset.examples_in_split
        for condition_index in range(len(self.dataset.conditions)):
            condition_start = condition_index * examples
            for offset in range(0, examples, self.batch_size):
                stop = min(offset + self.batch_size, examples)
                if self.drop_last and stop - offset < self.batch_size:
                    continue
                batches.append(list(range(condition_start + offset, condition_start + stop)))
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(batches), generator=generator).tolist()
        self.epoch += 1
        for index in order:
            yield batches[index]
