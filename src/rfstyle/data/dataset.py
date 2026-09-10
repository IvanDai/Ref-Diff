from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_list(value):
    if value is None:
        return None
    return {str(item) for item in value}


class PairedSignalDataset(Dataset):
    """Filtered access to paired HDF5 shards using manifest metadata."""

    def __init__(self, root: str | Path, split: str, target: str = "x_pure",
                 filters: dict | None = None):
        self.root = Path(root)
        self.target = target
        self.handles: dict[str, h5py.File] = {}
        filters = filters or {}
        modulations = _parse_list(filters.get("modulations"))
        profiles = _parse_list(filters.get("profiles"))
        esn0_range = filters.get("esn0_db", filters.get("snr_db"))
        delay_spreads = _parse_list(filters.get("delay_spreads"))
        self.rows = []
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                if modulations and row["modulation"] not in modulations:
                    continue
                if profiles and row["profile"] not in profiles:
                    continue
                value = float(row.get("esn0_db", row.get("snr_db", "nan")))
                if esn0_range and not float(esn0_range[0]) <= value <= float(esn0_range[1]):
                    continue
                if delay_spreads and str(float(row["delay_spread"])) not in {
                    str(float(item)) for item in delay_spreads
                }:
                    continue
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No samples match split={split!r} and filters={filters!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        return state

    def _handle(self, shard: str) -> h5py.File:
        if shard not in self.handles:
            self.handles[shard] = h5py.File(self.root / "shards" / shard, "r")
        return self.handles[shard]

    @staticmethod
    def _frequency(iq: np.ndarray) -> torch.Tensor:
        z = torch.complex(torch.from_numpy(iq[0]), torch.from_numpy(iq[1]))
        z = z / torch.sqrt(torch.mean(torch.abs(z) ** 2).clamp_min(1e-8))
        spectrum = torch.fft.fftshift(torch.fft.fft(z, norm="ortho"))
        return torch.stack((spectrum.real, spectrum.imag)).float()

    def __getitem__(self, index: int):
        metadata = self.rows[index]
        handle = self._handle(metadata["shard"])
        row = int(metadata["row"])
        condition = self._frequency(handle["y_rx"][row])
        target = self._frequency(handle[self.target][row])
        return {
            "condition": condition,
            "target": target,
            "sample_id": int(metadata["sample_id"]),
            "modulation": metadata["modulation"],
            "esn0_db": float(metadata.get("esn0_db", metadata.get("snr_db", "nan"))),
        }
