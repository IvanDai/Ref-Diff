from pathlib import Path

import h5py
import numpy as np
import torch

from refdiff_1_1 import ConditionalUNet1D, RID2026TimeDomainPairs, iq_to_time


def _fixture(path: Path) -> None:
    shape = (1, 1, 6, 2, 32)
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("modulation", data=np.asarray(["QPSK"], dtype=object), dtype=h5py.string_dtype())
        handle.create_dataset("esn0_db", data=np.asarray([10], dtype=np.float32))
        handle.create_dataset("y_rx", data=rng.normal(size=shape).astype(np.float32))
        handle.create_dataset("x_ref", data=rng.normal(size=shape).astype(np.float32))


def test_time_domain_loader_returns_normalized_raw_iq(tmp_path: Path):
    path = tmp_path / "pairs.h5"
    _fixture(path)
    dataset = RID2026TimeDomainPairs(path, "train", {"train": 4, "valid": 1, "test": 1})
    with h5py.File(path, "r") as handle:
        raw = handle["y_rx"][0, 0, 0]
    torch.testing.assert_close(dataset[0]["condition"], iq_to_time(raw))
    assert dataset[0]["condition"].shape == (2, 32)


def test_time_domain_model_forward_shape():
    model = ConditionalUNet1D(base_channels=8, channel_multipliers=(1, 2, 4))
    signal = torch.randn(2, 2, 32)
    output = model(signal, signal, torch.tensor([0, 3]))
    assert output.shape == signal.shape

