from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from refdiff.data import ConditionBatchSampler, RID2026Pairs, iq_to_spectrum


def _write_fixture(path: Path) -> None:
    shape = (2, 2, 10, 2, 64)
    rng = np.random.default_rng(12)
    received = rng.normal(size=shape).astype(np.float32)
    reference = rng.normal(size=shape).astype(np.float32)
    with h5py.File(path, "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        handle.create_dataset(
            "modulation",
            data=np.asarray(["BPSK", "QPSK"], dtype=object),
            dtype=string_type,
        )
        handle.create_dataset("esn0_db", data=np.asarray([0, 10], dtype=np.float32))
        handle.create_dataset("y_rx", data=received)
        handle.create_dataset("x_ref", data=reference)


def test_pair_reader_filters_and_preserves_split_coordinates(tmp_path: Path):
    path = tmp_path / "rid2026.h5"
    _write_fixture(path)
    dataset = RID2026Pairs(
        path,
        "valid",
        {"train": 6, "valid": 2, "test": 2},
        modulations=["QPSK"],
        esn0_values=[10],
    )

    assert len(dataset) == 2
    assert dataset.frame_length == 64
    item = dataset[0]
    assert item["condition"].shape == (2, 64)
    assert item["target"].shape == (2, 64)
    assert item["modulation"] == "QPSK"
    assert item["esn0_db"] == 10
    assert item["coordinate"].tolist() == [1, 1, 6]
    assert torch.isfinite(item["condition"]).all()


def test_spectrum_transform_is_unit_power_and_rejects_bad_shape():
    iq = np.stack((np.ones(32), np.zeros(32))).astype(np.float32)
    spectrum = iq_to_spectrum(iq)
    assert torch.allclose(spectrum.square().sum() / 32, torch.tensor(1.0))
    with pytest.raises(ValueError, match="expected IQ shape"):
        iq_to_spectrum(np.zeros((32, 2), dtype=np.float32))


def test_pair_reader_rejects_incorrect_split_counts(tmp_path: Path):
    path = tmp_path / "rid2026.h5"
    _write_fixture(path)
    with pytest.raises(ValueError, match="expected 10"):
        RID2026Pairs(path, "train", {"train": 5, "valid": 2, "test": 2})


def test_condition_batch_sampler_covers_split_with_contiguous_batches(tmp_path: Path):
    path = tmp_path / "rid2026.h5"
    _write_fixture(path)
    dataset = RID2026Pairs(path, "train", {"train": 6, "valid": 2, "test": 2})
    sampler = ConditionBatchSampler(dataset, batch_size=4, seed=3)
    batches = list(sampler)

    assert len(batches) == 8
    assert sorted(index for batch in batches for index in batch) == list(range(len(dataset)))
    assert all(batch == list(range(batch[0], batch[0] + len(batch))) for batch in batches)
    assert all(batch[0] // 6 == batch[-1] // 6 for batch in batches)
