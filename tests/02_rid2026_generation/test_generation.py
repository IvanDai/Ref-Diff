from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import yaml

from rid2026 import (
    ImpairmentParameters,
    RFSignalGenerator,
    RID2026DatasetGenerator,
    SignalConfig,
    derive_seed,
)


def load_test_config() -> dict:
    with Path(__file__).with_name("config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_high_level_generator_returns_repeatable_complete_result():
    config = SignalConfig(frame_length=512, samples_per_symbol=8)
    impairments = ImpairmentParameters(
        esn0_db=10.0,
        timing_offset=0.25,
        symbol_rate_offset=1e-4,
        phase_offset=0.2,
        carrier_offset=3e-4,
        delay_spread=0.0,
        channel_taps=np.array([1.0], dtype=np.complex64),
    )
    generator = RFSignalGenerator(config)
    first = generator.generate("QPSK", impairments, seed=123)
    second = generator.generate("QPSK", impairments, seed=123)
    assert first.modulation == "QPSK"
    assert first.y_rx.shape == (512,)
    assert first.x_ref.shape == (512,)
    assert first.parameters.x_tx.shape == (512,)
    np.testing.assert_array_equal(first.x_ref, second.x_ref)
    np.testing.assert_array_equal(first.y_rx, second.y_rx)
    assert [field.name for field in fields(first)] == [
        "y_rx", "modulation", "x_ref", "parameters"
    ]
    assert first.parameters.source_symbols is not None
    assert first.parameters.source_bits is not None


def test_coordinate_seeds_are_unique_and_stable():
    coordinates = [
        (modulation, snr, example)
        for modulation in range(24)
        for snr in range(26)
        for example in range(24)
    ]
    seeds = [derive_seed(233, *coordinate) for coordinate in coordinates]
    assert len(seeds) == len(set(seeds))
    assert derive_seed(233, 1, 2, 3) == derive_seed(233, 1, 2, 3)


def test_identity_receive_frame_matches_stored_transmitted_frame():
    result = RFSignalGenerator(
        SignalConfig(frame_length=512, samples_per_symbol=8)
    ).generate("QPSK", seed=45)
    np.testing.assert_allclose(result.y_rx, result.parameters.x_tx, atol=2e-6)


def test_reference_is_independent_of_receive_impairments():
    config = SignalConfig(frame_length=512, samples_per_symbol=8)
    clean = RFSignalGenerator(config).generate("QPSK", seed=233)
    severe = ImpairmentParameters(
        esn0_db=-20.0,
        timing_offset=4.5,
        symbol_rate_offset=1e-3,
        phase_offset=1.2,
        carrier_offset=2e-3,
        delay_spread=1.0,
        channel_taps=np.array([0.8 + 0.1j, 0.2 - 0.3j], dtype=np.complex64),
        channel_path_count=2,
    )
    impaired = RFSignalGenerator(config).generate("QPSK", severe, seed=233)

    np.testing.assert_array_equal(clean.x_ref, impaired.x_ref)
    assert not np.array_equal(clean.y_rx, impaired.y_rx)


def test_dataset_generator_writes_complete_contract(tmp_path: Path):
    config = load_test_config()
    config["signal"]["frame_length"] = 128
    config["signal"]["modulations"] = ["BPSK"]
    config["impairments"]["esn0_db"] = [0]
    config["examples_per_condition"] = 3

    output = tmp_path / "rid2026.h5"
    RID2026DatasetGenerator(config).generate(output)

    with h5py.File(output, "r") as handle:
        assert {"modulation", "esn0_db", "y_rx", "x_ref", "parameters"} == set(handle)
        assert not handle.attrs
        assert handle["y_rx"].shape == (1, 1, 3, 2, 128)
        assert handle["x_ref"].shape == handle["y_rx"].shape
        assert handle["y_rx"].compression is None
        assert handle["x_ref"].compression is None
        assert "x_tx" not in handle
        assert set(handle["parameters"]) == {
            "samples_per_symbol", "rrc_alpha", "timing_offset",
            "symbol_rate_offset", "phase_offset", "carrier_offset",
            "delay_spread", "channel_path_count", "channel_taps",
        }


def test_parallel_generation_matches_serial_generation(tmp_path: Path):
    config = load_test_config()
    config["signal"]["frame_length"] = 128
    config["signal"]["modulations"] = ["BPSK"]
    config["impairments"]["esn0_db"] = [0]
    config["examples_per_condition"] = 4
    serial_path = tmp_path / "serial.h5"
    parallel_path = tmp_path / "parallel.h5"

    config["workers"] = 1
    RID2026DatasetGenerator(config).generate(serial_path)
    config["workers"] = 2
    RID2026DatasetGenerator(config).generate(parallel_path)

    with h5py.File(serial_path, "r") as serial, h5py.File(parallel_path, "r") as parallel:
        for name in ("y_rx", "x_ref"):
            np.testing.assert_array_equal(serial[name][:], parallel[name][:])
        for name in serial["parameters"]:
            np.testing.assert_array_equal(
                serial["parameters"][name][:], parallel["parameters"][name][:]
            )


def test_gzip_compression_remains_available(tmp_path: Path):
    config = load_test_config()
    config["signal"]["frame_length"] = 128
    config["signal"]["modulations"] = ["BPSK"]
    config["impairments"]["esn0_db"] = [0]
    config["examples_per_condition"] = 1
    config["compression"] = "gzip"
    output = tmp_path / "compressed.h5"

    RID2026DatasetGenerator(config).generate(output)

    with h5py.File(output, "r") as handle:
        assert handle["y_rx"].compression == "gzip"
        assert handle["x_ref"].compression == "gzip"


def test_full_configuration_matches_rml2018_scale():
    config = load_test_config()
    assert RID2026DatasetGenerator(config).example_count() == 2_555_904
    assert config["signal"]["frame_length"] == 1024
    assert config["signal"]["rrc_alpha"] == [0.1, 0.4]
    assert config["impairments"]["esn0_db"] == list(range(-20, 31, 2))
    assert config["impairments"]["timing_offset"] == [0.0, 16.0]
    assert config["impairments"]["sigma_clk"] == 0.0001
    assert config["impairments"]["delay_spreads"] == [0.0, 0.5, 1.0, 2.0]
