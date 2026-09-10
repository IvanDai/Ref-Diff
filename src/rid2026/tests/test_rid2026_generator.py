from dataclasses import fields

import numpy as np

from rid2026 import (
    ImpairmentParameters,
    RFSignalGenerator,
    SignalConfig,
    derive_seed,
)


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
    assert first.x_pure.shape == (512,)
    assert first.parameters.x_tx.shape == (512,)
    np.testing.assert_array_equal(first.x_pure, second.x_pure)
    np.testing.assert_array_equal(first.y_rx, second.y_rx)
    assert [field.name for field in fields(first)] == [
        "y_rx", "modulation", "x_pure", "parameters"
    ]
    assert first.parameters.source_symbols is not None
    assert first.parameters.source_bits is not None


def test_coordinate_seeds_are_unique_and_stable():
    coordinates = [
        (split, profile, modulation, snr, repeat)
        for split in range(3)
        for profile in range(2)
        for modulation in range(24)
        for snr in range(26)
        for repeat in range(4)
    ]
    seeds = [derive_seed(233, *coordinate) for coordinate in coordinates]
    assert len(seeds) == len(set(seeds))
    assert derive_seed(233, 1, 2, 3) == derive_seed(233, 1, 2, 3)
