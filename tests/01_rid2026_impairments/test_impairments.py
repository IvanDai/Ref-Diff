import numpy as np
import pytest

from rid2026 import (
    ImpairmentParameters,
    apply_impairments,
    resample_with_offset,
    sample_channel,
    sample_path_delays,
    sample_parameters,
)


def _parameters(**overrides):
    values = dict(
        esn0_db=float("inf"),
        timing_offset=0.0,
        symbol_rate_offset=0.0,
        phase_offset=0.0,
        carrier_offset=0.0,
        delay_spread=0.0,
        channel_taps=np.array([1.0], dtype=np.complex64),
    )
    values.update(overrides)
    return ImpairmentParameters(**values)


def test_identity_chain_preserves_signal():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    actual = apply_impairments(signal, _parameters(), 8, rng)
    np.testing.assert_allclose(actual, signal, atol=2e-6)


def test_fractional_fir_resampler_tracks_a_tone():
    length = 4096
    frequency = 1 / 64
    signal = np.exp(1j * 2 * np.pi * frequency * np.arange(length))
    timing_offset = 0.37
    rate_offset = 2e-4
    actual = resample_with_offset(
        signal, timing_offset, rate_offset, length - 64, fir_taps=33
    )
    positions = timing_offset + np.arange(length - 64) * (1 + rate_offset)
    expected = np.exp(1j * 2 * np.pi * frequency * positions)
    np.testing.assert_allclose(actual[32:-32], expected[32:-32], atol=2e-4)


def test_phase_and_cfo_are_applied_before_multipath():
    length = 2048
    signal = np.exp(1j * 2 * np.pi * 0.03 * np.arange(length))
    taps = np.array([1.0, 0.4j, -0.2], dtype=np.complex64)
    taps /= np.linalg.norm(taps)
    phase_offset = 0.4
    carrier_offset = 0.007
    params = _parameters(
        phase_offset=phase_offset,
        carrier_offset=carrier_offset,
        delay_spread=1.0,
        channel_taps=taps,
    )
    actual = apply_impairments(signal, params, 8, np.random.default_rng(2))

    receiver_phase = np.exp(
        1j * (phase_offset + 2 * np.pi * carrier_offset * np.arange(length))
    )
    mixed = signal * receiver_phase
    expected = np.convolve(mixed, taps, mode="full")[:length]
    np.testing.assert_allclose(actual, expected, atol=3e-6)


@pytest.mark.parametrize("modulation_sps", [1, 4, 8])
@pytest.mark.parametrize("target_snr", [-10.0, 0.0, 10.0, 20.0])
def test_awgn_uses_the_requested_esn0_for_every_sps(modulation_sps, target_snr):
    rng = np.random.default_rng(10 + modulation_sps)
    signal = rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000)
    clean = apply_impairments(signal, _parameters(), modulation_sps, rng)
    noisy = apply_impairments(
        signal, _parameters(esn0_db=target_snr), modulation_sps, rng
    )
    noise = noisy - clean
    measured_sample_snr = 10 * np.log10(
        np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noise) ** 2)
    )
    measured_esn0 = measured_sample_snr + 10 * np.log10(modulation_sps)
    assert measured_esn0 == pytest.approx(target_snr, abs=0.08)


def test_sampled_channel_has_unit_energy():
    taps = sample_channel(1.0, 16, np.random.default_rng(2))
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0, atol=1e-6)


def test_path_delays_follow_configured_rayleigh_distribution():
    scale = 2.0
    delays = sample_path_delays(scale, 100_000, np.random.default_rng(20))
    expected_mean = scale * np.sqrt(np.pi / 2)
    expected_std = scale * np.sqrt((4 - np.pi) / 2)
    assert np.mean(delays) == pytest.approx(expected_mean, rel=0.01)
    assert np.std(delays) == pytest.approx(expected_std, rel=0.01)


def test_impairment_initialization_matches_table_i_distributions():
    sigma_clk = 0.01
    config = {
        "esn0_db": [0],
        "timing_offset": [0.0, 16.0],
        "sigma_clk": sigma_clk,
        "delay_spreads": [0.0],
        "enabled": {"multipath": False},
    }
    rng = np.random.default_rng(21)
    values = [sample_parameters(config, 1, rng) for _ in range(20_000)]
    timing = np.asarray([value.timing_offset for value in values])
    phase = np.asarray([value.phase_offset for value in values])
    sro = np.asarray([value.symbol_rate_offset for value in values])
    cfo = np.asarray([value.carrier_offset for value in values])
    assert np.mean(timing) == pytest.approx(8.0, abs=0.12)
    assert np.mean(phase) == pytest.approx(np.pi, abs=0.05)
    assert np.mean(sro) == pytest.approx(0.0, abs=2e-4)
    assert np.std(sro) == pytest.approx(sigma_clk, rel=0.03)
    assert np.mean(cfo) == pytest.approx(0.0, abs=2e-4)
    assert np.std(cfo) == pytest.approx(sigma_clk, rel=0.03)


def test_impairment_chain_does_not_wrap_record_boundaries():
    signal = np.zeros(64, dtype=np.complex64)
    signal[-1] = 1.0
    params = _parameters(channel_taps=np.array([0.0, 1.0], dtype=np.complex64))
    actual = apply_impairments(signal, params, 1, np.random.default_rng(2))
    np.testing.assert_allclose(actual, np.zeros_like(actual), atol=1e-15)


def test_sro_and_cfo_use_the_shared_sigma_clk():
    config = {
        "esn0_db": [10],
        "timing_offset": [0.0, 0.0],
        "sigma_clk": 0.2,
        "delay_spreads": [0.0],
        "enabled": {},
    }
    params = sample_parameters(config, 4, np.random.default_rng(5))
    assert params.symbol_rate_offset != 0.0
    assert params.carrier_offset != 0.0


def test_invalid_impairment_parameters_are_rejected():
    with pytest.raises(ValueError, match="greater than -1"):
        _parameters(symbol_rate_offset=-1.0)
    with pytest.raises(ValueError, match="non-zero energy"):
        _parameters(channel_taps=np.zeros(3))
    with pytest.raises(ValueError, match="odd integer"):
        resample_with_offset(np.ones(8), 0.0, 0.0, 8, fir_taps=8)
