import numpy as np
import pytest

from rid2026 import RML2018_MODULATIONS, constellation, generate_waveform, rrc_taps


def test_rrc_is_symmetric_unit_energy_and_nearly_nyquist():
    samples_per_symbol = 8
    taps = rrc_taps(0.35, samples_per_symbol)
    np.testing.assert_allclose(taps, taps[::-1], atol=1e-14)
    assert np.isclose(np.sum(taps**2), 1.0)

    raised_cosine = np.convolve(taps, taps)
    center = len(raised_cosine) // 2
    symbol_samples = raised_cosine[center - 4 * samples_per_symbol:
                                    center + 4 * samples_per_symbol + 1:
                                    samples_per_symbol]
    assert np.max(np.abs(np.delete(symbol_samples, 4))) < 0.01


@pytest.mark.parametrize("modulation", RML2018_MODULATIONS)
def test_all_modulations_generate_finite_unit_power_waveforms(modulation):
    waveform = generate_waveform(
        modulation,
        length=2048,
        samples_per_symbol=8,
        rrc_alpha=0.25,
        rng=np.random.default_rng(100),
    )
    for signal in (waveform.pure, waveform.transmitted):
        assert signal.shape == (2048,)
        assert signal.dtype == np.complex64
        assert np.isfinite(signal).all()
        assert np.isclose(np.mean(np.abs(signal) ** 2), 1.0, atol=2e-6)


def test_extended_constellations_have_expected_geometries():
    assert len(constellation("32QAM")) == 32
    assert len(constellation("128QAM")) == 128
    assert len(np.unique(np.round(np.abs(constellation("16APSK")), 6))) == 2
    assert len(np.unique(np.round(np.abs(constellation("32APSK")), 6))) == 3
    assert len(np.unique(np.round(np.abs(constellation("64APSK")), 6))) == 4
    assert len(np.unique(np.round(np.abs(constellation("128APSK")), 6))) == 5


def test_waveform_generation_is_reproducible():
    arguments = dict(
        name="16QAM",
        length=1024,
        samples_per_symbol=8,
        rrc_alpha=0.25,
    )
    first = generate_waveform(**arguments, rng=np.random.default_rng(42))
    second = generate_waveform(**arguments, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(first.transmitted, second.transmitted)
    np.testing.assert_array_equal(first.pure, second.pure)


def test_linear_digital_pure_waveform_uses_rectangular_pulses():
    waveform = generate_waveform(
        "16QAM", 1024, 8, 0.25, np.random.default_rng(42)
    )
    rectangular_symbols = waveform.pure.reshape(-1, 8)
    np.testing.assert_array_equal(
        rectangular_symbols,
        np.repeat(rectangular_symbols[:, :1], 8, axis=1),
    )
    assert waveform.pure_pulse_shape == "rectangular"


def test_gmsk_pure_waveform_retains_gaussian_shaping():
    waveform = generate_waveform(
        "GMSK", 1024, 8, 0.25, np.random.default_rng(42)
    )
    np.testing.assert_array_equal(waveform.pure, waveform.transmitted)
    assert waveform.pure_pulse_shape == "gaussian"


def test_oqpsk_rejects_odd_samples_per_symbol():
    with pytest.raises(ValueError, match="even"):
        generate_waveform(
            "OQPSK", 255, 5, 0.25, np.random.default_rng(1)
        )


def test_modulation_specific_metadata_is_recorded():
    gmsk = generate_waveform("GMSK", 256, 8, 0.25, np.random.default_rng(1))
    am = generate_waveform("AM-DSB-WC", 256, 8, 0.25, np.random.default_rng(1))
    fm = generate_waveform("FM", 256, 8, 0.25, np.random.default_rng(1))
    assert gmsk.rrc_alpha is None and gmsk.gmsk_bt == 0.3
    assert am.rrc_alpha is None and am.am_modulation_index == 1.0
    assert fm.rrc_alpha is None and fm.fm_deviation == 0.35


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length": 0}, "length"),
        ({"samples_per_symbol": 0}, "samples_per_symbol"),
        ({"rrc_alpha": 1.1}, "RRC"),
        ({"gmsk_bt": 0.0}, "gmsk_bt"),
        ({"message_bandwidth": 1.0}, "message_bandwidth"),
    ],
)
def test_invalid_waveform_parameters_are_rejected(kwargs, message):
    arguments = dict(
        name="QPSK",
        length=256,
        samples_per_symbol=8,
        rrc_alpha=0.25,
        rng=np.random.default_rng(1),
    )
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        generate_waveform(**arguments)
