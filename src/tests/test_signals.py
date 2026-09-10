import numpy as np

from rfstyle.data.signals import RML2018_MODULATIONS, generate_waveform, rrc_taps


def test_rrc_is_finite_and_unit_energy():
    taps = rrc_taps(0.35, 8)
    assert np.isfinite(taps).all()
    assert np.isclose(np.sum(taps**2), 1.0)


def test_all_rml2018_modulations_generate():
    for index, modulation in enumerate(RML2018_MODULATIONS):
        waveform = generate_waveform(
            modulation, length=1024, samples_per_symbol=8,
            rrc_alpha=0.25,
            rng=np.random.default_rng(index),
        )
        assert waveform.transmitted.shape == (1024,)
        assert waveform.pure.shape == (1024,)
        assert np.isfinite(waveform.transmitted).all()
        assert np.isclose(np.mean(np.abs(waveform.transmitted) ** 2), 1.0, atol=1e-5)
