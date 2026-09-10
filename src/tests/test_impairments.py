import numpy as np

from rfstyle.data.impairments import ImpairmentParameters, apply_impairments, sample_channel


def test_identity_channel_without_noise():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(128) + 1j * rng.standard_normal(128)).astype(np.complex64)
    taps = np.zeros(4, dtype=np.complex64)
    taps[0] = 1
    params = ImpairmentParameters(float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, taps)
    actual = apply_impairments(x, params, 1, rng)
    np.testing.assert_allclose(actual, x, atol=1e-6)


def test_channel_has_unit_energy():
    taps = sample_channel(1.0, 16, np.random.default_rng(2))
    assert np.isclose(np.sum(np.abs(taps) ** 2), 1.0, atol=1e-5)

