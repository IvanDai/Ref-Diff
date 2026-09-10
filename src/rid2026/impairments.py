from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ImpairmentParameters:
    """Parameters for a complex-baseband receive chain.

    Offsets use samples, fractional rate, radians, and cycles per sample.
    ``esn0_db`` is symbol-energy to noise-density ratio for digital signals.
    Analog signals use ``samples_per_symbol=1`` and therefore the equivalent
    sample-power SNR.
    """

    esn0_db: float
    timing_offset: float
    symbol_rate_offset: float
    phase_offset: float
    carrier_offset: float
    delay_spread: float
    channel_taps: np.ndarray

    def __post_init__(self) -> None:
        scalar_values = (
            self.timing_offset,
            self.symbol_rate_offset,
            self.phase_offset,
            self.carrier_offset,
            self.delay_spread,
        )
        if not np.isfinite(scalar_values).all():
            raise ValueError("impairment parameters must be finite")
        if not (np.isfinite(self.esn0_db) or self.esn0_db == float("inf")):
            raise ValueError("esn0_db must be finite or positive infinity")
        if self.symbol_rate_offset <= -1.0:
            raise ValueError("symbol_rate_offset must be greater than -1")
        if self.delay_spread < 0:
            raise ValueError("delay_spread cannot be negative")
        taps = np.asarray(self.channel_taps)
        if taps.ndim != 1 or taps.size == 0:
            raise ValueError("channel_taps must be a non-empty one-dimensional array")
        if not np.isfinite(taps).all():
            raise ValueError("channel_taps must contain only finite values")
        if not np.any(np.abs(taps) > 0):
            raise ValueError("channel_taps must have non-zero energy")

    @classmethod
    def identity(cls, channel_taps: int = 1) -> ImpairmentParameters:
        if channel_taps < 1:
            raise ValueError("channel_taps must be positive")
        taps = np.zeros(channel_taps, dtype=np.complex64)
        taps[0] = 1.0
        return cls(float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, taps)


def sample_path_delays(
    delay_spread: float, path_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw continuous path delays from ``Rayleigh(delay_spread)``."""
    if delay_spread < 0:
        raise ValueError("delay_spread cannot be negative")
    if path_count < 1:
        raise ValueError("path_count must be positive")
    if delay_spread == 0:
        return np.zeros(path_count, dtype=np.float64)
    return rng.rayleigh(delay_spread, size=path_count)


def sample_channel(
    delay_spread: float,
    max_taps: int,
    rng: np.random.Generator,
    *,
    path_count: int = 16,
) -> np.ndarray:
    """Sample the paper's discrete ``sum delta(t-Rayleigh_i(tau))`` channel.

    Rayleigh path delays are rounded to sample bins. Uniform random path phase
    supplies the complex coefficient that the paper's envelope-only notation
    leaves unspecified, and the final channel is normalized to unit energy.
    """
    if delay_spread < 0:
        raise ValueError("delay_spread cannot be negative")
    if max_taps < 1:
        raise ValueError("max_taps must be positive")
    if path_count < 1:
        raise ValueError("path_count must be positive")
    if delay_spread == 0:
        taps = np.zeros(max_taps, dtype=np.complex64)
        taps[0] = 1.0
        return taps

    delays = np.rint(sample_path_delays(delay_spread, path_count, rng)).astype(int)
    delays = np.clip(delays, 0, max_taps - 1)
    phases = rng.uniform(0.0, 2 * np.pi, size=path_count)
    taps = np.zeros(max_taps, dtype=np.complex128)
    np.add.at(taps, delays, np.exp(1j * phases))
    energy = float(np.sum(np.abs(taps) ** 2))
    if energy <= np.finfo(np.float64).tiny:
        # Exact cancellation is vanishingly unlikely, but keep generation total.
        taps[delays[0]] = 1.0
        energy = 1.0
    return (taps / np.sqrt(energy)).astype(np.complex64)


def resample_with_offset(
    x: np.ndarray,
    timing_offset: float,
    rate_offset: float,
    output_length: int,
    *,
    fir_taps: int = 33,
) -> np.ndarray:
    """Apply fractional timing and sample-rate offsets with a windowed-sinc FIR."""
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite values")
    if not np.isfinite((timing_offset, rate_offset)).all():
        raise ValueError("timing and rate offsets must be finite")
    if rate_offset <= -1.0:
        raise ValueError("rate_offset must be greater than -1")
    if output_length < 1:
        raise ValueError("output_length must be positive")
    if fir_taps < 3 or fir_taps % 2 == 0:
        raise ValueError("fir_taps must be an odd integer of at least 3")

    positions = timing_offset + np.arange(output_length) * (1.0 + rate_offset)
    half = fir_taps // 2
    offsets = np.arange(-half, half + 1)
    indices = np.floor(positions).astype(np.int64)[:, None] + offsets[None, :]
    distances = positions[:, None] - indices

    # A continuous Kaiser window keeps the FIR centered on each fraction.
    normalized_distance = distances / half
    inside_window = np.abs(normalized_distance) <= 1.0
    window = np.zeros_like(distances)
    window[inside_window] = np.i0(
        8.6 * np.sqrt(1 - normalized_distance[inside_window] ** 2)
    ) / np.i0(8.6)
    weights = np.sinc(distances) * window
    weights /= np.sum(weights, axis=1, keepdims=True)

    valid = (indices >= 0) & (indices < len(values))
    samples = np.zeros(indices.shape, dtype=np.complex128)
    samples[valid] = values[indices[valid]]
    return np.sum(samples * weights, axis=1)


def apply_impairments(
    x: np.ndarray,
    params: ImpairmentParameters,
    samples_per_symbol: int,
    rng: np.random.Generator,
    *,
    interpolation_taps: int = 33,
) -> np.ndarray:
    """Apply timing/SRO, phase/CFO, multipath, then Es/N0-calibrated AWGN."""
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite values")
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be positive")
    if interpolation_taps < 3 or interpolation_taps % 2 == 0:
        raise ValueError("interpolation_taps must be an odd integer of at least 3")

    length = len(values)
    shifted = resample_with_offset(
        values,
        params.timing_offset,
        params.symbol_rate_offset,
        length,
        fir_taps=interpolation_taps,
    )

    # Follow Figure 2 of arXiv:1712.04578: interpolation, mixer, convolution.
    sample_index = np.arange(length)
    phase = params.phase_offset + 2 * np.pi * params.carrier_offset * sample_index
    mixed = shifted * np.exp(1j * phase)
    deterministic = np.convolve(mixed, params.channel_taps, mode="full")[:length]

    signal_power = float(np.mean(np.abs(deterministic) ** 2))
    if signal_power <= np.finfo(np.float64).tiny:
        raise ValueError("deterministic received signal has zero power")
    if params.esn0_db == float("inf"):
        return deterministic.astype(np.complex64)

    esn0_linear = 10 ** (params.esn0_db / 10)
    symbol_energy = signal_power * samples_per_symbol
    noise_power = symbol_energy / esn0_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(length) + 1j * rng.standard_normal(length)
    )
    return (deterministic + noise).astype(np.complex64)


def sample_parameters(
    config: dict, max_taps: int, rng: np.random.Generator
) -> ImpairmentParameters:
    """Sample impairment parameters from a dataset configuration mapping."""
    enabled = config.get("enabled", {})
    esn0_values = config.get("esn0_db", config.get("snr_db"))
    if not esn0_values:
        raise ValueError("esn0_db must contain at least one value")

    esn0_db = (
        float(rng.choice(esn0_values))
        if enabled.get("awgn", True)
        else float("inf")
    )
    timing = (
        float(rng.uniform(*config["timing_offset"]))
        if enabled.get("timing_offset", True)
        else 0.0
    )
    sigma_clk = float(config.get("sigma_clk", config.get("clock_sigma", 0.0)))
    if sigma_clk < 0:
        raise ValueError("sigma_clk cannot be negative")
    symbol_rate_offset = (
        float(rng.normal(0, sigma_clk))
        if enabled.get("symbol_rate_offset", True)
        else 0.0
    )
    phase_offset = (
        float(rng.uniform(0, 2 * np.pi))
        if enabled.get("phase_offset", True)
        else 0.0
    )
    carrier_offset = (
        float(rng.normal(0, sigma_clk))
        if enabled.get("carrier_offset", True)
        else 0.0
    )
    delay_spread = (
        float(rng.choice(config["delay_spreads"]))
        if enabled.get("multipath", True)
        else 0.0
    )
    taps = sample_channel(
        delay_spread,
        max_taps,
        rng,
        path_count=int(config.get("channel_paths", 16)),
    )
    return ImpairmentParameters(
        esn0_db,
        timing,
        symbol_rate_offset,
        phase_offset,
        carrier_offset,
        delay_spread,
        taps,
    )
