from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, hilbert, lfilter


RML2018_MODULATIONS = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM",
    "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC",
    "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
)


@dataclass(frozen=True)
class Waveform:
    pure: np.ndarray
    transmitted: np.ndarray
    samples_per_symbol: int
    rrc_alpha: float | None
    source_symbols: np.ndarray | None = None
    source_bits: np.ndarray | None = None
    source_message: np.ndarray | None = None
    pure_pulse_shape: str = "rectangular"
    gmsk_bt: float | None = None
    message_bandwidth: float | None = None
    am_modulation_index: float | None = None
    fm_deviation: float | None = None


def unit_power(x: np.ndarray) -> np.ndarray:
    """Return a complex64 signal with mean squared magnitude equal to one."""
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("x must contain only finite values")
    power = float(np.mean(np.abs(values) ** 2))
    if power <= np.finfo(np.float64).tiny:
        raise ValueError("x must have non-zero power")
    return (values / np.sqrt(power)).astype(np.complex64)


def rrc_taps(alpha: float, samples_per_symbol: int, span_symbols: int = 10) -> np.ndarray:
    """Generate unit-energy root-raised-cosine FIR taps."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be positive")
    if span_symbols < 1:
        raise ValueError("span_symbols must be positive")

    half_length = span_symbols * samples_per_symbol / 2
    n = np.arange(-half_length, half_length + 1, dtype=np.float64)
    t = n / samples_per_symbol
    taps = np.empty_like(t)
    for index, time in enumerate(t):
        if abs(time) < 1e-12:
            taps[index] = 1 + alpha * (4 / math.pi - 1)
        elif alpha > 0 and abs(abs(time) - 1 / (4 * alpha)) < 1e-10:
            taps[index] = (alpha / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * alpha))
                + (1 - 2 / math.pi) * math.cos(math.pi / (4 * alpha))
            )
        else:
            numerator = (
                math.sin(math.pi * time * (1 - alpha))
                + 4 * alpha * time * math.cos(math.pi * time * (1 + alpha))
            )
            denominator = math.pi * time * (1 - (4 * alpha * time) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def _psk(order: int) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * np.arange(order) / order)


def _ask(order: int) -> np.ndarray:
    if order == 2:
        return np.array([0.0, 1.0], dtype=np.complex64)
    return np.arange(-(order - 1), order, 2, dtype=np.float64).astype(np.complex64)


def _rectangular_qam(order: int) -> np.ndarray:
    rows = 2 ** (int(math.log2(order)) // 2)
    columns = order // rows
    i_values = np.arange(-(columns - 1), columns, 2)
    q_values = np.arange(-(rows - 1), rows, 2)
    return (i_values[None, :] + 1j * q_values[:, None]).reshape(-1)


def _cross_qam(order: int) -> np.ndarray:
    """Generate the common cross-shaped 32-QAM or 128-QAM geometry."""
    side = {32: 6, 128: 12}[order]
    corner_width = {32: 1, 128: 2}[order]
    levels = np.arange(-(side - 1), side, 2)
    points = []
    for row, q_value in enumerate(levels):
        for column, i_value in enumerate(levels):
            in_i_corner = column < corner_width or column >= side - corner_width
            in_q_corner = row < corner_width or row >= side - corner_width
            if not (in_i_corner and in_q_corner):
                points.append(i_value + 1j * q_value)
    return np.asarray(points)


def _qam(order: int) -> np.ndarray:
    points = _cross_qam(order) if order in (32, 128) else _rectangular_qam(order)
    return unit_power(points)


def _apsk(order: int) -> np.ndarray:
    # RML2018 did not publish these coordinates. These layouts follow the
    # common DVB-S2/S2X ring populations and representative radius ratios.
    layouts = {
        16: ((4, 12), (1.0, 2.85)),
        32: ((4, 12, 16), (1.0, 2.84, 5.27)),
        64: ((4, 12, 20, 28), (1.0, 2.73, 4.52, 6.31)),
        128: ((4, 12, 20, 28, 64), (1.0, 2.64, 4.64, 6.54, 8.41)),
    }
    ring_sizes, radii = layouts[order]
    points = []
    for ring_index, (count, radius) in enumerate(zip(ring_sizes, radii, strict=True)):
        phase_offset = (ring_index % 2) * np.pi / count
        phases = 2 * np.pi * np.arange(count) / count + phase_offset
        points.extend(radius * np.exp(1j * phases))
    return unit_power(np.asarray(points))


def constellation(name: str) -> np.ndarray:
    """Return ideal points for a supported memoryless modulation."""
    if name == "OOK":
        return _ask(2)
    if name.endswith("ASK"):
        return _ask(int(name[:-3]))
    if name.endswith("APSK"):
        return _apsk(int(name[:-4]))
    if name in ("BPSK", "QPSK"):
        return _psk({"BPSK": 2, "QPSK": 4}[name])
    if name.endswith("PSK") and name != "OQPSK":
        return _psk(int(name[:-3]))
    if name.endswith("QAM"):
        return _qam(int(name[:-3]))
    raise ValueError(f"No memoryless constellation for {name}")


def _shape(symbols: np.ndarray, sps: int, alpha: float) -> np.ndarray:
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    taps = rrc_taps(alpha, sps)
    shaped = np.convolve(upsampled, taps, mode="full")
    delay = (len(taps) - 1) // 2
    return shaped[delay:delay + len(upsampled)]


def crop_signal(x: np.ndarray, start: int, length: int) -> np.ndarray:
    """Return an explicit signal window without padding or wraparound."""
    values = np.asarray(x)
    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if start < 0 or length < 1 or start + length > len(values):
        raise ValueError("crop must lie inside x")
    return values[start:start + length].copy()


def _indices_to_bits(indices: np.ndarray, order: int) -> np.ndarray:
    width = int(math.log2(order))
    shifts = np.arange(width - 1, -1, -1)
    return ((indices[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _gmsk(bits: np.ndarray, sps: int, bt: float) -> np.ndarray:
    impulses = np.repeat(bits, sps)
    span = 4 * sps
    time = np.arange(-span, span + 1) / sps
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * bt)
    gaussian = np.exp(-0.5 * (time / sigma) ** 2)
    gaussian /= gaussian.sum()
    frequency = np.convolve(impulses, gaussian, mode="same")
    phase = np.cumsum(frequency) * (np.pi / (2 * sps))
    return np.exp(1j * phase)


def _digital(
    name: str,
    length: int,
    sps: int,
    alpha: float,
    gmsk_bt: float,
    rng: np.random.Generator,
) -> Waveform:
    if length % sps:
        raise ValueError("digital waveform length must be divisible by samples_per_symbol")
    symbol_count = length // sps
    if name == "OQPSK":
        if sps % 2:
            raise ValueError("OQPSK requires an even samples_per_symbol value")
        i_symbols = rng.choice([-1.0, 1.0], symbol_count)
        q_symbols = rng.choice([-1.0, 1.0], symbol_count)
        i_tx = _shape(i_symbols, sps, alpha)
        q_tx = np.pad(_shape(q_symbols, sps, alpha), (sps // 2, 0))[:len(i_tx)]
        transmitted = i_tx + 1j * q_tx
        i_pure = np.repeat(i_symbols, sps)
        q_pure = np.pad(np.repeat(q_symbols, sps), (sps // 2, 0))[:len(i_pure)]
        pure = i_pure + 1j * q_pure
        symbols = i_symbols + 1j * q_symbols
        bits = np.column_stack((i_symbols > 0, q_symbols > 0)).astype(np.uint8).reshape(-1)
    elif name == "GMSK":
        bits = rng.choice([-1.0, 1.0], symbol_count)
        transmitted = _gmsk(bits, sps, gmsk_bt)
        pure = transmitted.copy()
        return Waveform(
            unit_power(pure),
            unit_power(transmitted),
            sps,
            None,
            source_symbols=bits.astype(np.complex64),
            source_bits=(bits > 0).astype(np.uint8),
            pure_pulse_shape="gaussian",
            gmsk_bt=gmsk_bt,
        )
    else:
        points = constellation(name)
        indices = rng.integers(0, len(points), symbol_count)
        symbols = points[indices]
        transmitted = _shape(symbols, sps, alpha)
        pure = np.repeat(symbols, sps)
        bits = _indices_to_bits(indices, len(points))
    return Waveform(
        unit_power(pure),
        unit_power(transmitted),
        sps,
        float(alpha),
        source_symbols=np.asarray(symbols, dtype=np.complex64),
        source_bits=bits,
    )


def _audio_source(
    length: int, bandwidth: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a bounded, audio-like low-pass message."""
    taps = firwin(129, bandwidth)
    white = rng.standard_normal(length + 4 * len(taps))
    message = lfilter(taps, [1.0], white)[4 * len(taps):]
    message -= message.mean()
    peak = float(np.max(np.abs(message)))
    if peak <= np.finfo(np.float64).tiny:
        raise RuntimeError("generated analog message has zero power")
    return message / peak


def _analog(
    name: str,
    length: int,
    message_bandwidth: float,
    am_modulation_index: float,
    fm_deviation: float,
    rng: np.random.Generator,
) -> Waveform:
    message = _audio_source(length, message_bandwidth, rng)
    if name == "FM":
        signal = np.exp(1j * np.cumsum(message) * fm_deviation)
        metadata = {"fm_deviation": fm_deviation}
    elif "SSB" in name:
        modulated = am_modulation_index * hilbert(message)
        signal = modulated + (1.0 if name.endswith("WC") else 0.0)
        metadata = {"am_modulation_index": am_modulation_index}
    elif "DSB" in name:
        modulated = am_modulation_index * message.astype(np.complex128)
        signal = modulated + (1.0 if name.endswith("WC") else 0.0)
        metadata = {"am_modulation_index": am_modulation_index}
    else:
        raise ValueError(f"Unsupported analog modulation: {name}")
    normalized = unit_power(signal)
    return Waveform(
        normalized,
        normalized.copy(),
        1,
        None,
        source_message=message.astype(np.float32),
        pure_pulse_shape="analog",
        message_bandwidth=message_bandwidth,
        **metadata,
    )


def generate_waveform(
    name: str,
    length: int,
    samples_per_symbol: int,
    rrc_alpha: float,
    rng: np.random.Generator,
    *,
    gmsk_bt: float = 0.3,
    message_bandwidth: float = 0.2,
    am_modulation_index: float = 1.0,
    fm_deviation: float = 0.35,
) -> Waveform:
    """Generate paired pure and clean-transmitted complex-baseband signals.

    Linear digital modulations use a rectangular-pulse ``pure`` target and an
    RRC-shaped ``transmitted`` waveform. GMSK and analog modulations retain
    their modulation-defining ideal waveform as ``pure``.
    """
    if name not in RML2018_MODULATIONS:
        raise ValueError(f"Unsupported modulation: {name}")
    if length < 1:
        raise ValueError("length must be positive")
    if samples_per_symbol < 1:
        raise ValueError("samples_per_symbol must be positive")
    if not 0.0 <= rrc_alpha <= 1.0:
        raise ValueError("RRC roll-off must be in [0, 1]")
    if gmsk_bt <= 0:
        raise ValueError("gmsk_bt must be positive")
    if not 0.0 < message_bandwidth < 1.0:
        raise ValueError("message_bandwidth must be in (0, 1), relative to Nyquist")
    if not 0.0 <= am_modulation_index <= 1.0:
        raise ValueError("am_modulation_index must be in [0, 1]")
    if fm_deviation <= 0:
        raise ValueError("fm_deviation must be positive")

    if name.startswith("AM-") or name == "FM":
        return _analog(
            name,
            length,
            message_bandwidth,
            am_modulation_index,
            fm_deviation,
            rng,
        )
    return _digital(
        name,
        length,
        samples_per_symbol,
        rrc_alpha,
        gmsk_bt,
        rng,
    )
