from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .impairments import ImpairmentParameters, apply_impairments
from .waveforms import crop_signal, generate_waveform, unit_power


def derive_seed(root_seed: int, *coordinates: int) -> int:
    """Derive a stable uint64 seed from independent sample coordinates."""
    if root_seed < 0 or any(value < 0 for value in coordinates):
        raise ValueError("seed components cannot be negative")
    sequence = np.random.SeedSequence(root_seed, spawn_key=coordinates)
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for one cropped observation frame and its guard region."""

    frame_length: int = 1024
    samples_per_symbol: int = 8
    guard_symbols: int = 32
    rrc_alpha: float = 0.35
    gmsk_bt: float = 0.3
    message_bandwidth: float = 0.2
    am_modulation_index: float = 1.0
    fm_deviation: float = 0.35

    def __post_init__(self) -> None:
        if self.frame_length < 1:
            raise ValueError("frame_length must be positive")
        if self.samples_per_symbol < 1:
            raise ValueError("samples_per_symbol must be positive")
        if self.frame_length % self.samples_per_symbol:
            raise ValueError("frame_length must be divisible by samples_per_symbol")
        if self.guard_symbols < 1:
            raise ValueError("guard_symbols must be positive")


@dataclass(frozen=True)
class WaveformParameters:
    samples_per_symbol: int
    frame_symbols: int | None
    pure_pulse_shape: str
    rrc_alpha: float | None
    gmsk_bt: float | None
    am_modulation_index: float | None
    fm_deviation: float | None


@dataclass(frozen=True)
class SimulationParameters:
    waveform: WaveformParameters
    impairments: ImpairmentParameters
    x_tx: np.ndarray
    source_symbols: np.ndarray | None
    source_bits: np.ndarray | None
    source_message: np.ndarray | None
    sample_seed: int | None
    guard_samples: int


@dataclass(frozen=True)
class SimulationResult:
    """The four-part record used by RID2026 dataset storage."""

    y_rx: np.ndarray
    modulation: str
    x_pure: np.ndarray
    parameters: SimulationParameters


def _crop_optional(
    values: np.ndarray | None, start: int, length: int
) -> np.ndarray | None:
    if values is None:
        return None
    return crop_signal(values, start, length)


class RFSignalGenerator:
    """Generate aligned pure, transmitted, and impaired complex-baseband frames."""

    def __init__(self, config: SignalConfig | None = None, seed: int | None = None):
        self.config = config or SignalConfig()
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        modulation: str,
        impairments: ImpairmentParameters | None = None,
        *,
        seed: int | None = None,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        config = self.config
        sps = config.samples_per_symbol
        guard_samples = config.guard_symbols * sps
        full_length = config.frame_length + 2 * guard_samples
        waveform = generate_waveform(
            modulation,
            full_length,
            sps,
            config.rrc_alpha,
            rng,
            gmsk_bt=config.gmsk_bt,
            message_bandwidth=config.message_bandwidth,
            am_modulation_index=config.am_modulation_index,
            fm_deviation=config.fm_deviation,
        )
        selected_impairments = impairments or ImpairmentParameters.identity()
        full_received = apply_impairments(
            waveform.transmitted,
            selected_impairments,
            waveform.samples_per_symbol,
            rng,
        )

        x_pure = unit_power(crop_signal(
            waveform.pure, guard_samples, config.frame_length
        ))
        x_tx = unit_power(crop_signal(
            waveform.transmitted, guard_samples, config.frame_length
        ))
        y_rx = crop_signal(full_received, guard_samples, config.frame_length).astype(
            np.complex64
        )

        if waveform.source_symbols is None:
            frame_symbols = None
            source_symbols = None
            source_bits = None
        else:
            frame_symbols = config.frame_length // sps
            symbol_start = config.guard_symbols
            source_symbols = _crop_optional(
                waveform.source_symbols, symbol_start, frame_symbols
            )
            bits_per_symbol = (
                len(waveform.source_bits) // len(waveform.source_symbols)
                if waveform.source_bits is not None
                else 0
            )
            source_bits = (
                _crop_optional(
                    waveform.source_bits,
                    symbol_start * bits_per_symbol,
                    frame_symbols * bits_per_symbol,
                )
                if bits_per_symbol
                else None
            )

        source_message = _crop_optional(
            waveform.source_message, guard_samples, config.frame_length
        )
        waveform_parameters = WaveformParameters(
            samples_per_symbol=waveform.samples_per_symbol,
            frame_symbols=frame_symbols,
            pure_pulse_shape=waveform.pure_pulse_shape,
            rrc_alpha=waveform.rrc_alpha,
            gmsk_bt=waveform.gmsk_bt,
            am_modulation_index=waveform.am_modulation_index,
            fm_deviation=waveform.fm_deviation,
        )
        parameters = SimulationParameters(
            waveform=waveform_parameters,
            impairments=selected_impairments,
            x_tx=x_tx,
            source_symbols=source_symbols,
            source_bits=source_bits,
            source_message=source_message,
            sample_seed=seed,
            guard_samples=guard_samples,
        )
        return SimulationResult(y_rx, modulation, x_pure, parameters)
