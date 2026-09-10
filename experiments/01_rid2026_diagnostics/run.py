from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rid2026-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rid2026-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rid2026 import (
    ImpairmentParameters,
    RFSignalGenerator,
    RML2018_MODULATIONS,
    SignalConfig,
    apply_impairments,
    constellation,
    generate_waveform,
    rrc_taps,
)


RESULTS_DIR = Path(__file__).parent / "results"


def _identity_taps(count: int = 1) -> np.ndarray:
    taps = np.zeros(count, dtype=np.complex64)
    taps[0] = 1.0
    return taps


def _params(**overrides) -> ImpairmentParameters:
    values = dict(
        esn0_db=float("inf"),
        timing_offset=0.0,
        symbol_rate_offset=0.0,
        phase_offset=0.0,
        carrier_offset=0.0,
        delay_spread=0.0,
        channel_taps=_identity_taps(),
    )
    values.update(overrides)
    return ImpairmentParameters(**values)


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(signal))
    spectrum = np.fft.fftshift(np.fft.fft(signal * window))
    power = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    power -= power.max()
    frequency = np.fft.fftshift(np.fft.fftfreq(len(signal)))
    return frequency, power


def plot_constellations() -> None:
    names = [
        name
        for name in RML2018_MODULATIONS
        if not name.startswith("AM-") and name not in ("FM", "GMSK", "OQPSK")
    ]
    figure, axes = plt.subplots(4, 5, figsize=(14, 11))
    for axis, name in zip(axes.flat, names, strict=False):
        points = constellation(name)
        axis.scatter(points.real, points.imag, s=18, color="#126782")
        axis.set_title(name)
        center_i = (points.real.max() + points.real.min()) / 2
        center_q = (points.imag.max() + points.imag.min()) / 2
        span = max(np.ptp(points.real), np.ptp(points.imag), 1.0) * 0.6
        axis.set(xlim=(center_i - span, center_i + span), ylim=(center_q - span, center_q + span))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(names):]:
        axis.axis("off")
    figure.suptitle("Ideal memoryless constellations")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(RESULTS_DIR / "01_constellations.png", dpi=170)
    plt.close(figure)


def plot_all_waveforms() -> None:
    figure, axes = plt.subplots(6, 4, figsize=(16, 16), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, RML2018_MODULATIONS, strict=True)):
        waveform = generate_waveform(
            name, 1024, 8, 0.25, np.random.default_rng(1000 + index)
        )
        axis.plot(waveform.transmitted.real[:160], linewidth=0.8, label="I")
        axis.plot(waveform.transmitted.imag[:160], linewidth=0.8, label="Q")
        axis.set_title(name)
        axis.set_ylim(-2.8, 2.8)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right", fontsize=7)
    figure.suptitle("Time-domain overview of all 24 modulations")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(RESULTS_DIR / "02_waveforms.png", dpi=160)
    plt.close(figure)


def plot_rrc_checks() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for alpha, color in zip((0.1, 0.25, 0.35, 0.5), ("#22223b", "#126782", "#e07a5f", "#6a994e"), strict=True):
        taps = rrc_taps(alpha, 8)
        axes[0].plot(np.arange(len(taps)) - len(taps) // 2, taps, label=f"alpha={alpha}", color=color)
        frequency, response = _spectrum(taps)
        axes[1].plot(frequency, response, label=f"alpha={alpha}", color=color)
    axes[0].set(title="RRC impulse response", xlabel="Sample", ylabel="Amplitude")
    axes[1].set(title="RRC magnitude response", xlabel="Cycles/sample", ylabel="Relative dB", ylim=(-80, 5))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "03_rrc_response.png", dpi=170)
    plt.close(figure)


def plot_impairment_chain() -> None:
    waveform = generate_waveform(
        "QPSK", 4096, 8, 0.25, np.random.default_rng(20)
    )
    multipath = np.array([1.0, 0.45j, -0.25, 0.12j], dtype=np.complex64)
    multipath /= np.linalg.norm(multipath)
    cases = {
        "Clean": _params(),
        "AWGN 5 dB": _params(esn0_db=5.0),
        "Phase +60 deg": _params(phase_offset=np.pi / 3),
        "CFO 0.01": _params(carrier_offset=0.01),
        "Timing + SRO": _params(timing_offset=0.5, symbol_rate_offset=1e-3),
        "Multipath": _params(delay_spread=1.0, channel_taps=multipath),
        "Combined": _params(
            esn0_db=5.0,
            timing_offset=0.5,
            symbol_rate_offset=1e-3,
            phase_offset=np.pi / 3,
            carrier_offset=0.01,
            delay_spread=1.0,
            channel_taps=multipath,
        ),
    }
    figure, axes = plt.subplots(3, len(cases), figsize=(21, 9))
    for column, (title, params) in enumerate(cases.items()):
        received = apply_impairments(
            waveform.transmitted, params, waveform.samples_per_symbol,
            np.random.default_rng(100 + column),
        )
        axes[0, column].plot(received.real[:160], linewidth=0.75, label="I")
        axes[0, column].plot(received.imag[:160], linewidth=0.75, label="Q")
        axes[0, column].set_title(title)
        axes[0, column].set_ylim(-2.5, 2.5)
        axes[1, column].scatter(received.real[::4], received.imag[::4], s=3, alpha=0.35)
        axes[1, column].set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), aspect="equal")
        frequency, power = _spectrum(received)
        axes[2, column].plot(frequency, power, linewidth=0.75)
        axes[2, column].set(xlim=(-0.15, 0.15), ylim=(-70, 3))
        for row in range(3):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Time-domain amplitude")
    axes[1, 0].set_ylabel("Constellation Q")
    axes[2, 0].set_ylabel("Spectrum (dB)")
    figure.suptitle("QPSK impairment diagnostics: time, constellation, and spectrum")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "04_impairments.png", dpi=160)
    plt.close(figure)


def plot_snr_validation() -> dict[str, dict[str, float]]:
    targets = (-10.0, 0.0, 10.0, 20.0)
    measured: dict[str, dict[str, float]] = {}
    figure, axes = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": (1.0, 1.35)}
    )
    error_rows = []
    for modulation_index, modulation in enumerate(RML2018_MODULATIONS):
        waveform = generate_waveform(
            modulation, 16384, 8, 0.25,
            np.random.default_rng(5000 + modulation_index),
        )
        clean = apply_impairments(
            waveform.transmitted, _params(), waveform.samples_per_symbol,
            np.random.default_rng(6000 + modulation_index),
        )
        values = []
        measured[modulation] = {}
        for target_index, target in enumerate(targets):
            noisy = apply_impairments(
                waveform.transmitted,
                _params(esn0_db=target),
                waveform.samples_per_symbol,
                np.random.default_rng(7000 + 10 * modulation_index + target_index),
            )
            noise = noisy - clean
            actual = float(10 * np.log10(
                np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noise) ** 2)
            ) + 10 * np.log10(waveform.samples_per_symbol))
            measured[modulation][str(target)] = actual
            values.append(actual)
        error_rows.append(np.asarray(values) - np.asarray(targets))
        axes[0].plot(targets, values, marker="o", markersize=2.5, linewidth=0.7, alpha=0.6)
    axes[0].plot(targets, targets, color="black", linewidth=2, label="Ideal")
    axes[0].set(
        title="Measured Es/N0 for all 24 modulations",
        xlabel="Configured Es/N0 (dB)",
        ylabel="Measured Es/N0 (dB)",
        xticks=targets,
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    image = axes[1].imshow(
        np.asarray(error_rows), aspect="auto", cmap="coolwarm", vmin=-0.15, vmax=0.15
    )
    axes[1].set(
        title="Measurement error (dB)",
        xlabel="Configured Es/N0 (dB)",
        ylabel="Modulation",
        xticks=np.arange(len(targets)),
        xticklabels=targets,
        yticks=np.arange(len(RML2018_MODULATIONS)),
        yticklabels=RML2018_MODULATIONS,
    )
    figure.colorbar(image, ax=axes[1], label="Measured - configured (dB)")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "05_snr_validation.png", dpi=170)
    plt.close(figure)
    return measured


def plot_high_level_example() -> None:
    generator = RFSignalGenerator(SignalConfig(frame_length=4096), seed=233)
    taps = np.array([1.0, 0.35j, -0.18], dtype=np.complex64)
    taps /= np.linalg.norm(taps)
    result = generator.generate(
        "16QAM",
        _params(
            esn0_db=10.0,
            timing_offset=0.35,
            symbol_rate_offset=2e-4,
            phase_offset=0.3,
            carrier_offset=5e-4,
            delay_spread=1.0,
            channel_taps=taps,
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes[0, 0].plot(result.x_ref.real[:256], label="I", linewidth=0.8)
    axes[0, 0].plot(result.x_ref.imag[:256], label="Q", linewidth=0.8)
    axes[0, 0].set_title("Pure waveform")
    axes[0, 1].plot(result.y_rx.real[:256], label="I", linewidth=0.8)
    axes[0, 1].plot(result.y_rx.imag[:256], label="Q", linewidth=0.8)
    axes[0, 1].set_title("Received waveform")
    axes[1, 0].scatter(
        result.parameters.x_tx.real[::4],
        result.parameters.x_tx.imag[::4],
        s=3,
        alpha=0.3,
    )
    axes[1, 0].set_title("Transmitted IQ samples")
    axes[1, 1].scatter(result.y_rx.real[::4], result.y_rx.imag[::4], s=3, alpha=0.3)
    axes[1, 1].set_title("Received IQ samples")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("End-to-end 16QAM simulation")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "06_end_to_end.png", dpi=170)
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_constellations()
    plot_all_waveforms()
    plot_rrc_checks()
    plot_impairment_chain()
    snr_metrics = plot_snr_validation()
    plot_high_level_example()
    (RESULTS_DIR / "snr_metrics.json").write_text(
        json.dumps(snr_metrics, indent=2), encoding="utf-8"
    )
    print(f"Wrote diagnostics to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
