"""RID2026 complex-baseband simulation core and dataset generator."""

from .dataset import RID2026DatasetGenerator, generate_dataset
from .generator import (
    RFSignalGenerator,
    SignalConfig,
    SimulationParameters,
    SimulationResult,
    WaveformParameters,
    derive_seed,
)
from .impairments import (
    ImpairmentParameters,
    apply_impairments,
    resample_with_offset,
    sample_channel,
    sample_path_delays,
    sample_parameters,
)
from .waveforms import (
    RML2018_MODULATIONS,
    Waveform,
    constellation,
    crop_signal,
    generate_waveform,
    rrc_taps,
    unit_power,
)

__all__ = [
    "ImpairmentParameters",
    "RFSignalGenerator",
    "RID2026DatasetGenerator",
    "RML2018_MODULATIONS",
    "SignalConfig",
    "SimulationResult",
    "SimulationParameters",
    "Waveform",
    "WaveformParameters",
    "apply_impairments",
    "constellation",
    "crop_signal",
    "derive_seed",
    "generate_waveform",
    "generate_dataset",
    "resample_with_offset",
    "rrc_taps",
    "sample_channel",
    "sample_path_delays",
    "sample_parameters",
    "unit_power",
]

__version__ = "0.1.0"
