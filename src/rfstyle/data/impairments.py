"""Compatibility imports for the standalone :mod:`rid2026` package."""

from rid2026.impairments import (
    ImpairmentParameters,
    apply_impairments,
    resample_with_offset,
    sample_channel,
    sample_parameters,
)

__all__ = [
    "ImpairmentParameters",
    "apply_impairments",
    "resample_with_offset",
    "sample_channel",
    "sample_parameters",
]
