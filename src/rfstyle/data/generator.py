"""Compatibility imports for the :mod:`rid2026` dataset generator."""

from rid2026.dataset import RID2026DatasetGenerator, complex_to_iq

RML2018Generator = RID2026DatasetGenerator

__all__ = ["RID2026DatasetGenerator", "RML2018Generator", "complex_to_iq"]
