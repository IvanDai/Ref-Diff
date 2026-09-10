# RID2026

RID2026 generates paired complex-baseband signals for reference restoration
research. The reusable package produces a clean reference waveform `x_ref`,
the impaired received waveform `y_rx`, and the signal-processing parameters
used for each sample.

## Repository Layout

```text
src/rid2026/                         Reusable simulation and HDF5 storage code
tests/00_rid2026_waveforms/          Waveform and constellation tests
tests/01_rid2026_impairments/        Channel and receiver impairment tests
tests/02_rid2026_generation/         Result and HDF5 contract tests
tests/03_rid2026_package_boundaries/ Dependency-boundary tests
experiments/00_generate_rid2026/     Smoke and full dataset generation
experiments/01_rid2026_diagnostics/  Diagnostic figures and metrics
datasets/rid2026/                    Local generated datasets
```

The repository has one root `pyproject.toml`. The `src/rid2026/` directory
contains no experiment configuration, test assembly, generated results, or
package-specific build metadata.

## Verification

Install this repository once into the configured Conda environment without
changing its dependencies:

```bash
conda run -n rfsig python -m pip install --no-deps -e .
```

Then run verification from the repository root:

```bash
conda run -n rfsig python -m pytest
```

See `src/rid2026/README.md` for the signal definitions, RML2018-compatible
parameter distributions, HDF5 contract, and generation commands.
