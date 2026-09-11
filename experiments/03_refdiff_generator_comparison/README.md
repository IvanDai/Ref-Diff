# RefDiff Generator Comparison

This experiment obtains one RID2026 frame, converts `y_rx` and `x_ref` to the
centered IQ spectra used during RefDiff training, runs DDIM restoration, and
writes a three-way time-domain real/imaginary comparison plot.

`parameters.modulation` and `parameters.esn0_db` are required. With no
`dataset`, a fresh signal is generated and omitted waveform/impairment values
follow the RID2026 default distributions. With a `dataset`, one stored sample
matching modulation and Es/N0 is selected, and all generator settings are
ignored. `--weights` and `--dataset` override YAML paths.

```bash
conda run --no-capture-output -n rfsig python \
  experiments/03_refdiff_generator_comparison/run.py \
  --config experiments/03_refdiff_generator_comparison/config.yaml
```

Each run creates `outputs/YYYYMMDD_HHMMSS/` with `comparison.png` and
`metadata.json`.
