# RefDiff Generator Comparison

This experiment obtains one RID2026 frame, applies the matching RefDiff
preprocessing, runs DDIM restoration, and writes a three-way time-domain
real/imaginary comparison plot.

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

For the time-domain RefDiff 1.1 model, use the copied entry point:

```bash
conda run --no-capture-output -n rfsig python \
  experiments/03_refdiff_generator_comparison/run_1_1.py \
  --config experiments/03_refdiff_generator_comparison/config.yaml \
  --weights experiments/02_train_refdiff_1_1_baseline/outputs/<run>/refdiff_1_1_best.pt \
  --dataset datasets/rid2026/rid2026.h5
```

`run_1_1.py` uses `refdiff_1_1.iq_to_time` and the `refdiff_1_1` U-Net;
it must be paired with `refdiff_1_1_*` checkpoints.
