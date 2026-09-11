# RefDiff 1.1 Time-Domain Baseline

This experiment trains `refdiff_1_1`, whose diffusion state, condition, and
restored result are all time-domain two-channel IQ. The loader performs only
unit-power normalization; it does not call FFT or IFFT.

The original frequency-domain experiment remains in
`experiments/02_train_refdiff_baseline/`. Checkpoints from the two experiments
are intentionally incompatible and use different names:

```text
refdiff_1_1_best.pt
refdiff_1_1_last.pt
```

Run locally with:

```bash
conda run --no-capture-output -n rfsig python \
  experiments/02_train_refdiff_1_1_baseline/run.py \
  --config experiments/02_train_refdiff_1_1_baseline/configs/debug.yaml
```

Each run writes its configuration, metrics, checkpoints, and tee'd log to
`outputs/YYYYMMDD_HHMMSS/`.

