# 00 Generate RID2026

This experiment materializes the reusable `rid2026` generator as one HDF5
dataset. It does not create train, validation, or test splits; downstream code
owns that decision.

After the root project has been installed in editable mode as described in the
root README, run the smoke configuration with:

```bash
conda run -n rfsig python experiments/00_generate_rid2026/run.py \
  --config experiments/00_generate_rid2026/configs/smoke.yaml \
  --output datasets/rid2026/rid2026_smoke.h5
```

Available configurations are:

- `configs/smoke.yaml`: three modulations, two Es/N0 levels, 18 examples per
  condition.
- `configs/debug.yaml`: all 24 modulations and 26 Es/N0 levels, 40 examples per
  condition (24,960 examples total).
- `configs/full.yaml`: all conditions, 4096 examples per condition (2,555,904
  examples total).

The generator uses multiple worker processes for signal calculation and keeps
HDF5 writing in the parent process. `workers` controls parallelism and
`write_batch_size` controls each bounded work item. The full and debug configs
use eight workers by default.

Compression is optional:

```yaml
compression: null  # fastest, approximately 40 GiB for the full dataset
compression: gzip  # smaller but substantially more CPU work
compression: lzf   # faster compression with a lower compression ratio
```

Command-line options override these two settings without editing YAML:

```bash
conda run -n rfsig python experiments/00_generate_rid2026/run.py \
  --config experiments/00_generate_rid2026/configs/full.yaml \
  --output datasets/rid2026/rid2026.h5 \
  --workers 8 --compression none
```

Every run creates `outputs/YYYYMMDD_HHMMSS/` containing `run.log`, the actual
`config.yaml`, and `output_path.txt`. Progress remains visible in the terminal
while the same output is written to the log.
