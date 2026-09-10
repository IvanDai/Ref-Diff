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

Use `configs/full.yaml` and `datasets/rid2026/rid2026.h5` for the complete
2,555,904-example dataset after confirming available disk space.
