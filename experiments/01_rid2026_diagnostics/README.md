# 01 RID2026 Diagnostics

This experiment visually checks constellation geometry, waveform generation,
RRC responses, individual impairments, Es/N0 accuracy, and the end-to-end
simulation chain.

Run it from the repository root:

```bash
conda run -n rfsig python experiments/01_rid2026_diagnostics/run.py
```

Generated plots and summary metrics are written to `results/`.
