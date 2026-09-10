# RefDiff Baseline Training

## Question

Can a conditional 1D diffusion model restore the paired RID2026 reference
spectrum from an impaired received spectrum?

The model receives only `y_rx`, the noisy diffusion state, and the timestep.
Modulation labels and impairment parameters are not model inputs.

## Metrics

- training and validation noise-prediction MSE;
- validation MSE in low, middle, and high timestep thirds;
- strict validation restoration NMSE from fixed samples and fixed initial noise;
- normalized raw-input NMSE on the same validation samples.

No target-derived gain, phase, or delay alignment is applied. The best
checkpoint is selected by validation restoration NMSE.

## Local Debug Run

From the repository root:

```bash
conda run --no-capture-output -n rfsig python \
  experiments/02_train_refdiff_baseline/run.py \
  --config experiments/02_train_refdiff_baseline/configs/debug.yaml
```

This is an execution check, not an effectiveness experiment. It uses two
training batches and four DDIM steps so it can finish locally.

## Full Training

After placing the full RID2026 file at the path in `configs/full.yaml`:

```bash
conda run --no-capture-output -n rfsig python \
  experiments/02_train_refdiff_baseline/run.py \
  --config experiments/02_train_refdiff_baseline/configs/full.yaml
```

Each invocation creates an ignored directory named from its local start time:

```text
outputs/YYYYMMDD_HHMMSS/
├── run.log
├── config.json
├── history.json
├── last.pt
└── best.pt
```

`run.log` receives both stdout and stderr while retaining live console output.
This keeps the complete training progress or traceback together with the
configuration and checkpoints that produced it.

The full configuration runs for at most 200 epochs. Starting at epoch 20,
training stops when 20 consecutive restoration evaluations fail to improve
strict validation NMSE by at least 0.1% relative to the current early-stopping
reference. `best.pt` always retains the checkpoint with the lowest observed
validation restoration NMSE.
