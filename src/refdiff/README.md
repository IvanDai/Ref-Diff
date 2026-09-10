# RefDiff

`refdiff` restores the paired RID2026 reference waveform from an impaired
received waveform. It does not consume modulation labels or impairment
parameters as model inputs.

The baseline represents complex IQ as two real channels and applies an
orthonormal, centered FFT. Diffusion noise is added only to `x_ref`; `y_rx`
remains available as the condition throughout the reverse process.

Reusable responsibilities are split across:

- `data.py`: RID2026 HDF5 reader and IQ representation;
- `model.py`: multi-scale conditional 1D U-Net;
- `diffusion.py`: forward diffusion and DDIM restoration;
- `metrics.py`: strict, unaligned restoration NMSE;
- `training.py`: optimization, validation, EMA, and checkpoints.

The runnable baseline is assembled in
`experiments/02_train_refdiff_baseline/`.

