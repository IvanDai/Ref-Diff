from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from rfstyle.data.dataset import PairedSignalDataset
from rfstyle.models import ConditionalUNet1D
from rfstyle.training import GaussianDiffusion
from scripts.train import choose_device


def as_complex(x: torch.Tensor) -> torch.Tensor:
    spectrum = torch.complex(x[:, 0], x[:, 1])
    return torch.fft.ifft(torch.fft.ifftshift(spectrum, dim=-1), norm="ortho")


def align(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    gain = (estimate.conj() * reference).sum(-1) / (estimate.abs().square().sum(-1).clamp_min(1e-8))
    return estimate * gain[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--ddim-steps", type=int, default=50)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    device = choose_device(cfg.get("device", "auto"))
    data_cfg = cfg["dataset"]
    dataset = PairedSignalDataset(data_cfg["root"], "test", data_cfg["target"], data_cfg.get("filters"))
    loader = DataLoader(dataset, batch_size=min(args.samples, cfg["training"]["batch_size"]), shuffle=False)
    model = ConditionalUNet1D(cfg["model"]["base_channels"], tuple(cfg["model"]["channel_mults"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(device=device, **cfg["diffusion"])

    nmse_values, evm_values = [], []
    example = None
    seen = 0
    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        estimate = diffusion.ddim_sample(model, condition, args.ddim_steps)
        target_time = as_complex(target)
        estimate_time = align(target_time, as_complex(estimate))
        error = (estimate_time - target_time).abs().square().sum(-1)
        power = target_time.abs().square().sum(-1).clamp_min(1e-8)
        nmse_values.extend((error / power).cpu().tolist())
        evm_values.extend(torch.sqrt(error / power).cpu().tolist())
        if example is None:
            example = (as_complex(condition[:1]).cpu(), estimate_time[:1].cpu(), target_time[:1].cpu())
        seen += len(condition)
        if seen >= args.samples:
            break

    print(f"samples={min(seen, args.samples)} nmse={np.mean(nmse_values):.6f} evm={np.mean(evm_values):.6f}")
    output = Path(args.checkpoint).parent
    received, estimate, target = (item[0].numpy() for item in example)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for row, (name, signal) in enumerate((("received", received), ("estimate", estimate), ("target", target))):
        axes[row, 0].plot(signal.real, label="I", linewidth=0.8)
        axes[row, 0].plot(signal.imag, label="Q", linewidth=0.8)
        axes[row, 0].set_title(f"{name}: time")
        spectrum = np.fft.fftshift(np.fft.fft(signal, norm="ortho"))
        axes[row, 1].plot(20 * np.log10(np.abs(spectrum) + 1e-8), linewidth=0.8)
        axes[row, 1].set_title(f"{name}: spectrum")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "reconstruction.png", dpi=160)


if __name__ == "__main__":
    main()

