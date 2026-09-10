from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rfstyle.config import load_config
from rfstyle.data.dataset import PairedSignalDataset
from rfstyle.models import ConditionalUNet1D
from rfstyle.training import GaussianDiffusion


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, diffusion, loader, device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        clean = batch["target"].to(device)
        condition = batch["condition"].to(device)
        t = torch.randint(diffusion.timesteps, (clean.shape[0],), device=device)
        noisy, noise = diffusion.q_sample(clean, t)
        total += F.mse_loss(model(noisy, condition, t), noise, reduction="sum").item()
        count += noise.numel()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frequency-domain conditional diffusion baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 233))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(cfg.get("device", "auto"))

    data_cfg = cfg["dataset"]
    train_set = PairedSignalDataset(data_cfg["root"], "train", data_cfg["target"], data_cfg.get("filters"))
    valid_set = PairedSignalDataset(data_cfg["root"], "valid", data_cfg["target"], data_cfg.get("filters"))
    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,
                              num_workers=cfg["training"].get("num_workers", 0))
    valid_loader = DataLoader(valid_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
                              num_workers=cfg["training"].get("num_workers", 0))

    model = ConditionalUNet1D(cfg["model"]["base_channels"], tuple(cfg["model"]["channel_mults"])).to(device)
    diffusion = GaussianDiffusion(device=device, **cfg["diffusion"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    best = float("inf")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            clean = batch["target"].to(device)
            condition = batch["condition"].to(device)
            t = torch.randint(diffusion.timesteps, (clean.shape[0],), device=device)
            noisy, noise = diffusion.q_sample(clean, t)
            loss = F.mse_loss(model(noisy, condition, t), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        valid_loss = validate(model, diffusion, valid_loader, device)
        print(f"epoch={epoch} train_loss={running / len(train_loader):.6f} valid_loss={valid_loss:.6f}")
        state = {"model": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss, "config": cfg}
        torch.save(state, output / "last.pt")
        if valid_loss < best:
            best = valid_loss
            torch.save(state, output / "best.pt")


if __name__ == "__main__":
    main()

