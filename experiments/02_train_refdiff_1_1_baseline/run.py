from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from refdiff import GaussianDiffusion
from refdiff.training import (
    EarlyStopping,
    ExponentialMovingAverage,
    save_checkpoint,
    train_epoch,
    validate_noise,
    validate_restoration,
    write_history,
)
from refdiff_1_1 import ConditionBatchSampler, ConditionalUNet1D, RID2026TimeDomainPairs


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def create_run_directory() -> tuple[Path, datetime]:
    started_at = datetime.now().astimezone()
    root = Path(__file__).with_name("outputs")
    name = started_at.strftime("%Y%m%d_%H%M%S")
    output = root / name
    suffix = 1
    while output.exists():
        output = root / f"{name}_{suffix:02d}"
        suffix += 1
    output.mkdir(parents=True)
    return output, started_at


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataset(config: dict, split: str) -> RID2026TimeDomainPairs:
    return RID2026TimeDomainPairs(
        config["path"], split, config["split_counts"],
        modulations=config.get("modulations"), esn0_values=config.get("esn0_values"),
    )


def run(config_path: Path, output: Path, started_at: datetime) -> None:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = int(config.get("seed", 233))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(config.get("device", "auto"))
    data_config = config["dataset"]
    train_set = make_dataset(data_config, "train")
    valid_set = make_dataset(data_config, "valid")
    training = config["training"]
    batch_size = int(training["batch_size"])
    loader_options = {
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    if training.get("condition_batched", True):
        sampler = ConditionBatchSampler(train_set, batch_size, seed=seed)
        train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_options)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, **loader_options)
    restoration_count = min(int(training["restoration_samples"]), len(valid_set))
    indices = np.linspace(0, len(valid_set) - 1, restoration_count, dtype=np.int64).tolist()
    restoration_loader = DataLoader(
        Subset(valid_set, indices), batch_size=batch_size, shuffle=False, **loader_options
    )
    model_config = config["model"]
    model = ConditionalUNet1D(
        int(model_config["base_channels"]), tuple(model_config["channel_multipliers"])
    ).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema = ExponentialMovingAverage(model, float(training["ema_decay"]))
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best_nmse = [], float("inf")
    early_config = training.get("early_stopping")
    early_stopping = EarlyStopping(
        patience=int(early_config["patience"]), min_epochs=int(early_config.get("min_epochs", 0)),
        min_relative_improvement=float(early_config.get("min_relative_improvement", 0.0)),
    ) if early_config and early_config.get("enabled", True) else None
    print(f"started_at={started_at.isoformat()}\nconfig={config_path.resolve()}\noutput_dir={output.resolve()}")
    print(f"version=refdiff_1_1 domain=time device={device} train_examples={len(train_set)} valid_examples={len(valid_set)}")
    total_epochs = int(training["epochs"])
    for epoch in range(1, total_epochs + 1):
        train_loss = train_epoch(
            model, diffusion, train_loader, optimizer, device, ema,
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            max_batches=training.get("max_train_batches"),
        )
        validation = validate_noise(
            ema.model, diffusion, valid_loader, device,
            max_batches=training.get("max_valid_batches"), seed=seed + 20_000,
        )
        metrics = {
            "epoch": epoch, "train_loss": train_loss, "valid_loss": validation.loss,
            "valid_low_t_loss": validation.low_t_loss,
            "valid_middle_t_loss": validation.middle_t_loss,
            "valid_high_t_loss": validation.high_t_loss,
        }
        should_stop = False
        if epoch % int(training["restoration_interval"]) == 0 or epoch == total_epochs:
            restoration = validate_restoration(
                ema.model, diffusion, restoration_loader, device,
                steps=int(training["ddim_steps"]), sample_count=restoration_count,
                seed=seed + 10_000,
            )
            metrics.update(restoration)
            if restoration["restoration_nmse"] < best_nmse:
                best_nmse = restoration["restoration_nmse"]
                save_checkpoint(output / "refdiff_1_1_best.pt", model, ema, optimizer, epoch, config, metrics)
            if early_stopping is not None:
                should_stop = early_stopping.update(restoration["restoration_nmse"], epoch)
                metrics.update({
                    "early_stopping_bad_evaluations": early_stopping.bad_evaluations,
                    "early_stopping_best_nmse": early_stopping.best,
                    "stopped_early": should_stop,
                })
        history.append(metrics)
        write_history(output / "history.json", history)
        save_checkpoint(output / "refdiff_1_1_last.pt", model, ema, optimizer, epoch, config, metrics)
        print(json.dumps(metrics, sort_keys=True))
        if should_stop:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RefDiff 1.1 in the time domain")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output, started_at = create_run_directory()
    try:
        with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            tee = Tee(sys.stdout, log)
            with redirect_stdout(tee), redirect_stderr(tee):
                try:
                    run(args.config, output, started_at)
                except Exception:
                    traceback.print_exc()
                    return 1
        return 0
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())

