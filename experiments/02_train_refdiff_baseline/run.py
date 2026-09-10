from __future__ import annotations

import argparse
import json
import random
import sys
import time
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

from refdiff import ConditionalUNet1D, GaussianDiffusion, RID2026Pairs
from refdiff.data import ConditionBatchSampler
from refdiff.training import (
    ExponentialMovingAverage,
    save_checkpoint,
    train_epoch,
    validate_noise,
    validate_restoration,
    write_history,
)


class Tee:
    def __init__(self, *streams) -> None:
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
    output_root = Path(__file__).with_name("outputs")
    base_name = started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / base_name
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir, started_at


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(config: dict, split: str) -> RID2026Pairs:
    return RID2026Pairs(
        config["path"],
        split,
        config["split_counts"],
        modulations=config.get("modulations"),
        esn0_values=config.get("esn0_values"),
    )


def run(config_path: Path, output_dir: Path, started_at: datetime) -> None:
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
    training_config = config["training"]
    train_set = build_dataset(data_config, "train")
    valid_set = build_dataset(data_config, "valid")
    loader_options = {
        "num_workers": int(training_config.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    batch_size = int(training_config["batch_size"])
    if training_config.get("condition_batched", True):
        batch_sampler = ConditionBatchSampler(train_set, batch_size, seed=seed)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **loader_options)
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, **loader_options
        )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, **loader_options
    )
    restoration_count = min(int(training_config["restoration_samples"]), len(valid_set))
    restoration_indices = np.linspace(
        0, len(valid_set) - 1, restoration_count, dtype=np.int64
    ).tolist()
    restoration_loader = DataLoader(
        Subset(valid_set, restoration_indices),
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    model_config = config["model"]
    model = ConditionalUNet1D(
        base_channels=int(model_config["base_channels"]),
        channel_multipliers=tuple(model_config["channel_multipliers"]),
    ).to(device)
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    ema = ExponentialMovingAverage(model, float(training_config["ema_decay"]))

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    history = []
    best_nmse = float("inf")
    print(f"started_at={started_at.isoformat()}")
    print(f"config={config_path.resolve()}")
    print(f"output_dir={output_dir.resolve()}")
    print(
        f"device={device} train_examples={len(train_set)} "
        f"valid_examples={len(valid_set)} parameters={sum(p.numel() for p in model.parameters())}"
    )

    for epoch in range(1, int(training_config["epochs"]) + 1):
        started = time.monotonic()
        train_loss = train_epoch(
            model,
            diffusion,
            train_loader,
            optimizer,
            device,
            ema,
            gradient_clip=float(training_config.get("gradient_clip", 1.0)),
            max_batches=training_config.get("max_train_batches"),
            log_interval=training_config.get("log_interval"),
        )
        validation = validate_noise(
            ema.model,
            diffusion,
            valid_loader,
            device,
            max_batches=training_config.get("max_valid_batches"),
            seed=seed + 20_000,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": validation.loss,
            "valid_low_t_loss": validation.low_t_loss,
            "valid_middle_t_loss": validation.middle_t_loss,
            "valid_high_t_loss": validation.high_t_loss,
        }
        interval = int(training_config["restoration_interval"])
        if epoch % interval == 0 or epoch == int(training_config["epochs"]):
            restoration = validate_restoration(
                ema.model,
                diffusion,
                restoration_loader,
                device,
                steps=int(training_config["ddim_steps"]),
                sample_count=restoration_count,
                seed=seed + 10_000,
            )
            metrics.update(restoration)
            if restoration["restoration_nmse"] < best_nmse:
                best_nmse = restoration["restoration_nmse"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    ema,
                    optimizer,
                    epoch,
                    config,
                    metrics,
                )
        metrics["elapsed_seconds"] = time.monotonic() - started
        history.append(metrics)
        write_history(output_dir / "history.json", history)
        save_checkpoint(
            output_dir / "last.pt", model, ema, optimizer, epoch, config, metrics
        )
        print(json.dumps(metrics, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the RefDiff baseline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_dir, started_at = create_run_directory()
    with (output_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                run(args.config, output_dir, started_at)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
