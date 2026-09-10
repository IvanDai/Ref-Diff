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
from typing import TextIO

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
    EarlyStopping,
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


class ProgressBar:
    """Single-line terminal progress whose completed line is persisted once."""

    def __init__(
        self,
        label: str,
        console: TextIO,
        log: TextIO,
        *,
        width: int = 24,
        unit: str = "batch",
        min_interval: float = 0.2,
    ) -> None:
        self.label = label
        self.console = console
        self.log = log
        self.width = width
        self.unit = unit
        self.min_interval = min_interval
        self.started = time.monotonic()
        self.last_rendered = 0.0
        self.last_line = ""

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(self, current: int, total: int, metrics: dict[str, float]) -> None:
        now = time.monotonic()
        if current < total and now - self.last_rendered < self.min_interval:
            return
        self.last_rendered = now
        elapsed = max(now - self.started, 1e-9)
        fraction = min(current / max(total, 1), 1.0)
        filled = min(int(fraction * self.width), self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = current / elapsed
        eta = (total - current) / rate if rate > 0 else 0.0
        details = []
        if "loss" in metrics:
            details.append(f"loss={metrics['loss']:.5f}")
        if "mean_loss" in metrics:
            details.append(f"mean={metrics['mean_loss']:.5f}")
        details.append(f"{rate:.2f} {self.unit}/s")
        details.append(f"ETA {self._duration(eta)}")
        self.last_line = (
            f"{self.label:<18} [{bar}] {current:>{len(str(total))}}/{total} "
            f"{fraction:6.1%}  " + "  ".join(details)
        )
        self.console.write(f"\r\x1b[2K{self.last_line}")
        self.console.flush()

    def close(self) -> None:
        if not self.last_line:
            return
        self.console.write("\n")
        self.console.flush()
        self.log.write(self.last_line + "\n")
        self.log.flush()


def format_epoch_summary(epoch: int, total_epochs: int, metrics: dict) -> str:
    summary = (
        f"Epoch {epoch}/{total_epochs} Summary | "
        f"train={metrics['train_loss']:.5f}  valid={metrics['valid_loss']:.5f}  "
        f"timestep(low/mid/high)={metrics['valid_low_t_loss']:.5f}/"
        f"{metrics['valid_middle_t_loss']:.5f}/{metrics['valid_high_t_loss']:.5f}"
    )
    if "restoration_nmse_db" in metrics:
        summary += (
            f"  NMSE(restored/raw)={metrics['restoration_nmse_db']:.2f}/"
            f"{metrics['raw_nmse_db']:.2f} dB"
        )
    summary += f"  elapsed={ProgressBar._duration(metrics['elapsed_seconds'])}"
    return summary


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


def run(
    config_path: Path,
    output_dir: Path,
    started_at: datetime,
    console: TextIO,
    log: TextIO,
) -> None:
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
    early_config = training_config.get("early_stopping")
    early_stopping = (
        EarlyStopping(
            patience=int(early_config["patience"]),
            min_epochs=int(early_config.get("min_epochs", 0)),
            min_relative_improvement=float(
                early_config.get("min_relative_improvement", 0.0)
            ),
        )
        if early_config and early_config.get("enabled", True)
        else None
    )
    print(f"started_at={started_at.isoformat()}")
    print(f"config={config_path.resolve()}")
    print(f"output_dir={output_dir.resolve()}")
    print(
        f"device={device} train_examples={len(train_set)} "
        f"valid_examples={len(valid_set)} parameters={sum(p.numel() for p in model.parameters())}"
    )

    total_epochs = int(training_config["epochs"])
    for epoch in range(1, total_epochs + 1):
        started = time.monotonic()
        train_progress = ProgressBar(
            f"Epoch {epoch}/{total_epochs} Train", console, log
        )
        try:
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
                progress_callback=train_progress.update,
            )
        finally:
            train_progress.close()

        valid_progress = ProgressBar(
            f"Epoch {epoch}/{total_epochs} Valid", console, log
        )
        try:
            validation = validate_noise(
                ema.model,
                diffusion,
                valid_loader,
                device,
                max_batches=training_config.get("max_valid_batches"),
                seed=seed + 20_000,
                progress_callback=valid_progress.update,
            )
        finally:
            valid_progress.close()
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": validation.loss,
            "valid_low_t_loss": validation.low_t_loss,
            "valid_middle_t_loss": validation.middle_t_loss,
            "valid_high_t_loss": validation.high_t_loss,
        }
        interval = int(training_config["restoration_interval"])
        should_stop = False
        is_best = False
        if epoch % interval == 0 or epoch == total_epochs:
            restore_progress = ProgressBar(
                f"Epoch {epoch}/{total_epochs} Restore",
                console,
                log,
                unit="step",
            )
            try:
                restoration = validate_restoration(
                    ema.model,
                    diffusion,
                    restoration_loader,
                    device,
                    steps=int(training_config["ddim_steps"]),
                    sample_count=restoration_count,
                    seed=seed + 10_000,
                    progress_callback=restore_progress.update,
                )
            finally:
                restore_progress.close()
            metrics.update(restoration)
            if restoration["restoration_nmse"] < best_nmse:
                best_nmse = restoration["restoration_nmse"]
                is_best = True
            if early_stopping is not None:
                should_stop = early_stopping.update(
                    restoration["restoration_nmse"], epoch
                )
                metrics["early_stopping_bad_evaluations"] = (
                    early_stopping.bad_evaluations
                )
                metrics["early_stopping_best_nmse"] = early_stopping.best
                metrics["stopped_early"] = should_stop
        metrics["elapsed_seconds"] = time.monotonic() - started
        history.append(metrics)
        write_history(output_dir / "history.json", history)
        if is_best:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                ema,
                optimizer,
                epoch,
                config,
                metrics,
            )
        save_checkpoint(
            output_dir / "last.pt", model, ema, optimizer, epoch, config, metrics
        )
        print(format_epoch_summary(epoch, total_epochs, metrics))
        if should_stop:
            print(
                "early_stopping: "
                f"epoch={epoch} patience={early_stopping.patience} "
                f"best_nmse={early_stopping.best:.8g}"
            )
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the RefDiff baseline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_dir, started_at = create_run_directory()
    console_stdout = sys.stdout
    with (output_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        stdout = Tee(sys.stdout, log)
        stderr = Tee(sys.stderr, log)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                run(args.config, output_dir, started_at, console_stdout, log)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
