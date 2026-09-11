from __future__ import annotations

import argparse
import json
import random
import sys
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


class ProgressBar:
    """Render progress in one terminal line and persist only its final state."""

    def __init__(self, label: str, console: TextIO, log: TextIO, *, unit: str = "batch"):
        self.label, self.console, self.log, self.unit = label, console, log, unit
        self.started = __import__("time").monotonic()
        self.last_rendered = 0.0
        self.last_line = ""

    @staticmethod
    def duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(self, current: int, total: int, metrics: dict[str, float]) -> None:
        import time
        now = time.monotonic()
        if current < total and now - self.last_rendered < 0.2:
            return
        self.last_rendered = now
        elapsed = max(now - self.started, 1e-9)
        fraction = min(current / max(total, 1), 1.0)
        width = 24
        filled = min(int(fraction * width), width)
        details = []
        if "loss" in metrics:
            details.append(f"loss={metrics['loss']:.5f}")
        if "mean_loss" in metrics:
            details.append(f"mean={metrics['mean_loss']:.5f}")
        rate = current / elapsed
        details.extend((f"{rate:.2f} {self.unit}/s", f"ETA {self.duration((total-current)/rate if rate else 0)}"))
        self.last_line = f"{self.label:<20} [{'#' * filled}{'-' * (width-filled)}] {current}/{total} {fraction:6.1%}  " + "  ".join(details)
        self.console.write("\r\x1b[2K" + self.last_line)
        self.console.flush()

    def close(self) -> None:
        if self.last_line:
            self.console.write("\n")
            self.console.flush()
            self.log.write(self.last_line + "\n")
            self.log.flush()


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


def open_console() -> tuple[TextIO, bool]:
    if sys.stdout.isatty():
        return sys.stdout, False
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1), True
    except OSError:
        return sys.stdout, False


def make_dataset(config: dict, split: str) -> RID2026TimeDomainPairs:
    return RID2026TimeDomainPairs(
        config["path"], split, config["split_counts"],
        modulations=config.get("modulations"), esn0_values=config.get("esn0_values"),
    )


def run(config_path: Path, output: Path, started_at: datetime, console: TextIO, log: TextIO) -> None:
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
    import time
    for epoch in range(1, total_epochs + 1):
        epoch_started = time.monotonic()
        train_progress = ProgressBar(f"Epoch {epoch}/{total_epochs} Train", console, log)
        try:
            train_loss = train_epoch(
            model, diffusion, train_loader, optimizer, device, ema,
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            max_batches=training.get("max_train_batches"),
            progress_callback=train_progress.update,
            )
        finally:
            train_progress.close()
        valid_progress = ProgressBar(f"Epoch {epoch}/{total_epochs} Valid", console, log)
        try:
            validation = validate_noise(
            ema.model, diffusion, valid_loader, device,
            max_batches=training.get("max_valid_batches"), seed=seed + 20_000,
            progress_callback=valid_progress.update,
            )
        finally:
            valid_progress.close()
        metrics = {
            "epoch": epoch, "train_loss": train_loss, "valid_loss": validation.loss,
            "valid_low_t_loss": validation.low_t_loss,
            "valid_middle_t_loss": validation.middle_t_loss,
            "valid_high_t_loss": validation.high_t_loss,
        }
        should_stop = False
        if epoch % int(training["restoration_interval"]) == 0 or epoch == total_epochs:
            restore_progress = ProgressBar(f"Epoch {epoch}/{total_epochs} Restore", console, log, unit="step")
            try:
                restoration = validate_restoration(
                ema.model, diffusion, restoration_loader, device,
                steps=int(training["ddim_steps"]), sample_count=restoration_count,
                seed=seed + 10_000,
                progress_callback=restore_progress.update,
                )
            finally:
                restore_progress.close()
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
        print(
            f"Epoch {epoch}/{total_epochs} Summary | train={train_loss:.5f} "
            f"valid={validation.loss:.5f} restored_nmse={metrics.get('restoration_nmse', float('nan')):.5f} "
            f"elapsed={ProgressBar.duration(time.monotonic() - epoch_started)}"
        )
        if should_stop:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RefDiff 1.1 in the time domain")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output, started_at = create_run_directory()
    console, close_console = open_console()
    try:
        with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            tee = Tee(console, log)
            with redirect_stdout(tee), redirect_stderr(tee):
                try:
                    run(args.config, output, started_at, console, log)
                except Exception:
                    traceback.print_exc()
                    return 1
        return 0
    finally:
        if close_console:
            console.close()


if __name__ == "__main__":
    raise SystemExit(main())
