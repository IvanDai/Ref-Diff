from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from rid2026 import RID2026DatasetGenerator


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def create_run_directory(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RID2026 dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.workers is not None:
        config["workers"] = args.workers
    if args.compression is not None:
        config["compression"] = (
            None if args.compression == "none" else args.compression
        )

    run_dir = create_run_directory(Path(__file__).parent / "outputs")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "output_path.txt").write_text(
        str(args.output.resolve()) + "\n", encoding="utf-8"
    )

    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, log)
        try:
            generator = RID2026DatasetGenerator(config)
            started = time.monotonic()
            next_percent = 0

            def report(completed: int, total: int) -> None:
                nonlocal next_percent
                percent = completed * 100 // total
                if percent < next_percent and completed != total:
                    return
                elapsed = time.monotonic() - started
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"progress={completed}/{total} ({percent}%) "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s"
                )
                next_percent = percent + 1

            print(f"run_dir={run_dir}")
            print(f"output={args.output}")
            print(
                f"workers={generator.workers} compression={generator.compression} "
                f"examples={generator.example_count()}"
            )
            generator.generate(args.output, progress=report)
            print(f"completed elapsed={time.monotonic() - started:.1f}s")
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
