from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rid2026 import RID2026DatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RID2026 dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    RID2026DatasetGenerator(config).generate(args.output)


if __name__ == "__main__":
    main()
