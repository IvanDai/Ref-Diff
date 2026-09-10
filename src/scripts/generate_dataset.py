from __future__ import annotations

import argparse

from rfstyle.config import load_config
from rid2026 import RID2026DatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the paired RID2026 dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    RID2026DatasetGenerator(load_config(args.config)).generate(args.output)


if __name__ == "__main__":
    main()
