"""Create deterministic train/val/test splits for the overhead-person dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import SplitConfig, build_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "overhead-person-detection"),
        help="Local root of the downloaded Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the split manifest JSON. Defaults to <dataset-root>/splits.json.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    manifest_path = build_split_manifest(
        args.dataset_root,
        output_path=args.output,
        config=config,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()

