"""Inspect the overhead-person dataset and verify the local loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import (  # noqa: E402
    OverheadPersonDetectionDataset,
    build_split_manifest,
    detection_collate_fn,
    load_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(ROOT / "overhead-person-detection"),
        help="Local root of the downloaded Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional split manifest path. If missing, the script will create one.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(args.dataset_root)

    manifest = load_split_manifest(manifest_path)
    split_sizes = {name: len(indices) for name, indices in manifest["splits"].items()}
    print(json.dumps({"stats": manifest["stats"], "split_sizes": split_sizes}, indent=2))

    dataset = OverheadPersonDetectionDataset(
        args.dataset_root,
        split=args.split,
        manifest_path=manifest_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=0,
    )
    images, targets = next(iter(loader))

    image_shapes = [tuple(image.shape) for image in images]
    box_counts = [int(target["boxes"].shape[0]) for target in targets]
    dtype_summary = sorted({str(image.dtype) for image in images})
    print(
        json.dumps(
            {
                "split": args.split,
                "batch_size": len(images),
                "image_shapes": image_shapes,
                "box_counts": box_counts,
                "image_dtypes": dtype_summary,
                "first_labels": targets[0]["labels"].tolist(),
                "first_boxes_shape": list(targets[0]["boxes"].shape),
                "all_grayscale": bool(all(image.shape[0] == 1 for image in images)),
                "all_finite": bool(all(torch.isfinite(image).all().item() for image in images)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
