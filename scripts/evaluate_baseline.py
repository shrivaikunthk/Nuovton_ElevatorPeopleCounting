"""Tune a count threshold on validation and report test metrics for a checkpoint."""

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
)
from elevator_counter.evaluation import (  # noqa: E402
    collect_count_predictions,
    compute_bucket_metrics,
    compute_count_metrics,
    counts_from_scores,
    select_best_threshold,
    sweep_thresholds,
)
from elevator_counter.models import build_grayscale_fasterrcnn_mobilenet  # noqa: E402
from elevator_counter.training import detect_device  # noqa: E402


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
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs" / "baseline_frcnn" / "best.pt"),
        help="Path to the checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for the evaluation summary.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def build_loader(dataset_root: str, manifest_path: Path, split: str, batch_size: int, num_workers: int):
    dataset = OverheadPersonDetectionDataset(
        dataset_root,
        split=split,
        manifest_path=manifest_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
    )


def build_thresholds(start: float, stop: float, step: float) -> list[float]:
    thresholds = []
    value = start
    while value <= stop + 1e-9:
        thresholds.append(round(value, 4))
        value += step
    return thresholds


def main() -> None:
    args = parse_args()
    device_name = detect_device(args.device)
    device = torch.device(device_name)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(
            args.dataset_root,
            output_path=manifest_path,
        )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    image_size = int(checkpoint_args.get("image_size", 192))

    model = build_grayscale_fasterrcnn_mobilenet(
        num_classes=2,
        image_size=image_size,
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_loader = build_loader(args.dataset_root, manifest_path, "val", args.batch_size, args.num_workers)
    test_loader = build_loader(args.dataset_root, manifest_path, "test", args.batch_size, args.num_workers)

    val_gt_counts, val_score_lists = collect_count_predictions(
        model,
        val_loader,
        device,
        max_batches=args.max_val_batches,
        desc="collect val",
    )
    thresholds = build_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep_results = sweep_thresholds(val_gt_counts, val_score_lists, thresholds)
    best_threshold_result = select_best_threshold(sweep_results)
    best_threshold = float(best_threshold_result["threshold"])

    test_gt_counts, test_score_lists = collect_count_predictions(
        model,
        test_loader,
        device,
        max_batches=args.max_test_batches,
        desc="collect test",
    )
    test_pred_counts = counts_from_scores(test_score_lists, best_threshold)
    test_metrics = compute_count_metrics(test_gt_counts, test_pred_counts)
    test_bucket_metrics = compute_bucket_metrics(test_gt_counts, test_pred_counts)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": device_name,
        "selected_threshold": best_threshold,
        "validation_threshold_search": {
            "best": best_threshold_result,
            "all": sweep_results,
        },
        "test": {
            "overall": test_metrics,
            "buckets": test_bucket_metrics,
        },
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
