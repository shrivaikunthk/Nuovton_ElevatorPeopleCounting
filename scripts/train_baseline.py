"""Train a small grayscale Faster R-CNN baseline on the overhead-person dataset."""

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
from elevator_counter.models import build_grayscale_fasterrcnn_mobilenet  # noqa: E402
from elevator_counter.training import (  # noqa: E402
    RandomHorizontalFlip,
    append_metrics,
    detect_device,
    evaluate_count_metrics,
    evaluate_detection_loss,
    seed_everything,
    train_one_epoch,
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
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "baseline_frcnn"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--val-loss-every",
        type=int,
        default=1,
        help="Run validation loss every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--count-val-every",
        type=int,
        default=2,
        help="Run count-based validation every N epochs. Use 0 to disable.",
    )
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace, manifest_path: Path):
    train_dataset = OverheadPersonDetectionDataset(
        args.dataset_root,
        split="train",
        manifest_path=manifest_path,
        transform=RandomHorizontalFlip(0.5),
    )
    val_dataset = OverheadPersonDetectionDataset(
        args.dataset_root,
        split="val",
        manifest_path=manifest_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device_name = detect_device(args.device)
    device = torch.device(device_name)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(
            args.dataset_root,
            output_path=manifest_path,
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "device": device_name,
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_loader, val_loader = build_dataloaders(args, manifest_path)

    model = build_grayscale_fasterrcnn_mobilenet(
        num_classes=2,
        image_size=args.image_size,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_mae = float("inf")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_batches=args.max_train_batches,
            epoch=epoch,
        )
        run_val_loss = args.val_loss_every > 0 and epoch % args.val_loss_every == 0
        run_val_count = args.count_val_every > 0 and epoch % args.count_val_every == 0

        val_loss_metrics = {"loss": None, "batches": 0}
        if run_val_loss:
            val_loss_metrics = evaluate_detection_loss(
                model,
                val_loader,
                device,
                max_batches=args.max_val_batches,
                epoch=epoch,
            )

        val_count_metrics = {
            "images": 0,
            "count_mae": None,
            "count_bias": None,
            "empty_false_positive_rate": None,
        }
        if run_val_count:
            val_count_metrics = evaluate_count_metrics(
                model,
                val_loader,
                device,
                score_threshold=args.score_threshold,
                max_batches=args.max_val_batches,
                epoch=epoch,
            )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_batches": train_metrics["batches"],
            "val_loss": val_loss_metrics["loss"],
            "val_batches": val_loss_metrics["batches"],
            "ran_val_loss": run_val_loss,
            "ran_val_count": run_val_count,
            **val_count_metrics,
        }
        append_metrics(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, epoch_metrics, args)
        if epoch_metrics["count_mae"] is not None and epoch_metrics["count_mae"] < best_val_mae:
            best_val_mae = epoch_metrics["count_mae"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, epoch_metrics, args)
        elif (
            epoch_metrics["count_mae"] is None
            and epoch_metrics["val_loss"] is not None
            and epoch_metrics["val_loss"] < best_val_loss
        ):
            best_val_loss = epoch_metrics["val_loss"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, epoch_metrics, args)


if __name__ == "__main__":
    main()
