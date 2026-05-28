"""Evaluate a trained FOMO checkpoint on val/test splits.

Reports per-image predicted vs ground-truth counts, overall MAE/RMSE/bias and
the empty-image false-positive rate. A CSV of per-image results is written and
a small grid of overlay heatmaps is saved for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import (  # noqa: E402
    OverheadPersonDetectionDataset,
    build_split_manifest,
    detection_collate_fn,
)
from elevator_counter.fomo import (  # noqa: E402
    build_fomo_mobilenetv2,
    count_from_heatmap,
)
from elevator_counter.training import detect_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(ROOT / "overhead-person-detection"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=str(ROOT / "runs" / "fomo" / "best.pt"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None, help="Defaults to <checkpoint dir>/eval_<split>.")
    parser.add_argument("--num-overlays", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def stack_images(images) -> torch.Tensor:
    return torch.stack(list(images), dim=0)


def save_overlays(samples, output_path: Path) -> None:
    """Save a small grid of heatmap overlays. Best-effort; matplotlib only."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[overlay] matplotlib unavailable ({exc}); skipping.")
        return
    if not samples:
        return
    n = len(samples)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i, (image, heatmap, gt, pred) in enumerate(samples):
        ax = axes[i // cols][i % cols]
        ax.imshow(image, cmap="gray")
        ax.imshow(
            heatmap,
            cmap="hot",
            alpha=0.45,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="nearest",
        )
        ax.set_title(f"gt={gt}, pred={pred}")
        ax.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(detect_device(args.device))

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(args.dataset_root, output_path=manifest_path)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    image_size = int(ckpt_args.get("image_size", 192))
    grid_size = int(ckpt_args.get("grid_size", 6))
    threshold = float(args.threshold) if args.threshold is not None else float(ckpt_args.get("threshold", 0.5))

    model = build_fomo_mobilenetv2(
        num_classes=int(ckpt_args.get("num_classes", 1)) if "num_classes" in ckpt_args else 1,
        grid_size=grid_size,
        image_size=image_size,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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
        num_workers=args.num_workers,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "per_image.csv"
    overlay_path = output_dir / "overlays.png"

    overlays: list = []
    rows_out: list = []

    total_images = 0
    total_abs = 0.0
    total_sq = 0.0
    total_signed = 0.0
    empty_images = 0
    empty_with_fp = 0

    with torch.no_grad(), csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "gt_count", "pred_count", "abs_error", "signed_error"])

        for batch_index, (images, targets) in enumerate(tqdm(loader, desc=f"eval {args.split}")):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = stack_images(images).to(device)
            probs = torch.sigmoid(model(batch))
            pred_counts = count_from_heatmap(probs, threshold=threshold).cpu().tolist()

            for i, t in enumerate(targets):
                gt = int(t["boxes"].shape[0])
                pred = int(pred_counts[i])
                err = pred - gt
                total_images += 1
                total_abs += abs(err)
                total_sq += err * err
                total_signed += err
                if gt == 0:
                    empty_images += 1
                    if pred > 0:
                        empty_with_fp += 1
                image_id = int(t["image_id"].item()) if "image_id" in t else batch_index * args.batch_size + i
                writer.writerow([image_id, gt, pred, abs(err), err])

                if len(overlays) < args.num_overlays:
                    img_np = images[i].squeeze(0).cpu().numpy()
                    hm = probs[i, 0].cpu().numpy()
                    overlays.append((img_np, hm, gt, pred))

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "threshold": threshold,
        "image_size": image_size,
        "grid_size": grid_size,
        "images": total_images,
        "count_mae": total_abs / max(total_images, 1),
        "count_rmse": math.sqrt(total_sq / max(total_images, 1)),
        "count_bias": total_signed / max(total_images, 1),
        "empty_false_positive_rate": empty_with_fp / max(empty_images, 1) if empty_images else 0.0,
        "empty_images": empty_images,
        "csv": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    save_overlays(overlays, overlay_path)


if __name__ == "__main__":
    main()
