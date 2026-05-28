"""Evaluate a Nuvoton YOLO checkpoint with count-focused diagnostics.

This script is meant to answer the practical question, "Is the model bad?"
It runs inference over a dataset split, compares predicted person counts to
ground-truth label counts, and saves:

- summary.json with aggregate metrics
- per_image_counts.csv with per-image count errors
- count_scatter.png (ground truth vs predicted)
- count_error_hist.png (error distribution)
- worst_cases/*.png overlays with GT and prediction boxes together
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.evaluation import compute_bucket_metrics, compute_count_metrics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Nuvoton YOLO checkpoint and generate count-focused diagnostics."
    )
    parser.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "nuvoton_yolo" / "five_hour_run" / "weights" / "best.pt"),
        help="Path to the YOLO checkpoint to evaluate.",
    )
    parser.add_argument(
        "--data",
        default=str(ROOT / "prepared_datasets" / "nuvoton_people_v1" / "dataset.yaml"),
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=192, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for counting detections.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS during inference.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on evaluated images.")
    parser.add_argument(
        "--worst-k",
        type=int,
        default=25,
        help="Number of worst-case overlay images to save.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the evaluation report. Defaults to runs/nuvoton_yolo_eval/<run>_<split>.",
    )
    return parser.parse_args()


def resolve_dataset_paths(data_yaml_path: Path, split: str) -> tuple[Path, Path, Path]:
    data_yaml = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    dataset_root = data_yaml_path.parent
    configured_root = data_yaml.get("path")
    if configured_root:
        configured_path = Path(configured_root)
        dataset_root = (
            configured_path
            if configured_path.is_absolute()
            else (data_yaml_path.parent / configured_path).resolve()
        )

    image_dir = (dataset_root / data_yaml[split]).resolve()
    label_dir = image_dir.parent / "labels"
    if image_dir.name != "images":
        label_dir = (dataset_root / split / "labels").resolve()
    return dataset_root, image_dir, label_dir


def iter_image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def read_yolo_label_file(label_path: Path, image_width: int, image_height: int) -> tuple[int, list[list[float]]]:
    if not label_path.exists():
        return 0, []

    boxes = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        _, x_center, y_center, width, height = map(float, parts[:5])
        box_width = width * image_width
        box_height = height * image_height
        center_x = x_center * image_width
        center_y = y_center * image_height
        x1 = center_x - box_width / 2.0
        y1 = center_y - box_height / 2.0
        x2 = center_x + box_width / 2.0
        y2 = center_y + box_height / 2.0
        boxes.append([x1, y1, x2, y2])
    return len(boxes), boxes


def predict_boxes(model: YOLO, image_path: Path, *, imgsz: int, conf: float, iou: float, device: str):
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        save=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], []

    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    return boxes, scores


def save_count_scatter(gt_counts: list[int], pred_counts: list[int], output_path: Path) -> None:
    max_count = max(gt_counts + pred_counts + [1])
    plt.figure(figsize=(6, 6))
    plt.scatter(gt_counts, pred_counts, alpha=0.55, s=18)
    plt.plot([0, max_count], [0, max_count], linestyle="--")
    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("Ground Truth vs Predicted Person Count")
    plt.xlim(0, max_count)
    plt.ylim(0, max_count)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_error_hist(errors: list[int], output_path: Path) -> None:
    lower = min(errors + [0]) - 0.5
    upper = max(errors + [0]) + 0.5
    bins = int(upper - lower)
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=bins, edgecolor="black")
    plt.xlabel("Prediction Error (predicted - ground truth)")
    plt.ylabel("Images")
    plt.title("Count Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def draw_boxes(
    image_path: Path,
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    output_path: Path,
    *,
    gt_count: int,
    pred_count: int,
    score_summary: str,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)

        for x1, y1, x2, y2 in gt_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(46, 204, 113), width=2)

        for x1, y1, x2, y2 in pred_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(231, 76, 60), width=2)

        header = (
            f"{image_path.name} | gt={gt_count} pred={pred_count} "
            f"error={pred_count - gt_count:+d} | {score_summary}"
        )
        draw.rectangle([0, 0, image.width, 20], fill=(0, 0, 0))
        draw.text((4, 4), header, fill=(255, 255, 255))
        image.save(output_path)


def choose_output_dir(weights_path: Path, split: str, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    run_name = weights_path.parents[1].name if len(weights_path.parents) > 1 else weights_path.stem
    return (ROOT / "runs" / "nuvoton_yolo_eval" / f"{run_name}_{split}").resolve()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    data_yaml_path = Path(args.data).expanduser().resolve()
    output_dir = choose_output_dir(weights_path, args.split, args.output_dir)
    overlays_dir = output_dir / "worst_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, image_dir, label_dir = resolve_dataset_paths(data_yaml_path, args.split)
    image_paths = iter_image_paths(image_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in split directory: {image_dir}")

    model = YOLO(str(weights_path))

    records = []
    gt_counts: list[int] = []
    pred_counts: list[int] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size

        label_path = label_dir / f"{image_path.stem}.txt"
        gt_count, gt_boxes = read_yolo_label_file(label_path, width, height)
        pred_boxes, pred_scores = predict_boxes(
            model,
            image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )
        pred_count = len(pred_boxes)
        error = pred_count - gt_count

        record = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "gt_count": gt_count,
            "pred_count": pred_count,
            "error": error,
            "abs_error": abs(error),
            "mean_conf": round(sum(pred_scores) / len(pred_scores), 6) if pred_scores else 0.0,
            "max_conf": round(max(pred_scores), 6) if pred_scores else 0.0,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
        }
        records.append(record)
        gt_counts.append(gt_count)
        pred_counts.append(pred_count)

    count_metrics = compute_count_metrics(gt_counts, pred_counts)
    bucket_metrics = compute_bucket_metrics(gt_counts, pred_counts)

    errors = [record["error"] for record in records]
    save_count_scatter(gt_counts, pred_counts, output_dir / "count_scatter.png")
    save_error_hist(errors, output_dir / "count_error_hist.png")

    records_sorted = sorted(
        records,
        key=lambda item: (item["abs_error"], item["gt_count"], item["max_conf"]),
        reverse=True,
    )
    for index, record in enumerate(records_sorted[: args.worst_k], start=1):
        score_summary = (
            f"mean_conf={record['mean_conf']:.3f} max_conf={record['max_conf']:.3f}"
            if record["pred_scores"]
            else "no detections"
        )
        output_name = (
            f"{index:02d}_err_{record['abs_error']:02d}_"
            f"gt_{record['gt_count']:02d}_pred_{record['pred_count']:02d}_{record['image_name']}"
        )
        draw_boxes(
            Path(record["image_path"]),
            record["gt_boxes"],
            record["pred_boxes"],
            overlays_dir / output_name,
            gt_count=record["gt_count"],
            pred_count=record["pred_count"],
            score_summary=score_summary,
        )

    csv_fields = [
        "image_name",
        "image_path",
        "label_path",
        "gt_count",
        "pred_count",
        "error",
        "abs_error",
        "mean_conf",
        "max_conf",
    ]
    csv_path = output_dir / "per_image_counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for record in records_sorted:
            writer.writerow({key: record[key] for key in csv_fields})

    summary = {
        "weights": str(weights_path),
        "data_yaml": str(data_yaml_path),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "images_evaluated": len(records),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "count_metrics": count_metrics,
        "bucket_metrics": bucket_metrics,
        "worst_cases_saved": min(args.worst_k, len(records_sorted)),
        "artifacts": {
            "per_image_counts_csv": str(csv_path),
            "count_scatter_png": str((output_dir / "count_scatter.png").resolve()),
            "count_error_hist_png": str((output_dir / "count_error_hist.png").resolve()),
            "worst_cases_dir": str(overlays_dir.resolve()),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
