"""Export annotated image previews from the overhead-person dataset."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import build_split_manifest, load_local_dataset, load_split_manifest  # noqa: E402


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
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "preview_labels"),
        help="Directory to write annotated preview images into.",
    )
    return parser.parse_args()


def clamp_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x1 = max(0.0, min(float(width - 1), x))
    y1 = max(0.0, min(float(height - 1), y))
    x2 = max(x1 + 1.0, min(float(width), x + w))
    y2 = max(y1 + 1.0, min(float(height), y + h))
    return x1, y1, x2, y2


def draw_preview(image_bytes: bytes, boxes: list[list[float]], categories: list[int], sample_index: int) -> PILImage.Image:
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    valid_count = 0
    for box, category in zip(boxes, categories):
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        valid_count += 1
        x1, y1, x2, y2 = clamp_box(box, image.width, image.height)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
        label = f"person:{category}"
        text_y = max(0, y1 - 10)
        draw.text((x1 + 2, text_y), label, fill=(255, 255, 0), font=font)

    header = f"sample={sample_index} count={valid_count}"
    draw.rectangle((0, 0, image.width, 14), fill=(0, 0, 0))
    draw.text((3, 2), header, fill=(255, 255, 255), font=font)
    return image


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(args.dataset_root)

    manifest = load_split_manifest(manifest_path)
    split_indices = list(manifest["splits"][args.split])
    rng = random.Random(args.seed)
    rng.shuffle(split_indices)
    selected_indices = split_indices[: min(args.num_samples, len(split_indices))]

    dataset = load_local_dataset(args.dataset_root, decode_images=False)
    output_dir = Path(args.output_dir).expanduser().resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for export_index, dataset_index in enumerate(selected_indices, start=1):
        row = dataset[int(dataset_index)]
        image_payload = row["image"]
        objects = row["objects"]
        preview = draw_preview(
            image_payload["bytes"],
            objects["bbox"],
            objects["category"],
            int(dataset_index),
        )
        filename = f"{export_index:03d}_idx_{int(dataset_index):05d}_count_{sum(1 for b in objects['bbox'] if b[2] > 0 and b[3] > 0)}.png"
        destination = output_dir / filename
        preview.save(destination)
        exported.append(str(destination))

    summary = {
        "split": args.split,
        "exported": len(exported),
        "output_dir": str(output_dir),
        "files": exported,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
