#!/usr/bin/env python3
"""Prepare a merged one-class YOLO dataset for Nuvoton training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_counter.data import load_local_dataset, load_split_manifest


PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a merged one-class YOLO dataset for Nuvoton YOLOv8 training."
    )
    parser.add_argument(
        "--overhead-root",
        type=Path,
        default=Path("overhead-person-detection"),
        help="Path to the parquet-backed overhead-person dataset snapshot.",
    )
    parser.add_argument(
        "--overhead-manifest",
        type=Path,
        default=Path("overhead-person-detection/splits.json"),
        help="Path to the overhead split manifest.",
    )
    parser.add_argument(
        "--passenger-root",
        type=Path,
        default=Path("Passenger Counter.yolov8"),
        help="Path to the Roboflow YOLO export for Passenger Counter.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_datasets/nuvoton_people_v1"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--passenger-val-fraction",
        type=float,
        default=0.1,
        help="Fraction of Passenger Counter images to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for Passenger Counter train/val assignment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing output dataset root before regenerating.",
    )
    return parser.parse_args()


def ensure_clean_output_root(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --force to overwrite."
            )
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def xywh_to_yolo_line(box: list[float], image_width: int, image_height: int) -> str | None:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None

    x_center = (x + (w / 2.0)) / image_width
    y_center = (y + (h / 2.0)) / image_height
    width = w / image_width
    height = h / image_height

    if width <= 0 or height <= 0:
        return None

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return (
        f"{PERSON_CLASS_ID} "
        f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    )


def write_label_file(path: Path, lines: list[str]) -> None:
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def export_overhead_dataset(
    overhead_root: Path,
    manifest_path: Path,
    output_root: Path,
) -> dict[str, int]:
    dataset = load_local_dataset(overhead_root, decode_images=False)
    manifest = load_split_manifest(manifest_path)
    split_lookup = {}
    for split, indices in manifest["splits"].items():
        for source_index in indices:
            split_lookup[int(source_index)] = split

    counts = {"images": 0, "labels": 0, "skipped_boxes": 0}

    for source_index, row in enumerate(dataset):
        split = split_lookup[source_index]
        image_payload = row["image"]
        with PILImage.open(io.BytesIO(image_payload["bytes"])) as image:
            image = image.convert("L")
            image_width, image_height = image.size

            stem = f"overhead_{source_index:05d}"
            image_path = output_root / split / "images" / f"{stem}.png"
            label_path = output_root / split / "labels" / f"{stem}.txt"

            image.save(image_path)

        label_lines = []
        for box in row["objects"]["bbox"]:
            line = xywh_to_yolo_line(box, image_width, image_height)
            if line is None:
                counts["skipped_boxes"] += 1
                continue
            label_lines.append(line)

        write_label_file(label_path, label_lines)
        counts["images"] += 1
        counts["labels"] += len(label_lines)

    return counts


def stable_bucket(name: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def normalize_passenger_label_line(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    values = parts[1:]

    if len(values) == 4:
        x_center, y_center, width, height = values
        return f"{PERSON_CLASS_ID} {x_center} {y_center} {width} {height}"

    if len(values) >= 6 and len(values) % 2 == 0:
        coordinates = [float(value) for value in values]
        xs = coordinates[0::2]
        ys = coordinates[1::2]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        x_center = (min_x + max_x) / 2.0
        y_center = (min_y + max_y) / 2.0
        width = max_x - min_x
        height = max_y - min_y
        return (
            f"{PERSON_CLASS_ID} "
            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    raise ValueError(f"Unexpected YOLO label line: {line!r}")


def export_passenger_counter_dataset(
    passenger_root: Path,
    output_root: Path,
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, int]:
    image_root = passenger_root / "train" / "images"
    label_root = passenger_root / "train" / "labels"
    if not image_root.exists() or not label_root.exists():
        raise FileNotFoundError(
            f"Expected Passenger Counter train/images and train/labels under {passenger_root}"
        )

    counts = {"images": 0, "labels": 0, "empty_labels": 0}

    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file():
            continue

        split = "val" if stable_bucket(image_path.name, seed) < val_fraction else "train"
        stem = f"passenger_{image_path.stem}"
        output_image_path = output_root / split / "images" / f"{stem}{image_path.suffix.lower()}"
        output_label_path = output_root / split / "labels" / f"{stem}.txt"
        shutil.copy2(image_path, output_image_path)

        source_label_path = label_root / f"{image_path.stem}.txt"
        label_lines: list[str] = []
        if source_label_path.exists():
            for raw_line in source_label_path.read_text(encoding="utf-8").splitlines():
                normalized = normalize_passenger_label_line(raw_line)
                if normalized is not None:
                    label_lines.append(normalized)

        write_label_file(output_label_path, label_lines)
        counts["images"] += 1
        counts["labels"] += len(label_lines)
        if not label_lines:
            counts["empty_labels"] += 1

    return counts


def write_dataset_yaml(output_root: Path) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                # Do not write an explicit dataset root. In this Ultralytics fork,
                # relative `path:` values are resolved against the global
                # `datasets_dir` setting rather than the YAML file location.
                # Omitting `path` makes the loader use the YAML file's parent
                # directory, which stays portable across machines and workspaces.
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "",
                "nc: 1",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def write_metadata(
    output_root: Path,
    *,
    overhead_stats: dict[str, int],
    passenger_stats: dict[str, int],
    passenger_val_fraction: float,
    seed: int,
) -> Path:
    split_counts = {}
    for split in ("train", "val", "test"):
        split_counts[split] = {
            "images": len(list((output_root / split / "images").glob("*"))),
            "labels": len(list((output_root / split / "labels").glob("*.txt"))),
        }

    metadata = {
        "goal": "Nuvoton YOLOv8 one-class person detection at 192x192",
        "class_names": [PERSON_CLASS_NAME],
        "image_size": [192, 192],
        "sources": {
            "overhead_person_detection": str(Path("overhead-person-detection").resolve()),
            "passenger_counter": str(Path("Passenger Counter.yolov8").resolve()),
        },
        "notes": [
            "Passenger Counter labels normalized to a single class id 0 (person).",
            "Passenger Counter contributes to train/val only; test remains the overhead-person held-out split.",
            "Overhead images are exported from the parquet snapshot as grayscale PNGs.",
            "Use this dataset with relu6-yolov8.yaml and imgsz 192 for Nuvoton compatibility.",
        ],
        "passenger_val_fraction": passenger_val_fraction,
        "seed": seed,
        "stats": {
            "overhead_export": overhead_stats,
            "passenger_export": passenger_stats,
            "final_split_counts": split_counts,
        },
        "recommended_training": {
            "model_cfg": "ultralytics/cfg/models/v8/relu6-yolov8.yaml",
            "imgsz": 192,
            "epochs_min": 200,
        },
    }

    metadata_path = output_root / "prep_summary.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    ensure_clean_output_root(args.output_root, force=args.force)

    overhead_stats = export_overhead_dataset(
        args.overhead_root,
        args.overhead_manifest,
        args.output_root,
    )
    passenger_stats = export_passenger_counter_dataset(
        args.passenger_root,
        args.output_root,
        val_fraction=args.passenger_val_fraction,
        seed=args.seed,
    )
    dataset_yaml = write_dataset_yaml(args.output_root)
    metadata_path = write_metadata(
        args.output_root,
        overhead_stats=overhead_stats,
        passenger_stats=passenger_stats,
        passenger_val_fraction=args.passenger_val_fraction,
        seed=args.seed,
    )

    print(json.dumps(
        {
            "output_root": str(args.output_root.resolve()),
            "dataset_yaml": str(dataset_yaml.resolve()),
            "metadata": str(metadata_path.resolve()),
            "overhead_export": overhead_stats,
            "passenger_export": passenger_stats,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
