"""tf.data input pipeline for the overhead-person-detection dataset.

Reads the same Hugging Face parquet snapshot as the PyTorch path and the same
JSON split manifest produced by `scripts/build_splits.py`. Each example is
converted to (grayscale_image, presence_heatmap) pairs.

The dataset module is intentionally self-contained: it depends on
``datasets``, ``Pillow`` and ``numpy`` only, no PyTorch.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Manifest + parquet loading
# ---------------------------------------------------------------------------


def load_manifest(dataset_root: str | Path, manifest_path: str | Path | None = None) -> dict:
    """Return the JSON manifest. Falls back to `<dataset_root>/splits.json`."""

    if manifest_path is None:
        manifest_path = Path(dataset_root) / "splits.json"
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Run scripts/build_splits.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_parquet_rows(dataset_root: str | Path):
    """Lazy import: load the HF dataset (kept undecoded so we get raw bytes)."""

    from datasets import Image as HFImage, load_dataset

    parquet_files = sorted((Path(dataset_root) / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {Path(dataset_root) / 'data'}")
    cache_dir = Path(dataset_root).parent / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "parquet",
        data_files=[str(p) for p in parquet_files],
        split="train",
        cache_dir=str(cache_dir),
    )
    return dataset.cast_column("image", HFImage(decode=False))


# ---------------------------------------------------------------------------
# Heatmap target generation
# ---------------------------------------------------------------------------


def boxes_xywh_to_heatmap(
    boxes_xywh: Iterable[Iterable[float]],
    *,
    src_width: int,
    src_height: int,
    grid_size: int,
) -> np.ndarray:
    """Build a (grid_size, grid_size, 1) heatmap from xywh boxes (source pixels)."""

    heatmap = np.zeros((grid_size, grid_size, 1), dtype=np.float32)
    for box in boxes_xywh:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / max(src_width, 1)
        cy = (y + h / 2.0) / max(src_height, 1)
        gx = int(np.clip(np.floor(cx * grid_size), 0, grid_size - 1))
        gy = int(np.clip(np.floor(cy * grid_size), 0, grid_size - 1))
        heatmap[gy, gx, 0] = 1.0
    return heatmap


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _decode_image_bytes(payload_bytes: bytes, image_size: int) -> np.ndarray:
    """Return a (H, W, 3) float32 image scaled to [-1, 1] from grayscale bytes."""

    with PILImage.open(io.BytesIO(payload_bytes)) as image:
        image = image.convert("L")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), PILImage.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    # Tile to 3 channels and scale to [-1, 1] for MobileNetV2 preprocessing.
    rgb = np.stack([arr, arr, arr], axis=-1)
    return rgb * 2.0 - 1.0


class _SplitGenerator:
    """Picklable callable that yields (image, heatmap) pairs for a split."""

    def __init__(self, dataset_root: str, indices: list[int], image_size: int, grid_size: int, augment: bool) -> None:
        self.dataset_root = dataset_root
        self.indices = indices
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment

    def __call__(self):
        rows = _load_parquet_rows(self.dataset_root)
        for source_index in self.indices:
            row = rows[int(source_index)]
            payload = row["image"]
            with PILImage.open(io.BytesIO(payload["bytes"])) as image:
                src_w, src_h = image.size
            image_arr = _decode_image_bytes(payload["bytes"], self.image_size)
            boxes_xywh = row["objects"]["bbox"] or []
            heatmap = boxes_xywh_to_heatmap(
                boxes_xywh,
                src_width=src_w,
                src_height=src_h,
                grid_size=self.grid_size,
            )
            yield image_arr, heatmap


def _augment_factory(brightness: float, contrast: float):
    """Build an augmentation function. Image tensors are in [-1, 1] (3 channels)."""

    def _augment(image: tf.Tensor, heatmap: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Horizontal flip applied jointly.
        do_flip = tf.random.uniform(()) < 0.5
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        heatmap = tf.cond(do_flip, lambda: tf.image.flip_left_right(heatmap), lambda: heatmap)

        # Brightness / contrast jitter operate in [0, 1] for stable behaviour.
        if brightness > 0 or contrast > 0:
            img01 = (image + 1.0) * 0.5
            if brightness > 0:
                img01 = tf.image.random_brightness(img01, max_delta=brightness)
            if contrast > 0:
                img01 = tf.image.random_contrast(img01, lower=max(0.0, 1 - contrast), upper=1 + contrast)
            img01 = tf.clip_by_value(img01, 0.0, 1.0)
            image = img01 * 2.0 - 1.0
        return image, heatmap

    return _augment


def build_dataset(
    dataset_root: str | Path,
    *,
    split: str,
    manifest_path: str | Path | None = None,
    image_size: int = 192,
    grid_size: int = 6,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = True,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
) -> tf.data.Dataset:
    """Return a batched, prefetching tf.data.Dataset for the requested split."""

    manifest = load_manifest(dataset_root, manifest_path)
    if split not in manifest["splits"]:
        raise ValueError(f"Unknown split: {split}")
    indices = list(manifest["splits"][split])

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(grid_size, grid_size, 1), dtype=tf.float32),
    )

    generator = _SplitGenerator(str(dataset_root), indices, image_size, grid_size, augment)
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(indices), 1024), reshuffle_each_iteration=True)
    if augment:
        aug_fn = _augment_factory(brightness_jitter, contrast_jitter)
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def split_size(dataset_root: str | Path, split: str, manifest_path: str | Path | None = None) -> int:
    return len(load_manifest(dataset_root, manifest_path)["splits"][split])
