"""Evaluate a trained Keras FOMO model (or its TFLite export) on a split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_fomo.data import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--model", required=True, help=".keras model file OR a .tflite file.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--nms-kernel", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def local_maxima_count(heatmap: np.ndarray, threshold: float, kernel: int) -> int:
    """heatmap: (G, G, 1) float. Returns count of local maxima >= threshold."""

    h = heatmap[..., 0]
    pad = kernel // 2
    padded = np.pad(h, pad, mode="edge")
    G = h.shape[0]
    pooled = np.zeros_like(h)
    for y in range(G):
        for x in range(G):
            pooled[y, x] = padded[y : y + kernel, x : x + kernel].max()
    keep = (h == pooled) & (h >= threshold)
    return int(keep.sum())


def predict_keras(model: keras.Model, batch: np.ndarray) -> np.ndarray:
    return model.predict(batch, verbose=0)


def predict_tflite(interpreter: "tf.lite.Interpreter", batch: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    outs = []
    for i in range(batch.shape[0]):
        x = batch[i : i + 1]
        if input_details["dtype"] == np.int8:
            scale, zero = input_details["quantization"]
            x_q = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details["index"], x_q)
        else:
            interpreter.set_tensor(input_details["index"], x.astype(input_details["dtype"]))
        interpreter.invoke()
        y = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            scale, zero = output_details["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        outs.append(y)
    return np.concatenate(outs, axis=0)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()

    is_tflite = model_path.suffix == ".tflite"
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        predict = lambda b: predict_tflite(interpreter, b)  # noqa: E731
    else:
        model = keras.models.load_model(model_path, compile=False)
        predict = lambda b: predict_keras(model, b)  # noqa: E731

    ds = build_dataset(
        args.dataset_root,
        split=args.split,
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else model_path.parent / f"eval_{args.split}_{model_path.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_image.csv"

    total = 0
    abs_err = 0.0
    sq_err = 0.0
    signed = 0.0
    empty = 0
    empty_fp = 0

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_index", "gt_count", "pred_count", "abs_error", "signed_error"])

        sample_index = 0
        for batch_images, batch_targets in ds:
            preds = predict(batch_images.numpy())
            for i in range(preds.shape[0]):
                gt = int(batch_targets[i].numpy().sum())
                pred_count = local_maxima_count(preds[i], args.threshold, args.nms_kernel)
                err = pred_count - gt
                total += 1
                abs_err += abs(err)
                sq_err += err * err
                signed += err
                if gt == 0:
                    empty += 1
                    if pred_count > 0:
                        empty_fp += 1
                writer.writerow([sample_index, gt, pred_count, abs(err), err])
                sample_index += 1

    summary = {
        "model": str(model_path),
        "split": args.split,
        "threshold": args.threshold,
        "images": total,
        "count_mae": abs_err / max(total, 1),
        "count_rmse": math.sqrt(sq_err / max(total, 1)),
        "count_bias": signed / max(total, 1),
        "empty_false_positive_rate": empty_fp / max(empty, 1) if empty else 0.0,
        "empty_images": empty,
        "csv": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
