"""Sweep classification thresholds for a trained FOMO model.

Works for both:
  * PyTorch checkpoints (``runs/.../best.pt``)
  * Keras models (``keras_fomo/runs/.../best.keras``)
  * TFLite files (``*.tflite``)

For every threshold in the sweep we compute count MAE/RMSE/bias and the
empty-image false-positive rate on the validation split, then pick the best
threshold (lowest MAE, ties broken by lower empty-FP rate) and report the
matching test metrics.

Outputs a JSON summary next to the model file (or to ``--output``) and a CSV
of the full sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(ROOT / "overhead-person-detection"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--model", required=True,
                        help="Path to .pt (PyTorch), .keras or .tflite (Keras).")
    parser.add_argument("--thresholds", default="0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nms-kernel", type=int, default=3)
    parser.add_argument("--output", default=None,
                        help="JSON path for the sweep summary. Defaults next to the model file.")
    return parser.parse_args()


def _is_pytorch(path: Path) -> bool:
    return path.suffix == ".pt"


def _eval_pytorch(model_path: Path, dataset_root: str, manifest: str | None,
                  split: str, threshold: float, batch_size: int, nms_kernel: int,
                  device: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from elevator_counter.data import (
        OverheadPersonDetectionDataset,
        build_split_manifest,
        detection_collate_fn,
    )
    from elevator_counter.fomo import build_fomo_mobilenetv2, count_from_heatmap
    from elevator_counter.training import detect_device

    manifest_path = Path(manifest).expanduser().resolve() if manifest else None
    if manifest_path is None or not manifest_path.exists():
        manifest_path = build_split_manifest(dataset_root, output_path=manifest_path)

    ckpt = torch.load(model_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    image_size = int(ckpt_args.get("image_size", 192))
    grid_size = int(ckpt_args.get("grid_size", 6))

    model = build_fomo_mobilenetv2(num_classes=1, grid_size=grid_size,
                                    image_size=image_size, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    dev = torch.device(detect_device(device))
    model.to(dev).eval()

    dataset = OverheadPersonDetectionDataset(dataset_root, split=split,
                                              manifest_path=manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=detection_collate_fn)

    total = abs_e = sq_e = signed = 0
    abs_e = sq_e = signed = 0.0
    empty = empty_fp = 0
    with torch.no_grad():
        for images, targets in loader:
            batch = torch.stack(list(images), dim=0).to(dev)
            probs = torch.sigmoid(model(batch))
            counts = count_from_heatmap(probs, threshold=threshold,
                                         nms_kernel=nms_kernel).cpu().tolist()
            for t, pred in zip(targets, counts):
                gt = int(t["boxes"].shape[0])
                err = int(pred) - gt
                total += 1
                abs_e += abs(err); sq_e += err*err; signed += err
                if gt == 0:
                    empty += 1
                    if pred > 0:
                        empty_fp += 1
    return _summarise(total, abs_e, sq_e, signed, empty, empty_fp, threshold)


def _eval_keras(model_path: Path, dataset_root: str, manifest: str | None,
                split: str, threshold: float, batch_size: int, nms_kernel: int) -> dict:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    from keras_fomo.data import build_dataset

    is_tflite = model_path.suffix == ".tflite"
    if is_tflite:
        interp = tf.lite.Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]

        def predict(batch_np):
            outs = []
            for i in range(batch_np.shape[0]):
                x = batch_np[i:i+1]
                if in_d["dtype"] == np.int8:
                    s, z = in_d["quantization"]
                    x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
                else:
                    x = x.astype(in_d["dtype"])
                interp.set_tensor(in_d["index"], x)
                interp.invoke()
                y = interp.get_tensor(out_d["index"])
                if out_d["dtype"] == np.int8:
                    s, z = out_d["quantization"]
                    y = (y.astype(np.float32) - z) * s
                outs.append(y)
            return np.concatenate(outs, axis=0)
    else:
        model = keras.models.load_model(model_path, compile=False)
        predict = lambda b: model.predict(b, verbose=0)

    # Infer image_size / grid_size from the dataset (default 192/6).
    ds = build_dataset(dataset_root, split=split, manifest_path=manifest,
                       image_size=192, grid_size=6, batch_size=batch_size,
                       shuffle=False, augment=False)

    def local_max_count(hm: np.ndarray) -> int:
        h = hm[..., 0]
        G = h.shape[0]
        pad = nms_kernel // 2
        padded = np.pad(h, pad, mode="edge")
        pooled = np.zeros_like(h)
        for y in range(G):
            for x in range(G):
                pooled[y, x] = padded[y:y+nms_kernel, x:x+nms_kernel].max()
        return int(((h == pooled) & (h >= threshold)).sum())

    total = 0
    abs_e = sq_e = signed = 0.0
    empty = empty_fp = 0
    for batch_imgs, batch_tgt in ds:
        preds = predict(batch_imgs.numpy())
        for i in range(preds.shape[0]):
            gt = int(batch_tgt[i].numpy().sum())
            pred = local_max_count(preds[i])
            err = pred - gt
            total += 1
            abs_e += abs(err); sq_e += err*err; signed += err
            if gt == 0:
                empty += 1
                if pred > 0:
                    empty_fp += 1
    return _summarise(total, abs_e, sq_e, signed, empty, empty_fp, threshold)


def _summarise(total, abs_e, sq_e, signed, empty, empty_fp, threshold):
    return {
        "threshold": threshold,
        "images": total,
        "count_mae": abs_e / max(total, 1),
        "count_rmse": math.sqrt(sq_e / max(total, 1)),
        "count_bias": signed / max(total, 1),
        "empty_false_positive_rate": empty_fp / max(empty, 1) if empty else 0.0,
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]

    is_pytorch = _is_pytorch(model_path)
    backend = "pytorch" if is_pytorch else "keras"

    val_results = []
    for t in thresholds:
        if is_pytorch:
            r = _eval_pytorch(model_path, args.dataset_root, args.manifest,
                               "val", t, args.batch_size, args.nms_kernel, args.device)
        else:
            r = _eval_keras(model_path, args.dataset_root, args.manifest,
                             "val", t, args.batch_size, args.nms_kernel)
        val_results.append(r)
        print(f"[val] thr={t:.2f}  mae={r['count_mae']:.4f}  bias={r['count_bias']:+.3f}  empty_fp={r['empty_false_positive_rate']:.3f}")

    best = min(val_results, key=lambda r: (r["count_mae"], r["empty_false_positive_rate"]))
    best_threshold = best["threshold"]
    print(f"[val] best threshold = {best_threshold:.2f}  mae={best['count_mae']:.4f}")

    if is_pytorch:
        test = _eval_pytorch(model_path, args.dataset_root, args.manifest, "test",
                              best_threshold, args.batch_size, args.nms_kernel, args.device)
    else:
        test = _eval_keras(model_path, args.dataset_root, args.manifest, "test",
                            best_threshold, args.batch_size, args.nms_kernel)

    print(f"[test] mae={test['count_mae']:.4f}  rmse={test['count_rmse']:.4f}  bias={test['count_bias']:+.3f}  empty_fp={test['empty_false_positive_rate']:.3f}")

    summary = {
        "model": str(model_path),
        "backend": backend,
        "thresholds": thresholds,
        "val_sweep": val_results,
        "best_threshold": best_threshold,
        "best_val": best,
        "test_at_best": test,
    }
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else model_path.parent / f"sweep_{model_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "count_mae", "count_rmse", "count_bias", "empty_false_positive_rate"])
        for r in val_results:
            w.writerow([r["threshold"], r["count_mae"], r["count_rmse"], r["count_bias"], r["empty_false_positive_rate"]])
    print(f"summary -> {output_path}")


if __name__ == "__main__":
    main()
