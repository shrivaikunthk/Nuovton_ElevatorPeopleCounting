"""Export the merged-dataset YOLOv8 and FOMO checkpoints to int8 TFLite +
Vela-compiled .tflite for the Nuvoton M55M1 (Ethos-U55-256) board.

Outputs are dropped into ``windows_deployment/sd_card/`` so they can be
copied straight onto the board's SD card / embedded into the firmware.

Usage::

    python3 scripts/export_models_for_board.py
        # picks up runs/fomo_merged/best.pt and the latest YOLOv8 run

Requires (already pinned in requirements / pip-installed on demand):
    tensorflow, onnx, onnx2tf, ultralytics, ethos-u-vela
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def collect_calib_images(images_dir: Path, n_img: int, image_size: int) -> np.ndarray:
    """Return a (N, H, W, 3) float32 RGB calibration tensor in [0, 1]."""

    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(paths) < n_img:
        raise RuntimeError(f"Only {len(paths)} images in {images_dir} (need {n_img})")
    random.seed(0)
    random.shuffle(paths)
    paths = paths[:n_img]

    stack = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        stack.append(img.astype(np.float32) / 255.0)
    return np.stack(stack, axis=0)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[run]", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


# ---------------------------------------------------------------------------
# YOLOv8 export
# ---------------------------------------------------------------------------


def export_yolov8_int8(
    weights: Path,
    image_size: int,
    calib_images_dir: Path,
    n_calib: int,
    output_dir: Path,
) -> Path:
    """Run the Nuvoton-style ONNX -> int8 TFLite export.

    Reuses the logic of ``ML_YOLO/yolov8_ultralytics/nu_export_tflite_int8.py``
    but kept inline so we can drive both formats from one script.
    """

    import onnx2tf
    from ultralytics import YOLO

    weights = weights.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[yolo] loading {weights}")
    model = YOLO(str(weights))

    print("[yolo] export -> ONNX")
    model.export(
        format="onnx",
        imgsz=image_size,
        simplify=True,
        device="cpu",
    )
    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export failed; expected {onnx_path}")

    print(f"[yolo] preparing {n_calib} calib images from {calib_images_dir}")
    calib = collect_calib_images(calib_images_dir, n_calib, image_size)
    calib_npy = output_dir / "yolo_calib.npy"
    np.save(calib_npy, calib)

    np_data = [["images", str(calib_npy), [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]]

    print("[yolo] onnx2tf -> int8 TFLite")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(weights.parent),
        not_use_onnxsim=True,
        verbosity="warn",
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",
        custom_input_op_name_np_data_path=np_data,
        input_quant_dtype="int8",
        output_quant_dtype="int8",
    )

    # onnx2tf writes <stem>_full_integer_quant.tflite into weights.parent.
    src_tflite = weights.parent / f"{weights.stem}_full_integer_quant.tflite"
    if not src_tflite.exists():
        cands = list(weights.parent.glob(f"{weights.stem}*_integer_quant.tflite"))
        if not cands:
            raise RuntimeError(f"No int8 TFLite produced under {weights.parent}")
        src_tflite = cands[0]

    dst_tflite = output_dir / "yolov8_merged_int8.tflite"
    shutil.copy2(src_tflite, dst_tflite)
    print(f"[yolo] OK -> {dst_tflite}")
    return dst_tflite


# ---------------------------------------------------------------------------
# FOMO export (PyTorch -> ONNX -> onnx2tf -> int8 TFLite)
# ---------------------------------------------------------------------------


def export_fomo_int8(
    checkpoint: Path,
    image_size: int,
    grid_size: int,
    calib_images_dir: Path,
    n_calib: int,
    output_dir: Path,
) -> Path:
    import onnx2tf
    import torch

    from elevator_counter.fomo import build_fomo_mobilenetv2

    checkpoint = checkpoint.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    image_size = int(args.get("image_size", image_size))
    grid_size = int(args.get("grid_size", grid_size))

    print(f"[fomo] loading {checkpoint} (image_size={image_size}, grid_size={grid_size})")
    model = build_fomo_mobilenetv2(
        num_classes=1, grid_size=grid_size, image_size=image_size, pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Wrap to (a) take 3-channel RGB input (so int8 path matches the firmware
    # camera tensor) and (b) apply sigmoid so the deployed graph emits
    # probabilities directly.
    # Export in native PyTorch NCHW without sigmoid (firmware post-processing
    # applies dequantize -> sigmoid -> threshold -> count).
    class FomoDeploy(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):  # x: (B, 3, H, W) float in [0,1]
            gray = x.mean(dim=1, keepdim=True)   # 3ch -> 1ch
            return self.base(gray)                 # logits (B, 1, G, G)

    deploy = FomoDeploy(model).eval()
    onnx_path = output_dir / "fomo_merged.onnx"
    example = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    print(f"[fomo] exporting ONNX (NCHW, no sigmoid) -> {onnx_path}")
    torch.onnx.export(
        deploy, example, str(onnx_path),
        input_names=["images"], output_names=["heatmap"],
        opset_version=13,
        dynamic_axes={"images": {0: "batch"}, "heatmap": {0: "batch"}},
    )

    print(f"[fomo] preparing {n_calib} calib images from {calib_images_dir}")
    calib = collect_calib_images(calib_images_dir, n_calib, image_size)
    calib_npy = output_dir / "fomo_calib.npy"
    np.save(calib_npy, calib)

    np_data = [["images", str(calib_npy), [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]]

    print("[fomo] onnx2tf -> int8 TFLite")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        not_use_onnxsim=True,
        verbosity="warn",
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",
        custom_input_op_name_np_data_path=np_data,
        input_quant_dtype="int8",
        output_quant_dtype="int8",
    )

    cands = list(output_dir.glob("fomo_merged*_full_integer_quant.tflite"))
    if not cands:
        cands = list(output_dir.glob("fomo_merged*integer_quant.tflite"))
    if not cands:
        raise RuntimeError(f"No int8 TFLite produced under {output_dir}")
    src_tflite = cands[0]

    dst_tflite = output_dir / "fomo_merged_int8.tflite"
    if src_tflite != dst_tflite:
        shutil.copy2(src_tflite, dst_tflite)
    print(f"[fomo] OK -> {dst_tflite}")
    return dst_tflite


# ---------------------------------------------------------------------------
# Vela compile
# ---------------------------------------------------------------------------


def vela_compile(tflite_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vela_ini = ROOT / "ML_YOLO/yolov8_ultralytics/vela/Tool/vela/default_vela.ini"
    if not vela_ini.exists():
        raise FileNotFoundError(f"vela ini not found: {vela_ini}")

    cmd = [
        "vela", str(tflite_path),
        "--accelerator-config=ethos-u55-256",
        "--optimise", "Size",
        "--config", str(vela_ini),
        "--memory-mode=Shared_Sram",
        "--system-config=Ethos_U55_High_End_Embedded",
        "--output-dir", str(output_dir),
    ]
    if shutil.which("vela") is None:
        cmd[0:1] = [sys.executable, "-m", "ethosu.vela"]
    print("[vela]", " ".join(cmd))
    subprocess.check_call(cmd)

    out = output_dir / f"{tflite_path.stem}_vela.tflite"
    if not out.exists():
        cands = list(output_dir.glob(f"{tflite_path.stem}*_vela.tflite"))
        if cands:
            out = cands[0]
        else:
            raise RuntimeError(f"vela did not produce expected output for {tflite_path}")
    print(f"[vela] OK -> {out}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def latest_yolo_run(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    cand = sorted(runs_root.glob("nuvoton_people_v*/weights/best.pt"))
    return cand[-1] if cand else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=192)
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--calib-images-dir", type=Path,
                        default=ROOT / "prepared_datasets/nuvoton_people_v1/train/images")
    parser.add_argument("--yolo-weights", type=Path, default=None)
    parser.add_argument("--fomo-checkpoint", type=Path,
                        default=ROOT / "runs/fomo_merged/best.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "windows_deployment/sd_card")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-fomo", action="store_true")
    parser.add_argument("--skip-vela", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.calib_images_dir.exists():
        raise SystemExit(
            f"calibration image dir missing: {args.calib_images_dir}\n"
            "Run scripts/prepare_nuvoton_yolo_dataset.py first."
        )

    artifacts: dict[str, Path] = {}

    if not args.skip_yolo:
        weights = args.yolo_weights or latest_yolo_run(ROOT / "runs/nuvoton_yolo")
        if weights is None or not weights.exists():
            print("[yolo] no checkpoint found; skipping")
        else:
            tflite = export_yolov8_int8(
                weights=weights,
                image_size=args.imgsz,
                calib_images_dir=args.calib_images_dir,
                n_calib=args.n_calib,
                output_dir=args.output_dir,
            )
            artifacts["yolov8_int8"] = tflite
            if not args.skip_vela:
                artifacts["yolov8_vela"] = vela_compile(tflite, args.output_dir / "vela")

    if not args.skip_fomo and args.fomo_checkpoint.exists():
        tflite = export_fomo_int8(
            checkpoint=args.fomo_checkpoint,
            image_size=args.imgsz,
            grid_size=6,
            calib_images_dir=args.calib_images_dir,
            n_calib=args.n_calib,
            output_dir=args.output_dir,
        )
        artifacts["fomo_int8"] = tflite
        if not args.skip_vela:
            artifacts["fomo_vela"] = vela_compile(tflite, args.output_dir / "vela")
    elif not args.skip_fomo:
        print(f"[fomo] checkpoint not found: {args.fomo_checkpoint}")

    print("\n=== artifacts ===")
    for name, path in artifacts.items():
        size_kb = path.stat().st_size / 1024.0
        print(f"  {name:14s} {path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
