# Nuvoton YOLOv8 Training Report

## Overview
Custom 192x192 ReLU6-activated YOLOv8-nano for overhead person detection targeting edge deployment.

## Datasets

| Dataset | Source | Split | Images | Labels |
|---------|--------|-------|--------|--------|
| Overhead Person Detection | [bdanko/overhead-person-detection](https://huggingface.co/datasets/bdanko/overhead-person-detection) | train/val/test | 13,448 | 28,989 |
| Passenger Counter | [Roboflow](https://universe.roboflow.com/passenger-counter-project/passenger-counter) | train/val only | 4,906 | 14,969 |

### Merged Dataset (prepared_datasets/nuvoton_people_v1/)

| Split | Images | Composition |
|-------|--------|-------------|
| Train | 15,163 | Overhead train + Passenger train (90%) |
| Val   | 1,845  | Overhead val + Passenger val (10%) |
| Test  | 1,346  | Overhead test only (held-out) |

- **Class:** `person` (id 0)
- **Format:** Grayscale PNGs (overhead), RGB (Passenger Counter)
- **Input size:** 192x192

## Data Splitting

**Overhead:** Deterministic SHA1-based split via `scripts/build_splits.py` — 80/10/10, seed=42.

**Passenger Counter:** Deterministic hash-based assignment — 90/10 train/val, seed=42. Does not contribute to test set.

## Model Architecture

| Property | Value |
|----------|-------|
| Base | YOLOv8-nano (yolov8n.pt) |
| Config | relu6-yolov8.yaml |
| Activation | nn.ReLU6() |
| Input | 192x192 |
| Classes | 1 (person) |
| Params | ~3.16M |
| GFLOPs | ~8.9 |

## Training Parameters (BEST_EmKnight_run)

| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | **300** | Full convergence run |
| batch | **64** | Nominal batch size |
| imgsz | **192** | Fixed for Nuvoton edge target |
| device | **0** | NVIDIA GPU |
| workers | **0** | Single-GPU local training |
| optimizer | **SGD** | momentum=0.937, weight_decay=5e-4 |
| lr0 | **0.01** | Initial LR |
| lrf | **0.01** | Final LR = lr0 * lrf |
| patience | **30** | Early stopping |
| save_period | **25** | Save every 25 epochs |
| close_mosaic | **3** | Disable mosaic final 3 epochs |
| pretrained | True | Transfer from yolov8n.pt |
| seed | **0** | Random seed |
| deterministic | **True** | Deterministic mode |
| amp | **True** | Automatic Mixed Precision |
| iou | **0.7** | NMS IoU threshold |
| max_det | **300** | Max detections per image |

### Augmentations

| Aug | Value | Description |
|-----|-------|-------------|
| hsv_h | 0.015 | Hue shift |
| hsv_s | 0.7 | Saturation |
| hsv_v | 0.4 | Brightness |
| translate | 0.1 | Translation |
| scale | 0.5 | Scale +/- 50% |
| fliplr | 0.5 | Horizontal flip |
| mosaic | 1.0 | Mosaic (disabled last 3 epochs) |
| degrees | 0.0 | Rotation disabled |
| shear | 0.0 | Shear disabled |
| perspective | 0.0 | Perspective disabled |
| flipud | 0.0 | Vertical flip disabled |
| mixup | 0.0 | Mixup disabled |
| copy_paste | 0.0 | Copy-paste disabled |

### Loss Weights

| Component | Weight |
|-----------|--------|
| box | 7.5 |
| cls | 0.5 |
| dfl | 1.5 |

### Learning Rate Schedule

- **Warmup:** 3.0 epochs, warmup momentum 0.8, warmup bias LR 0.1
- **Scheduler:** Linear (default Ultralytics)
- **Final LR (epoch 300):** ~0.000133

## Training Results (Validation — Epoch 300)

| Metric | Value |
|--------|-------|
| **Precision** | **0.9494** |
| **Recall** | **0.9405** |
| **mAP@0.5** | **0.9776** |
| **mAP@0.5:0.95** | **0.6862** |

### Convergence Summary

- Model converged early and maintained high precision (~0.95) and recall (~0.94) from epoch 275–300.
- mAP@0.5 stabilized near **0.977–0.978** in the final 25 epochs.
- mAP@0.5:0.95 stabilized near **0.686** in the final epochs.
- Training losses decreased monotonically through epoch 300.
- Validation losses remained stable in the final epochs, indicating no overfitting.

### Saved Checkpoints

```
weights/
  best.pt        # Best validation metric checkpoint
  last.pt        # Final epoch 300 checkpoint
  epoch25.pt, epoch50.pt, epoch75.pt, epoch100.pt
  epoch125.pt, epoch150.pt, epoch175.pt, epoch200.pt
  epoch225.pt, epoch250.pt, epoch275.pt, epoch300.pt
```

## Training Commands

### 1-Epoch Smoke Test

```bash
repo="$(pwd)"
export MPLCONFIGDIR="$repo/.matplotlib"
export YOLO_CONFIG_DIR="$repo/.ultralytics"
export TEMP="$repo/.tmp"
export TMP="$repo/.tmp"
export TMPDIR="$repo/.tmp"
mkdir -p "$repo/.tmp"

cd ML_YOLO/yolov8_ultralytics

python dg_train.py \
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml" \
  --imgsz 192 \
  --weights yolov8n.pt \
  --epochs 1 \
  --patience 30 \
  --device 0 \
  --workers 0 \
  --save-period 1 \
  --project "$repo/runs/nuvoton_yolo" \
  --name smoke_1epoch_gpu
```

### Full 300-Epoch Run (matching BEST_EmKnight_run)

```bash
repo="$(pwd)"
export MPLCONFIGDIR="$repo/.matplotlib"
export YOLO_CONFIG_DIR="$repo/.ultralytics"
export TEMP="$repo/.tmp"
export TMP="$repo/.tmp"
export TMPDIR="$repo/.tmp"
mkdir -p "$repo/.tmp"

cd ML_YOLO/yolov8_ultralytics

python dg_train.py \
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml" \
  --imgsz 192 \
  --weights yolov8n.pt \
  --epochs 300 \
  --patience 30 \
  --device 0 \
  --workers 0 \
  --save-period 25 \
  --batch 64 \
  --project "$repo/runs/nuvoton_yolo" \
  --name nuvoton_people_v1_relu6_192_e300
```

### Evaluate After Training

```bash
# Validation set evaluation
python scripts/evaluate_nuvoton_yolo.py \
  --weights runs/nuvoton_yolo/nuvoton_people_v1_relu6_192_e300/weights/best.pt \
  --data prepared_datasets/nuvoton_people_v1/dataset.yaml \
  --split val \
  --imgsz 192 \
  --conf 0.25 \
  --device 0

# Test set evaluation
python scripts/evaluate_nuvoton_yolo.py \
  --weights runs/nuvoton_yolo/nuvoton_people_v1_relu6_192_e300/weights/best.pt \
  --data prepared_datasets/nuvoton_people_v1/dataset.yaml \
  --split test \
  --imgsz 192 \
  --conf 0.25 \
  --device 0
```

## Notes

- Use absolute `--data` path to prevent Ultralytics from resolving against global `datasets_dir`.
- Default batch=64 is safe on 24GB+ GPUs. Reduce with `--batch 16` or `--batch 8` if OOM.
- `yolov8n.pt` downloads automatically on first use.
- Runs produce: `runs/nuvoton_yolo/<name>/weights/best.pt` and `last.pt`.
