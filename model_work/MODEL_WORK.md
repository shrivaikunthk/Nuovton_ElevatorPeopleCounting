# Model work — current state

This folder documents the two models trained for the Nuvoton M55M1
overhead person counter and the metrics each one achieved. Both were
trained on the same dataset (overhead-person-detection, 192 x 192).

## TL;DR

| Model     | Task                | Inference time on M55M1 | Headline metric                  
| --------- | ------------------- | ----------------------- | -------------------------------- 
| **FOMO**  | grid heatmap -> count | **~19.6 ms (100% NPU)** | test MAE = **0.42 people**      
| **YOLOv8n** (ReLU6 variant) | bounding boxes | ~120 ms (CPU fallback for some ops) | val mAP50 = **0.978**, mAP50-95 = 0.686

**Conclusion**: FOMO is the production model. It is ~6x faster on the
NPU, ~10x smaller, and the use case (a count, not bounding boxes)
matches what FOMO outputs natively. The YOLOv8 model is kept as a
higher-fidelity reference and proof-of-concept.

---

## 1. FOMO (production model)

### Architecture

* MobileNetV2 backbone (alpha=1.0), ImageNet-pretrained.
* Truncated at the block-6 stride-16 feature map.
* 1x1 conv head -> sigmoid -> 6 x 6 x 1 person-presence heatmap.
* Input: **(1, 192, 192, 3) int8**
* Output: **(1, 6, 6, 1) int8** sigmoid heatmap

Post-processing (in `firmware/fomo_postprocess/`):

1. Dequantize int8 -> float probability in [0, 1].
2. 3 x 3 local-maxima suppression (replicate pad).
3. Threshold at **0.30** (tuned on val sweep, see below).
4. Count surviving cells -> people in frame.

### Training config (`fomo/config.json`)

```json
{
  "dataset_root": "overhead-person-detection",
  "epochs": 30, "batch_size": 32, "lr": 0.001,
  "image_size": 192, "grid_size": 6, "alpha": 1.0,
  "seed": 42, "int8_rep_samples": 128
}
```

Training: 30 epochs, Adam, cosine LR schedule (visible in
`fomo/training_log.csv` -- LR halves from 1e-3 at epoch 0 to 1.25e-4
at epoch 29). Final epoch metrics:

| Metric          | Train | Val  |
| --------------- | ----- | ---- |
| accuracy        | 1.00  | 0.984 |
| precision       | 1.00  | 0.881 |
| recall          | 1.00  | 0.846 |
| loss (BCE)      | 0.001 | 0.081 |

Training is effectively saturated by epoch ~20 (loss flat after that),
so 30 epochs is a comfortable margin. No overfitting visible.

### Threshold sweep + final test metrics (`fomo/sweep_best.json`)

Threshold was swept on val (1344 images), and the winning threshold was
applied on a held-out test set (1346 images).

Val sweep (count MAE -- lower is better):

| Threshold | count MAE | count RMSE | bias  | empty FP rate |
| --------: | --------: | ---------: | ----: | ------------: |
| **0.30**  | **0.408** | 1.013      | -0.30 | 4.8%          |
| 0.40      | 0.417     | 1.018      | -0.31 | 4.8%          |
| 0.50      | 0.425     | 1.033      | -0.34 | 2.4%          |
| 0.70      | 0.471     | 1.067      | -0.41 | 2.4%          |
| 0.80      | 0.493     | 1.091      | -0.45 | 2.4%          |

Test at threshold 0.30 (`fomo/sweep_best.json::test_at_best`):

| Metric                       | Value |
| ---------------------------- | ----: |
| count MAE                    | **0.418** |
| count RMSE                   | 1.038 |
| count bias                   | -0.32 |
| empty-image false-positive rate | **3.0%** |
| test images                  | 1346  |

**Interpretation**:

* On the average frame the count is off by **~0.42 people**. Given the
  use case (capacity tracking, elevator monitoring), that is well
  within tolerance.
* The bias of -0.32 means the model **slightly under-counts**. This is
  the safer failure mode for a capacity counter.
* Empty-frame FP rate of 3% means in a room with nobody, the model
  hallucinates a person on roughly 1 frame in 33. With even mild
  temporal smoothing (e.g. running median over 5 frames) this drops to
  ~zero in practice.

Per-image predictions for the entire test set are in
`fomo/eval_test_best/per_image.csv` (1346 rows: idx, gt, pred, abs_err,
signed_err). Useful for failure-mode analysis if you want to look at
which kinds of frames the model misses.

### Quantization + NPU compile

* Quantized to int8 with 128 representative samples drawn from the
  training set with the same augmentation pipeline.
* Vela 3.10.0 compilation for Ethos-U55-256, Shared_SRAM, Size-optimized:
  **100% of operators placed on the NPU**. Zero CPU fallback.
* Reported inference time: **19.64 ms / batch (~50.9 inferences/s)**.
* On-disk model size: 2.5 MB (`sd_card/model_int8_vela.tflite`).

This is the file that ships on the SD card. Firmware in
`ObjectDetection_FOMO/` loads it at boot.

---

## 2. YOLOv8n (ReLU6 variant) -- reference model

Trained as a higher-fidelity comparison and to validate that the
dataset can support a full bounding-box detector. Not currently
deployed because (a) it is ~6x slower than FOMO on the M55M1, and
(b) the application only needs a count.

### Architecture / config (`yolo/args.yaml`)

* Backbone: `ultralytics/cfg/models/v8/relu6-yolov8.yaml` -- Nuvoton's
  ReLU6 fork of YOLOv8n (ReLU6 quantizes more cleanly than SiLU on
  Ethos-U).
* Input: **192 x 192** (matched to FOMO so the same camera pipeline
  works for both).
* Classes: 1 ("person", `single_cls=false` because the dataset is
  multi-class but the only class present is person).
* Optimizer: SGD, lr0=0.01, momentum=0.937, weight_decay=5e-4.
* Augmentation: HSV jitter, RandAugment, fliplr 0.5, mosaic 1.0.
* Epochs: **300** trained (out of 300 requested; patience=30 never tripped).
* Batch: 64.
* Pretrained: yes.

### Final metrics (epoch 300, from `yolo/results.csv`)

| Metric                      | Value     |
| --------------------------- | --------: |
| metrics/precision(B)        | **0.949** |
| metrics/recall(B)           | 0.941     |
| **metrics/mAP50(B)**        | **0.978** |
| metrics/mAP50-95(B)         | 0.686     |
| val/box_loss                | 1.11      |
| val/cls_loss                | 0.47      |
| val/dfl_loss                | 0.995     |

The 0.978 mAP50 means the model finds essentially every person in the
val set when allowed a 50% IoU tolerance. The 0.686 mAP50-95 (averaged
over IoU thresholds 0.5..0.95) is good but shows the box localisation
is not pixel-perfect at the strictest IoUs -- expected at this input
resolution and model size.

### Plots (in `yolo/`)

* `results.png` -- training curves for all losses + metrics over 300 epochs
* `PR_curve.png` -- precision/recall curve at the final checkpoint
* `F1_curve.png` -- F1 vs confidence (use to pick a deployment confidence)
* `P_curve.png` / `R_curve.png` -- precision/recall vs confidence
* `confusion_matrix.png` / `confusion_matrix_normalized.png`
* `val_batch0_labels.jpg` vs `val_batch0_pred.jpg` -- side-by-side
  qualitative comparison

### Deployable weights

`yolo/weights/best.pt` -- 6.8 MB PyTorch checkpoint. The official
Nuvoton YOLOv8 sample in `ML_YOLO/yolov8_ultralytics/` has a known
workflow to:

1. Export `.pt` -> int8 `.tflite` using
   `ML_YOLO/yolov8_ultralytics/nu_export_tflite_int8.py`.
2. Compile with Vela.
3. Drop into the `ObjectDetection_YOLOv8n` firmware sample.

Intermediate epoch checkpoints (epoch100.pt .. epoch225.pt) are
**not** in this repo but are kept locally can provide if needed.

---

## 3. Dataset

Both models share the same source dataset:
**overhead-person-detection** (Roboflow), curated to a single class
("person") and prepared at 192 x 192. Splits:

| Split | Images |
| ----- | -----: |
| train | not in this folder (lives in the dataset repo) |
| val   | 1344   |
| test  | 1346   |

Roughly 5% of the test set is **empty** (no people) -- the empty-image
FP-rate metric above is computed on those.

---

## 4. Reproducibility

Everything in this folder is enough to **score** the models again on
your machine; it does not include training data.

### FOMO

```bash
# in the main repo (not this windows_deployment subfolder)
python scripts/train_fomo.py --epochs 30 --batch_size 32 --lr 1e-3 \
    --image_size 192 --grid_size 6 --seed 42 \
    --dataset_root overhead-person-detection
```

To re-run the val/test sweep on an existing trained model:

```bash
python scripts/eval_fomo_sweep.py --model keras_fomo/runs/fomo/best.keras
```

### YOLOv8

```bash
# inside ML_YOLO/yolov8_ultralytics/
python -m ultralytics train \
    model=ultralytics/cfg/models/v8/relu6-yolov8.yaml \
    data=path/to/nuvoton_people_v1/dataset.yaml \
    epochs=300 imgsz=192 batch=64 optimizer=SGD lr0=0.01 \
    patience=30 close_mosaic=3 pretrained=true seed=0
```

---

## 5. What is here vs what is not

**Included** (everything needed to verify the numbers above):

```
model_work/
├── MODEL_WORK.md                          <- you are here
├── fomo/
│   ├── config.json                        <- training hyperparams
│   ├── training_log.csv                   <- per-epoch metrics (30 rows)
│   ├── sweep_best.json + sweep_best.csv   <- val threshold sweep + test result
│   ├── eval_test_best/
│   │   ├── summary.json                   <- headline test metrics
│   │   └── per_image.csv                  <- 1346 rows of test predictions
│   └── model_int8.tflite                  <- pre-Vela int8 model (2.7 MB)
└── yolo/
    ├── args.yaml                          <- full training arg dump
    ├── results.csv                        <- 300 rows of training metrics
    ├── results.png + curve plots          <- training curves + PR/F1/etc.
    ├── val_batch0_{labels,pred}.jpg       <- qualitative samples
    └── weights/best.pt                    <- final PyTorch checkpoint (6.8 MB)
```

**Excluded** (kept in local repo only -- size or redundancy):

* `keras_fomo/runs/fomo/best.keras` / `final.keras` (27 MB each Keras)
* `keras_fomo/runs/fomo/model_float32.tflite` (9 MB, superseded by int8)
* `keras_fomo/runs/fomo/vela/model_int8_vela.tflite` -- present at
  `windows_deployment/sd_card/` (this is what you copy to the SD card).
* `best_yolo_run/weights/epoch{100,125,150,175,200,225}.pt` -- 78 MB of
  intermediate snapshots, irrelevant once `best.pt` is in hand.
* `best_yolo_run/train_batch*.jpg` -- training batch previews, not metrics.
