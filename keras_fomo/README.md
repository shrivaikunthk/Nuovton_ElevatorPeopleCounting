# Keras FOMO pipeline

Standalone TensorFlow / Keras implementation of a FOMO-style (Faster Objects,
More Objects) grid detector for overhead person counting. Independent from
the main repo's PyTorch code so the two paths can be benchmarked side by
side.

## Layout

- `model.py` &mdash; `build_fomo_keras(image_size, grid_size, alpha, pretrained)`
  built on `tf.keras.applications.MobileNetV2`. Grayscale inputs are tiled to
  3 channels and rescaled to `[-1, 1]` before the backbone; the head is a
  `1x1` conv with sigmoid producing a `(grid, grid, 1)` heatmap.
- `data.py` &mdash; `tf.data.Dataset` pipeline that reads the same parquet
  snapshot used by the PyTorch path and the JSON manifest written by
  `scripts/build_splits.py` at the repo root. Boxes are converted to per-cell
  presence targets on the fly. Supports a joint horizontal-flip augmentation.
- `train.py` &mdash; `model.compile/fit` with `EarlyStopping`,
  `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger`. Saves
  `best.keras`, `final.keras`, `model_float32.tflite` and (when supported)
  `model_int8.tflite` with representative-dataset calibration.
- `evaluate.py` &mdash; counts persons with thresholding + local-maxima NMS
  on the heatmap and reports MAE, RMSE, bias and the empty-image false
  positive rate. Works on both `.keras` and `.tflite` files.

## Setup (isolated env recommended)

```bash
python -m venv .venv-keras
source .venv-keras/bin/activate
pip install -r keras_fomo/requirements.txt
```

> The Keras pipeline does **not** install PyTorch and will not interfere with
> the main repo's `requirements.txt`.

## Build the split manifest (one-off, shared with PyTorch path)

```bash
python scripts/build_splits.py --dataset-root overhead-person-detection
```

## Train

```bash
python -m keras_fomo.train \
    --dataset-root overhead-person-detection \
    --epochs 30 --batch-size 32 \
    --image-size 192 --grid-size 6 \
    --output-dir keras_fomo/runs/fomo
```

## Evaluate

```bash
# Keras model
python -m keras_fomo.evaluate \
    --dataset-root overhead-person-detection \
    --model keras_fomo/runs/fomo/best.keras --split test

# TFLite model
python -m keras_fomo.evaluate \
    --dataset-root overhead-person-detection \
    --model keras_fomo/runs/fomo/model_int8.tflite --split test
```

## Counting logic

For each predicted heatmap we keep cells equal to the local maximum within a
`nms-kernel`x`nms-kernel` neighbourhood and above `--threshold`. The count is
the number of surviving cells. This avoids double-counting when a person's
centre falls near a cell boundary.

# Evaluate the .keras model on test split
python -m keras_fomo.evaluate \
    --dataset-root overhead-person-detection \
    --model keras_fomo/runs/fomo/best.keras \
    --split test \
    --threshold 0.5

# Or evaluate the INT8 TFLite model
python -m keras_fomo.evaluate \
    --dataset-root overhead-person-detection \
    --model keras_fomo/runs/fomo/model_int8.tflite \
    --split test

# Re-sweep everything that exists; this also re-runs PyTorch sweeps quickly
for model in \
    runs/fomo/best.pt \
    runs/pt_v1_lower_posweight/best.pt \
    runs/pt_v2_focal/best.pt \
    runs/pt_v3_grid12/best.pt \
    runs/pt_v4_full/best.pt \
    keras_fomo/runs/fomo/best.keras \
    keras_fomo/runs/ks_v1_jitter/best.keras \
    keras_fomo/runs/ks_v2_count_aux/best.keras \
    keras_fomo/runs/ks_v3_grid12/best.keras \
    keras_fomo/runs/ks_v4_full/best.keras; do
  [ -f "$model" ] && python scripts/sweep_threshold_fomo.py \
      --dataset-root overhead-person-detection --model "$model" || true
done

python scripts/compare_fomo_runs.py

# 60 epochs with stronger regularization
python -m keras_fomo.train \
    --dataset-root overhead-person-detection \
    --epochs 60 --batch-size 32 \
    --lr 1e-3 \
    --output-dir keras_fomo/runs/ks_60ep