# FOMO Training Results - All Datasets

**Training Date:** June 7, 2026  
**Run ID:** fomo_all_20260607_102448  
**Model:** FOMO-s (small)  
**Epochs:** 100  
**Batch Size:** 16  
**Learning Rate:** 0.0003  

## Summary

Trained FOMO models on 5 datasets with automatic metrics logging.

| Dataset | Status | Best F1 | Best Epoch | Images |
|---------|--------|---------|------------|--------|
| overhead | ⚠️ Skipped | - | - | - |
| passenger | ✅ Completed | See summary.json | - | ~5,959 |
| scene-1 | ✅ Completed | See summary.json | - | ~580 |
| scene-2 | ✅ Completed | See summary.json | - | ~350 |
| scene-3 | ✅ Completed | See summary.json | - | ~607 |
| top-down | ✅ Completed | See summary.json | - | ~507 |

## Directory Structure

```
fomo_all_20260607_102448/
├── README.md                    # This file
├── training_summary.json        # Overall training summary
├── passenger_s/
│   ├── results.csv              # Per-epoch metrics
│   ├── summary.json             # Training summary
│   ├── train_config.yaml        # Training configuration
│   └── weights/
│       ├── best.pt              # Best model checkpoint
│       └── last.pt              # Last epoch checkpoint
├── scene-1_s/
│   └── ... (same structure)
├── scene-2_s/
│   └── ... (same structure)
├── scene-3_s/
│   └── ... (same structure)
└── top-down_s/
    └── ... (same structure)
```

## Files Included

For each dataset:
- ✅ `results.csv` - Per-epoch training and validation metrics
- ✅ `summary.json` - Final training summary with best metrics
- ✅ `train_config.yaml` - Complete training configuration
- ✅ `weights/best.pt` - Best model checkpoint (based on F1 score)
- ✅ `weights/last.pt` - Last epoch checkpoint

## Metrics

Each `results.csv` contains:
- epoch, train_loss, loss (validation)
- F1, precision, recall
- mean_dist, threshold
- TP, FP, FN (True/False Positives, False Negatives)
- learning_rate

Each `summary.json` contains:
- total_epochs, best_epoch, best_F1
- final_epoch metrics
- All per-epoch metrics

## Usage

### Load a Trained Model

```python
from fomo import FOMO

# Load best model for a dataset
model = FOMO.load("runs/fomo_all_20260607_102448/scene-1_s/weights/best.pt")

# Evaluate on test set
results = model.evaluate(
    data="prepared_datasets/fomo_datasets/sjsu-headcount-scene-1.yolo/data.yaml",
    split="test"
)
print(results)
```

### View Metrics

```python
import pandas as pd
import json

# Load results CSV
df = pd.read_csv("runs/fomo_all_20260607_102448/scene-1_s/results.csv")
print(df.head())

# Load summary JSON
with open("runs/fomo_all_20260607_102448/scene-1_s/summary.json") as f:
    summary = json.load(f)
print(f"Best F1: {summary['best_F1']} at epoch {summary['best_epoch']}")
```

### Deploy a Model

```python
from fomo import FOMO

# Load model
model = FOMO.load("runs/fomo_all_20260607_102448/passenger_s/weights/best.pt")

# Run inference
import cv2
image = cv2.imread("test_image.jpg")
detections = model.predict(image)
print(f"Detected {len(detections)} people")
```

## Training Command

This run was generated with:

```bash
python scripts/train_all_fomo_models.py --epochs 100
```

## Notes

- **overhead** dataset was skipped (not downloaded locally)
- All other datasets trained successfully with metrics saved
- Class index fix was applied before training (class 1 → 0)
- Intermediate checkpoints (epoch_*.pt) were removed to save space

## Related Documentation

- `FOMO_TRAINING_GUIDE.md` - Complete FOMO training guide
- `FOMO_METRICS_GUIDE.md` - Metrics and results guide
- `CLASS_INDEX_FIX.md` - Class index fix documentation
- `TRAIN_ALL_WITH_METRICS.md` - Training all datasets with metrics

## Citation

If you use these models, please cite:

```
Nuvoton People Counting with FOMO
Training Date: June 7, 2026
Model: FOMO-s
Datasets: Passenger Counter, SJSU Headcount Scenes 1-3, Top-Down People
```
