# Evaluation

This repo evaluates models with count-focused metrics, not only box-level detection quality.

## Baseline Evaluation

Windows PowerShell:

```powershell
python scripts\evaluate_baseline.py `
  --dataset-root overhead-person-detection `
  --checkpoint runs\baseline_frcnn\best.pt `
  --output runs\baseline_frcnn\eval_summary.json
```

Linux:

```bash
python scripts/evaluate_baseline.py \
  --dataset-root overhead-person-detection \
  --checkpoint runs/baseline_frcnn/best.pt \
  --output runs/baseline_frcnn/eval_summary.json
```

## Nuvoton YOLO Evaluation

Windows PowerShell:

```powershell
python scripts\evaluate_nuvoton_yolo.py `
  --weights runs\nuvoton_yolo\<run-name>\weights\best.pt `
  --data prepared_datasets\nuvoton_people_v1\dataset.yaml `
  --split val `
  --imgsz 192 `
  --conf 0.25 `
  --device 0
```

Linux:

```bash
python scripts/evaluate_nuvoton_yolo.py \
  --weights runs/nuvoton_yolo/<run-name>/weights/best.pt \
  --data prepared_datasets/nuvoton_people_v1/dataset.yaml \
  --split val \
  --imgsz 192 \
  --conf 0.25 \
  --device 0
```

Common options:

- `--split val` or `--split test`
- `--conf 0.25` changes the counting confidence threshold
- `--max-images N` evaluates only the first N images
- `--device 0` uses the first GPU; `--device cpu` runs on CPU

The YOLO evaluator writes `summary.json`, `per_image_counts.csv`, plots, and worst-case overlays.

Next: [Troubleshooting](TROUBLESHOOTING.md)
