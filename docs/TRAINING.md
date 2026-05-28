# Training

Run commands from the repository root unless a step explicitly changes directories.

## Nuvoton YOLO Training

The Nuvoton path uses the merged dataset from `scripts/prepare_nuvoton_yolo_dataset.py`, the local Ultralytics fork, and `relu6-yolov8.yaml` at 192x192 input size.

### 1-Epoch GPU Smoke Test

Windows PowerShell:

```powershell
$repo = (Get-Location).Path
$env:MPLCONFIGDIR = "$repo\.matplotlib"
$env:YOLO_CONFIG_DIR = "$repo\.ultralytics"
$env:TEMP = "$repo\.tmp"
$env:TMP = "$repo\.tmp"
$env:TMPDIR = "$repo\.tmp"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null

cd .\ML_YOLO\yolov8_ultralytics

python dg_train.py `
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml `
  --data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml" `
  --imgsz 192 `
  --weights yolov8n.pt `
  --epochs 1 `
  --patience 30 `
  --device 0 `
  --workers 0 `
  --save-period 1 `
  --project "$repo\runs\nuvoton_yolo" `
  --name smoke_1epoch_gpu
```

Linux:

```bash
repo="$(pwd)"
export MPLCONFIGDIR="$repo/.matplotlib"
export YOLO_CONFIG_DIR="$repo/.ultralytics"
export TMPDIR="$repo/.tmp"
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$TMPDIR"

cd ./ML_YOLO/yolov8_ultralytics

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

Expected successful smoke-test signs:

- log shows `CUDA:0` with the NVIDIA GPU name
- AMP checks pass
- train and val label caches are created
- epoch finishes and checkpoints are saved

Expected output:

```text
runs/nuvoton_yolo/smoke_1epoch_gpu/weights/best.pt
runs/nuvoton_yolo/smoke_1epoch_gpu/weights/last.pt
```

### Full Training Run

PowerShell:

```powershell
python dg_train.py `
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml `
  --data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml" `
  --imgsz 192 `
  --weights yolov8n.pt `
  --epochs 200 `
  --patience 30 `
  --device 0 `
  --workers 0 `
  --save-period 1 `
  --project "$repo\runs\nuvoton_yolo" `
  --name nuvoton_people_v1_relu6_192_e200
```

Linux:

```bash
python dg_train.py \
  --model-cfg ultralytics/cfg/models/v8/relu6-yolov8.yaml \
  --data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml" \
  --imgsz 192 \
  --weights yolov8n.pt \
  --epochs 200 \
  --patience 30 \
  --device 0 \
  --workers 0 \
  --save-period 1 \
  --project "$repo/runs/nuvoton_yolo" \
  --name nuvoton_people_v1_relu6_192_e200
```

Notes:

- use an absolute `--data` path
- `--device 0` uses the first NVIDIA GPU
- add `--batch 16` or `--batch 8` if CUDA runs out of memory
- `yolov8n.pt` downloads automatically on first use

## Baseline Training

Windows PowerShell:

```powershell
python scripts\build_splits.py --dataset-root overhead-person-detection
python scripts\inspect_dataset.py --dataset-root overhead-person-detection --split train
python scripts\train_baseline.py `
  --dataset-root overhead-person-detection `
  --output-dir runs\baseline_frcnn `
  --epochs 10 `
  --batch-size 4 `
  --image-size 192 `
  --device auto
```

Linux:

```bash
python scripts/build_splits.py --dataset-root overhead-person-detection
python scripts/inspect_dataset.py --dataset-root overhead-person-detection --split train
python scripts/train_baseline.py \
  --dataset-root overhead-person-detection \
  --output-dir runs/baseline_frcnn \
  --epochs 10 \
  --batch-size 4 \
  --image-size 192 \
  --device auto
```

Expected outputs:

- `runs/baseline_frcnn/run_config.json`
- `runs/baseline_frcnn/metrics.jsonl`
- `runs/baseline_frcnn/last.pt`
- `runs/baseline_frcnn/best.pt`

Next: [Evaluation](EVALUATION.md)
