# Troubleshooting

## Dataset Root Does Not Exist

```text
FileNotFoundError: Dataset root does not exist: ...\overhead-person-detection
```

Fix: download the datasets and place them in the folders documented in [Dataset Setup](DATASETS.md).

## YOLO Looks For Images In The Wrong Directory

Fix: pass an absolute `--data` path.

Windows PowerShell:

```powershell
$repo = (Get-Location).Path
--data "$repo\prepared_datasets\nuvoton_people_v1\dataset.yaml"
```

Linux:

```bash
repo="$(pwd)"
--data "$repo/prepared_datasets/nuvoton_people_v1/dataset.yaml"
```

Avoid relying on relative `../../prepared_datasets/...` paths from inside `ML_YOLO/yolov8_ultralytics`.

## CUDA Is Not Available

Windows PowerShell or Linux:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Expected for GPU training:

```text
True
```

## CUDA Out Of Memory

Add a smaller batch size to `dg_train.py`:

```bash
--batch 16
```

If needed, use `--batch 8`.

## Virtual Environment Activation Issues

PowerShell direct Python:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build_splits.py --dataset-root overhead-person-detection
```

Linux direct Python:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_splits.py --dataset-root overhead-person-detection
```

Back to: [README](../README.md)
