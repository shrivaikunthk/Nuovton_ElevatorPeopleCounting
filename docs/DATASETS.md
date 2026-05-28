# Dataset Setup

The datasets are not committed to Git. Download or export them into the expected local folders before training.

## Datasets Used For Training

1. [bdanko/overhead-person-detection](https://huggingface.co/datasets/bdanko/overhead-person-detection)
2. [Passenger Counter](https://universe.roboflow.com/passenger-counter-project/passenger-counter)

Passenger Counter may require Roboflow project access to download. Training does not require Roboflow access after the exported files are placed locally.

## Download Locally

Download the datasets from the source links above and place them in the expected local folders below.

For Passenger Counter, download/export the dataset in YOLOv8 format. If Roboflow provides a ZIP, unzip it so the local folder is named `Passenger Counter.yolov8/` and contains `train/images` and `train/labels`.

## Expected Local Layout

Place both datasets in the repository root with these exact folder names:

```text
Nuvoton_Model/
  overhead-person-detection/
    data/
      train-00000-of-00001.parquet
    README.md
    splits.json                  # optional; regenerated if missing

  Passenger Counter.yolov8/
    train/
      images/
      labels/
    data.yaml
    README.roboflow.txt
```

The Passenger Counter export used during development had 3,508 images and 3,508 label files.

## Validate Dataset Placement

Windows PowerShell:

```powershell
Test-Path .\overhead-person-detection\data
Test-Path '.\Passenger Counter.yolov8\train\images'
Test-Path '.\Passenger Counter.yolov8\train\labels'
Get-ChildItem .\overhead-person-detection\data -Filter *.parquet
Get-ChildItem '.\Passenger Counter.yolov8\train\images' -File | Measure-Object
Get-ChildItem '.\Passenger Counter.yolov8\train\labels' -File | Measure-Object
```

Linux:

```bash
test -d overhead-person-detection/data
test -d "Passenger Counter.yolov8/train/images"
test -d "Passenger Counter.yolov8/train/labels"
find overhead-person-detection/data -name "*.parquet"
find "Passenger Counter.yolov8/train/images" -type f | wc -l
find "Passenger Counter.yolov8/train/labels" -type f | wc -l
```

Expected results:

- the dataset directories exist
- the overhead dataset contains at least one parquet file
- Passenger Counter image and label counts match

## Build Splits

Windows PowerShell:

```powershell
python scripts\build_splits.py --dataset-root overhead-person-detection
```

Linux:

```bash
python scripts/build_splits.py --dataset-root overhead-person-detection
```

## Prepare The Merged YOLO Dataset

Windows PowerShell:

```powershell
python scripts\prepare_nuvoton_yolo_dataset.py --force
```

Linux:

```bash
python scripts/prepare_nuvoton_yolo_dataset.py --force
```

This creates `prepared_datasets/nuvoton_people_v1/` with `dataset.yaml`, `prep_summary.json`, and train/val/test folders.

Next: [Training](TRAINING.md)
