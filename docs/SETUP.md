# Fresh Setup

This guide brings a new machine from a clean clone to a ready-to-train environment.

## Requirements

- Windows PowerShell or Linux shell
- Python 3.12
- NVIDIA GPU and working NVIDIA driver for GPU training
- At least 12 GB GPU VRAM for the default YOLO smoke-test batch size, or reduce `--batch` if memory is lower
- At least 15 GB free disk space for datasets, prepared exports, checkpoints, and caches
- Internet access for Python dependencies and the first `yolov8n.pt` download

## Clone The Repo

Windows PowerShell:

```powershell
git clone <repo-url> Nuvoton_Model
cd Nuvoton_Model
```

Linux:

```bash
git clone <repo-url> Nuvoton_Model
cd Nuvoton_Model
```

## Create The Python Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs CUDA-enabled PyTorch wheels and installs the local Ultralytics fork from `ML_YOLO/yolov8_ultralytics` in editable mode.

## Verify CUDA

Windows PowerShell or Linux:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

For GPU training, the Python check should print `cuda: True` and the NVIDIA GPU name.


Next: [Dataset Setup](DATASETS.md)
