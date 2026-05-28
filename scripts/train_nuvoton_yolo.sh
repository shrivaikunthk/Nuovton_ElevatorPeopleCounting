#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="${NUVOTON_YOLO_REPO:-$ROOT/ML_YOLO/yolov8_ultralytics}"
DATASET_YAML="$ROOT/prepared_datasets/nuvoton_people_v1/dataset.yaml"
MODEL_CFG="ultralytics/cfg/models/v8/relu6-yolov8.yaml"
WEIGHTS="${1:-yolov8n.pt}"
EPOCHS="${2:-200}"
IMGSZ="${3:-192}"
RUN_NAME="${4:-nuvoton_people_v1_relu6_192_e${EPOCHS}}"
PROJECT_DIR="${NUVOTON_RUNS_DIR:-$ROOT/runs/nuvoton_yolo}"
DEVICE="${NUVOTON_DEVICE:-cpu}"
WORKERS="${NUVOTON_WORKERS:-0}"
PATIENCE="${NUVOTON_PATIENCE:-30}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT/.ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
export TEMP="${TEMP:-$ROOT/.tmp}"
export TMP="${TMP:-$ROOT/.tmp}"
export TMPDIR="${TMPDIR:-$ROOT/.tmp}"
mkdir -p "$TEMP"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT/.venv/Scripts/python.exe"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python"
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing Nuvoton YOLO repo at: $REPO_ROOT" >&2
  echo "Clone OpenNuvoton/ML_YOLO and point NUVOTON_YOLO_REPO to ML_YOLO/yolov8_ultralytics." >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PYTHON" dg_train.py \
  --model-cfg "$MODEL_CFG" \
  --data "$DATASET_YAML" \
  --imgsz "$IMGSZ" \
  --weights "$WEIGHTS" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --save-period 1 \
  --project "$PROJECT_DIR" \
  --name "$RUN_NAME"
