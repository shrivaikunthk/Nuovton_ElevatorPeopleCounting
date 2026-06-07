#!/usr/bin/env bash
# Compile YOLOv8 merged model for Nuvoton M55M1 board
# This script properly exports YOLO with hardware optimization flags for Vela compatibility

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
YOLO_WEIGHTS="$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best.pt"
OUTPUT_DIR="$ROOT/windows_deployment/sd_card"
VELA_OUTPUT_DIR="$OUTPUT_DIR/vela"
IMG_SIZE=192
CALIB_DIR="$ROOT/prepared_datasets/nuvoton_people_v1/train/images"

echo "=== YOLOv8 Nuvoton Board Compilation ==="
echo "Weights: $YOLO_WEIGHTS"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if weights exist
if [[ ! -f "$YOLO_WEIGHTS" ]]; then
    echo "ERROR: YOLO weights not found at $YOLO_WEIGHTS"
    echo "Please train the model first or specify correct path"
    exit 1
fi

# Check calibration images
if [[ ! -d "$CALIB_DIR" ]]; then
    echo "ERROR: Calibration images not found at $CALIB_DIR"
    echo "Run: python scripts/prepare_nuvoton_yolo_dataset.py"
    exit 1
fi

cd "$ROOT/ML_YOLO/yolov8_ultralytics"

# Step 1: Export to ONNX with hardware optimization
echo "[1/4] Exporting YOLO to ONNX with hardware optimization flags..."
python nu_export_tflite_int8.py \
    --format onnx \
    --weights "$YOLO_WEIGHTS" \
    --imgsz $IMG_SIZE \
    --device cpu

ONNX_FILE="$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best.onnx"
if [[ ! -f "$ONNX_FILE" ]]; then
    echo "ERROR: ONNX export failed"
    exit 1
fi
echo "✓ ONNX export complete"

# Step 2: Generate calibration data
echo "[2/4] Generating calibration data..."
CALIB_NPY="$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/calib_data_${IMG_SIZE}x${IMG_SIZE}.npy"
python generate_calib_data.py \
    --img-size $IMG_SIZE $IMG_SIZE \
    --n-img 200 \
    -o "$CALIB_NPY" \
    --img-dir "$CALIB_DIR"
echo "✓ Calibration data ready"

# Step 3: Convert ONNX to INT8 TFLite
echo "[3/4] Converting ONNX to INT8 TFLite..."
python -m onnx2tf \
    -i "$ONNX_FILE" \
    -oiqt \
    -qt per-tensor \
    -cind images "$CALIB_NPY" "[[[[0,0,0]]]]" "[[[[1,1,1]]]]" \
    -o "$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights"

TFLITE_FILE="$ROOT/runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best_full_integer_quant.tflite"
if [[ ! -f "$TFLITE_FILE" ]]; then
    echo "ERROR: TFLite conversion failed"
    exit 1
fi

# Copy to sd_card directory
cp "$TFLITE_FILE" "$OUTPUT_DIR/yolov8_merged_int8.tflite"
echo "✓ INT8 TFLite model created"

# Step 4: Compile with Vela
echo "[4/4] Compiling with Vela for Ethos-U55-256..."
mkdir -p "$VELA_OUTPUT_DIR"

VELA_INI="$ROOT/ML_YOLO/yolov8_ultralytics/vela/Tool/vela/default_vela.ini"
if [[ ! -f "$VELA_INI" ]]; then
    echo "ERROR: Vela config not found at $VELA_INI"
    exit 1
fi

python -m ethosu.vela \
    "$OUTPUT_DIR/yolov8_merged_int8.tflite" \
    --accelerator-config=ethos-u55-256 \
    --optimise Size \
    --config "$VELA_INI" \
    --memory-mode=Shared_Sram \
    --system-config=Ethos_U55_High_End_Embedded \
    --output-dir="$VELA_OUTPUT_DIR"

VELA_FILE="$VELA_OUTPUT_DIR/yolov8_merged_int8_vela.tflite"
if [[ ! -f "$VELA_FILE" ]]; then
    echo "ERROR: Vela compilation failed"
    exit 1
fi

echo ""
echo "=== SUCCESS ==="
echo "✓ INT8 TFLite: $OUTPUT_DIR/yolov8_merged_int8.tflite"
echo "✓ Vela NPU-ready: $VELA_FILE"
echo ""
echo "Next steps:"
echo "1. Copy $VELA_FILE to your SD card as 'model_int8_vela.tflite'"
echo "2. Insert SD card into Nuvoton M55M1 board"
echo "3. Flash the firmware and run"
