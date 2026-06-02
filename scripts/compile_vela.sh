#!/usr/bin/env bash
# Compile an INT8 TFLite model with ethos-u-vela for the Nuvoton M55M1
# (Ethos-U55-256 NPU). Operators that the NPU supports are placed on the
# NPU; unsupported ops fall back to the Cortex-M55 CPU automatically.
#
# Usage:
#   bash scripts/compile_vela.sh <input_int8.tflite> [output_dir]
#
# Defaults match Nuvoton's official YOLO sample variables.bat:
#   --accelerator-config=ethos-u55-256
#   --optimise Size
#   --memory-mode=Shared_Sram
#   --system-config=Ethos_U55_High_End_Embedded
#
# The resulting <name>_vela.tflite is what gets embedded into the firmware.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input_int8.tflite> [output_dir]"
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="${2:-$(dirname "$INPUT")/vela}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: input model not found: $INPUT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Install ethos-u-vela on demand (pinned to the version Nuvoton uses).
if ! python -c "import ethosu.vela" 2>/dev/null; then
    echo "[vela] installing ethos-u-vela 3.10.0 ..."
    python -m pip install --quiet "ethos-u-vela==3.10.0"
fi

# Reuse the vela.ini that ships with the Nuvoton ML_YOLO sample so the
# memory layout matches the M55M1 firmware.
VELA_INI="$ROOT/ML_YOLO/yolov8_ultralytics/vela/Tool/vela/default_vela.ini"
if [[ ! -f "$VELA_INI" ]]; then
    echo "ERROR: vela ini not found at $VELA_INI"
    echo "       (it ships with ML_YOLO; make sure that submodule is present)"
    exit 1
fi

echo "[vela] compiling $INPUT -> $OUTPUT_DIR"
python -m ethosu.vela \
    "$INPUT" \
    --accelerator-config=ethos-u55-256 \
    --optimise Size \
    --config "$VELA_INI" \
    --memory-mode=Shared_Sram \
    --system-config=Ethos_U55_High_End_Embedded \
    --output-dir="$OUTPUT_DIR"

OUT_FILE="$OUTPUT_DIR/$(basename "${INPUT%.tflite}")_vela.tflite"
if [[ -f "$OUT_FILE" ]]; then
    SIZE=$(stat -c%s "$OUT_FILE")
    echo "[vela] OK  -> $OUT_FILE  (${SIZE} bytes)"
else
    echo "[vela] FAILED: expected output file not produced"
    exit 2
fi
