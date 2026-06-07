# YOLOv8 Deployment Guide for Nuvoton M55M1 Board

## Problem Identified

The current `yolov8_merged_int8.tflite` in the `sd_card/` directory was exported without proper hardware optimization flags, causing Vela compilation errors:
```
ValueError: cannot reshape array of size 127 into shape (1,)
```

This happens because the standard YOLOv8 export doesn't include the `separate_outputs=True` and `export_hw_optimized=True` flags required for Ethos-U NPU compatibility.

## Solution: Proper Export Workflow

### Prerequisites

1. **Python environment** with required packages:
   ```bash
   pip install ultralytics onnx2tf ethos-u-vela==3.10.0
   ```

2. **Trained YOLO weights** at:
   ```
   runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best.pt
   ```

3. **Calibration images** at:
   ```
   prepared_datasets/nuvoton_people_v1/train/images/
   ```
   (If missing, run: `python scripts/prepare_nuvoton_yolo_dataset.py`)

### Compilation Steps

#### **On Linux:**

```bash
# From the repository root
bash compile_yolo_for_board.sh
```

#### **On Windows:**

```batch
REM From the repository root
cd windows_deployment
compile_yolo_for_board.bat
```

### What the Script Does

The compilation process has 4 steps:

1. **ONNX Export** - Exports PyTorch model to ONNX with hardware optimization flags
   - Uses `separate_outputs=True` for Vela compatibility
   - Uses `export_hw_optimized=True` for NPU optimization
   
2. **Calibration Data** - Generates 200 calibration images for INT8 quantization
   - Samples from training dataset
   - Normalizes to [0, 1] range
   
3. **INT8 TFLite Conversion** - Converts ONNX to fully quantized TFLite
   - Uses `onnx2tf` with per-tensor quantization
   - Input/output dtype: int8
   
4. **Vela NPU Compilation** - Compiles for Ethos-U55-256 NPU
   - Accelerator: `ethos-u55-256`
   - Optimization: `Size` (for memory-constrained MCU)
   - Memory mode: `Shared_Sram`
   - System config: `Ethos_U55_High_End_Embedded`

### Expected Output

After successful compilation, you'll have:

```
windows_deployment/sd_card/
├── yolov8_merged_int8.tflite              # INT8 quantized model (~3 MB)
└── vela/
    └── yolov8_merged_int8_vela.tflite     # NPU-ready model (~3 MB)
```

## Testing Before Board Deployment

**IMPORTANT**: Test the compiled model on your host machine before flashing to the board.

### Test 1: Basic Model Validation

Verify the model loads and runs without errors:

```bash
# Pick any test image from your dataset
python test_yolo_tflite.py \
    --model windows_deployment/sd_card/yolov8_merged_int8.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg
```

**Expected output:**
- ✓ Model loads successfully
- ✓ Input/output shapes are correct
- ✓ Inference completes in <100ms on CPU
- ✓ Output contains detection data

### Test 2: Visual Inspection

Generate visualization to verify detections look reasonable:

```bash
python visualize_yolo_tflite.py \
    --model windows_deployment/sd_card/yolov8_merged_int8.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg \
    --output yolo_test_result.jpg
```

Open `yolo_test_result.jpg` and verify:
- ✓ People are detected with bounding boxes
- ✓ Confidence scores are reasonable (>0.5)
- ✓ No excessive false positives

### Test 3: Quantization Quality Check (Optional)

Compare INT8 vs PyTorch accuracy:

```bash
python compare_yolo_quantization.py \
    --pytorch runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best.pt \
    --tflite windows_deployment/sd_card/yolov8_merged_int8.tflite \
    --n-images 20
```

**Acceptable results:**
- Average detection count difference < 1.0 person per image
- If difference > 2.0, increase calibration images (--n-img 500)

### Test 4: Vela Model Validation

After Vela compilation, test the NPU-optimized model:

```bash
python test_yolo_tflite.py \
    --model windows_deployment/sd_card/vela/yolov8_merged_int8_vela.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg
```

**Note**: Vela models can only run on ARM NPU hardware or emulator. On x86, this test verifies the model structure is valid but won't execute NPU ops.

## Deployment to Board

### Step 1: Prepare SD Card

Copy the Vela-compiled model to your SD card:

```bash
# The firmware expects this exact filename
cp windows_deployment/sd_card/vela/yolov8_merged_int8_vela.tflite /path/to/sdcard/model_int8_vela.tflite
```

### Step 2: Flash Firmware

The YOLO firmware is located at:
```
ML_M55M1_SampleCode-master/M55M1BSP-3.01.003/SampleCode/NuEdgeWise/ObjectDetection_YOLOv8n/
```

1. Open `KEIL/ObjectDetection.uvprojx` in Keil uVision5
2. Build the project (F7)
3. Flash to the board via Nu-Link

### Step 3: Run

1. Insert SD card with `model_int8_vela.tflite`
2. Power on the board
3. Open serial terminal at 115200 baud
4. You should see detection output on LCD and serial

## Performance Expectations

Based on the merged dataset training (`runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/`):

- **mAP50**: 0.9707
- **Precision**: 0.9517
- **Recall**: 0.9204
- **Image size**: 192×192
- **Classes**: 1 (person)

Expected inference performance on M55M1:
- **NPU utilization**: ~100% (most ops run on Ethos-U55)
- **Inference time**: ~20-30 ms per frame
- **FPS**: ~30-50 FPS

## Troubleshooting

### Vela compilation fails with reshape error

**Cause**: Model exported without hardware optimization flags

**Solution**: Use the provided `compile_yolo_for_board.sh` script, which includes the correct export flags

### Model file too large for SD card

**Cause**: Using float32 or float16 model instead of INT8

**Solution**: Ensure you're using the `_full_integer_quant.tflite` output from onnx2tf

### Poor detection accuracy after quantization

**Cause**: Insufficient or non-representative calibration data

**Solution**: 
- Increase `--n-img` from 200 to 500 in calibration step
- Ensure calibration images are from the same distribution as deployment

### Firmware crashes or hangs

**Cause**: Model tensor arena size exceeds available memory

**Solution**: Check `ACTIVATION_BUF_SZ` in KEIL project settings (should be ≥ 1 MB)

## Alternative: Use FOMO Instead

If YOLO compilation continues to have issues, consider using the FOMO model which is already successfully deployed:

```bash
# FOMO is already compiled and working
ls windows_deployment/sd_card/vela/fomo_merged_int8_vela.tflite
```

FOMO advantages:
- ✓ Smaller model size (2.3 MB vs 3 MB)
- ✓ Faster inference (~19.6 ms)
- ✓ Already verified on hardware
- ✓ Better for counting applications

FOMO limitations:
- ✗ No bounding boxes (only heatmap)
- ✗ Lower spatial resolution (6×6 grid)
- ✗ Cannot distinguish overlapping people in same cell

See `docs/DEPLOYMENT_M55M1.md` for FOMO deployment details.

## References

- Nuvoton YOLO export script: `ML_YOLO/yolov8_ultralytics/nu_export_tflite_int8.py`
- Vela documentation: https://github.com/nxp-imx/ethos-u-vela
- Training results: `docs/FOMO_vs_YOLO_MERGED.md`
