# Quick Testing Guide - YOLO Model Validation

Test your YOLO model **before** deploying to the Nuvoton board to catch issues early.

## Prerequisites

```bash
pip install tensorflow opencv-python numpy
```

## Quick Test (30 seconds)

Pick any test image and run:

```bash
# Basic validation
python test_yolo_tflite.py \
    --model windows_deployment/sd_card/yolov8_merged_int8.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg
```

### ✓ Success Indicators

```
[Model Info]
Input shape: [1, 192, 192, 3]
Input dtype: <class 'numpy.int8'>
✓ Inference completed in XX.XX ms (CPU)
✓ Model test PASSED
```

### ✗ Failure Indicators

- **"ValueError: cannot reshape"** → Model export failed, re-run `compile_yolo_for_board.sh`
- **"Model not found"** → Run compilation first
- **Inference time > 500ms** → Model may be too large or not optimized

## Visual Verification (1 minute)

See what the model actually detects:

```bash
python visualize_yolo_tflite.py \
    --model windows_deployment/sd_card/yolov8_merged_int8.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg \
    --output test_output.jpg

# View the result
xdg-open test_output.jpg  # Linux
# or just open test_output.jpg in your file browser
```

### What to Look For

- ✓ Green boxes around people
- ✓ Confidence scores > 0.5
- ✓ No boxes on empty areas (false positives)
- ✗ Missing obvious people → Quantization issue
- ✗ Boxes everywhere → Model broken

## Quantization Quality Check (2 minutes)

Compare INT8 vs original PyTorch model:

```bash
python compare_yolo_quantization.py --n-images 10
```

### Acceptable Results

```
Average detection count difference: 0.5
✓ GOOD: Quantization preserved model accuracy
```

### Warning Signs

```
Average detection count difference: 3.2
✗ WARNING: Significant quantization degradation
```

**Fix**: Increase calibration images in `compile_yolo_for_board.sh` from 200 to 500

## Test Multiple Images

```bash
# Test on 5 random images
for img in prepared_datasets/nuvoton_people_v1/test/images/*.jpg | head -5; do
    echo "Testing: $img"
    python visualize_yolo_tflite.py \
        --model windows_deployment/sd_card/yolov8_merged_int8.tflite \
        --image "$img" \
        --output "test_$(basename $img)"
done
```

## Common Issues

### Issue: "No module named 'tensorflow'"

```bash
pip install tensorflow
```

### Issue: Model loads but no detections

**Cause**: Output parsing may not match your YOLO export format

**Debug**: Check output shapes in test output:
```
Output 0: shape=(1, 2100, 6)  # [batch, detections, [x,y,w,h,conf,class]]
```

Adjust parsing logic in `visualize_yolo_tflite.py` if needed.

### Issue: Detections look wrong

**Cause**: Coordinate format mismatch (normalized vs pixel, center vs corner)

**Fix**: Check the `parse_yolo_output()` function and adjust coordinate conversion.

## Ready for Board?

✓ All tests pass → **Proceed to board deployment**

✗ Tests fail → **Debug on host first** (much faster than board debugging)

## Next Steps

After successful testing:

1. Copy Vela model to SD card:
   ```bash
   cp windows_deployment/sd_card/vela/yolov8_merged_int8_vela.tflite /path/to/sdcard/model_int8_vela.tflite
   ```

2. Flash firmware (see `windows_deployment/YOLO_BOARD_DEPLOYMENT.md`)

3. Test on board with live camera

## Troubleshooting on Board

If model works in tests but fails on board:

1. **Check SD card** - File must be named exactly `model_int8_vela.tflite`
2. **Check firmware** - Ensure using YOLO firmware, not FOMO
3. **Check serial output** - Look for model loading errors
4. **Check memory** - Vela model may exceed available RAM

## Alternative: Test FOMO Model

FOMO is already validated and working:

```bash
python test_yolo_tflite.py \
    --model windows_deployment/sd_card/vela/fomo_merged_int8_vela.tflite \
    --image prepared_datasets/nuvoton_people_v1/test/images/image_001.jpg
```

FOMO is recommended if YOLO continues having issues.
