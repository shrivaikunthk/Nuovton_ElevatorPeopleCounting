#!/usr/bin/env python3
"""
Test YOLOv8 INT8 TFLite model before deploying to Nuvoton board.
Validates model loading, inference, and output shapes.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def load_test_image(image_path: Path, img_size: int = 192) -> np.ndarray:
    """Load and preprocess image for YOLO input."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1] then convert to int8 range
    img = img.astype(np.float32) / 255.0
    
    return img


def test_tflite_model(model_path: Path, image_path: Path, img_size: int = 192):
    """Test TFLite model inference."""
    
    print(f"Testing model: {model_path}")
    print(f"Test image: {image_path}")
    print("-" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n[Model Info]")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Input quantization: scale={input_details[0]['quantization'][0]}, "
          f"zero_point={input_details[0]['quantization'][1]}")
    
    print(f"\nNumber of outputs: {len(output_details)}")
    for i, out in enumerate(output_details):
        print(f"Output {i}: shape={out['shape']}, dtype={out['dtype']}, "
              f"scale={out['quantization'][0]}, zero_point={out['quantization'][1]}")
    
    # Load and preprocess image
    img = load_test_image(image_path, img_size)
    
    # Quantize input to int8
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    
    if input_details[0]['dtype'] == np.int8:
        # Quantize: q = round(f / scale) + zero_point
        img_quantized = (img / input_scale + input_zero_point).astype(np.int8)
    else:
        img_quantized = img.astype(input_details[0]['dtype'])
    
    # Add batch dimension
    input_data = np.expand_dims(img_quantized, axis=0)
    
    print(f"\n[Input Data]")
    print(f"Preprocessed shape: {input_data.shape}")
    print(f"Preprocessed dtype: {input_data.dtype}")
    print(f"Value range: [{input_data.min()}, {input_data.max()}]")
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    print("\n[Running Inference]")
    start_time = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✓ Inference completed in {inference_time:.2f} ms (CPU)")
    
    # Get outputs
    print("\n[Output Analysis]")
    outputs = []
    for i, out_detail in enumerate(output_details):
        output_data = interpreter.get_tensor(out_detail['index'])
        outputs.append(output_data)
        
        # Dequantize if needed
        if out_detail['dtype'] == np.int8:
            scale = out_detail['quantization'][0]
            zero_point = out_detail['quantization'][1]
            output_float = (output_data.astype(np.float32) - zero_point) * scale
        else:
            output_float = output_data
        
        print(f"\nOutput {i}:")
        print(f"  Shape: {output_data.shape}")
        print(f"  Quantized range: [{output_data.min()}, {output_data.max()}]")
        print(f"  Dequantized range: [{output_float.min():.4f}, {output_float.max():.4f}]")
        
        # For YOLO, typically output is [batch, num_detections, 6] or similar
        # where 6 = [x, y, w, h, confidence, class]
        if len(output_data.shape) == 3 and output_data.shape[-1] >= 5:
            # Count detections above threshold
            conf_idx = 4  # confidence is typically at index 4
            threshold = 0.5
            
            if out_detail['dtype'] == np.int8:
                # Convert threshold to quantized space
                threshold_q = int(threshold / scale + zero_point)
                num_detections = np.sum(output_data[0, :, conf_idx] > threshold_q)
            else:
                num_detections = np.sum(output_float[0, :, conf_idx] > threshold)
            
            print(f"  Detections (conf > {threshold}): {num_detections}")
    
    print("\n" + "=" * 60)
    print("✓ Model test PASSED")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If inference completed without errors, model is valid")
    print("2. Check output shapes match expected YOLO format")
    print("3. Ready to deploy to Nuvoton board")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 TFLite model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("windows_deployment/sd_card/yolov8_merged_int8.tflite"),
        help="Path to TFLite model"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to test image"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=192,
        help="Input image size"
    )
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        print("\nRun compilation first:")
        print("  bash compile_yolo_for_board.sh")
        return 1
    
    if not args.image.exists():
        print(f"ERROR: Test image not found: {args.image}")
        return 1
    
    try:
        test_tflite_model(args.model, args.image, args.imgsz)
        return 0
    except Exception as e:
        print(f"\n✗ Model test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
