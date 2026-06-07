#!/usr/bin/env python3
"""
Compare YOLOv8 PyTorch vs INT8 TFLite accuracy on test images.
Helps verify quantization quality before board deployment.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
from ultralytics import YOLO


def run_pytorch_inference(model_path: Path, image_path: Path, img_size: int):
    """Run inference with PyTorch YOLO model."""
    model = YOLO(str(model_path))
    results = model.predict(str(image_path), imgsz=img_size, verbose=False)
    
    boxes = results[0].boxes
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(conf),
            'class': int(cls)
        })
    
    return detections


def run_tflite_inference(model_path: Path, image_path: Path, img_size: int):
    """Run inference with TFLite INT8 model."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    
    # Quantize input
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    img_quantized = (img / input_scale + input_zero_point).astype(np.int8)
    input_data = np.expand_dims(img_quantized, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Parse outputs (format depends on YOLO export)
    # This is a simplified parser - adjust based on actual output format
    detections = []
    for out_detail in output_details:
        output_data = interpreter.get_tensor(out_detail['index'])
        
        # Dequantize
        scale = out_detail['quantization'][0]
        zero_point = out_detail['quantization'][1]
        output_float = (output_data.astype(np.float32) - zero_point) * scale
        
        # Parse detections (simplified - adjust for your model)
        if len(output_float.shape) == 3:
            for detection in output_float[0]:
                if len(detection) >= 5 and detection[4] > 0.5:  # confidence threshold
                    detections.append({
                        'bbox': detection[:4].tolist(),
                        'confidence': float(detection[4]),
                        'class': int(detection[5]) if len(detection) > 5 else 0
                    })
    
    return detections


def compare_models(pytorch_path: Path, tflite_path: Path, test_images: list, img_size: int):
    """Compare PyTorch and TFLite model outputs."""
    
    print("=" * 60)
    print("YOLOv8 Quantization Quality Check")
    print("=" * 60)
    
    results = []
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        
        # PyTorch inference
        pytorch_dets = run_pytorch_inference(pytorch_path, img_path, img_size)
        print(f"  PyTorch detections: {len(pytorch_dets)}")
        
        # TFLite inference
        tflite_dets = run_tflite_inference(tflite_path, img_path, img_size)
        print(f"  TFLite detections: {len(tflite_dets)}")
        
        # Compare
        diff = abs(len(pytorch_dets) - len(tflite_dets))
        print(f"  Detection count diff: {diff}")
        
        results.append({
            'image': img_path.name,
            'pytorch_count': len(pytorch_dets),
            'tflite_count': len(tflite_dets),
            'diff': diff
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    avg_diff = np.mean([r['diff'] for r in results])
    print(f"Average detection count difference: {avg_diff:.2f}")
    
    if avg_diff < 1.0:
        print("✓ GOOD: Quantization preserved model accuracy")
    elif avg_diff < 2.0:
        print("⚠ ACCEPTABLE: Minor quantization degradation")
    else:
        print("✗ WARNING: Significant quantization degradation")
        print("  Consider using more calibration images or per-channel quantization")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch",
        type=Path,
        default=Path("runs/nuvoton_yolo/nuvoton_people_v2_relu6_192_e200/weights/best.pt"),
        help="PyTorch model path"
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=Path("windows_deployment/sd_card/yolov8_merged_int8.tflite"),
        help="TFLite model path"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("prepared_datasets/nuvoton_people_v1/test/images"),
        help="Test images directory"
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=10,
        help="Number of test images"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=192
    )
    
    args = parser.parse_args()
    
    # Get test images
    test_images = sorted(args.test_dir.glob("*.jpg"))[:args.n_images]
    if not test_images:
        test_images = sorted(args.test_dir.glob("*.png"))[:args.n_images]
    
    if not test_images:
        print(f"ERROR: No test images found in {args.test_dir}")
        return 1
    
    compare_models(args.pytorch, args.tflite, test_images, args.imgsz)
    return 0


if __name__ == "__main__":
    exit(main())
