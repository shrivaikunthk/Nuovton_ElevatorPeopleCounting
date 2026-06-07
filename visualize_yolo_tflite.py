#!/usr/bin/env python3
"""
Visualize YOLOv8 TFLite INT8 model predictions.
Helps verify model works correctly before board deployment.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def draw_detections(image, detections, conf_threshold=0.5):
    """Draw bounding boxes on image."""
    img_draw = image.copy()
    h, w = image.shape[:2]
    
    for det in detections:
        confidence = det.get('confidence', 0)
        if confidence < conf_threshold:
            continue
        
        bbox = det['bbox']
        # Convert normalized coords to pixel coords if needed
        if all(0 <= x <= 1 for x in bbox):
            x1, y1, x2, y2 = int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h)
        else:
            x1, y1, x2, y2 = map(int, bbox)
        
        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"person {confidence:.2f}"
        cv2.putText(img_draw, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_draw


def parse_yolo_output(output_data, output_details, img_size=192):
    """Parse YOLO TFLite output to detections."""
    detections = []
    
    for i, (out_data, out_detail) in enumerate(zip(output_data, output_details)):
        # Dequantize
        if out_detail['dtype'] == np.int8:
            scale = out_detail['quantization'][0]
            zero_point = out_detail['quantization'][1]
            out_float = (out_data.astype(np.float32) - zero_point) * scale
        else:
            out_float = out_data
        
        # Parse based on shape
        # YOLOv8 typically outputs [batch, num_boxes, 5+num_classes]
        # where 5 = [x, y, w, h, confidence]
        if len(out_float.shape) == 3:
            batch_size, num_boxes, box_dim = out_float.shape
            
            for box_idx in range(num_boxes):
                box = out_float[0, box_idx]
                
                if box_dim >= 5:
                    x, y, w, h, conf = box[:5]
                    
                    # Convert center format to corner format
                    x1 = (x - w/2) / img_size
                    y1 = (y - h/2) / img_size
                    x2 = (x + w/2) / img_size
                    y2 = (y + h/2) / img_size
                    
                    cls = 0
                    if box_dim > 5:
                        cls = int(np.argmax(box[5:]))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': cls
                    })
    
    return detections


def visualize_tflite_model(model_path: Path, image_path: Path, output_path: Path, img_size: int = 192):
    """Run inference and visualize results."""
    
    print(f"Loading model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Loading image: {image_path}")
    img_orig = cv2.imread(str(image_path))
    if img_orig is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Preprocess
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    
    # Quantize
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]
    img_quantized = (img / input_scale + input_zero_point).astype(np.int8)
    input_data = np.expand_dims(img_quantized, axis=0)
    
    # Inference
    print("Running inference...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Get outputs
    outputs = []
    for out_detail in output_details:
        output_data = interpreter.get_tensor(out_detail['index'])
        outputs.append(output_data)
    
    # Parse detections
    detections = parse_yolo_output(outputs, output_details, img_size)
    print(f"Found {len(detections)} raw detections")
    
    # Filter by confidence
    conf_threshold = 0.5
    filtered_dets = [d for d in detections if d['confidence'] >= conf_threshold]
    print(f"Detections above {conf_threshold}: {len(filtered_dets)}")
    
    # Draw on resized image for visualization
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_vis = draw_detections(img_resized, filtered_dets, conf_threshold)
    
    # Save
    img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_vis_bgr)
    print(f"✓ Saved visualization to: {output_path}")
    
    return filtered_dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("windows_deployment/sd_card/yolov8_merged_int8.tflite"),
        help="TFLite model path"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_tflite_test_output.jpg"),
        help="Output visualization path"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=192
    )
    
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        return 1
    
    if not args.image.exists():
        print(f"ERROR: Image not found: {args.image}")
        return 1
    
    try:
        visualize_tflite_model(args.model, args.image, args.output, args.imgsz)
        print("\n✓ Test completed successfully")
        print(f"View results: {args.output}")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
