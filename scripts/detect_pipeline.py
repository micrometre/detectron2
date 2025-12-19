#!/usr/bin/env python3
"""
License Plate Detection Pipeline
Stage 1: Detectron2 detects vehicles
Stage 2: EasyOCR reads text from vehicle regions
"""

import cv2
import argparse
from pathlib import Path
import easyocr
import numpy as np

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


def setup_vehicle_detector(confidence_threshold=0.7):
    """
    Set up Detectron2 predictor for vehicle detection.
    """
    cfg = get_cfg()
    
    # Use Faster R-CNN with ResNet-50 FPN backbone
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cpu"
    
    return DefaultPredictor(cfg), cfg


def detect_vehicles(image, predictor, cfg):
    """
    Detect vehicles in the image.
    Returns list of vehicle bounding boxes and class names.
    """
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Get class names
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    
    # Filter for vehicle classes (car, truck, bus, motorcycle)
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
    vehicles = []
    
    for box, cls, score in zip(boxes, classes, scores):
        class_name = class_names[cls]
        if class_name in vehicle_classes:
            vehicles.append({
                'box': box,
                'class': class_name,
                'score': score
            })
    
    return vehicles


def read_text_from_region(image, box, reader, expand_ratio=0.1):
    """
    Read text from a specific region of the image.
    Expands the box slightly to catch plates near vehicle edges.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    h, w = image.shape[:2]
    
    # Expand box to catch plates that might be at edges
    expand_x = int((x2 - x1) * expand_ratio)
    expand_y = int((y2 - y1) * expand_ratio)
    
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)
    
    # Crop region
    region = image[y1:y2, x1:x2]
    
    if region.size == 0:
        return []
    
    # Run OCR on the region
    results = reader.readtext(region)
    
    # Adjust coordinates back to full image
    adjusted_results = []
    for bbox, text, confidence in results:
        # Adjust bbox coordinates
        adjusted_bbox = [[pt[0] + x1, pt[1] + y1] for pt in bbox]
        adjusted_results.append((adjusted_bbox, text, confidence))
    
    return adjusted_results


def process_image(image_path, output_path=None, languages=['en'], 
                  vehicle_conf=0.7, text_conf=0.5, expand_ratio=0.1):
    """
    Full pipeline: detect vehicles, then read text from vehicle regions.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Stage 1: Detect vehicles
    print("\n[Stage 1] Detecting vehicles...")
    vehicle_predictor, vehicle_cfg = setup_vehicle_detector(vehicle_conf)
    vehicles = detect_vehicles(image, vehicle_predictor, vehicle_cfg)
    print(f"Found {len(vehicles)} vehicle(s)")
    
    if len(vehicles) == 0:
        print("No vehicles detected. Skipping OCR stage.")
        if output_path:
            cv2.imwrite(str(output_path), image)
        return image, []
    
    # Stage 2: Read text from vehicle regions
    print(f"\n[Stage 2] Reading text from vehicle regions...")
    print(f"Languages: {languages}")
    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    
    all_text_results = []
    
    for i, vehicle in enumerate(vehicles, 1):
        box = vehicle['box']
        vehicle_class = vehicle['class']
        vehicle_score = vehicle['score']
        
        print(f"\n  Vehicle {i}: {vehicle_class} ({vehicle_score:.1%})")
        
        # Read text from this vehicle region
        text_results = read_text_from_region(image, box, reader, expand_ratio)
        
        # Filter by confidence
        text_results = [(bbox, text, conf) for bbox, text, conf in text_results 
                       if conf >= text_conf]
        
        if text_results:
            print(f"    Found {len(text_results)} text region(s):")
            for bbox, text, conf in text_results:
                print(f"      - '{text}' ({conf:.1%})")
                all_text_results.append({
                    'vehicle_box': box,
                    'vehicle_class': vehicle_class,
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox
                })
        else:
            print(f"    No text detected")
    
    # Draw results
    output_image = image.copy()
    
    # Draw vehicle boxes (blue)
    for vehicle in vehicles:
        box = vehicle['box']
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        label = f"{vehicle['class']} {vehicle['score']:.0%}"
        cv2.putText(output_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw text boxes (green)
    for result in all_text_results:
        bbox = result['bbox']
        text = result['text']
        conf = result['confidence']
        
        # Convert bbox to rectangle
        pts = np.array([[int(p[0]), int(p[1])] for p in bbox], np.int32)
        cv2.polylines(output_image, [pts], True, (0, 255, 0), 2)
        
        # Draw text label
        x1, y1 = pts[0]
        label = f"{text} ({conf:.0%})"
        cv2.putText(output_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save result
    if output_path:
        cv2.imwrite(str(output_path), output_image)
        print(f"\nResult saved to: {output_path}")
    
    return output_image, all_text_results


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage pipeline: Vehicle detection → Text recognition"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Path to save output image")
    parser.add_argument("-l", "--languages", type=str, nargs='+', default=['en'],
                       help="Languages for OCR (e.g., en ar fr)")
    parser.add_argument("--vehicle-conf", type=float, default=0.7,
                       help="Vehicle detection confidence threshold (0-1)")
    parser.add_argument("--text-conf", type=float, default=0.5,
                       help="Text recognition confidence threshold (0-1)")
    parser.add_argument("--expand", type=float, default=0.1,
                       help="Expand vehicle box by this ratio to catch edge plates")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.image)
        args.output = str(input_path.parent / f"{input_path.stem}_pipeline{input_path.suffix}")
    else:
        # Ensure output path has an extension
        output_path = Path(args.output)
        if not output_path.suffix:
            # No extension provided, use same as input
            input_path = Path(args.image)
            args.output = str(output_path.parent / f"{output_path.name}{input_path.suffix}")
    
    process_image(args.image, args.output, args.languages, 
                 args.vehicle_conf, args.text_conf, args.expand)


if __name__ == "__main__":
    main()
