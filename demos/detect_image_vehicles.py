#!/usr/bin/env python3
"""
License Plate Detection Pipeline
Stage 1: Detectron2 detects vehicles
"""

import cv2
import argparse
from pathlib import Path
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
    vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'}
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





def process_image(image_path, output_path=None, languages=['en'], 
                  vehicle_conf=0.7, output_only=False):
    """
    Vehicle detection only: detect vehicles and draw boxes.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    print(f"Processing: {image_path}")
    print(f"Image shape: {image.shape}")

    # Detect vehicles
    print("\n[Stage 1] Detecting vehicles...")
    vehicle_predictor, vehicle_cfg = setup_vehicle_detector(vehicle_conf)
    vehicles = detect_vehicles(image, vehicle_predictor, vehicle_cfg)
    print(f"Found {len(vehicles)} vehicle(s)")

    if len(vehicles) == 0:
        print("No vehicles detected.")
        if output_path:
            cv2.imwrite(str(output_path), image)
        return image, []

    # Draw results
    output_image = image.copy()

    # Draw vehicle boxes (blue)
    for vehicle in vehicles:
        box = vehicle['box']
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{vehicle['class']} {vehicle['score']:.0%}"
        cv2.putText(output_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result
    if output_path:
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)
        print(f"\nResult saved to: {output_path}")

    return output_image, vehicles


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle detection pipeline (Detectron2 only)"
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--vehicle-conf", type=float, default=0.7,
                       help="Vehicle detection confidence threshold (0-1)")

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="results_images",
        help="Directory to save detected frames"
    )
    args = parser.parse_args()

    # Set default output path
    if args.output_dir  is None:
        input_path = Path(args.image)
        #args.output = str(input_path.parent / f"{input_path.stem}_pipeline{input_path.suffix}")
        output_path = Path(args.output_dir) / f"{input_path.stem}_detected_vehicles{input_path.suffix}"
    else:
        # Ensure output path has an extension
        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_detected_vehicles"
        if not output_path.suffix:
            # No extension provided, use same as input
            input_path = Path(args.image)
            args.output_dir = str(output_path.parent / f"{output_path.name}{input_path.suffix}")

    process_image(args.image, args.output_dir, vehicle_conf=args.vehicle_conf)


if __name__ == "__main__":
    main()
