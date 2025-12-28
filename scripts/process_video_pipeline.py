#!/usr/bin/env python3
"""
License Plate Detection Pipeline for Video
Stage 1: Detectron2 detects vehicles
Stage 2: EasyOCR reads text from vehicle regions
Processes video files frame by frame
"""

import cv2
import argparse
from pathlib import Path
import easyocr
import numpy as np
from tqdm import tqdm

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


def setup_vehicle_detector(confidence_threshold=0.7, device="cpu"):
    """
    Set up Detectron2 predictor for vehicle detection.
    """
    cfg = get_cfg()
    
    # Use Faster R-CNN with ResNet-50 FPN backbone
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = device
    
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


def process_frame(frame, predictor, cfg, reader, text_conf=0.5, expand_ratio=0.1, vehicles_only=False):
    """
    Process a single frame: detect vehicles and read text.
    Returns annotated frame and detection results.
    """
    # Stage 1: Detect vehicles
    vehicles = detect_vehicles(frame, predictor, cfg)
    
    # If vehicles_only mode and no vehicles detected, return None
    if vehicles_only and len(vehicles) == 0:
        return None, []
    
    # Stage 2: Read text from vehicle regions
    all_text_results = []
    
    for vehicle in vehicles:
        box = vehicle['box']
        vehicle_class = vehicle['class']
        vehicle_score = vehicle['score']
        
        # Read text from this vehicle region
        text_results = read_text_from_region(frame, box, reader, expand_ratio)
        
        # Filter by confidence
        text_results = [(bbox, text, conf) for bbox, text, conf in text_results 
                       if conf >= text_conf]
        
        for bbox, text, conf in text_results:
            all_text_results.append({
                'vehicle_box': box,
                'vehicle_class': vehicle_class,
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })
    
    # Draw results
    output_frame = frame.copy()
    
    # Draw vehicle boxes (blue)
    for vehicle in vehicles:
        box = vehicle['box']
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        label = f"{vehicle['class']} {vehicle['score']:.0%}"
        cv2.putText(output_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw text boxes (green)
    for result in all_text_results:
        bbox = result['bbox']
        text = result['text']
        conf = result['confidence']
        
        # Convert bbox to rectangle
        pts = np.array([[int(p[0]), int(p[1])] for p in bbox], np.int32)
        cv2.polylines(output_frame, [pts], True, (0, 255, 0), 2)
        
        # Draw text label
        x1, y1 = pts[0]
        label = f"{text} ({conf:.0%})"
        cv2.putText(output_frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return output_frame, all_text_results


def process_video(video_path, output_dir="images", frame_skip=1, threshold=0.6, 
                 device="cpu", save_video=False, model=None, size=640, nms=0.45,
                 vehicles_only=False, languages=['en'], text_conf=0.5):
    """
    Process video file frame by frame.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Frame skip: {frame_skip} (processing every {frame_skip}th frame)")
    print(f"Detection threshold: {threshold}")
    print(f"Device: {device}")
    print(f"Vehicles only mode: {vehicles_only}")
    
    # Setup detector and OCR
    print("\nInitializing models...")
    predictor, cfg = setup_vehicle_detector(threshold, device)
    reader = easyocr.Reader(languages, gpu=(device == "cuda"), verbose=False)
    print("Models loaded successfully")
    
    # Setup video writer if needed
    video_writer = None
    if save_video:
        output_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_video_path}")
    
    # Process frames
    frame_count = 0
    saved_count = 0
    
    print(f"\nProcessing frames...")
    pbar = tqdm(total=total_frames // frame_skip, desc="Processing")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Process frame
        annotated_frame, detections = process_frame(
            frame, predictor, cfg, reader, text_conf, 0.1, vehicles_only
        )
        
        # Save frame if detections found (or if not in vehicles_only mode)
        if annotated_frame is not None:
            frame_filename = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_filename), annotated_frame)
            saved_count += 1
            
            # Write to video if enabled
            if video_writer is not None:
                video_writer.write(annotated_frame)
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if video_writer is not None:
        video_writer.release()
        print(f"\nOutput video saved to: {output_video_path}")
    
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count // frame_skip}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {output_dir}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Process video files for object detection using Detectron2 + EasyOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (mp4, avi, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="images",
        help="Directory to save detected frames"
    )
    
    parser.add_argument(
        "-s", "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames, 5 = every 5th frame)"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.6,
        help="Detection confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output as video file (in addition to frames)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolox_s.pth",
        help="Path to YOLOX model weights (not used with Detectron2, kept for compatibility)"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Input size for model (not used with Detectron2, kept for compatibility)"
    )
    
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="NMS threshold (not used with Detectron2, kept for compatibility)"
    )
    
    parser.add_argument(
        "--vehicles-only",
        action="store_true",
        help="Only save frames with vehicles detected"
    )
    
    parser.add_argument(
        "-l", "--languages",
        type=str,
        nargs='+',
        default=['en'],
        help="Languages for OCR (e.g., en ar fr)"
    )
    
    parser.add_argument(
        "--text-conf",
        type=float,
        default=0.5,
        help="Text recognition confidence threshold (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.output_dir,
        args.frame_skip,
        args.threshold,
        args.device,
        args.save_video,
        args.model,
        args.size,
        args.nms,
        args.vehicles_only,
        args.languages,
        args.text_conf
    )


if __name__ == "__main__":
    main()
