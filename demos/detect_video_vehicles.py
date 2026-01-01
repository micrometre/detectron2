#!/usr/bin/env python3
"""
Vehicle Detection Pipeline for Video
Detects vehicles using Detectron2
Processes video files frame by frame
"""

import cv2
import argparse
from pathlib import Path
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


def process_frame(frame, predictor, cfg, vehicles_only=False):
    """
    Process a single frame: detect vehicles.
    Returns annotated frame and detection results.
    """
    # Detect vehicles
    vehicles = detect_vehicles(frame, predictor, cfg)
    
    # If vehicles_only mode and no vehicles detected, return None
    if vehicles_only and len(vehicles) == 0:
        return None, []
    
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
    
    return output_frame, vehicles


def process_video(video_path, output_dir="images", frame_skip=1, threshold=0.6, 
                 device="cpu", save_video=False, vehicles_only=False):
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
    
    # Setup detector
    print("\nInitializing model...")
    predictor, cfg = setup_vehicle_detector(threshold, device)
    print("Model loaded successfully")
    
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
            frame, predictor, cfg, vehicles_only
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
        description="Process video files for vehicle detection using Detectron2",
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
        default="video_images",
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
        default=True,
        help="Save output as video file (in addition to frames)"
    )
    
    parser.add_argument(
        "--vehicles-only",
        action="store_true",
        help="Only save frames with vehicles detected"
    )
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.output_dir,
        args.frame_skip,
        args.threshold,
        args.device,
        args.save_video,
        args.vehicles_only
    )


if __name__ == "__main__":
    main()
