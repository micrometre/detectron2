#!/usr/bin/env python3
"""
Convert CV detection results to COCO annotations.
This creates initial annotations that can be manually refined.
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np


def detect_plates_cv(image, min_area=500, max_area=50000, min_aspect=1.5, max_aspect=6):
    """
    Detect license plates using computer vision techniques.
    (Copied from detect_plate_cv.py to avoid import issues)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edges = cv2.Canny(bilateral, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    detections = []
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area < min_area or area > max_area:
            continue
        
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue
        
        if len(approx) >= 4:
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                variance = np.var(roi)
                if variance < 100:
                    continue
                
                area_score = min(1.0, area / 5000)
                aspect_score = min(1.0, aspect_ratio / 3.5)
                variance_score = min(1.0, variance / 1000)
                corner_score = min(1.0, len(approx) / 4)
                
                confidence = (area_score * 0.3 + aspect_score * 0.3 + 
                            variance_score * 0.2 + corner_score * 0.2)
                
                detections.append((x, y, x + w, y + h, confidence))
    
    # Simple NMS
    if len(detections) > 1:
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = [detections[0]]
        for det in detections[1:]:
            overlap = False
            for kept in keep:
                # Check IoU
                x1 = max(det[0], kept[0])
                y1 = max(det[1], kept[1])
                x2 = min(det[2], kept[2])
                y2 = min(det[3], kept[3])
                if x2 > x1 and y2 > y1:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        detections = keep
    
    return detections


def create_annotations_from_cv(images_dir, output_dir, confidence_threshold=0.5):
    """
    Use CV-based plate detection to create initial COCO annotations.
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in images_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Detecting plates with CV method...")
    print(f"Confidence threshold: {confidence_threshold}\n")
    
    # COCO format structure
    coco_data = {
        "info": {
            "description": "Auto-generated from CV detection",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "license_plate",
                "supercategory": "object"
            }
        ]
    }
    
    annotation_id = 1
    total_plates = 0
    
    for img_id, img_file in enumerate(image_files, 1):
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"⚠ Skipping {img_file.name} (could not load)")
            continue
        
        height, width = image.shape[:2]
        
        # Detect plates
        detections = detect_plates_cv(image)
        
        # Filter by confidence
        detections = [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf in detections 
                      if conf >= confidence_threshold]
        
        if detections:
            print(f"  {img_file.name}: {len(detections)} plate(s)")
            total_plates += len(detections)
            
            # Add image info
            coco_data["images"].append({
                "id": img_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            })
            
            # Add annotations
            for x1, y1, x2, y2, conf in detections:
                w = x2 - x1
                h = y2 - y1
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [int(x1), int(y1), int(w), int(h)],
                    "area": int(w * h),
                    "iscrowd": 0,
                    "confidence": float(conf)  # Extra field for review
                })
                annotation_id += 1
    
    if total_plates == 0:
        print("\n❌ No plates detected!")
        print("   Try lowering --confidence threshold")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save annotations
    annotations_file = output_path / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ Created annotations:")
    print(f"   Images: {len(coco_data['images'])}")
    print(f"   Plates: {total_plates}")
    print(f"   Saved to: {annotations_file}")
    print(f"\n⚠ IMPORTANT: Review and correct annotations manually!")
    print(f"   Edit {annotations_file} to fix any incorrect detections")
    print(f"\nNext steps:")
    print(f"   1. Review {annotations_file}")
    print(f"   2. Copy images to {output_path}/images/")
    print(f"   3. Split into train/val if needed")
    print(f"   4. Run training")


def main():
    parser = argparse.ArgumentParser(
        description="Create COCO annotations from CV detection results"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory with images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for annotations")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    create_annotations_from_cv(args.input, args.output, args.confidence)


if __name__ == "__main__":
    main()
