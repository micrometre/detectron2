#!/usr/bin/env python3
"""
License Plate Detection using Computer Vision
Detects license plate bounding boxes using color and edge detection.
No OCR - only bounding box detection.
"""

import cv2
import argparse
from pathlib import Path
import numpy as np


def detect_plates_cv(image, min_area=500, max_area=50000, min_aspect=1.5, max_aspect=6):
    """
    Detect license plates using computer vision techniques.
    
    Args:
        image: Input image (BGR)
        min_area: Minimum plate area in pixels
        max_area: Maximum plate area in pixels
        min_aspect: Minimum width/height ratio
        max_aspect: Maximum width/height ratio
    
    Returns:
        List of detections: [(x1, y1, x2, y2, confidence), ...]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edges = cv2.Canny(bilateral, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    detections = []
    
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Filter by aspect ratio (plates are typically wide)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue
        
        # Check if contour has 4 corners (rectangular)
        if len(approx) >= 4:
            # Extract ROI for additional validation
            roi = gray[y:y+h, x:x+w]
            
            # Check for text-like patterns (high variance in pixel values)
            if roi.size > 0:
                variance = np.var(roi)
                
                # Plates typically have high contrast (text on background)
                if variance < 100:  # Too uniform, probably not a plate
                    continue
                
                # Calculate confidence based on multiple factors
                area_score = min(1.0, area / 5000)  # Normalize area
                aspect_score = min(1.0, aspect_ratio / 3.5)  # Ideal aspect ~3:1
                variance_score = min(1.0, variance / 1000)
                corner_score = min(1.0, len(approx) / 4)  # Prefer 4-corner shapes
                
                confidence = (area_score * 0.3 + aspect_score * 0.3 + 
                            variance_score * 0.2 + corner_score * 0.2)
                
                detections.append((x, y, x + w, y + h, confidence))
    
    # Non-maximum suppression to remove overlapping detections
    detections = non_max_suppression(detections, overlap_thresh=0.3)
    
    return detections


def non_max_suppression(detections, overlap_thresh=0.3):
    """
    Remove overlapping bounding boxes.
    """
    if len(detections) == 0:
        return []
    
    # Convert to numpy array
    boxes = np.array([[x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf in detections])
    
    # Sort by confidence
    idxs = np.argsort(boxes[:, 4])[::-1]
    
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / ((boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * 
                            (boxes[idxs[1:], 3] - boxes[idxs[1:], 1]))
        
        # Remove overlapping boxes
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return [detections[i] for i in keep]


def detect_plates(image_path, output_path=None, conf_threshold=0.3,
                  min_area=500, max_area=50000):
    """
    Detect license plates in an image.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing: {image_path}")
    print(f"Image shape: {image.shape}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Detect plates
    detections = detect_plates_cv(image, min_area=min_area, max_area=max_area)
    
    # Filter by confidence
    detections = [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf in detections 
                  if conf >= conf_threshold]
    
    print(f"\nDetected {len(detections)} license plate(s):")
    
    # Draw results
    output_image = image.copy()
    
    for i, (x1, y1, x2, y2, conf) in enumerate(detections, 1):
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"Plate {i} ({conf:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(output_image, (x1, y1 - text_h - 10), 
                     (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(output_image, label, (x1, y1 - 5),
                   font, font_scale, (0, 0, 0), thickness)
        
        print(f"  Plate {i}: [{x1}, {y1}, {x2}, {y2}] confidence: {conf:.1%}")
    
    # Save result
    if output_path:
        cv2.imwrite(str(output_path), output_image)
        print(f"\nResult saved to: {output_path}")
    
    return output_image, detections


def process_folder(folder_path, output_folder, conf_threshold=0.3):
    """
    Process all images in a folder.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Create output folder
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all images
    image_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in {folder_path}")
    print(f"Output folder: {output_dir}\n")
    
    total_detections = 0
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        output_path = output_dir / f"{image_file.stem}_detected{image_file.suffix}"
        
        try:
            _, detections = detect_plates(image_file, output_path, conf_threshold)
            total_detections += len(detections)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        
        print()  # Blank line between images
    
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"✓ Total plates detected: {total_detections}")
    print(f"✓ Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection (Bounding Boxes Only - No OCR)"
    )
    parser.add_argument("image", type=str, nargs='?', help="Path to input image")
    parser.add_argument("-f", "--folder", type=str, help="Process all images in a folder")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Path to save output image/folder")
    parser.add_argument("-c", "--confidence", type=float, default=0.3,
                       help="Confidence threshold (default: 0.3)")
    parser.add_argument("--min-area", type=int, default=500,
                       help="Minimum plate area in pixels (default: 500)")
    parser.add_argument("--max-area", type=int, default=50000,
                       help="Maximum plate area in pixels (default: 50000)")
    
    args = parser.parse_args()
    
    # Check if folder mode or single image mode
    if args.folder:
        # Folder processing mode
        output_folder = args.output if args.output else "plate_detections"
        process_folder(args.folder, output_folder, args.confidence)
    elif args.image:
        # Single image mode
        if args.output is None:
            input_path = Path(args.image)
            args.output = str(input_path.parent / f"{input_path.stem}_plate_detected{input_path.suffix}")
        detect_plates(args.image, args.output, args.confidence, 
                     args.min_area, args.max_area)
    else:
        parser.error("Either provide an image path or use --folder option")


if __name__ == "__main__":
    main()
