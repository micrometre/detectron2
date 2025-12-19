#!/usr/bin/env python3
"""
License Plate Detector using EasyOCR
Simple one-package solution for detection + OCR.
"""

import cv2
import argparse
from pathlib import Path
import easyocr
import os


def detect_and_read(image_path, output_path=None, languages=['en'], reader=None):
    """
    Detect text regions and read text from an image.
    EasyOCR handles both detection and recognition in one step.
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None, []
    
    print(f"Processing: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Initialize EasyOCR reader if not provided
    if reader is None:
        print(f"Languages: {languages}")
        print("Loading EasyOCR model (first run downloads models)...")
        reader = easyocr.Reader(languages, gpu=False)
    
    # Detect and read text
    results = reader.readtext(image)
    
    print(f"Detected {len(results)} text regions:")
    
    # Draw results on image
    for i, (bbox, text, confidence) in enumerate(results, 1):
        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        pts = [[int(p[0]), int(p[1])] for p in bbox]
        x1, y1 = pts[0]
        x2, y2 = pts[2]
        
        print(f"  {i}. '{text}' ({confidence:.1%}) at [{x1}, {y1}, {x2}, {y2}]")
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw text label above box
        label = f"{text} ({confidence:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        # Background for text
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), 1)
    
    # Save result
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"Result saved to: {output_path}\n")
    
    return image, results


def process_folder(folder_path, output_folder, languages=['en']):
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
    print(f"Output folder: {output_dir}")
    print(f"Languages: {languages}")
    print("Loading EasyOCR model (first run downloads models)...\n")
    
    # Initialize reader once for all images
    reader = easyocr.Reader(languages, gpu=False)
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        output_path = output_dir / f"{image_file.stem}_detected{image_file.suffix}"
        detect_and_read(image_file, output_path, languages, reader)
    
    print(f"\nProcessed {len(image_files)} images. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EasyOCR Text Detection & Recognition")
    parser.add_argument("image", type=str, nargs='?', help="Path to input image")
    parser.add_argument("-f", "--folder", type=str, help="Process all images in a folder")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Path to save output image or output folder (for --folder mode)")
    parser.add_argument("-l", "--languages", type=str, nargs='+', default=['en'],
                        help="Languages to detect (e.g., en ar fr de)")
    
    args = parser.parse_args()
    
    # Check if folder mode or single image mode
    if args.folder:
        # Folder processing mode
        output_folder = args.output if args.output else "output"
        process_folder(args.folder, output_folder, args.languages)
    elif args.image:
        # Single image mode
        if args.output is None:
            input_path = Path(args.image)
            args.output = str(input_path.parent / f"{input_path.stem}_easyocr{input_path.suffix}")
        detect_and_read(args.image, args.output, args.languages)
    else:
        parser.error("Either provide an image path or use --folder option")


if __name__ == "__main__":
    main()
