#!/usr/bin/env python3
"""
Video License Plate Detector using EasyOCR
Processes video files with frame skipping for efficiency.
"""

import cv2
import argparse
from pathlib import Path
import easyocr
from tqdm import tqdm


def detect_and_read_frame(image, reader, languages=['en']):
    """
    Detect text regions and read text from a single frame.
    Returns annotated frame and text results.
    """
    # Detect and read text
    results = reader.readtext(image)
    
    # Draw results on image
    output_frame = image.copy()
    
    for bbox, text, confidence in results:
        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        pts = [[int(p[0]), int(p[1])] for p in bbox]
        x1, y1 = pts[0]
        x2, y2 = pts[2]
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        thickness = 2
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw text label above box
        label = f"{text} ({confidence:.0%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 2)
        
        # Background for text
        cv2.rectangle(output_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(output_frame, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), 2)
    
    return output_frame, results


def process_video(video_path, output_path=None, languages=['en'], 
                  frame_skip=5, min_confidence=0.5):
    """
    Process video file with frame skipping.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        languages: Languages for OCR
        frame_skip: Process every Nth frame (1 = every frame, 5 = every 5th frame)
        min_confidence: Minimum confidence threshold for text detection
    """
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
    print(f"Languages: {languages}")
    print("Loading EasyOCR model...\n")
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    
    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Track all detected plates
    all_detections = {}  # {text: count}
    frame_count = 0
    processed_count = 0
    
    # Process video
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                output_frame, results = detect_and_read_frame(frame, reader, languages)
                processed_count += 1
                
                # Track detections
                for bbox, text, confidence in results:
                    if confidence >= min_confidence:
                        all_detections[text] = all_detections.get(text, 0) + 1
                
                # Write processed frame
                if output_path:
                    out.write(output_frame)
            else:
                # Write original frame
                if output_path:
                    out.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    
    # Print summary
    print(f"\n✓ Processed {processed_count}/{total_frames} frames")
    
    if all_detections:
        print(f"\n📋 Detected license plates (sorted by frequency):")
        sorted_detections = sorted(all_detections.items(), key=lambda x: x[1], reverse=True)
        for text, count in sorted_detections:
            print(f"  • {text}: {count} times")
    else:
        print("\n⚠ No license plates detected")
    
    if output_path:
        print(f"\n💾 Output saved to: {output_path}")
    
    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Video License Plate Detection with EasyOCR")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("-o", "--output", type=str, default=None, 
                       help="Path to save output video")
    parser.add_argument("-l", "--languages", type=str, nargs='+', default=['en'],
                       help="Languages for OCR (e.g., en ar fr)")
    parser.add_argument("-s", "--skip", type=int, default=5,
                       help="Process every Nth frame (default: 5)")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Minimum confidence threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        input_path = Path(args.video)
        args.output = str(input_path.parent / f"{input_path.stem}_detected.mp4")
    
    process_video(args.video, args.output, args.languages, args.skip, args.confidence)


if __name__ == "__main__":
    main()
