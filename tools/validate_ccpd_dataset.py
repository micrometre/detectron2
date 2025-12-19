#!/usr/bin/env python3
"""
Validate CCPD (Chinese City Parking Dataset) before training.

CCPD uses filename-based annotations. This script validates:
- Image files exist and are readable
- Bounding box coordinates are valid
- Dataset statistics
- Corrupted images

Filename format:
{area}-{tilt}_{brightness}-{x1}&{y1}_{x2}&{y2}-{x2}&{y2}_{x3}&{y3}_{x1}&{y1}_{x4}&{y4}-{plate_chars}-{brightness_level}-{blur_level}.jpg

Example:
00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from collections import defaultdict
import sys


class CCPDValidator:
    """Validator for CCPD dataset."""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.stats = {
            'train': {'total': 0, 'valid': 0, 'corrupted': 0, 'invalid_bbox': 0},
            'val': {'total': 0, 'valid': 0, 'corrupted': 0, 'invalid_bbox': 0},
            'test': {'total': 0, 'valid': 0, 'corrupted': 0, 'invalid_bbox': 0}
        }
        self.bbox_sizes = []
        self.image_sizes = []
        self.errors = []
        
    def parse_ccpd_filename(self, filename: str) -> Optional[Dict]:
        """
        Parse CCPD filename to extract annotations.
        
        Format: {area}-{tilt}_{brightness}-{x1}&{y1}_{x2}&{y2}-{corners}-{plate}-{brightness}-{blur}.jpg
        
        Returns:
            Dictionary with parsed data or None if parsing fails
        """
        try:
            # Remove .jpg extension
            name = filename.replace('.jpg', '')
            
            # Split by '-'
            parts = name.split('-')
            
            if len(parts) < 7:
                return None
            
            # Extract tilt and brightness
            tilt_brightness = parts[1].split('_')
            if len(tilt_brightness) != 2:
                return None
            
            tilt = int(tilt_brightness[0])
            brightness = int(tilt_brightness[1])
            
            # Extract top-left and bottom-right corners (simplified bbox)
            bbox_parts = parts[2].split('_')
            if len(bbox_parts) != 2:
                return None
            
            x1, y1 = map(int, bbox_parts[0].split('&'))
            x2, y2 = map(int, bbox_parts[1].split('&'))
            
            # Extract all four corners
            corners_str = parts[3]
            corners = []
            for corner in corners_str.split('_'):
                if '&' in corner:
                    x, y = map(int, corner.split('&'))
                    corners.append((x, y))
            
            # Extract plate characters (encoded as numbers)
            plate_chars = parts[4].split('_')
            
            # Extract brightness and blur levels
            brightness_level = int(parts[5]) if len(parts) > 5 else 0
            blur_level = int(parts[6]) if len(parts) > 6 else 0
            
            return {
                'tilt': tilt,
                'brightness': brightness,
                'bbox': (x1, y1, x2, y2),
                'corners': corners,
                'plate_chars': plate_chars,
                'brightness_level': brightness_level,
                'blur_level': blur_level
            }
            
        except (ValueError, IndexError) as e:
            return None
    
    def validate_image(self, image_path: Path, split: str) -> bool:
        """
        Validate a single image file.
        
        Returns:
            True if image is valid, False otherwise
        """
        self.stats[split]['total'] += 1
        
        # Check if file exists
        if not image_path.exists():
            self.errors.append(f"Missing file: {image_path}")
            self.stats[split]['corrupted'] += 1
            return False
        
        # Parse filename
        annotation = self.parse_ccpd_filename(image_path.name)
        if annotation is None:
            self.errors.append(f"Invalid filename format: {image_path.name}")
            self.stats[split]['invalid_bbox'] += 1
            return False
        
        # Try to read image
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                self.errors.append(f"Corrupted image: {image_path}")
                self.stats[split]['corrupted'] += 1
                return False
            
            height, width = img.shape[:2]
            self.image_sizes.append((width, height))
            
            # Validate bounding box
            x1, y1, x2, y2 = annotation['bbox']
            
            # Check if bbox is within image bounds
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                self.errors.append(f"Bbox out of bounds: {image_path.name} - bbox({x1},{y1},{x2},{y2}) img({width},{height})")
                self.stats[split]['invalid_bbox'] += 1
                return False
            
            # Check if bbox has valid dimensions
            if x2 <= x1 or y2 <= y1:
                self.errors.append(f"Invalid bbox dimensions: {image_path.name} - ({x1},{y1},{x2},{y2})")
                self.stats[split]['invalid_bbox'] += 1
                return False
            
            # Store bbox size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            self.bbox_sizes.append((bbox_width, bbox_height))
            
            self.stats[split]['valid'] += 1
            return True
            
        except Exception as e:
            self.errors.append(f"Error reading {image_path}: {str(e)}")
            self.stats[split]['corrupted'] += 1
            return False
    
    def validate_split(self, split: str) -> bool:
        """Validate all images in a dataset split."""
        split_dir = self.dataset_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  {split.upper()} split not found: {split_dir}")
            return True  # Not an error if split doesn't exist
        
        print(f"\n{'='*60}")
        print(f"Validating {split.upper()} split...")
        print(f"{'='*60}")
        
        # Get all jpg files
        image_files = list(split_dir.glob('*.jpg'))
        
        if not image_files:
            print(f"⚠️  No images found in {split_dir}")
            return True
        
        print(f"Found {len(image_files)} images")
        
        # Validate each image
        valid_count = 0
        for i, img_path in enumerate(image_files):
            if self.validate_image(img_path, split):
                valid_count += 1
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images...")
        
        # Print results
        print(f"\n✓ Validation complete:")
        print(f"  Total images: {self.stats[split]['total']}")
        print(f"  Valid images: {self.stats[split]['valid']}")
        print(f"  Corrupted images: {self.stats[split]['corrupted']}")
        print(f"  Invalid bboxes: {self.stats[split]['invalid_bbox']}")
        
        return self.stats[split]['corrupted'] == 0 and self.stats[split]['invalid_bbox'] == 0
    
    def print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        
        # Total counts
        total_images = sum(s['total'] for s in self.stats.values())
        total_valid = sum(s['valid'] for s in self.stats.values())
        total_corrupted = sum(s['corrupted'] for s in self.stats.values())
        total_invalid_bbox = sum(s['invalid_bbox'] for s in self.stats.values())
        
        print(f"\nTotal images: {total_images}")
        print(f"  Train: {self.stats['train']['total']}")
        print(f"  Val:   {self.stats['val']['total']}")
        print(f"  Test:  {self.stats['test']['total']}")
        
        print(f"\nValid images: {total_valid} ({100*total_valid/total_images:.1f}%)")
        print(f"Corrupted images: {total_corrupted}")
        print(f"Invalid bboxes: {total_invalid_bbox}")
        
        # Image size statistics
        if self.image_sizes:
            widths = [w for w, h in self.image_sizes]
            heights = [h for w, h in self.image_sizes]
            
            print(f"\nImage dimensions:")
            print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.0f}")
            print(f"  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.0f}")
        
        # Bbox size statistics
        if self.bbox_sizes:
            bbox_widths = [w for w, h in self.bbox_sizes]
            bbox_heights = [h for w, h in self.bbox_sizes]
            
            print(f"\nBounding box dimensions:")
            print(f"  Width:  min={min(bbox_widths)}, max={max(bbox_widths)}, avg={np.mean(bbox_widths):.0f}")
            print(f"  Height: min={min(bbox_heights)}, max={max(bbox_heights)}, avg={np.mean(bbox_heights):.0f}")
    
    def print_sample_annotations(self):
        """Print sample parsed annotations."""
        print(f"\n{'='*60}")
        print("SAMPLE ANNOTATIONS")
        print(f"{'='*60}")
        
        # Get a sample image from train split
        train_dir = self.dataset_dir / 'train'
        if train_dir.exists():
            sample_files = list(train_dir.glob('*.jpg'))[:3]
            
            for img_path in sample_files:
                annotation = self.parse_ccpd_filename(img_path.name)
                if annotation:
                    print(f"\nFile: {img_path.name}")
                    print(f"  Bbox: {annotation['bbox']}")
                    print(f"  Corners: {annotation['corners'][:2]}...")  # Show first 2 corners
                    print(f"  Plate chars: {annotation['plate_chars']}")
                    print(f"  Tilt: {annotation['tilt']}°, Brightness: {annotation['brightness']}")
    
    def print_errors(self, max_errors: int = 10):
        """Print validation errors."""
        if not self.errors:
            return
        
        print(f"\n{'='*60}")
        print(f"VALIDATION ERRORS (showing first {min(len(self.errors), max_errors)})")
        print(f"{'='*60}")
        
        for error in self.errors[:max_errors]:
            print(f"  ❌ {error}")
        
        if len(self.errors) > max_errors:
            print(f"\n  ... and {len(self.errors) - max_errors} more errors")
    
    def validate(self) -> bool:
        """
        Run full validation on the dataset.
        
        Returns:
            True if dataset is valid, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"CCPD DATASET VALIDATION")
        print(f"{'='*60}")
        print(f"Dataset directory: {self.dataset_dir}")
        
        # Check if dataset directory exists
        if not self.dataset_dir.exists():
            print(f"\n❌ Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Print sample annotations first
        self.print_sample_annotations()
        
        # Validate each split
        all_valid = True
        for split in ['train', 'val', 'test']:
            if not self.validate_split(split):
                all_valid = False
        
        # Print statistics
        self.print_statistics()
        
        # Print errors
        self.print_errors()
        
        # Final verdict
        print(f"\n{'='*60}")
        if all_valid and not self.errors:
            print("✅ DATASET VALIDATION PASSED")
            print(f"{'='*60}")
            print("\nYour dataset is ready for training!")
            print("\nNext steps:")
            print("  1. Convert to COCO format (if needed):")
            print("     python tools/ccpd_to_coco.py --input datasets/CCPD2020/ccpd_green --output datasets/plates")
            print("  2. Train the model:")
            print("     python tools/train_plate_detector.py --dataset datasets/plates")
            return True
        else:
            print("❌ DATASET VALIDATION FAILED")
            print(f"{'='*60}")
            print(f"\nFound {len(self.errors)} errors. Please fix them before training.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate CCPD dataset before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate CCPD dataset
  python tools/validate_ccpd_dataset.py --dataset datasets/CCPD2020/ccpd_green
  
  # Validate with verbose output
  python tools/validate_ccpd_dataset.py --dataset datasets/CCPD2020/ccpd_green --verbose
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to CCPD dataset directory (should contain train/val/test folders)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output'
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = CCPDValidator(args.dataset)
    is_valid = validator.validate()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()
