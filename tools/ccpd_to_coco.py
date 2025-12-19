#!/usr/bin/env python3
"""
Convert CCPD dataset (filename-based annotations) to COCO format.

CCPD filename format:
{area}-{tilt}_{brightness}-{x1}&{y1}_{x2}&{y2}-{corners}-{plate_chars}-{brightness}-{blur}.jpg
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil
from tqdm import tqdm


def parse_ccpd_filename(filename):
    """Parse CCPD filename to extract bounding box."""
    try:
        name = filename.replace('.jpg', '')
        parts = name.split('-')
        
        if len(parts) < 3:
            return None
        
        # Extract top-left and bottom-right corners
        bbox_parts = parts[2].split('_')
        if len(bbox_parts) != 2:
            return None
        
        x1, y1 = map(int, bbox_parts[0].split('&'))
        x2, y2 = map(int, bbox_parts[1].split('&'))
        
        # COCO format: [x, y, width, height]
        width = x2 - x1
        height = y2 - y1
        
        return [x1, y1, width, height]
        
    except (ValueError, IndexError):
        return None


def convert_split(input_dir, output_dir, split_name):
    """Convert one split (train/val/test) to COCO format."""
    
    input_split_dir = input_dir / split_name
    if not input_split_dir.exists():
        print(f"⚠️  {split_name} split not found, skipping...")
        return None
    
    # Create output directories
    output_split_dir = output_dir / split_name
    output_images_dir = output_split_dir / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize COCO format structure
    coco_data = {
        "info": {
            "description": f"CCPD {split_name} dataset converted to COCO format",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "license_plate",
                "supercategory": "vehicle"
            }
        ]
    }
    
    # Get all images
    image_files = sorted(list(input_split_dir.glob('*.jpg')))
    
    print(f"\n{'='*60}")
    print(f"Converting {split_name.upper()} split...")
    print(f"{'='*60}")
    print(f"Found {len(image_files)} images")
    
    annotation_id = 1
    valid_count = 0
    skipped_count = 0
    
    # Process each image
    for image_id, img_path in enumerate(tqdm(image_files, desc=f"Processing {split_name}"), start=1):
        # Parse filename to get bbox
        bbox = parse_ccpd_filename(img_path.name)
        
        if bbox is None:
            skipped_count += 1
            continue
        
        # Validate bbox
        if bbox[2] <= 0 or bbox[3] <= 0:
            skipped_count += 1
            continue
        
        # Copy image to output directory
        output_img_path = output_images_dir / img_path.name
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": 720,  # CCPD standard size
            "height": 1160,  # CCPD standard size
        })
        
        # Add annotation
        area = bbox[2] * bbox[3]
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "segmentation": []
        })
        
        annotation_id += 1
        valid_count += 1
    
    # Save annotations.json
    annotations_path = output_split_dir / 'annotations.json'
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ {split_name.upper()} conversion complete:")
    print(f"  Valid images: {valid_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Annotations saved: {annotations_path}")
    
    return valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert CCPD dataset to COCO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CCPD dataset
  python tools/ccpd_to_coco.py \\
    --input datasets/CCPD2020/ccpd_green \\
    --output datasets/plates
  
  # Convert only train and val splits
  python tools/ccpd_to_coco.py \\
    --input datasets/CCPD2020/ccpd_green \\
    --output datasets/plates \\
    --splits train val
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CCPD dataset directory (contains train/val/test folders)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for COCO format dataset'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to convert (default: train val test)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    print(f"\n{'='*60}")
    print("CCPD TO COCO CONVERTER")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Splits: {', '.join(args.splits)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each split
    total_images = 0
    for split in args.splits:
        count = convert_split(input_dir, output_dir, split)
        if count:
            total_images += count
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images converted: {total_images}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Train the model:")
    print(f"     source venv/bin/activate")
    print(f"     python tools/train_plate_detector.py --dataset {output_dir}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == '__main__':
    exit(main())
