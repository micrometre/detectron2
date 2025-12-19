#!/usr/bin/env python3
"""
Prepare dataset for Detectron2 training.
Converts annotations to COCO format and splits into train/val sets.
"""

import json
import argparse
from pathlib import Path
import shutil
import random
from datetime import datetime


def create_coco_dataset(image_files, annotations, output_dir, split_name):
    """
    Create COCO format dataset.
    
    Args:
        image_files: List of image paths
        annotations: List of annotation dicts
        output_dir: Output directory
        split_name: 'train' or 'val'
    """
    output_path = Path(output_dir) / split_name
    images_dir = output_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO format structure
    coco_data = {
        "info": {
            "description": "License Plate Detection Dataset",
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
                "supercategory": "object"
            }
        ]
    }
    
    annotation_id = 1
    
    for img_id, (img_file, img_annotations) in enumerate(zip(image_files, annotations), 1):
        # Copy image to output directory
        dest_img = images_dir / img_file.name
        shutil.copy2(img_file, dest_img)
        
        # Add image info
        import cv2
        img = cv2.imread(str(img_file))
        height, width = img.shape[:2]
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": width,
            "height": height
        })
        
        # Add annotations for this image
        for bbox in img_annotations:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],  # COCO format: [x, y, width, height]
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1
    
    # Save annotations
    annotations_file = output_path / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✓ Created {split_name} set:")
    print(f"  - Images: {len(image_files)}")
    print(f"  - Annotations: {annotation_id - 1}")
    print(f"  - Saved to: {output_path}")
    
    return coco_data


def load_labelimg_annotations(xml_file):
    """
    Load annotations from LabelImg XML format.
    Returns list of bounding boxes: [(x1, y1, x2, y2), ...]
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall('object'):
        # Only process objects labeled as 'license_plate' or 'plate'
        name = obj.find('name').text.lower()
        if 'plate' not in name:
            continue
        
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        
        boxes.append((x1, y1, x2, y2))
    
    return boxes


def prepare_dataset(input_dir, output_dir, val_split=0.2, seed=42):
    """
    Prepare dataset from annotated images.
    
    Args:
        input_dir: Directory containing images and XML annotations
        output_dir: Output directory for COCO format dataset
        val_split: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducibility
    """
    input_path = Path(input_dir)
    
    # Find all images with corresponding XML annotations
    image_extensions = {'.jpg', '.jpeg', '.png'}
    annotated_images = []
    
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            xml_file = img_file.with_suffix('.xml')
            if xml_file.exists():
                annotated_images.append((img_file, xml_file))
    
    if not annotated_images:
        print("❌ No annotated images found!")
        print("   Make sure you have .xml files alongside your images.")
        print("   Use LabelImg to create annotations:")
        print("   pip install labelImg")
        print("   labelImg")
        return
    
    print(f"Found {len(annotated_images)} annotated images")
    
    # Load all annotations
    all_data = []
    for img_file, xml_file in annotated_images:
        boxes = load_labelimg_annotations(xml_file)
        if boxes:  # Only include images with at least one plate
            all_data.append((img_file, boxes))
    
    print(f"Images with plate annotations: {len(all_data)}")
    
    if len(all_data) < 2:
        print("❌ Need at least 2 annotated images (1 train, 1 val)")
        return
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_data)
    
    split_idx = max(1, int(len(all_data) * (1 - val_split)))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")
    
    # Create COCO datasets
    train_images, train_annotations = zip(*train_data)
    create_coco_dataset(train_images, train_annotations, output_dir, 'train')
    
    if val_data:
        val_images, val_annotations = zip(*val_data)
        create_coco_dataset(val_images, val_annotations, output_dir, 'val')
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"   1. Review the dataset in {output_dir}")
    print(f"   2. Run training: python tools/train_plate_detector.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Detectron2 training")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory with images and XML annotations")
    parser.add_argument("--output", type=str, default="datasets/plates",
                       help="Output directory for COCO format dataset")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_dataset(args.input, args.output, args.val_split, args.seed)


if __name__ == "__main__":
    main()
