#!/usr/bin/env python3
"""
License Plate Detector using Facebook's Detectron2
Uses a pre-trained model for general object detection.
"""

import cv2
import argparse
from pathlib import Path

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def setup_predictor(confidence_threshold=0.5):
    """
    Set up Detectron2 predictor with pre-trained Faster R-CNN model.
    """
    cfg = get_cfg()
    
    # Use Faster R-CNN with ResNet-50 FPN backbone (good balance of speed/accuracy)
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    # Use pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    
    # Set confidence threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    
    # Force CPU mode (no GPU available)
    cfg.MODEL.DEVICE = "cpu"
    
    return DefaultPredictor(cfg), cfg


def detect_objects(image_path, output_path=None, confidence=0.5):
    """
    Run object detection on an image.
    
    Note: Detectron2's COCO model detects 80 object classes including 'car'.
    For specific license plate detection, you would need to:
    1. Fine-tune on a license plate dataset, or
    2. Use detected vehicles and apply OCR/plate-specific detection
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Processing: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Setup predictor
    predictor, cfg = setup_predictor(confidence)
    
    # Run detection
    outputs = predictor(image)
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Get class names from COCO metadata
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        class_name = class_names[cls]
        print(f"  {i+1}. {class_name}: {score:.2%} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(instances)
    result_image = out.get_image()[:, :, ::-1]
    
    # Save or return result
    if output_path:
        # Create output directory if it doesn't exist
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), result_image)
        print(f"\nResult saved to: {output_path}")
    
    return result_image, outputs


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Object Detection")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="results-images",
        help="Directory to save detected frames"
    )
    
    args = parser.parse_args()
    
    # Generate output path using output directory
    input_path = Path(args.image)
    output_path = Path(args.output_dir) / f"{input_path.stem}_detected{input_path.suffix}"
    
    detect_objects(args.image, str(output_path), args.confidence)


if __name__ == "__main__":
    main()
