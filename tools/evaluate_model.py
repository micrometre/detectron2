#!/usr/bin/env python3
"""
Evaluate trained Detectron2 model on validation set.
"""

import argparse
from pathlib import Path
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def setup_config(model_path):
    """
    Setup config for inference.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    
    return cfg


def evaluate(model_path, dataset_dir, visualize=False, output_dir="eval_results"):
    """
    Evaluate model on validation set.
    """
    dataset_path = Path(dataset_dir)
    val_json = dataset_path / 'val' / 'annotations.json'
    val_images = dataset_path / 'val' / 'images'
    
    if not val_json.exists():
        print(f"❌ Validation annotations not found: {val_json}")
        return
    
    # Register validation dataset
    register_coco_instances(
        "plates_val_eval",
        {},
        str(val_json),
        str(val_images)
    )
    
    # Setup config
    cfg = setup_config(model_path)
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {val_json}")
    print(f"{'='*60}\n")
    
    evaluator = COCOEvaluator("plates_val_eval", output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "plates_val_eval")
    results = inference_on_dataset(DefaultPredictor(cfg).model, val_loader, evaluator)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for key, value in results['bbox'].items():
        print(f"{key}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Visualize predictions
    if visualize:
        visualize_predictions(cfg, val_images, output_dir)
    
    return results


def visualize_predictions(cfg, images_dir, output_dir, max_images=5):
    """
    Visualize predictions on sample images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("plates_val_eval")
    
    image_files = list(Path(images_dir).glob("*.jpg"))[:max_images]
    
    print(f"\nVisualizing predictions on {len(image_files)} images...")
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        outputs = predictor(img)
        
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        result_img = out.get_image()[:, :, ::-1]
        output_file = output_path / f"pred_{img_file.name}"
        cv2.imwrite(str(output_file), result_img)
        
        print(f"  ✓ {img_file.name} -> {output_file}")
    
    print(f"\nVisualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained plate detection model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.pth file)")
    parser.add_argument("--dataset", type=str, default="datasets/plates",
                       help="Dataset directory")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize predictions on validation images")
    parser.add_argument("--output", type=str, default="eval_results",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    evaluate(args.model, args.dataset, args.visualize, args.output)


if __name__ == "__main__":
    main()
