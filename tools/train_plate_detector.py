#!/usr/bin/env python3
"""
Train Detectron2 model for license plate detection.
"""

import os
import argparse
from pathlib import Path
import json
import logging

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import EventStorage, CommonMetricPrinter, JSONWriter, TensorboardXWriter


class PlateDetectionTrainer(DefaultTrainer):
    """
    Custom trainer with evaluation during training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def setup_config(dataset_dir, output_dir, max_iter=3000, batch_size=2, base_lr=0.001):
    """
    Setup Detectron2 configuration for plate detection.
    """
    cfg = get_cfg()
    
    # Load base config
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset
    cfg.DATASETS.TRAIN = ("plates_train",)
    cfg.DATASETS.TEST = ("plates_val",)
    
    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Solver (training parameters)
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))  # LR decay steps
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    cfg.SOLVER.CHECKPOINT_PERIOD = 500  # Save checkpoint every N iterations
    
    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class: license_plate
    cfg.MODEL.DEVICE = "cpu"  # Use CPU (change to "cuda" if GPU available)
    
    # Output
    cfg.OUTPUT_DIR = output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = 500  # Evaluate every N iterations
    
    # Logging - print metrics every 20 iterations for better visibility
    cfg.SOLVER.LOG_PERIOD = 20
    
    return cfg


def register_datasets(dataset_dir):
    """
    Register train and validation datasets.
    """
    dataset_path = Path(dataset_dir)
    
    # Register training set
    train_json = dataset_path / 'train' / 'annotations.json'
    train_images = dataset_path / 'train' / 'images'
    
    if train_json.exists():
        register_coco_instances(
            "plates_train",
            {},
            str(train_json),
            str(train_images)
        )
        print(f"✓ Registered training dataset: {train_json}")
    else:
        raise FileNotFoundError(f"Training annotations not found: {train_json}")
    
    # Register validation set
    val_json = dataset_path / 'val' / 'annotations.json'
    val_images = dataset_path / 'val' / 'images'
    
    if val_json.exists():
        register_coco_instances(
            "plates_val",
            {},
            str(val_json),
            str(val_images)
        )
        print(f"✓ Registered validation dataset: {val_json}")
    else:
        print(f"⚠ Validation annotations not found: {val_json}")
        print("  Training without validation set")


def train(dataset_dir, output_dir, max_iter=3000, batch_size=2, base_lr=0.001, resume=False):
    """
    Train the model.
    """
    # Setup logger with verbose output
    setup_logger(output=None, distributed_rank=0, name="detectron2")
    logger = logging.getLogger("detectron2")
    logger.setLevel(logging.INFO)
    
    # Register datasets
    register_datasets(dataset_dir)
    
    # Setup config
    cfg = setup_config(dataset_dir, output_dir, max_iter, batch_size, base_lr)
    
    # Calculate approximate epochs
    try:
        train_dataset = DatasetCatalog.get("plates_train")
        num_images = len(train_dataset)
        iterations_per_epoch = num_images / batch_size
        num_epochs = max_iter / iterations_per_epoch
    except:
        num_images = "unknown"
        num_epochs = "unknown"
    
    # Print training info
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Training images: {num_images}")
    print(f"Max iterations: {max_iter}")
    print(f"Batch size: {batch_size}")
    print(f"Base learning rate: {base_lr}")
    print(f"Device: {cfg.MODEL.DEVICE}")
    if num_epochs != "unknown":
        print(f"Approximate epochs: {num_epochs:.2f}")
        print(f"Iterations per epoch: ~{int(iterations_per_epoch)}")
    print(f"Checkpoint every: {cfg.SOLVER.CHECKPOINT_PERIOD} iterations")
    print(f"Evaluate every: {cfg.TEST.EVAL_PERIOD} iterations")
    print(f"Log metrics every: {cfg.SOLVER.LOG_PERIOD} iterations")
    print("="*70)
    
    print("\n📊 WHAT TO WATCH DURING TRAINING:")
    print("  • total_loss: Should decrease over time (target: < 0.5)")
    print("  • loss_cls: Classification loss (should decrease)")
    print("  • loss_box_reg: Bounding box regression loss (should decrease)")
    print("  • lr: Learning rate (will decay at 70% and 90% of training)")
    print("  • Every 500 iters: Validation metrics (AP, AP50, AP75)")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = PlateDetectionTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    
    # Start training
    print("🚀 Starting training...\n")
    print("💡 TIP: Training progress will be logged every 20 iterations")
    print("💡 Press Ctrl+C to stop training early (model will be saved)\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"   Latest checkpoint saved in: {output_dir}")
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}/model_final.pth")
    print(f"\nNext steps:")
    print(f"   1. Evaluate: python tools/evaluate_model.py --model {output_dir}/model_final.pth")
    print(f"   2. Inference: python detect.py --model {output_dir}/model_final.pth --image test.jpg")


def main():
    parser = argparse.ArgumentParser(description="Train Detectron2 for license plate detection")
    parser.add_argument("--dataset", type=str, default="datasets/plates",
                       help="Dataset directory (COCO format)")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory for model and logs")
    parser.add_argument("--max-iter", type=int, default=3000,
                       help="Maximum training iterations (default: 3000)")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Base learning rate (default: 0.001)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    
    args = parser.parse_args()
    
    train(args.dataset, args.output, args.max_iter, args.batch_size, args.lr, args.resume)


if __name__ == "__main__":
    main()
