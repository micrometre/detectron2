#!/bin/bash
# Quick training script with proper environment activation

echo "🚀 Starting License Plate Detection Training"
echo "=============================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Run training
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --output output \
  --max-iter 3000 \
  --batch-size 2 \
  --lr 0.001

echo ""
echo "✅ Training script completed!"
