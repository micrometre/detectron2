#!/bin/bash
# Automated training workflow using CV-based annotations
# No manual annotation required!

set -e

echo "=========================================="
echo "Automated Detectron2 Training"
echo "Using CV-based plate detection"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if images exist
if [ ! -d "images" ] || [ -z "$(ls -A images/*.jpg 2>/dev/null)" ]; then
    echo "❌ No images found in images/ directory"
    echo "   Please add some images first"
    exit 1
fi

echo "✓ Found images in images/ directory"
echo ""

# Step 1: Auto-generate annotations
echo "STEP 1: Auto-Generate Annotations"
echo "----------------------------------"
echo "Using CV-based plate detection..."
echo ""

python tools/cv_to_annotations.py \
    --input images/ \
    --output datasets/plates/auto \
    --confidence 0.4

echo ""

# Copy images to dataset
echo "Copying images to dataset..."
mkdir -p datasets/plates/auto/images
cp images/*.jpg datasets/plates/auto/images/ 2>/dev/null || true

# Split into train/val
echo ""
echo "STEP 2: Split Dataset"
echo "---------------------"

python -c "
import json
import shutil
from pathlib import Path
import random

# Load annotations
with open('datasets/plates/auto/annotations.json') as f:
    data = json.load(f)

# Split 80/20
random.seed(42)
images = data['images']
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

# Create train set
train_data = data.copy()
train_data['images'] = train_images
train_img_ids = {img['id'] for img in train_images}
train_data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in train_img_ids]

Path('datasets/plates/train/images').mkdir(parents=True, exist_ok=True)
with open('datasets/plates/train/annotations.json', 'w') as f:
    json.dump(train_data, f, indent=2)

for img in train_images:
    src = Path('datasets/plates/auto/images') / img['file_name']
    dst = Path('datasets/plates/train/images') / img['file_name']
    shutil.copy2(src, dst)

# Create val set
if val_images:
    val_data = data.copy()
    val_data['images'] = val_images
    val_img_ids = {img['id'] for img in val_images}
    val_data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in val_img_ids]
    
    Path('datasets/plates/val/images').mkdir(parents=True, exist_ok=True)
    with open('datasets/plates/val/annotations.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    for img in val_images:
        src = Path('datasets/plates/auto/images') / img['file_name']
        dst = Path('datasets/plates/val/images') / img['file_name']
        shutil.copy2(src, dst)

print(f'✓ Split: {len(train_images)} train, {len(val_images)} val')
"

echo ""

# Step 3: Train
echo "STEP 3: Train Model"
echo "-------------------"
echo "Starting training..."
echo ""

python tools/train_plate_detector.py \
    --dataset datasets/plates \
    --output output \
    --max-iter 1000 \
    --batch-size 2 \
    --lr 0.001

echo ""

# Step 4: Evaluate
echo "STEP 4: Evaluate Model"
echo "----------------------"
python tools/evaluate_model.py \
    --model output/model_final.pth \
    --dataset datasets/plates \
    --visualize \
    --output eval_results

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: output/model_final.pth"
echo "Evaluation results: eval_results/"
echo ""
echo "Test your model:"
echo "  python detect.py test.jpg  # Uses trained model"
echo ""
