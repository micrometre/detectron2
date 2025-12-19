#!/bin/bash
# Quick start script for Detectron2 fine-tuning

set -e

echo "=========================================="
echo "Detectron2 Fine-Tuning Quick Start"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if images exist
if [ ! -d "images" ] || [ -z "$(ls -A images/*.jpg 2>/dev/null)" ]; then
    echo "❌ No images found in images/ directory"
    echo "   Please add some images to annotate"
    exit 1
fi

echo "✓ Found images in images/ directory"
echo ""

# Step 1: Annotate
echo "STEP 1: Annotate Images"
echo "------------------------"
echo "Launching LabelImg..."
echo ""
echo "Instructions:"
echo "  1. Open Dir: Select 'images/' folder"
echo "  2. Change Save Dir: Select 'images/' folder"
echo "  3. Draw boxes around license plates (press W)"
echo "  4. Label as 'license_plate'"
echo "  5. Save (Ctrl+S) and next image (D)"
echo "  6. Close LabelImg when done"
echo ""
read -p "Press Enter to launch LabelImg..."

# Launch LabelImg without arguments to avoid the directory bug
labelImg

# Check if annotations were created
XML_COUNT=$(find images/ -name "*.xml" | wc -l)
if [ "$XML_COUNT" -lt 2 ]; then
    echo ""
    echo "❌ Need at least 2 annotated images"
    echo "   Found: $XML_COUNT annotations"
    exit 1
fi

echo ""
echo "✓ Found $XML_COUNT annotated images"
echo ""

# Step 2: Prepare dataset
echo "STEP 2: Prepare Dataset"
echo "------------------------"
python tools/prepare_dataset.py \
    --input images/ \
    --output datasets/plates \
    --val-split 0.2

echo ""

# Step 3: Train
echo "STEP 3: Train Model"
echo "-------------------"
echo "Starting training (this will take a while)..."
echo ""

python tools/train_plate_detector.py \
    --dataset datasets/plates \
    --output output \
    --max-iter 500 \
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
echo "Next steps:"
echo "  1. Check evaluation metrics above"
echo "  2. View visualizations in eval_results/"
echo "  3. Use model: python detect.py --model output/model_final.pth test.jpg"
echo ""
