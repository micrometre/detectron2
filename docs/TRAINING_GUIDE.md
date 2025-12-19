# Detectron2 Fine-Tuning Guide

Complete guide to fine-tune Detectron2 for license plate detection.

## Prerequisites

- ✅ Detectron2 installed
- ✅ Images to annotate (in `images/` folder)
- ⚠️ Annotation tool needed (LabelImg)

## Step 1: Install Annotation Tool

```bash
source venv/bin/activate
pip install labelImg
```

## Step 2: Annotate Images

### Launch LabelImg

```bash
# Launch without arguments (avoids a bug)
labelImg
```

Or if you prefer, launch from the GUI menu if installed system-wide.

### Annotation Instructions

1. **Open Dir**: Select your `images/` folder
2. **Change Save Dir**: Select same `images/` folder
3. **For each image**:
   - Click "Create RectBox" (or press `W`)
   - Draw box around license plate
   - Label it as `license_plate`
   - Press `Ctrl+S` to save
   - Press `D` for next image

4. **Tips**:
   - Be precise with bounding boxes
   - Include entire plate (don't cut off edges)
   - Label all plates in multi-plate images
   - Skip images without visible plates

### Minimum Requirements

- **Quick test**: 10-15 annotated images
- **Good results**: 30-50 annotated images  
- **Production**: 100+ annotated images

## Step 3: Prepare Dataset

Convert annotations to COCO format:

```bash
source venv/bin/activate

python tools/prepare_dataset.py \
  --input images/ \
  --output datasets/plates \
  --val-split 0.2
```

**Output**:
```
datasets/plates/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

## Step 4: Train Model

### Quick Training (Testing)

```bash
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --output output \
  --max-iter 500 \
  --batch-size 2 \
  --lr 0.001
```

**Time**: ~5-10 minutes on CPU  
**Purpose**: Verify pipeline works

### Production Training

```bash
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --output output \
  --max-iter 3000 \
  --batch-size 2 \
  --lr 0.001
```

**Time**: ~30-60 minutes on CPU  
**Purpose**: Get good accuracy

### Resume Training

If training is interrupted:

```bash
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --output output \
  --resume
```

## Step 5: Evaluate Model

```bash
python tools/evaluate_model.py \
  --model output/model_final.pth \
  --dataset datasets/plates \
  --visualize \
  --output eval_results
```

**Metrics to check**:
- **AP (Average Precision)**: Target 70-90%
- **AP50**: Target 80-95%
- **AP75**: Target 60-80%

## Step 6: Use Trained Model

Update `detect.py` to use your custom model:

```python
# In detect.py, change:
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Your trained model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # One class: license_plate
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
```

Then run detection:

```bash
python detect.py test_image.jpg
```

## Training Parameters

### Hyperparameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-iter` | 3000 | Training iterations |
| `--batch-size` | 2 | Images per batch (lower for less memory) |
| `--lr` | 0.001 | Learning rate (lower = slower but stable) |
| `--val-split` | 0.2 | Validation set ratio |

### Tuning Tips

**If loss is not decreasing**:
- Lower learning rate: `--lr 0.0001`
- Train longer: `--max-iter 5000`

**If overfitting** (train good, val bad):
- More training data
- Lower `--max-iter`
- Add data augmentation

**If underfitting** (both train and val bad):
- Train longer: `--max-iter 5000`
- Higher learning rate: `--lr 0.01`
- More training data

## Monitoring Training

Training logs show:

```
[00:01:23] eta: 0:15:42  iter: 100  total_loss: 0.8234  loss_cls: 0.3421  loss_box_reg: 0.4813
```

**What to watch**:
- `total_loss`: Should decrease over time
- `loss_cls`: Classification loss
- `loss_box_reg`: Bounding box regression loss
- `eta`: Estimated time remaining

**Good training**:
- Loss starts high (~1.5-2.0)
- Decreases steadily
- Stabilizes around 0.3-0.5

## Expected Results

### Before Fine-Tuning (COCO Model)
- Detects: Cars, trucks, buses
- Does NOT detect: License plates

### After Fine-Tuning (Your Model)
- Detects: License plates specifically
- Typical performance:
  - **mAP@0.5**: 75-90%
  - **Precision**: 85-95%
  - **Recall**: 80-90%

## Troubleshooting

### "No annotated images found"
- Make sure `.xml` files exist alongside images
- Check LabelImg saved to correct directory

### "Training is very slow"
- Normal on CPU (~10-20 sec/iteration)
- Consider using GPU or reducing `--max-iter`

### "Out of memory"
- Reduce `--batch-size` to 1
- Use smaller images

### "Model not detecting plates"
- Train longer (`--max-iter 5000`)
- Check annotations are correct
- Lower detection threshold in `detect.py`

## Advanced: Data Augmentation

To improve robustness, add augmentation in training config:

```python
# In train_plate_detector.py, add:
from detectron2.data import transforms as T

cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333
```

## Next Steps

1. **Collect more data**: More annotations = better model
2. **Test on new images**: Verify generalization
3. **Deploy**: Use trained model in production
4. **Iterate**: Collect failure cases, annotate, retrain
