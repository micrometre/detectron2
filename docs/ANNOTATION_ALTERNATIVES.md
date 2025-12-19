# Alternative Annotation Methods

LabelImg has compatibility issues with PyQt5. Here are better alternatives:

## Option 1: CVAT (Recommended)

**CVAT** is a professional web-based annotation tool.

### Quick Setup (Docker)

```bash
# Install Docker if not already installed
# Then run CVAT:
docker run -d -p 8080:8080 --name cvat cvat/ui:latest

# Open browser: http://localhost:8080
# Create account and start annotating
```

### Export Format
- Export as "COCO 1.0"
- Place exported `annotations.json` in `datasets/plates/train/`

## Option 2: Roboflow (Web-Based, Free Tier)

1. Go to https://roboflow.com
2. Create free account
3. Upload images
4. Annotate in browser
5. Export as "COCO JSON"

## Option 3: Manual JSON Creation

For small datasets (5-10 images), create annotations manually:

### Simple Annotation Script

```bash
python tools/simple_annotator.py
```

This will launch a simple OpenCV-based annotator (no PyQt dependencies).

## Option 4: Use Pre-Annotated Dataset

Download CCPD (Chinese City Parking Dataset):

```bash
# Download subset (or full dataset from official source)
wget https://example.com/ccpd_subset.zip
unzip ccpd_subset.zip -C datasets/
```

### CCPD Dataset Format

CCPD uses **filename-based annotations**. Each filename encodes the bounding box and plate information:

```
Format: {area}-{tilt}_{brightness}-{x1}&{y1}_{x2}&{y2}-{corners}-{plate_chars}-{brightness}-{blur}.jpg

Example: 00360785590278-91_265-311&485_406&524-406&524_313&520_311&485_402&489-0_0_3_24_28_24_31_33-117-16.jpg
```

### Validate Before Training

**IMPORTANT**: Always validate the dataset before training:

```bash
# Validate CCPD dataset
python tools/validate_ccpd_dataset.py --dataset datasets/CCPD2020/ccpd_green

# This will check:
# - All images are readable
# - Bounding boxes are valid
# - Dataset statistics (train/val/test counts)
# - Corrupted or missing files
```

### Convert to COCO Format (if needed)

If your training script requires COCO format, convert the dataset:

```bash
# Convert CCPD to COCO format
python tools/ccpd_to_coco.py --input datasets/CCPD2020/ccpd_green --output datasets/plates

# Then train
python tools/train_plate_detector.py --dataset datasets/plates
```

## Option 5: Use Existing CV Detection Results

Use the CV-based detector to create initial annotations:

```bash
# Generate initial annotations from CV detector
python tools/cv_to_annotations.py --input images/ --output datasets/plates/
```

Then manually review and correct in any text editor.

## Recommended Workflow

**For quick testing (recommended):**
```bash
# Use CV detector to create initial annotations
python tools/cv_to_annotations.py --input images/ --output datasets/plates/

# Review annotations.json in text editor
# Correct any wrong boxes

# Train
python tools/train_plate_detector.py
```

**For production:**
- Use CVAT (web-based, professional)
- Or Roboflow (easiest, cloud-based)
