# License Plate Detector

A comprehensive license plate detection training system with Detectron2.

## 🚀 Features

- **Custom Model Training**: Train on CCPD dataset with Detectron2
- **Dataset Validation**: Pre-training validation and COCO conversion tools
- **Production Ready**: Apache 2.0 licensed, commercially friendly

## 📦 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/micrometre/Detectron2
cd plate_detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Detectron2
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

## 🎯 Usage

### Detection (Inference)


# Object detection on images

```bash
python demos/detect_image_objects.py test-images/public.jpg 
```

# Vehicle detection on images

```bash
python demos/detect_image_vehicles.py test-images/public.jpg 
```

# Adjust thresholds

```bash
python demos/detect_image_vehicles.py test-images/public.jpg  --vehicle-conf 0.7 
```


# Object detection with a videos 

```bash
python demos/detect_video_obects.py ~/Videos/uk_road.mp4 -s 10

```

# Vehicle detection with videos

```bash
python demos/detect_video_vehicles.py ~/Videos/uk_road.mp4 -s 10

```



### Training Custom Models

#### 1. Download CCPD Dataset
```bash
# Download CCPD2020 dataset
wget <ccpd-dataset-url>
unzip ccpd_dataset.zip -d datasets/CCPD2020/
```

#### 2. Validate Dataset
```bash
# Check dataset integrity before training
python tools/validate_ccpd_dataset.py --dataset datasets/CCPD2020/ccpd_green

# Expected output:
# ✅ 11,776 total images (5,769 train, 1,001 val, 5,006 test)
# ✅ 99.97% valid images
```

#### 3. Convert to COCO Format
```bash
# Convert CCPD filename annotations to COCO JSON
python tools/ccpd_to_coco.py \
  --input datasets/CCPD2020/ccpd_green \
  --output datasets/plates \
  --splits train val
```

#### 4. Train Model
```bash
# Quick test (1 epoch, ~8 hours on CPU)
./train.sh

# Or manually with custom settings
source venv/bin/activate
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --output output \
  --max-iter 3000 \
  --batch-size 2 \
  --lr 0.001

# For better results (10 epochs)
python tools/train_plate_detector.py \
  --dataset datasets/plates \
  --max-iter 28850 \
  --batch-size 2
```

#### 5. Monitor Training
```bash
# Training logs show progress every 20 iterations:
# [12/19 14:31:31] iter: 19  total_loss: 0.814  loss_cls: 0.662  lr: 1.99e-05
# [12/19 14:34:53] iter: 39  total_loss: 0.625  loss_cls: 0.453  lr: 3.99e-05

# Watch for:
# - total_loss decreasing (target: < 0.5)
# - Validation AP scores every 500 iterations
# - ETA and time per iteration
```

## � Training Performance

| Device | Speed | 3000 Iterations | 10 Epochs |
|--------|-------|-----------------|-----------|
| CPU (Intel i7) | ~10s/iter | ~8 hours | ~80 hours |
| Intel Iris Xe | ~4s/iter | ~3 hours | ~30 hours |
| NVIDIA GPU | ~1s/iter | ~1 hour | ~8 hours |

## 🛠️ Tools

| Tool | Purpose |
|------|---------|
| `validate_ccpd_dataset.py` | Validate CCPD dataset before training |
| `ccpd_to_coco.py` | Convert CCPD to COCO format |
| `train_plate_detector.py` | Train Detectron2 model |
| `detect_easyocr.py` | EasyOCR-only detection |
| `detect_video.py` | Video processing with frame skip |
| `detect_pipeline.py` | Two-stage vehicle→text pipeline |

## � Project Structure

```
plate_detector/
├── demos/              # Detection scripts
│   ├── detect_image_objects.py
│   └── detect_image_vehicles.py
│   ├── detect_video_objects.py
│   ├── detect_video_vehicles.py
├── tools/                # Training and conversion tools
│   ├── validate_ccpd_dataset.py
│   ├── ccpd_to_coco.py
│   └── train_plate_detector.py
├── datasets/             # Training datasets (gitignored)
│   ├── CCPD2020/        # Original CCPD dataset
│   └── plates/          # Converted COCO format
├── output/              # Model checkpoints (gitignored)
├── docs/                # Documentation
├── train.sh            # Training convenience script
└── requirements.txt    # Python dependencies
```

## 📝 Dataset Information

### CCPD (Chinese City Parking Dataset)

- **Format**: Filename-based annotations
- **Size**: ~12,000 images (720×1160)
- **Splits**: Train (5,769), Val (1,001), Test (5,006)
- **Annotations**: Bounding boxes encoded in filenames

**Filename Format**:
```
{area}-{tilt}_{brightness}-{x1}&{y1}_{x2}&{y2}-{corners}-{plate_chars}-{brightness}-{blur}.jpg

Example:
00360785590278-91_265-311&485_406&524-...jpg
```

## 🎓 Training Tips

### Good Training Indicators
- ✅ **total_loss** decreases from ~1.5 → 0.3-0.5
- ✅ **AP** (Average Precision) > 60%
- ✅ **AP50** > 85%

### Troubleshooting
- **Out of memory?** Reduce `--batch-size 1`
- **Training too slow?** Use GPU or reduce `--max-iter`
- **Want to resume?** Add `--resume` flag

## 📄 License

- **Detectron2**: Apache 2.0 (Facebook)
- **PyTorch**: BSD-style (Facebook)

All components are **commercially friendly** with permissive licenses.

## 🔗 References

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [CCPD Dataset](https://github.com/detectRecog/CCPD)

## 📚 Documentation

- [Annotation Alternatives](docs/ANNOTATION_ALTERNATIVES.md) - Dataset annotation methods
- [Training Instructions](docs/TRAINING_GUIDE.md) - Detailed training guide
- See `tools/` scripts for `--help` documentation
