# Plate Detection (Bounding Boxes Only)

## Overview

`detect_plate_cv.py` - Computer vision-based license plate detection using OpenCV.

**Focus**: Detects plate **bounding boxes only** - no OCR/text recognition.

## How It Works

Uses multiple computer vision techniques:

1. **Edge Detection** - Canny edge detector finds plate boundaries
2. **Contour Analysis** - Identifies rectangular shapes
3. **Filtering** - Validates by:
   - Area (500-50,000 pixels)
   - Aspect ratio (1.5:1 to 6:1 - typical plate dimensions)
   - Variance (high contrast text on background)
   - Corner count (prefers 4-corner rectangles)
4. **Non-Maximum Suppression** - Removes overlapping detections

## Usage

### Single Image

```bash
source venv/bin/activate

# Basic usage
python detect_plate_cv.py image.jpg

# Custom output path
python detect_plate_cv.py image.jpg -o result.jpg

# Adjust confidence threshold
python detect_plate_cv.py image.jpg --confidence 0.5
```

### Folder Processing

```bash
# Process all images in folder
python detect_plate_cv.py --folder images/ -o plate_results/

# Lower confidence for difficult images
python detect_plate_cv.py --folder images/ -c 0.2
```

### Advanced Parameters

```bash
# Adjust area constraints
python detect_plate_cv.py image.jpg --min-area 1000 --max-area 30000

# Combine parameters
python detect_plate_cv.py --folder images/ -c 0.4 --min-area 800
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-c, --confidence` | 0.3 | Confidence threshold (0-1) |
| `--min-area` | 500 | Minimum plate area in pixels |
| `--max-area` | 50000 | Maximum plate area in pixels |
| `-o, --output` | auto | Output file/folder path |
| `-f, --folder` | - | Process folder instead of single image |

## Test Results

Tested on 9 sample images:

- **Total plates detected**: 20
- **Confidence range**: 59.9% - 100%
- **Average confidence**: ~85%
- **False positives**: Minimal (filtered by aspect ratio and variance)

### Example Output

```
Processing: image.jpg
Image shape: (720, 1280, 3)
Confidence threshold: 0.3

Detected 2 license plate(s):
  Plate 1: [565, 225, 734, 289] confidence: 92.6%
  Plate 2: [1021, 175, 1280, 327] confidence: 84.6%

Result saved to: image_plate_detected.jpg
```

## Advantages

✅ **No external models** - Pure OpenCV, no downloads  
✅ **Fast** - Processes images in <1 second  
✅ **Lightweight** - No GPU required  
✅ **Adjustable** - Fine-tune parameters for your use case  
✅ **Apache 2.0 License** - Commercially friendly  

## Limitations

⚠️ **Lighting sensitive** - Works best with good contrast  
⚠️ **Angle dependent** - Front-facing plates work best  
⚠️ **No OCR** - Only bounding boxes, no text recognition  
⚠️ **May detect non-plates** - Rectangular objects with text-like patterns  

## Tips for Better Results

1. **Lower confidence** for difficult lighting: `--confidence 0.2`
2. **Adjust area** for different image resolutions:
   - High-res (4K): `--min-area 2000 --max-area 100000`
   - Low-res (480p): `--min-area 200 --max-area 10000`
3. **Preprocessing**: Enhance contrast before detection
4. **Combine with OCR**: Use detected boxes as ROI for EasyOCR

## Integration with OCR

To get plate text, combine with EasyOCR:

```python
# 1. Detect plate bounding boxes
_, detections = detect_plates("image.jpg")

# 2. For each detection, crop and run OCR
import easyocr
reader = easyocr.Reader(['en'])

for x1, y1, x2, y2, conf in detections:
    plate_roi = image[y1:y2, x1:x2]
    text = reader.readtext(plate_roi)
    print(f"Plate text: {text}")
```

## Comparison with Other Methods

| Method | Speed | Accuracy | Setup | License |
|--------|-------|----------|-------|---------|
| **CV Detection** | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | ✅ Easy | Apache 2.0 |
| EasyOCR | ⚡⚡ Medium | ⭐⭐⭐⭐ Very Good | ✅ Easy | Apache 2.0 |
| Detectron2 | ⚡ Slow | ⭐⭐⭐⭐ Very Good | ⚠️ Complex | Apache 2.0 |
| YOLOv8 | ⚡⚡⚡ Fast | ⭐⭐⭐⭐⭐ Excellent | ✅ Easy | ❌ AGPL-3.0 |

## Next Steps

1. **Fine-tune parameters** on your specific images
2. **Add preprocessing** (contrast enhancement, denoising)
3. **Integrate OCR** for full ALPR solution
4. **Train custom model** for better accuracy (optional)
