# Video Processing Guide

## Quick Start

```bash
source venv/bin/activate

# Process a video file
python detect_video.py your_video.mp4
```

## Frame Skip Strategy

The `--skip` parameter controls how many frames to process:

| Skip Value | Frames Processed | Use Case | Speed |
|------------|------------------|----------|-------|
| `1` | Every frame | High accuracy needed | Slowest |
| `5` | Every 5th frame | **Recommended default** | Balanced |
| `10` | Every 10th frame | Fast preview | Fastest |
| `30` | Every 30th frame | 1 frame/second @ 30fps | Very fast |

## Example Scenarios

### Dashcam Footage (Moving Camera)
```bash
# Process more frames to catch plates
python detect_video.py dashcam.mp4 --skip 3
```

### Parking Lot Surveillance (Stationary Camera)
```bash
# Process fewer frames (vehicles move slowly)
python detect_video.py parking.mp4 --skip 10
```

### Traffic Camera (Fast-Moving Vehicles)
```bash
# Process every frame for best results
python detect_video.py traffic.mp4 --skip 1 --confidence 0.4
```

## Output

The script provides:

1. **Progress bar** - Real-time processing status
2. **Detection summary** - All unique plates found with frequency
3. **Annotated video** - Output video with bounding boxes and text

Example output:
```
Processing video: traffic.mp4
Resolution: 1920x1080
FPS: 30
Total frames: 3000
Frame skip: 5 (processing every 5th frame)

Processing frames: 100%|████████████| 3000/3000 [02:15<00:00, 22.1frames/s]

✓ Processed 600/3000 frames

📋 Detected license plates (sorted by frequency):
  • ABC123: 45 times
  • XYZ789: 32 times
  • DEF456: 18 times

💾 Output saved to: traffic_detected.mp4
```

## Performance Tips

1. **Frame skip = FPS / desired_samples_per_second**
   - 30 FPS video, want 6 samples/sec → `--skip 5`
   - 60 FPS video, want 6 samples/sec → `--skip 10`

2. **Memory usage**: Processing is done frame-by-frame (low memory)

3. **Speed**: On CPU, expect ~2-5 frames/second processing speed

4. **Accuracy vs Speed**:
   - Lower skip = More accurate but slower
   - Higher skip = Faster but may miss plates

## Common Issues

**Video won't open?**
- Ensure OpenCV supports the codec
- Try converting to standard MP4: `ffmpeg -i input.mov -c:v libx264 output.mp4`

**No plates detected?**
- Lower confidence: `--confidence 0.3`
- Process more frames: `--skip 1`
- Check if plates are visible/readable

**Output video too large?**
- Use lower skip value (fewer processed frames)
- Compress output: `ffmpeg -i output.mp4 -crf 23 compressed.mp4`
