# YOLOv8 GPU Feature Extraction Guide

## Overview

The improved YOLOv8 GPU extraction script now supports:
- **Parallel processing** with multiple GPU workers
- **Clean logging** to files and console
- **Real-time GPU monitoring**
- **Automatic checkpointing** for resume capability
- **Better error handling** and recovery

## Quick Start

### 1. Run the extraction

```bash
python run_yolov8_extraction.py
```

This will:
- Check GPU availability
- Start parallel extraction with optimal settings
- Show real-time progress

### 2. Monitor progress (in another terminal)

```bash
python extraction-monitor.py
```

This shows:
- Extraction progress for each split
- GPU utilization
- Worker process status
- ETA and performance metrics

## Command Line Options

### Basic extraction:
```bash
python src/extract_features_yolov8_gpu.py --split train
```

### Full parallel extraction with custom settings:
```bash
python src/extract_features_yolov8_gpu.py \
    --split all \
    --batch-size 16 \
    --workers 2 \
    --model yolov8m-pose.pt \
    --checkpoint-interval 50
```

### Options:
- `--split`: train, dev, test, or all
- `--batch-size`: Frames per batch (default: 8, recommended: 16 for GPU)
- `--workers`: Number of parallel workers (default: auto based on GPU count)
- `--model`: YOLOv8 model size (n/s/m/l/x)
- `--no-hands`: Skip hand extraction for faster processing
- `--checkpoint-interval`: Save progress every N samples

## Performance Tips

1. **Batch Size**: Higher is better for GPU utilization
   - RTX 3090/4090: Use 16-32
   - RTX 3080/3070: Use 8-16
   - Older GPUs: Use 4-8

2. **Workers**: 1-2 workers per GPU is optimal
   - Single GPU: 2 workers
   - Multi-GPU: 2 workers per GPU

3. **Model Size**: 
   - yolov8m-pose.pt: Good balance (default)
   - yolov8l-pose.pt: Better accuracy, slower
   - yolov8s-pose.pt: Faster, less accurate

## Expected Performance

With RTX 3090 and optimal settings:
- **Train split (4259 videos)**: ~2-3 hours
- **Average FPS**: 50-100 depending on video length
- **GPU Utilization**: 80-95%

## Troubleshooting

### "Error collecting result" messages
These are normal timeout messages when workers are busy. The extraction continues normally.

### Low GPU utilization
- Increase batch size
- Check if using parallel mode (`--parallel`)
- Ensure YOLOv8 is using GPU (check logs)

### Out of memory errors
- Reduce batch size
- Use fewer workers
- Use smaller model (yolov8s-pose.pt)

## Resume from checkpoint

If extraction is interrupted, it automatically resumes from the last checkpoint when restarted.

## Output Files

Features are saved to:
```
data/processed/
├── train/
│   ├── *.npy                          # Feature files
│   ├── extraction_summary_yolov8.json  # Summary statistics
│   ├── extraction_metrics_yolov8.csv   # Detailed metrics
│   └── gpu_stats_yolov8.json          # GPU utilization data
├── dev/
├── test/
└── logs/
    └── yolov8_extraction_*.log        # Detailed logs
```

## Feature Format

Each .npy file contains a numpy array of shape (num_frames, num_features):
- Without hands: (num_frames, 51) - YOLOv8 pose only
- With hands: (num_frames, 177) - YOLOv8 pose + MediaPipe hands

Features layout:
- 0-50: YOLOv8 pose keypoints (17 points × 3 coordinates)
- 51-176: MediaPipe hand landmarks (2 hands × 21 points × 3 coordinates)
