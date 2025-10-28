# GPU-Accelerated Feature Extraction - Environment Ready

**Date**: October 14, 2025
**Status**: READY TO RUN

---

## Environment Setup Complete

### GPU Configuration
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
- **VRAM**: 8188 MB total (6487 MB free)
- **CUDA Version**: 12.6
- **PyTorch CUDA**: Available and working
- **GPUtil**: Installed and detecting GPU

### Python Environment
- **Python**: 3.9.13
- **Required packages**: ALL INSTALLED
  - torch 2.8.0+cu126
  - mediapipe 0.10.21
  - opencv-python 4.11.0.86
  - gputil 1.4.0
  - pynvml 11.5.0
  - setproctitle 1.3.3
  - pandas, numpy, tqdm, psutil

### Files Ready
- `src/extract_full_dataset_parallel.py` - GPU-accelerated extraction script
- `extraction-monitor.py` - Real-time monitoring dashboard

---

## Current Extraction Status

### Train Split Progress
- **Completed**: 517 / 4,376 samples (11.8%)
- **Remaining**: 3,859 samples
- **Previous processes**: Both killed (sequential and 2-worker CPU)

### Dev and Test Splits
- **Dev**: 0 / 111 samples (0%)
- **Test**: 0 / 180 samples (0%)

---

## How to Run GPU Extraction

### Option 1: Train Split Only (Recommended First)
```bash
# Resume train split extraction with GPU acceleration
python src/extract_full_dataset_parallel.py --split train --workers 2
```

**Expected Performance**:
- Workers: 2 GPU workers
- Speed: ~0.5-1.0 samples/second
- Time: ~1-2 hours for remaining 3,859 samples
- GPU Load: 60-80%
- VRAM Usage: ~3-4 GB per worker

### Option 2: All Splits
```bash
# Extract train, dev, and test splits
python src/extract_full_dataset_parallel.py --split all --workers 2
```

### Option 3: CPU Fallback (If GPU Issues)
```bash
# Disable GPU and use CPU only
python src/extract_full_dataset_parallel.py --split train --workers 2 --no-gpu
```

---

## Monitoring Progress

### Terminal 1: Run Extraction
```bash
python src/extract_full_dataset_parallel.py --split train --workers 2
```

**You will see**:
- Worker initialization messages
- Real-time progress bar with FPS and GPU load
- Checkpoint saves every 50 samples
- GPU statistics in the logs

### Terminal 2: Monitor GPU (Optional)
```bash
# Watch GPU usage in real-time
nvidia-smi -l 1
```

### Terminal 3: Dashboard Monitor (Optional)
```bash
# Run the monitoring dashboard
python extraction-monitor.py
```

**Dashboard features**:
- Live progress bars for each split
- GPU load, memory, and temperature
- System CPU and RAM usage
- Active worker count
- ETA calculations

---

## Output Files

### During Extraction
```
data/processed/train/
├── *.npy                           # Feature files (517 already exist)
├── checkpoint_50.json              # Checkpoint at 50 samples
├── checkpoint_100.json             # Checkpoint at 100 samples
├── checkpoint_150.json             # ... and so on
└── logs/
    └── extraction_YYYYMMDD_HHMMSS.log  # Detailed logs
```

### After Completion
```
data/processed/train/
├── *.npy                           # All 4,376 feature files
├── extraction_metrics_gpu.csv      # Per-sample statistics
├── extraction_summary_gpu.json     # Overall performance metrics
├── gpu_stats.json                  # GPU usage timeline
└── failed_samples_gpu.json         # Any failures (if any)
```

---

## Expected Performance Comparison

| Mode | Workers | Device | Speed | Time (3,859 samples) |
|------|---------|--------|-------|----------------------|
| Sequential | 1 | CPU | ~0.1/sec | ~10.7 hours |
| Parallel | 2 | CPU | ~0.25/sec | ~4.3 hours |
| Parallel | 4 | CPU | ~0.46/sec | ~2.3 hours |
| **Parallel** | **2** | **GPU** | **~0.5-1.0/sec** | **~1-2 hours** |

**GPU Advantages**:
- Higher model complexity (2 vs 1)
- Batch processing (16 frames at once)
- Better landmark detection quality
- Real-time GPU monitoring

---

## Troubleshooting

### Issue: GPU Out of Memory
**Symptom**: CUDA out of memory errors
**Solution**:
```bash
# Reduce to 1 worker
python src/extract_full_dataset_parallel.py --split train --workers 1
```

### Issue: GPU Not Being Used
**Symptom**: GPU load stays at 0%
**Check**:
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```
**Solution**: Use `--no-gpu` flag if needed

### Issue: Process Killed
**Symptom**: Process stops unexpectedly
**Cause**: System OOM (out of memory)
**Solution**:
```bash
# Close other GPU applications
# OR use CPU mode
python src/extract_full_dataset_parallel.py --split train --workers 2 --no-gpu
```

### Issue: Slow Extraction
**Check**:
- GPU temperature (should be < 80°C)
- GPU load (should be 60-80%)
- Disk speed (SSD vs HDD)

---

## Resume Capability

If extraction is interrupted:
1. **Features are saved** immediately after each sample
2. **Checkpoints** are saved every 50 samples
3. **Just rerun** the same command:
   ```bash
   python src/extract_full_dataset_parallel.py --split train --workers 2
   ```
4. Script will **automatically skip** already-extracted samples

---

## Next Steps After Completion

### 1. Validate Extraction
```bash
# Check all files exist
python -c "from pathlib import Path; train = list(Path('data/processed/train').glob('*.npy')); print(f'Train: {len(train)}/4376'); dev = list(Path('data/processed/dev').glob('*.npy')); print(f'Dev: {len(dev)}/111'); test = list(Path('data/processed/test').glob('*.npy')); print(f'Test: {len(test)}/180')"
```

### 2. Review Extraction Report
```bash
# View GPU extraction summary
python -c "import json; summary = json.load(open('data/processed/train/extraction_summary_gpu.json')); print('Speedup:', summary['speedup_factor'], 'x'); print('Avg FPS:', summary['average_fps']); print('GPU Load:', summary.get('avg_gpu_load', 'N/A'))"
```

### 3. Test PyTorch Dataset Loader
```bash
python src/phoenix_dataset.py
```

### 4. Begin Week 2 EDA
- Temporal analysis on extracted features
- Spatial analysis (keypoint stability)
- Vocabulary analysis (class imbalance)

---

## Command Reference

### Basic Extraction
```bash
# Train only (resume from 517/4376)
python src/extract_full_dataset_parallel.py --split train

# Specific worker count
python src/extract_full_dataset_parallel.py --split train --workers 2

# All splits
python src/extract_full_dataset_parallel.py --split all

# CPU mode
python src/extract_full_dataset_parallel.py --split train --no-gpu
```

### Monitoring
```bash
# GPU usage
nvidia-smi -l 1

# Dashboard
python extraction-monitor.py

# Check progress
ls data/processed/train/*.npy | wc -l
```

---

## Safety Features

1. **Graceful Shutdown**: Ctrl+C stops cleanly
2. **Checkpointing**: Progress saved every 50 samples
3. **Resume Support**: Skips existing files
4. **Error Handling**: Individual failures don't crash system
5. **GPU Monitoring**: Prevents overheating and overload
6. **Logging**: Detailed logs for debugging

---

## Ready to Run!

Your environment is fully configured and ready for GPU-accelerated extraction.

**Recommended command to start**:
```bash
python src/extract_full_dataset_parallel.py --split train --workers 2
```

**Monitor in separate terminal**:
```bash
nvidia-smi -l 1
```

**Estimated completion**: ~1-2 hours for remaining train samples

---

*Environment prepared: October 14, 2025*
*GPU: NVIDIA RTX 4070 Laptop (8GB)*
*All dependencies installed and verified*
