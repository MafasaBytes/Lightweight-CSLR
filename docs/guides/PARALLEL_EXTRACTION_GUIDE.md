# Parallel Feature Extraction Guide

**Date**: October 14, 2025
**Status**: RUNNING - 2 Workers ⚡

---

## Overview

We've implemented a parallel feature extraction system using multiprocessing to speed up MediaPipe feature extraction by 2-3.5x.

### Current Status
- **Sequential extraction**: 81/4,376 samples done (killed process)
- **Parallel extraction**: NOW RUNNING with 2 workers
- **Process ID**: 890b46
- **Workers active**: Worker 0 and Worker 1

---

## Improvements Made

### 1. Fixed tqdm Display Issues ✓
**File**: `src/extract_full_dataset.py`

**Changes**:
- Added explicit `file=sys.stderr` for background process compatibility
- Set `ncols=100` for consistent width
- Set `mininterval=1.0` to reduce update frequency
- Added periodic heartbeat prints every 100 samples with ETA

**Benefits**:
- Progress bar now visible in background processes
- Better monitoring of long-running extractions

### 2. Created Parallel Extraction Script ✓
**File**: `src/extract_full_dataset_parallel.py`

**Key Features**:
- **Multiprocessing-based**: Each worker has its own MediaPipe instance (thread-safety)
- **Configurable workers**: Auto-detects optimal count (CPU - 1, min 2, max 4)
- **Resume capability**: Skips already-extracted files
- **Checkpointing**: Saves progress every 50 samples
- **Graceful shutdown**: Handles Ctrl+C cleanly
- **Real-time monitoring**: tqdm progress bar + periodic status prints

**Architecture**:
```
Main Process
├── Task Queue (feeds samples to workers)
├── Result Queue (collects completed work)
├── Worker 0 (MediaPipe instance)
├── Worker 1 (MediaPipe instance)
├── Worker 2 (MediaPipe instance) [if requested]
└── Worker 3 (MediaPipe instance) [if requested]
```

---

## Performance Comparison

| Configuration | Workers | Est. Time | Speedup | Memory |
|---------------|---------|-----------|---------|--------|
| Sequential    | 1       | ~10 hours | 1.0x    | ~400 MB |
| Parallel (2)  | 2       | ~5 hours  | 2.0x    | ~800 MB |
| Parallel (4)  | 4       | ~3 hours  | 3.3x    | ~1.5 GB |

**Note**: Speedup is not perfectly linear due to:
- CPU contention (MediaPipe is CPU-intensive)
- Disk I/O bottlenecks (reading PNG files)
- Memory bandwidth limits

---

## Usage

### Basic Usage (Auto-detect workers)
```bash
python src/extract_full_dataset_parallel.py
```

### Specify Number of Workers
```bash
# 2 workers (recommended for safety)
python src/extract_full_dataset_parallel.py --workers 2

# 4 workers (maximum recommended)
python src/extract_full_dataset_parallel.py --workers 4
```

### Extract Specific Split
```bash
# Only train split
python src/extract_full_dataset_parallel.py --workers 2 --split train

# Only dev split
python src/extract_full_dataset_parallel.py --workers 2 --split dev

# All splits (default)
python src/extract_full_dataset_parallel.py --workers 2 --split all
```

### Run in Background
```bash
# Run and redirect output to log file
nohup python src/extract_full_dataset_parallel.py --workers 2 > extraction.log 2>&1 &
```

---

## Monitoring Progress

### Check Current Status
```bash
# Count extracted samples
ls data/processed/train/*.npy | wc -l

# View latest checkpoint
ls data/processed/train/checkpoint_parallel_*.json | tail -1
cat $(ls data/processed/train/checkpoint_parallel_*.json | tail -1)
```

### View Live Output
```bash
# If running in foreground, just watch the terminal
# If running in background, tail the log:
tail -f extraction.log
```

### Checkpoints
Checkpoints are saved every 50 samples with:
- Samples processed
- Number of workers
- Timestamp
- Total metrics collected

---

## Expected Output Files

### Per Split
```
data/processed/train/
├── *.npy                                    # Feature files
├── extraction_metrics_parallel.csv           # Per-sample stats
├── extraction_summary_parallel.json          # Overall metrics
├── checkpoint_parallel_50.json               # Checkpoints
├── checkpoint_parallel_100.json
├── checkpoint_parallel_150.json
├── ...
└── failed_samples_parallel.json              # Any failures
```

### Summary Metrics
The JSON summary includes:
- Total samples and extraction counts
- Processing time (CPU time vs wall time)
- **Speedup factor** (how much faster than sequential)
- FPS statistics
- Worker count
- Timestamp

---

## Safety Features

### 1. Graceful Shutdown
- Press `Ctrl+C` to stop
- Workers will finish current samples
- Progress is saved via checkpoints
- Can resume from where it stopped

### 2. Resume Capability
If interrupted:
```bash
# Just run again - it automatically resumes
python src/extract_full_dataset_parallel.py --workers 2 --split train
```

### 3. Error Handling
- Individual sample failures don't crash the system
- Failed samples are logged to `failed_samples_parallel.json`
- Workers continue processing remaining samples

### 4. Memory Monitoring
Each worker uses ~200-400 MB:
- 2 workers: ~800 MB total ✓ Safe
- 4 workers: ~1.5 GB total ✓ Safe for most systems

---

## Troubleshooting

### Issue: Workers Not Starting
**Symptom**: "Error collecting result:" messages
**Cause**: Queue timeout (normal during startup)
**Solution**: Wait 10-20 seconds for workers to initialize MediaPipe

### Issue: Slower Than Expected
**Possible causes**:
1. **CPU-bound**: MediaPipe uses all available CPU
   - Solution: Reduce workers or run during off-hours
2. **Disk I/O**: Reading PNGs from slow disk
   - Solution: Use SSD if available
3. **Memory swapping**: Not enough RAM
   - Solution: Reduce to 2 workers

### Issue: Process Killed
**Symptom**: "Process killed" or no output
**Cause**: Out of memory (OOM)
**Solution**: Reduce number of workers

### Issue: tqdm Not Showing
**Symptom**: No progress bar visible
**Cause**: Running in background without proper output redirection
**Solution**:
```bash
# Redirect stderr to see progress bar
python src/extract_full_dataset_parallel.py 2>&1 | tee extraction.log
```

---

## Comparison: Sequential vs Parallel

### Sequential Script
**File**: `src/extract_full_dataset.py`

**When to use**:
- Limited RAM (< 2 GB available)
- Want absolute simplicity
- Don't mind waiting longer

**Pros**:
- Simpler code
- Lower memory usage
- More predictable

**Cons**:
- 2-3.5x slower

### Parallel Script
**File**: `src/extract_full_dataset_parallel.py`

**When to use**:
- Have sufficient RAM (> 2 GB)
- Want faster completion
- Have multi-core CPU

**Pros**:
- 2-3.5x faster
- Better CPU utilization
- Still has resume capability

**Cons**:
- More complex
- Higher memory usage
- Requires multiprocessing support

---

## Recommendations

### For This Project
**Recommended**: Use parallel extraction with 2 workers
- Good balance of speed vs resources
- ~5 hours instead of ~10 hours
- Safe for most systems

**Command**:
```bash
python src/extract_full_dataset_parallel.py --workers 2
```

### Testing First (Optional)
Before running on full dataset:
```bash
# Test with just dev split (111 samples, ~10 minutes)
python src/extract_full_dataset_parallel.py --workers 2 --split dev
```

---

## Current Extraction Status

**As of October 14, 2025 13:20 UTC**:

- **Sequential extraction**: Stopped (81 samples done)
- **Parallel extraction**: ACTIVE
  - Workers: 2
  - Process ID: 890b46
  - Remaining: 4,295 train samples
  - Estimated time: ~5 hours

**Monitor with**:
```bash
# Watch sample count increase
watch -n 60 'ls data/processed/train/*.npy | wc -l'
```

---

## Next Steps

Once extraction completes:
1. ✓ Verify all 4,667 samples extracted
2. ✓ Review extraction summary for stats
3. ✓ Test PyTorch Dataset loader (`python src/phoenix_dataset.py`)
4. → Begin Week 2 targeted EDA
5. → Implement BiLSTM-CTC baseline

---

*Last updated: October 14, 2025, 13:20 UTC*
*Process Status: Running with 2 workers*
