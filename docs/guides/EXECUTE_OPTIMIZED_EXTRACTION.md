# Execute Optimized CPU Extraction - Action Plan

**Date**: October 14, 2025
**Status**: READY TO EXECUTE
**Strategy**: CPU-optimized MediaPipe (advisor approved)

---

## ✅ Strategic Decision Approved

**MediaPipe has NO GPU support in Python** - this is architectural, not fixable.

**Decision**: Continue with **CPU extraction** using optimized settings.

See full analysis in: `RESEARCH_DECISION_MediaPipe_CPU.md`

---

## 🎯 Immediate Actions

### Step 1: Kill Old Processes

You have 2 background processes running with suboptimal settings:

```bash
# Check running processes
# Process 1b84af: python src/extract_full_dataset.py (sequential, slow)
# Process 890b46: python src/extract_full_dataset_parallel.py --workers 2 (2 workers, slow)

# These should be stopped
```

**Note**: Both are using non-optimized settings (confidence=0.5, model_complexity varied)

### Step 2: Run Optimized Extraction

**Recommended command for overnight run**:

```bash
python src/extract_full_dataset_parallel.py --workers 8 --no-gpu --split train
```

**What this does**:
- 8 CPU workers (maximum parallelism)
- CPU-optimized settings:
  - `model_complexity=1` (vs 2)
  - `min_detection_confidence=0.3` (vs 0.5)
  - `min_tracking_confidence=0.3` (vs 0.5)
  - `smooth_landmarks=False` (disabled)
  - `refine_face_landmarks=False` (disabled)

**Expected performance**:
- Speed improvement: 10-20% faster than current 7.7 FPS
- Estimated: ~8-9 FPS per worker
- Total throughput: ~64-72 FPS (8 workers)
- Time for 3,859 samples: **~8-10 hours**
- Completion: Tomorrow morning (October 15)

---

## 📊 Alternative Options

### Option A: Maximum Speed (Recommended)
```bash
python src/extract_full_dataset_parallel.py --workers 8 --no-gpu --split train
```
- **Pros**: Fastest completion (~8 hours)
- **Cons**: High CPU usage (may affect system responsiveness)
- **Best for**: Overnight/weekend runs

### Option B: Balanced (If using computer during extraction)
```bash
python src/extract_full_dataset_parallel.py --workers 4 --no-gpu --split train
```
- **Pros**: Moderate CPU usage, still reasonably fast
- **Cons**: Slower (~12 hours)
- **Best for**: Background extraction while working

### Option C: Conservative (System with limited RAM)
```bash
python src/extract_full_dataset_parallel.py --workers 2 --no-gpu --split train
```
- **Pros**: Low resource usage
- **Cons**: Slowest (~16-18 hours)
- **Best for**: Resource-constrained systems

---

## 🖥️ Monitoring Setup

### Terminal 1: Run Extraction
```bash
# Navigate to project directory
cd C:\Users\Masia\OneDrive\Desktop\sign-language-recognition

# Run extraction (choose your option)
python src/extract_full_dataset_parallel.py --workers 8 --no-gpu --split train
```

### Terminal 2: Monitor Progress (Optional)
```bash
# Run monitoring dashboard
python extraction-monitor.py
```

**Dashboard shows**:
- Progress bars (train/dev/test)
- System CPU, Memory, Disk usage
- Active worker count
- Estimated completion time

### Terminal 3: Watch System (Optional)
```bash
# Monitor CPU usage
taskmgr

# Or use PowerShell
Get-Process python | Select-Object CPU,PM,ProcessName
```

---

## 📁 Output Structure

### During Extraction
```
data/processed/train/
├── *.npy                    # Feature files (517 exist, adding 3,859 more)
├── checkpoint_550.json      # Checkpoints every 50 samples
├── checkpoint_600.json
├── checkpoint_650.json
├── ...
└── logs/
    └── extraction_YYYYMMDD_HHMMSS.log
```

### After Completion
```
data/processed/train/
├── *.npy                           # All 4,376 feature files
├── extraction_metrics_gpu.csv      # Per-sample statistics
├── extraction_summary_gpu.json     # Performance summary
└── failed_samples_gpu.json         # Any failures (hopefully empty)
```

---

## 🔍 Progress Tracking

### Check Progress Manually
```powershell
# Count extracted files
(Get-ChildItem data\processed\train\*.npy).Count

# Expected: starts at 517, ends at 4,376
```

### Check Latest Checkpoint
```powershell
# Find latest checkpoint
Get-ChildItem data\processed\train\checkpoint_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# View checkpoint content
Get-Content (Get-ChildItem data\processed\train\checkpoint_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

### Calculate ETA
```python
# Quick ETA calculation
import json
from pathlib import Path
from datetime import datetime, timedelta

# Load latest checkpoint
checkpoints = sorted(Path("data/processed/train").glob("checkpoint_*.json"))
if checkpoints:
    with open(checkpoints[-1]) as f:
        cp = json.load(f)

    checkpoint_time = datetime.fromisoformat(cp["timestamp"])
    elapsed = (datetime.now() - checkpoint_time).total_seconds() / 3600
    samples_done = cp["samples_processed"]
    remaining = 3859 - samples_done
    rate = samples_done / elapsed if elapsed > 0 else 0
    eta_hours = remaining / rate if rate > 0 else 0

    print(f"Samples done: {samples_done}/3,859")
    print(f"Rate: {rate:.1f} samples/hour")
    print(f"ETA: {eta_hours:.1f} hours")
```

---

## 🚨 Troubleshooting

### Issue: Process Killed
**Symptom**: Extraction stops unexpectedly
**Cause**: Out of memory
**Solution**:
```bash
# Reduce workers
python src/extract_full_dataset_parallel.py --workers 4 --no-gpu --split train
```

### Issue: Very Slow (<5 FPS)
**Symptom**: Progress bar shows <5 FPS
**Check**:
- CPU usage (should be 70-90%)
- Disk speed (check for HDD vs SSD)
- Background applications

**Solution**:
```bash
# Close other applications
# Check if antivirus is scanning files
# Consider using fewer workers to reduce I/O contention
```

### Issue: High Error Rate
**Symptom**: Many "failed" samples in progress
**Check**: Log file in `data/processed/logs/extraction_*.log`
**Common causes**:
- Missing frame folders
- Corrupted PNG files
- Permission issues

---

## ✅ Success Criteria

After extraction completes, verify:

### 1. File Count
```powershell
# Should be 4,376
(Get-ChildItem data\processed\train\*.npy).Count
```

### 2. Check Summary
```powershell
Get-Content data\processed\train\extraction_summary_gpu.json
```

**Expected**:
```json
{
  "split": "train",
  "total_samples": 4376,
  "newly_extracted": 3859,
  "failed": 0,  // Or very low (<10)
  "average_fps": 8-9,
  "total_wall_time": 28800-36000  // 8-10 hours
}
```

### 3. Test Dataset Loader
```bash
python src/phoenix_dataset.py
```

**Expected output**:
```
Loaded 4376/4376 samples for train split
Vocabulary size: 1135
Dataset test complete!
```

---

## 📅 Next Steps (Tomorrow Morning)

### After Extraction Completes

1. **Validate Extraction** ✓
   ```bash
   python -c "from pathlib import Path; print(f'Train: {len(list(Path(\"data/processed/train\").glob(\"*.npy\")))}/4376')"
   ```

2. **Extract Dev & Test Splits**
   ```bash
   # Dev split (~5-10 minutes)
   python src/extract_full_dataset_parallel.py --workers 4 --no-gpu --split dev

   # Test split (~8-15 minutes)
   python src/extract_full_dataset_parallel.py --workers 4 --no-gpu --split test
   ```

3. **Begin Week 2 Work**
   - Start BiLSTM-CTC baseline training
   - Prototype MMPose extractor (2-3 hours)
   - Design feature abstraction layer

---

## 📖 Documentation Created

1. **GPU_EXTRACTION_READY.md** - Environment setup (now obsolete)
2. **RESEARCH_DECISION_MediaPipe_CPU.md** - Strategic decision rationale
3. **EXECUTE_OPTIMIZED_EXTRACTION.md** - This file (action plan)

---

## 🎯 Command to Execute NOW

**Recommended** (overnight run):
```bash
python src/extract_full_dataset_parallel.py --workers 8 --no-gpu --split train
```

**Monitor** (optional, separate terminal):
```bash
python extraction-monitor.py
```

---

## 📊 Expected Timeline

| Time | Event |
|------|-------|
| Today 16:00 | Start extraction |
| Today 24:00 | ~50% complete (checkpoint 2,400) |
| Tomorrow 02:00 | ~90% complete (checkpoint 4,000) |
| Tomorrow 04:00 | **COMPLETE** (4,376/4,376) |
| Tomorrow 09:00 | Extract dev split |
| Tomorrow 09:30 | Extract test split |
| Tomorrow 10:00 | Begin baseline training |

---

*Created: October 14, 2025*
*Status: READY TO EXECUTE*
*Strategy: CPU-optimized extraction (research advisor approved)*
