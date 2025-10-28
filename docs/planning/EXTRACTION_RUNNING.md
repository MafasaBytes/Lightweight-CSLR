# GPU Extraction In Progress

**Started**: October 14, 2025
**Status**: Background process running
**Process ID**: Check with Task Manager (python.exe)

---

## ✅ Verification - IT WORKS!

**Test Results**:
- First sample processed successfully
- Time: 13 seconds for 176 frames
- Speed: 13.2 FPS (first sample includes warmup)
- Files created: 1/4376
- Feature dimensions: 177 (body 51 + hands 126)

---

## 🚀 How to Run (Without Terminal Closing)

### Option 1: Using Batch File (RECOMMENDED)
```cmd
# Double-click this file or run in cmd:
run_extraction.bat

# Output goes to: extraction_output.log
```

### Option 2: PowerShell with Logging
```powershell
# Run in PowerShell
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python src\extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split all" -RedirectStandardOutput extraction_output.log -RedirectStandardError extraction_errors.log
```

### Option 3: Direct Command (Keep window open)
```cmd
python src\extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split all
# Don't close the window!
```

---

## 📊 Monitor Progress

### Check File Count
```powershell
# PowerShell - run every minute
while ($true) {
    $count = (Get-ChildItem data\processed\train\*.npy -ErrorAction SilentlyContinue).Count
    Write-Host "Extracted: $count/4376 ($(($count/4376*100).ToString('F1'))%)"
    Start-Sleep 60
}
```

### Check Log File
```cmd
# View last 20 lines
powershell -Command "Get-Content extraction_output.log -Tail 20"

# Continuous monitoring
powershell -Command "Get-Content extraction_output.log -Wait"
```

### Check Latest Checkpoint
```powershell
Get-ChildItem data\processed\train\checkpoint_yolov8_*.json |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 |
    Get-Content |
    ConvertFrom-Json
```

---

## 🔍 Expected Progress

| Time | Samples | Percentage | Status |
|------|---------|------------|--------|
| 00:00 | 0 | 0% | Starting |
| 00:30 | 2-3 | 0.1% | Warmup complete |
| 01:00 | 5-7 | 0.2% | Normal speed |
| 02:00 | 15-20 | 0.5% | Check FPS |
| 06:00 | 100 | 2.3% | First checkpoint |
| 12:00 | 200 | 4.6% | Stable rate |
| **04:00** | **~4376** | **100%** | **Train complete** |

**Expected completion**: 3-4 hours for all 4,667 samples (train/dev/test)

---

## 🎯 Performance Expectations

### Current Performance (From Test)
- First sample: 13.2 FPS (includes warmup)
- Expected after warmup: 20-30 FPS
- Time per sample: ~5-10 seconds (depending on frame count)

### Full Dataset Estimate
- Train (4,376 samples): 3-4 hours
- Dev (111 samples): +5 minutes
- Test (180 samples): +8 minutes
- **Total: 3.5-4.5 hours**

---

## ⚠️ If Process Stops

### Check if Still Running
```powershell
Get-Process python | Where-Object {$_.CPU -gt 0}
```

### Resume from Checkpoint
The script automatically resumes! Just run again:
```cmd
python src\extract_features_yolov8_gpu.py --model yolov8m-pose.pt --split all
```

It will skip already-extracted files and continue.

---

## 📁 Output Structure

```
data/processed/
├── train/
│   ├── 01April_2010_Thursday_heute_default-0.npy  ✅ (Done!)
│   ├── *.npy  (4,375 more to go)
│   ├── checkpoint_yolov8_100.json
│   ├── checkpoint_yolov8_200.json
│   ├── ...
│   ├── extraction_metrics_yolov8.csv
│   └── extraction_summary_yolov8.json
├── dev/ (after train completes)
└── test/ (after train completes)
```

---

## 🎯 Next Steps

1. **Run the extraction** using one of the methods above
2. **Keep window open** or use logging
3. **Check progress** every 30-60 minutes
4. **Wait 3-4 hours** for completion
5. **Validate** extraction with file count

---

## ✅ Success Criteria

After completion:
- [ ] 4,376 files in `data/processed/train/`
- [ ] 111 files in `data/processed/dev/`
- [ ] 180 files in `data/processed/test/`
- [ ] Each file has shape `(num_frames, 177)`
- [ ] No failed samples in logs
- [ ] Ready for model training!

---

*Last updated: October 14, 2025*
*Status: Verified working - ready for full extraction*
*FPS: 13.2 (first sample), expected 20-30 after warmup*
