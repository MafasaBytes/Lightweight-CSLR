# Week 2: Pose + Dominant Hand Features

**Status:** Ready to extract
**Date:** 2025-10-20
**Expected WER:** 50-60% (down from 73.68% pose-only)

---

## Phase Summary

Adding dominant hand features to improve recognition performance.

**Feature Enhancement:**
- **Pose-only (Week 1):** 66 dims → 73.68% WER ❌
- **Pose + Hand (Week 2):** 108 dims → Target 50-60% WER ✅

---

## Step 1: Feature Extraction (~6-8 hours)

### Run Extraction

```bash
# Extract all splits at once
.\extract_all_features.bat

# Or extract individually:
python src/baseline/extract_features.py --split train  # 5,672 videos
python src/baseline/extract_features.py --split dev    # 540 videos
python src/baseline/extract_features.py --split test   # 629 videos
```

### Feature Breakdown

**108-dimensional features:**
- **Pose:** 66 dims (33 MediaPipe pose keypoints × 2 coords)
- **Dominant Hand:** 42 dims (21 hand keypoints × 2 coords)

**Dominant Hand Selection:**
- Uses right hand if detected (most signers are right-handed)
- Falls back to left hand if right not detected
- Zero-pads if no hand detected

### Expected Output

```
data/features/
├── train/
│   ├── 01April_2010_Thursday_heute-4467.npz
│   ├── ... (5,672 files)
├── dev/
│   ├── ... (540 files)
└── test/
    └── ... (629 files)
```

Each `.npz` file contains:
- `features`: (num_frames, 108) array
- `metadata`: Detection rates for pose and hand
- `video_id`: Video identifier

---

## Step 2: Update Configuration

**File:** `configs/bilstm_config.yaml`

Update `input_dim` from 66 → 108:

```yaml
model:
  architecture:
    input_dim: 108              # Changed from 66
    hidden_dim: 192
    num_layers: 3
    vocab_size: 1229
    dropout: 0.3
    projection_dim: 128
    subsample_factor: 2

data:
  features_dir: "data/features"  # Changed from data/baseline_features
  vocabulary_path: "data/baseline_vocabulary/vocabulary.txt"
```

---

## Step 3: Update Dataset Loader

**File:** `src/baseline/dataset.py`

Update to load from `data/features` instead of `data/baseline_features`.

---

## Step 4: Train Model (~2-3 hours)

```bash
python src/baseline/train.py --config configs/bilstm_config.yaml
```

**Expected Training:**
- Epochs: ~50-70 (early stopping)
- Duration: 2-3 hours
- Target WER: 50-60%

---

## Step 5: Evaluation

Compare results to pose-only baseline:

| Metric | Pose-Only (Week 1) | Pose+Hand (Week 2) | Improvement |
|--------|-------------------|-------------------|-------------|
| Val WER | 73.68% | 50-60% (target) | ~15-25% |
| Test WER | 72.19% | TBD | TBD |
| Blank Ratio | 86.13% | TBD | TBD |
| Vocab Coverage | 19.1% (234/1226) | TBD | TBD |

---

## Decision Point

**If Val WER < 65%:** ✅ Proceed to Week 3 (full holistic features)
**If Val WER > 70%:** Document results and move to Phase II

---

## Files Created

- ✅ `src/baseline/extract_features.py` - Extraction script
- ✅ `extract_all_features.bat` - Batch extraction
- ✅ `WEEK2_PLAN.md` - This plan
- ⏳ `configs/bilstm_config.yaml` - Need to create/update
- ⏳ Updated `src/baseline/dataset.py` - Need to update path

---

## Cleaned Up Files

**Archived to `archive/`:**
- `extract_pose_features_simple.py`
- `extract_pose_features_sequential.py`
- `extract_pose_features_debug.py`
- `extract_pose_features_fixed.py`

**Renamed (clean conventions):**
- `train_bilstm_optimized.py` → `train.py`
- `bilstm_optimized.py` → `bilstm.py`
- `create_optimized_bilstm_model()` → `create_model()`

---

## Next Actions

1. **Run feature extraction** (~6-8 hours)
   ```bash
   .\extract_all_features.bat
   ```

2. **Update config** (5 minutes)
   - Change `input_dim: 108`
   - Change `features_dir: "data/features"`

3. **Train model** (~2-3 hours)
   ```bash
   python src/baseline/train.py --config configs/bilstm_config.yaml
   ```

4. **Evaluate and compare**

---

**Ready to start extraction when you are!**
