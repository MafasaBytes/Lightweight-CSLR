# BiLSTM-CTC Baseline Training - Quick Start Guide

**Status:** ✅ Training infrastructure complete and tested
**Date:** 2025-10-20

---

## What's Been Completed

### ✅ All Components Ready

1. **✓ Vocabulary created** (1,229 glosses with CTC tokens)
2. **✓ Features extracted** (6,841 videos, 66-dim pose features)
3. **✓ Model architecture optimized** (12.57 MB, 3.3M parameters)
4. **✓ Dataset loader created and tested**
5. **✓ Training script ready**
6. **✓ Configuration file prepared**

---

## Training Command

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Start training
python src/baseline/train_bilstm_optimized.py --config configs/bilstm_baseline_config.yaml
```

---

## Expected Training Behavior

### Initial Output
```
======================================================================
BILSTM-CTC BASELINE TRAINING
======================================================================
Device: cuda
Config: bilstm_baseline_optimized

[1/4] Loading datasets...
Vocabulary loaded: 1229 glosses
Loaded 5672 samples from train.corpus.csv
Loaded 540 samples from dev.corpus.csv
Loaded 629 samples from test.corpus.csv

DataLoaders created:
  Train batches: 178
  Dev batches:   17
  Test batches:  20

[2/4] Creating optimized BiLSTM model...
[Model summary with 3.3M parameters]

[3/4] Setting up optimizer and scheduler...

[4/4] Training setup complete!

Training configuration:
  Epochs: 100
  Batch size: 32
  Learning rate: 0.001
  Gradient clip: 5.0
  Vocab size: 1229

======================================================================
STARTING TRAINING
======================================================================
```

### During Training
```
Epoch 1/100: [progress bar with loss]
  Train Loss: X.XXXX

Running validation...
  Val Loss: X.XXXX
  Val WER: XX.XX%

Example predictions (first 2):
  Ref: MORGEN REGEN WOLKE ...
  Pred: MORGEN REGEN SONNE ...
```

---

## Expected Performance

### Phase I Baseline Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Training WER** | 30-40% | By epoch 50 |
| **Validation WER** | 35-45% | Final baseline |
| **Test WER** | 35-45% | Comparable to Koller et al. |
| **Training time** | 2-3 hours | 5,672 sequences × 100 epochs |
| **Model size** | 12.57 MB | ✅ Under 100MB constraint |

### Training Timeline

- **Epochs 1-10:** Loss decreases rapidly, WER ~80-90%
- **Epochs 10-30:** Steady improvement, WER reaches 50-60%
- **Epochs 30-60:** Slower improvement, WER reaches 40-50%
- **Epochs 60-100:** Fine-tuning, WER stabilizes at 35-45%

---

## Monitoring Training

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir logs/bilstm_baseline_optimized
```

Access at: http://localhost:6006

**Metrics to monitor:**
- `train/loss_step` - Training loss per batch
- `train/loss_epoch` - Average training loss per epoch
- `val/wer` - Validation Word Error Rate
- `val/loss` - Validation loss
- `train/lr` - Learning rate (should decrease when WER plateaus)

### Checkpoints

Models saved in `models/bilstm_baseline/`:
- `checkpoint_latest.pt` - Most recent epoch
- `checkpoint_best.pt` - Best validation WER
- `checkpoint_epoch_XXX.pt` - Periodic saves every 5 epochs

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   batch_size: 32  →  24  →  16
   ```

2. Reduce max_sequence_length:
   ```yaml
   max_sequence_length: 300  →  250  →  200
   ```

3. Enable gradient accumulation (already configured):
   ```yaml
   gradient_accumulation_steps: 2  # Effective batch = 64
   ```

### Issue 2: Very High WER (>80% after 20 epochs)

**Possible causes:**
- Features not normalized properly
- Learning rate too high/low
- CTC blank index mismatch

**Debug steps:**
1. Check first batch decoding (printed during training)
2. Verify vocabulary size matches (1229)
3. Check CTC blank index is 0
4. Try adjusting learning rate: 0.001 → 0.0005 or 0.002

### Issue 3: Training Too Slow

**Expected speed:** ~1-2 minutes per epoch

If slower:
1. Reduce num_workers in config:
   ```yaml
   num_workers: 4  →  2  →  0
   ```

2. Check GPU utilization:
   ```bash
   nvidia-smi  # Should show ~70-90% GPU usage
   ```

3. Reduce logging frequency:
   ```yaml
   log_every_n_steps: 50  →  100
   ```

### Issue 4: NaN Loss

**Symptoms:**
```
Warning: Invalid loss detected: nan
```

**Solutions:**
1. Reduce learning rate:
   ```yaml
   learning_rate: 0.001  →  0.0005
   ```

2. Check gradient clipping (already enabled):
   ```yaml
   gradient_clip_norm: 5.0
   ```

3. Verify feature normalization in dataset

---

## After Training Completes

### 1. Check Results

```bash
# View final results
cat models/bilstm_baseline/training_results.json
```

Expected content:
```json
{
  "best_val_wer": 38.5,
  "test_wer": 40.2,
  "test_loss": 3.45,
  "total_epochs": 100,
  "training_time_hours": 2.5
}
```

### 2. Evaluate Best Model

```python
# Load best checkpoint
checkpoint = torch.load('models/bilstm_baseline/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
test_wer = ...  # Run evaluation
```

### 3. Analyze Predictions

Check example predictions in the training log:
- Are common signs recognized correctly?
- What types of errors occur (substitution/insertion/deletion)?
- Are short sequences better than long ones?

### 4. Next Steps (Phase II)

If baseline achieves 35-45% WER:
- ✅ Proceed to Phase II optimizations
- Add attention mechanisms
- Implement knowledge distillation
- Optimize for real-time inference

If baseline is worse (>50% WER):
- Review feature extraction quality
- Check data preprocessing
- Consider architectural adjustments
- Analyze failure cases

---

## Quick Reference

### File Locations

| Component | Path |
|-----------|------|
| Training script | `src/baseline/train_bilstm_optimized.py` |
| Dataset loader | `src/baseline/dataset.py` |
| Model | `src/models/bilstm_optimized.py` |
| Config | `configs/bilstm_baseline_config.yaml` |
| Vocabulary | `data/baseline_vocabulary/vocabulary.txt` |
| Features | `data/baseline_features/{train,dev,test}/*.npz` |
| Checkpoints | `models/bilstm_baseline/*.pt` |
| Logs | `logs/bilstm_baseline_optimized/` |

### Key Configuration Parameters

```yaml
# Model
input_dim: 66              # Pose features
hidden_dim: 192            # LSTM hidden size
num_layers: 3              # BiLSTM depth
vocab_size: 1229           # Vocabulary
subsample_factor: 2        # Temporal reduction

# Training
batch_size: 32             # Sequences per batch
learning_rate: 0.001       # AdamW LR
gradient_clip_norm: 5.0    # Gradient clipping
num_epochs: 100            # Maximum epochs
early_stopping_patience: 15  # Stop after 15 epochs without improvement

# CTC
blank_index: 0             # <BLANK> token
```

---

## Success Criteria

Training is considered successful if:
- ✅ Training completes without errors
- ✅ Validation WER reaches 35-45%
- ✅ Test WER is within 2-3% of validation WER
- ✅ Model size remains under 100MB (currently 12.57 MB)
- ✅ Training time is reasonable (2-3 hours)

---

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review TensorBoard logs
3. Examine training output for error messages
4. Verify all file paths are correct
5. Check GPU memory with `nvidia-smi`

---

**Status:** Ready to train!
**Expected Duration:** 2-3 hours
**GPU Required:** Yes (CUDA-capable)
**Expected WER:** 35-45%

---

**Last Updated:** 2025-10-20
**Phase:** I - Baseline Development
**Next Milestone:** Phase II - Architecture Optimization
