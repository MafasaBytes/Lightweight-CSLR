# Week 2: Knowledge Distillation Implementation Guide

**Date Started**: 2025-10-23
**Status**: Ready to Begin
**Expected Duration**: 6.5-8.5 days
**Current WER**: 45.99% (with beam search alpha=0.9)
**Target WER**: 45.5-46.8% WER (conservative) or 44.5-45.5% (optimistic)

---

## Quick Start

### Step 1: Train Teacher Model (START NOW)

```bash
# Activate environment
.\venv\Scripts\activate

# Train teacher model (5-layer, 384 hidden)
python src/baseline/train.py --config configs/teacher_config.yaml

# This will take 2.5-3 days
# Expected teacher WER: 45.5-46.5%
```

### Step 2: Verify Teacher Performance (After Training)

```bash
# Evaluate teacher on test set
python src/baseline/evaluate.py \
    --checkpoint models/teacher_bilstm/checkpoint_best.pt \
    --split test

# Expected: Test WER < 47%
# If WER >= 47%, extend training or adjust hyperparameters
```

### Step 3: Implement Distillation Components (While Teacher Trains)

You can work on implementation while teacher is training. I'll create these files for you:
1. `src/models/distillation_loss.py` - Distillation loss module
2. `src/baseline/train_distill.py` - Distillation training script
3. Modify `src/models/bilstm.py` - Add hidden state extraction

### Step 4: Train with Distillation (After Teacher Complete)

```bash
# Train student with teacher guidance
python src/baseline/train_distill.py --config configs/distillation_config.yaml

# This will take 3-4 days
# Expected distilled WER: 45.5-46.8%
```

### Step 5: Final Evaluation with Beam Search

```bash
# Evaluate distilled model with optimal beam search
python src/baseline/evaluate_beam.py \
    --checkpoint models/distilled_student/checkpoint_best.pt \
    --split test \
    --lm_weight 0.9 \
    --compare_greedy

# Expected: Test WER 44.5-45.5% with beam search
```

---

## Architecture Overview

### Teacher Model (NEW)
- **Architecture**: 5-layer BiLSTM, 384 hidden units
- **Parameters**: ~15.2M (2× student)
- **Model Size**: ~58 MB
- **Purpose**: Learn better representations to transfer to student
- **Expected Performance**: 45.5-46.5% WER (greedy)

### Student Model (EXISTING)
- **Architecture**: 4-layer BiLSTM, 256 hidden units
- **Parameters**: 7.47M (unchanged)
- **Model Size**: 28.5 MB (unchanged)
- **Current Performance**: 48.41% WER (greedy), 45.99% WER (beam)
- **Target After Distillation**: 45.5-46.8% WER (greedy)

---

## Knowledge Distillation Strategy

### Loss Function

```
L_total = 0.7 × L_distill + 0.2 × L_hard + 0.1 × L_feature
```

**Component 1: Soft Target Distillation (70%)**
- Frame-level KL divergence between teacher and student predictions
- Temperature scaling: T=4.0 → 3.5 → 3.0 (annealed)
- Teaches student the "soft" probability distributions

**Component 2: Hard CTC Loss (20%)**
- Standard CTC loss with ground truth labels
- Maintains connection to actual targets
- Prevents drift from correct answers

**Component 3: Feature Matching (10%)**
- MSE loss on intermediate LSTM hidden states
- Teacher layers 2&4 → Student layers 2&3
- Aligns internal representations

### Temperature Annealing

```
Epochs 1-30:   T = 4.0  (smooth distributions)
Epochs 31-70:  T = 3.5  (medium smoothing)
Epochs 71-100: T = 3.0  (sharper distributions)
```

---

## Implementation Files Created

### ✅ Configuration Files (READY)

1. **`configs/teacher_config.yaml`** ✅
   - Teacher architecture (5L, 384H)
   - Training hyperparameters
   - Expected to train in 2.5-3 days

2. **`configs/distillation_config.yaml`** ✅
   - Distillation loss weights
   - Temperature schedule
   - Feature matching configuration

### 📋 To Be Created (While Teacher Trains)

3. **`src/models/distillation_loss.py`**
   - CTCDistillationLoss class
   - Temperature annealing logic
   - Feature matching loss

4. **`src/baseline/train_distill.py`**
   - Distillation training loop
   - Dual model management (teacher + student)
   - Loss computation and logging

5. **Modification to `src/models/bilstm.py`**
   - Add `forward_with_hidden_states()` method
   - Extract intermediate LSTM outputs

---

## Expected Timeline

### Phase 1: Teacher Training (2.5-3.5 days) ⏰
- **Start**: Now
- **Action**: Run `python src/baseline/train.py --config configs/teacher_config.yaml`
- **Monitoring**: Check TensorBoard every 12 hours
- **Completion**: When dev WER < 47%
- **Deliverable**: `models/teacher_bilstm/checkpoint_best.pt`

### Phase 2: Implementation (0.5 day) - PARALLEL
- **Start**: While teacher trains
- **Action**: Create distillation loss, training script, modify model
- **Deliverable**: Ready-to-run distillation training script

### Phase 3: Distillation Training (3-4 days)
- **Start**: After teacher complete
- **Action**: Run `python src/baseline/train_distill.py --config configs/distillation_config.yaml`
- **Monitoring**: Check dev WER every 10 epochs, compare to teacher
- **Completion**: When dev WER stops improving (60-80 epochs)
- **Deliverable**: `models/distilled_student/checkpoint_best.pt`

### Phase 4: Evaluation (0.5 day)
- **Action**: Test set evaluation with beam search
- **Expected**: 44.5-45.5% WER (beam search, alpha=0.9)
- **Deliverable**: Final results report

**Total Calendar Time**: 6.5-8.5 days (7 days realistic)

---

## Monitoring & Success Criteria

### Teacher Training Checkpoints

**Day 1** (After 24 hours):
- Check: Dev WER should be < 52%
- If not: Verify data loading, check for NaN losses

**Day 2** (After 48 hours):
- Check: Dev WER should be < 49%
- If not: May need to extend training

**Day 3** (Completion):
- **Success**: Dev WER < 47%, Test WER < 47.5%
- **Proceed**: Start distillation
- **Failure**: Dev WER >= 47%
  - Extend training 20 more epochs OR
  - Adjust dropout to 0.2 and retrain

### Distillation Training Checkpoints

**After 20 epochs**:
- Check: Student dev WER should be < 47.5%
- Loss components should all be decreasing

**After 50 epochs**:
- Check: Student dev WER should be < 47%
- At least 1% improvement over baseline

**After 80 epochs**:
- **Success**: Student dev WER < 46.5%
- **Target met**: Proceed to final evaluation
- **Needs tuning**: Adjust alpha to 0.8, extend 20 epochs

### Final Evaluation Success Criteria

- [ ] Test WER (greedy) < 47%
- [ ] Test WER (beam search) < 45.5%
- [ ] Model size unchanged (28.5 MB)
- [ ] Inference speed unchanged (~30 FPS)
- [ ] Improvement consistent across dev and test

---

## Current Progress Summary

### Week 1 Results (COMPLETED) ✅
- **Baseline student**: 48.41% WER (greedy)
- **With beam search**: 47.00% WER (alpha=0.5)
- **Tuned beam search**: **45.99% WER** (alpha=0.9) ⭐
- **Improvement**: 2.42 pp from baseline

### Week 2 Goals (IN PROGRESS)
- **Teacher training**: Target 45.5-46.5% WER
- **Distillation**: Target 45.5-46.8% WER (greedy)
- **With beam search**: Target 44.5-45.5% WER
- **Total improvement**: 3.5-4.0 pp from baseline

---

## Troubleshooting Guide

### Problem: Teacher Not Converging

**Symptoms**: Teacher WER > 47% after 100 epochs

**Solutions**:
1. Extend training to 140 epochs
2. Reduce dropout from 0.3 to 0.2
3. Lower learning rate to 0.0002
4. Check data augmentation (disable if enabled)

### Problem: OOM During Teacher Training

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch_size from 20 to 16
2. Increase gradient_accumulation_steps to 5
3. Enable gradient checkpointing in config
4. Disable feature caching

### Problem: Distillation Not Improving

**Symptoms**: Student WER not decreasing after 30 epochs

**Solutions**:
1. Increase alpha from 0.7 to 0.8
2. Reduce beta from 0.2 to 0.1
3. Lower learning rate to 0.0001
4. Increase temperature from 4.0 to 5.0

### Problem: Student Worse Than Teacher

**Symptoms**: Student WER > Teacher WER + 1%

**Root Cause**: Student not learning from teacher effectively

**Solutions**:
1. Reduce learning rate (prevent catastrophic forgetting)
2. Increase distillation weight alpha to 0.85
3. Check teacher checkpoint is loaded correctly
4. Verify temperature scaling is applied

---

## Next Immediate Actions

### Action 1: Start Teacher Training (DO NOW) 🚀

```bash
# Make sure you're in the right directory
cd C:\Users\Masia\OneDrive\Desktop\sign-language-recognition

# Activate environment
.\venv\Scripts\activate

# Start training
python src/baseline/train.py --config configs/teacher_config.yaml
```

**Monitor with TensorBoard**:
```bash
# In another terminal
tensorboard --logdir logs
# Open browser to http://localhost:6006
```

### Action 2: Verify Setup (WHILE TRAINING)

```bash
# Check that teacher is training
# Look for output like:
# Epoch 1/120: train_loss=..., val_wer=...

# Verify checkpoint directory exists
ls models/teacher_bilstm/

# Check TensorBoard is logging
ls logs/teacher_bilstm_5L_384H/
```

### Action 3: Parallel Implementation Work

While teacher trains (next 2-3 days), I can create:
1. Distillation loss module
2. Training script for distillation
3. Model modifications for hidden state extraction
4. Unit tests for distillation components

**Would you like me to create these implementation files now?**

---

## Research Context

### Why Knowledge Distillation?

**Teacher-Student Framework**:
- Large teacher model learns rich representations
- Small student model mimics teacher's behavior
- Result: Student performance approaches teacher (compression)

**Advantages**:
1. **No architecture change**: Student stays lightweight (7.5M params)
2. **Better than training alone**: Teacher acts as strong regularizer
3. **Proven technique**: 2-3% WER improvement typical
4. **Edge-compatible**: Student remains deployable (<100 MB)

### Key Insight for CTC Models

Standard distillation uses softmax predictions, but CTC uses frame-level log probabilities. Our approach:
- Apply temperature scaling to CTC log-probs
- Use KL divergence at frame level (not sequence level)
- Add feature matching for intermediate representations

This is **novel for sign language recognition** - most distillation work is on classification tasks.

---

## References

### Distillation Papers
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Romero et al. (2015): "FitNets: Hints for Thin Deep Nets" (feature matching)
- Polino et al. (2018): "Model compression via distillation and quantization"

### CTC-Specific Distillation
- Kim & Rush (2016): "Sequence-Level Knowledge Distillation"
- Fukuda et al. (2017): "Efficient Knowledge Distillation from an Ensemble of Teachers"

---

## Contact & Support

### Files Created So Far
- ✅ `configs/teacher_config.yaml`
- ✅ `configs/distillation_config.yaml`
- ✅ `WEEK2_KNOWLEDGE_DISTILLATION_GUIDE.md` (this file)

### Next Files to Create
- ⏳ `src/models/distillation_loss.py`
- ⏳ `src/baseline/train_distill.py`
- ⏳ Modifications to `src/models/bilstm.py`

**Ready to proceed with implementation?** Let me know and I'll create all the remaining files!

---

**Last Updated**: 2025-10-23
**Status**: 🚀 Ready to start teacher training
**Next Milestone**: Teacher converged (<47% WER) in 2.5-3 days
