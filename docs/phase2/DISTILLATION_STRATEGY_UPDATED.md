# Updated Distillation Strategy - Self-Distillation

**Date**: 2025-10-24
**Issue**: Teacher (5L-384H) performing worse than student (4L-256H)
**Solution**: Use self-distillation instead of teacher-student

---

## Problem Summary

### Teacher Training Results
- **Attempt 1**: 51.23% WER (57 epochs)
- **Attempt 2**: 51.18% WER (68 epochs)
- **Student**: 48.41% WER ✅ (better!)

### Why Teacher Failed
1. **Larger model harder to train**: 15.2M vs 7.5M params
2. **Training from scratch**: No good initialization
3. **Dimension mismatch**: Can't transfer weights (256 vs 384 hidden)
4. **Early stopping**: Stopped before convergence

### Conclusion
Training a larger teacher that outperforms the student is **not feasible** with current resources and time.

---

## New Strategy: Self-Distillation

Instead of teacher-student, use **self-distillation** where the model learns from its own predictions.

### What is Self-Distillation?

The student model serves as its own teacher:
1. **Forward pass**: Get predictions from current model
2. **Soft targets**: Use model's own probability distribution
3. **Hard targets**: Use ground truth labels
4. **Combined loss**: Mix both with temperature scaling

### Benefits
1. ✅ **No separate teacher needed**: Use existing 48.41% WER model
2. ✅ **Proven technique**: Self-distillation works well in practice
3. ✅ **Label smoothing effect**: Soft targets regularize learning
4. ✅ **Faster training**: Only one model to train

### Research Backing
- "Be Your Own Teacher" (Zhang et al., 2019)
- "Self-Distillation Amplifies Regularization in Hilbert Space" (Mobahi et al., 2020)
- Used successfully in BERT, ResNet, and other SOTA models

---

## Implementation Approach

### Option A: Self-Distillation (Recommended)

Train a **new student** that learns from the **current best student**:

```python
# Pseudo-code
teacher = load_checkpoint("models/bilstm_baseline/checkpoint_best.pt")
teacher.eval()  # Freeze

student = create_fresh_model(same_architecture)  # 4L-256H
student.train()

for batch in dataloader:
    # Teacher predictions (soft targets)
    with torch.no_grad():
        teacher_logits = teacher(batch)
        teacher_probs = softmax(teacher_logits / temperature)

    # Student predictions
    student_logits = student(batch)
    student_probs = softmax(student_logits / temperature)

    # Combined loss
    loss = alpha * KL_div(student_probs, teacher_probs) + \\
           beta * CTC_loss(student_logits, ground_truth)
```

**Expected outcome**: 46.5-47.5% WER (1-2 pp improvement)

### Option B: Label Smoothing + Mixup

Use data augmentation instead of distillation:
- Label smoothing: 0.1
- Mixup: α=0.2
- Curriculum learning

**Expected outcome**: 47.0-47.5% WER (1.0-1.5 pp improvement)

### Option C: Ensemble Methods

Train 3-5 student models with different:
- Random seeds
- Dropout rates
- Learning rates

Average their predictions at inference.

**Expected outcome**: 46.0-47.0% WER (1.5-2.5 pp improvement)

---

## Recommended Path Forward

### Phase II Week 2 (Revised)

**Day 1-2: Self-Distillation Training**

1. **Modify distillation config**:
```yaml
distillation:
  teacher_checkpoint: "models/bilstm_baseline/checkpoint_best.pt"  # Use student as teacher
  student_checkpoint: null  # Train from scratch

  model:  # Same architecture as teacher
    architecture:
      input_dim: 512
      hidden_dim: 256  # Same as teacher (not 384)
      num_layers: 4    # Same as teacher (not 5)
```

2. **Train with self-distillation**:
```bash
python src/baseline/train_distill.py --config configs/self_distillation_config.yaml
```

3. **Expected**: 1-2 days training, WER 46.5-47.5%

**Day 3: Combine with Beam Search**

```bash
python src/baseline/evaluate_beam.py \\
    --checkpoint models/distilled_student/checkpoint_best.pt \\
    --split test \\
    --lm_weight 0.9
```

**Expected**: 45.0-45.5% WER (combined improvement)

---

## Updated Timeline

### Original Plan (Failed)
- Day 1-3: Train teacher (5L-384H) → 45.5% WER
- Day 4-7: Distill to student → 45.0% WER
- **Total**: 7 days

### Revised Plan (Achievable)
- Day 1-2: Self-distillation (4L-256H) → 46.5-47.5% WER
- Day 3: Beam search evaluation → 45.0-45.5% WER
- **Total**: 3 days

**Time saved**: 4 days
**Risk**: Lower
**Expected result**: Similar or better

---

## Alternative: Skip Distillation, Focus on Beam Search

Since we already have **45.99% WER** with beam search, we could:

1. **Optimize beam search further**:
   - Try different LM weights (0.5-1.5)
   - Larger beam widths (20, 50)
   - Better language model (use lmplz properly)

   **Expected**: 44.5-45.5% WER

2. **Add simple attention mechanism**:
   - Selective attention on LSTM outputs
   - 1-2 days implementation

   **Expected**: 44.0-45.0% WER

3. **Ensemble current model**:
   - Use different checkpoints (epochs 70, 75, 79)
   - Average predictions

   **Expected**: 44.5-45.5% WER

---

## Recommendation

**Option 1: Self-Distillation** (if thesis requires distillation)
- Aligns with proposal
- Legitimate research technique
- Expected: 46.5-47.5% greedy, 45.0-45.5% beam

**Option 2: Optimize Beam Search** (fastest, least risk)
- We already have 45.99%
- Further tune hyperparameters
- Expected: 44.5-45.5% beam

**Option 3: Add Attention** (novel contribution)
- Lightweight selective attention
- Good for thesis novelty
- Expected: 44.0-45.0% greedy

---

## Next Steps

**Decision Point**: Which approach to take?

1. **Self-distillation** → Create `configs/self_distillation_config.yaml`
2. **Beam search optimization** → Hyperparameter sweep
3. **Attention mechanism** → Implement selective attention layer

**All three** can achieve the target <45% WER in different ways.

Which would you prefer for the thesis?

---

**Last Updated**: 2025-10-24
**Status**: Awaiting decision on distillation strategy
**Current Best**: 45.99% WER (beam search)
**Target**: <45% WER
