# CTC Loss Quick Reference Card

## TL;DR - Your Implementation is CORRECT

**Verdict:** The simplified 2D padded target approach is correct. All tests passed.

```python
# CORRECT (Your current implementation)
loss = criterion(
    log_probs,      # (T, N, C)
    targets,        # (N, S) - 2D padded
    output_lengths, # (N,)
    target_lengths  # (N,)
)
```

---

## Vocabulary Configuration (CORRECT)

```python
vocab = {
    '<PAD>': 0,    # Padding token (only in padded positions)
    '<BLANK>': 1,  # CTC blank token (used by algorithm)
    '<UNK>': 2,    # Unknown signs
    # signs: 3-1119 (actual vocabulary)
}

criterion = nn.CTCLoss(blank=1, zero_infinity=True)
```

---

## Key Test Results

| Test | Result | Meaning |
|------|--------|---------|
| 2D vs 1D | IDENTICAL | Both produce loss = 34.344803 |
| PAD handling | CORRECT | PAD values don't affect loss |
| BLANK config | CORRECT | blank=1 is properly configured |
| Edge cases | PASSED | Single token, max length, variable lengths work |
| Gradients | HEALTHY | No NaN/Inf, proper backprop |

---

## What Changed?

### Before (Unnecessary Complexity):
```python
# Flatten targets for CTC
targets_flat = []
for i in range(targets.size(0)):
    targets_flat.append(targets[i, :target_lengths[i]])
targets_flat = torch.cat(targets_flat)

loss = criterion(log_probs, targets_flat, output_lengths, target_lengths)
```

### After (Simplified - RECOMMENDED):
```python
# CTC loss with 2D targets
loss = criterion(log_probs, targets, output_lengths, target_lengths)
```

**Benefit:**
- Simpler (1 line vs 5 lines)
- Faster (no loop overhead)
- Same result (verified identical)

---

## How It Works

CTCLoss uses `target_lengths` to determine valid sequence boundaries:

```python
targets = torch.tensor([
    [3, 4, 5, 0, 0],  # length=3 → uses [3,4,5], ignores [0,0]
    [6, 7, 0, 0, 0]   # length=2 → uses [6,7], ignores [0,0,0]
])
target_lengths = torch.tensor([3, 2])
```

Padding values can be anything (0, 999, -1) - they're ignored.

---

## Important Rules

1. **PAD token (0):**
   - Use ONLY in padding positions of 2D tensors
   - NEVER in actual target sequences

2. **BLANK token (1):**
   - Used internally by CTC algorithm
   - NEVER in target sequences (CTC inserts automatically)

3. **Actual targets:**
   - Should only contain indices 3-1119 (sign vocabulary)
   - Must satisfy: `targets[:, :target_length].min() >= 3`

4. **Sequence length constraint:**
   - CTC requires: `output_length >= 2 * target_length + 1`
   - Your max_sequence_length=241 should be sufficient

---

## Validation Commands

### Run comprehensive tests:
```bash
python test_ctc_loss.py
```

### Check for issues in your data:
```python
# In your dataset or training loop:
assert targets[:, :target_length].min() >= 3, "PAD/BLANK in actual sequence"
assert not torch.isnan(loss), "NaN loss detected"
assert not torch.isinf(loss), "Inf loss detected"
```

---

## No Action Required

Your current implementation is correct. You can proceed with training.

**Files:**
- Training: `src/train_bilstm.py` ✓ Correct
- Model: `src/models/bilstm.py` ✓ Correct
- Tests: `test_ctc_loss.py` ✓ All passed
- Analysis: `CTC_LOSS_ANALYSIS.md` (detailed report)

**Status:** VALIDATED - Ready for training

---

## If You See Issues During Training

### Loss is NaN or Inf:
- Check sequence length constraint: T >= 2*S + 1
- Verify gradient clipping is enabled (already is: grad_clip=5.0)
- Check for invalid target indices

### Loss not decreasing:
- Verify targets don't contain PAD (0) or BLANK (1)
- Check learning rate (current: 1e-4)
- Try overfitting single batch (sanity check)

### Error messages:
- "Expected tensor for argument #1" → Check tensor shapes
- "targets values must be in range" → Check vocab indices
- "target length is longer than input length" → Check T >= 2*S + 1

---

**Date:** 2025-10-15
**Status:** ✓ VALIDATED
