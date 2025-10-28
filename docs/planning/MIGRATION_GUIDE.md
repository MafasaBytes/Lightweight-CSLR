# BiLSTM Architecture Migration Guide

## Quick Start: Migrating from Original to Optimized BiLSTM

### 1. Update Model Import

**Before:**
```python
from src.models.bilstm import BiLSTMModel, create_bilstm_model

model = create_bilstm_model(
    input_dim=177,      # Wrong dimension
    hidden_dim=256,
    num_layers=2,
    vocab_size=1120,    # Wrong vocab size
    dropout=0.3
)
```

**After:**
```python
from src.models.bilstm_optimized import OptimizedBiLSTMModel, create_optimized_bilstm_model

model = create_optimized_bilstm_model(
    input_dim=66,       # Correct: MediaPipe pose features
    hidden_dim=192,     # Optimized for efficiency
    num_layers=3,       # Deeper for better temporal modeling
    vocab_size=1229,    # Correct: actual vocabulary size
    dropout=0.3,
    projection_dim=128,
    subsample_factor=2  # Reduces CTC computation
)
```

### 2. Key Parameter Changes

| Parameter | Original | Optimized | Reason |
|-----------|----------|-----------|---------|
| `input_dim` | 177 | **66** | Actual pose feature dimension |
| `vocab_size` | 1120 | **1229** | Actual vocabulary size |
| `hidden_dim` | 256 | **192** | Better efficiency with 3 layers |
| `num_layers` | 2 | **3** | Improved temporal modeling |
| `projection_dim` | N/A | **128** | Gradual feature scaling |
| `subsample_factor` | N/A | **2** | 2x faster CTC computation |

### 3. Training Configuration

Use the provided configuration file:
```yaml
# configs/bilstm_baseline_config.yaml
model:
  architecture:
    input_dim: 66
    hidden_dim: 192
    num_layers: 3
    vocab_size: 1229
    dropout: 0.3
    projection_dim: 128
    subsample_factor: 2

training:
  optimizer:
    type: "AdamW"
    learning_rate: 0.001
  gradient_clip_norm: 5.0  # Important for stability
```

### 4. Output Differences

**Original Model:**
- Output shape: `(T, B, V)` where T = input sequence length
- No subsampling, full temporal resolution

**Optimized Model:**
- Output shape: `(T//2, B, V)` due to temporal subsampling
- Remember to adjust target sequence lengths accordingly:

```python
# Training loop adjustment
log_probs, output_lengths = model(features, input_lengths)
# output_lengths are automatically adjusted by the model
# Use these for CTC loss computation
```

### 5. CTC Loss Compatibility

Both models output in the correct format for CTC loss:
```python
ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# Both models work the same way:
loss = ctc_loss(
    log_probs,      # (T, B, V) format
    targets,        # Target sequences
    output_lengths, # From model.forward()
    target_lengths  # Actual target lengths
)
```

### 6. Performance Improvements

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Model Size | 10.95 MB | 12.57 MB | +14.8% (acceptable) |
| Inference Speed | 24.58 ms | 21.82 ms | 13% faster |
| Memory Usage (batch=32) | 30.9 MB | 24.3 MB | **21% less** |
| CTC Computation | Full sequence | Half sequence | **2x faster** |
| Training Stability | Basic | Enhanced | Layer norm + residuals |

### 7. Backward Compatibility

If you need to load weights from an old model:
```python
# Not directly compatible due to architecture changes
# You'll need to retrain from scratch with the optimized model
# This is recommended anyway for the baseline phase
```

### 8. Common Issues and Solutions

**Issue 1: Dimension Mismatch**
- Ensure input features are 66-dimensional (pose only)
- Check vocabulary size is 1229

**Issue 2: Output Length Confusion**
- Remember outputs are subsampled by factor of 2
- Use `output_lengths` from model, not input lengths

**Issue 3: Memory Issues**
- Optimized model uses LESS memory despite more parameters
- Can increase batch size from 32 to 40+ with same VRAM

### 9. Training Script Template

```python
import torch
from src.models.bilstm_optimized import create_optimized_bilstm_model

# Create model
model = create_optimized_bilstm_model(
    input_dim=66,
    vocab_size=1229,
    device='cuda'
)

# Setup optimizer with gradient clipping
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

# CTC Loss
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        features, targets, feat_lengths, target_lengths = batch

        # Forward pass
        log_probs, output_lengths = model(features, feat_lengths)

        # Compute loss
        loss = ctc_loss(log_probs, targets, output_lengths, target_lengths)

        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
```

### 10. Next Steps

1. **Test the optimized model**: Run `python src/models/bilstm_optimized.py`
2. **Compare architectures**: Run `python scripts/compare_architectures.py`
3. **Start training**: Use the optimized model with provided config
4. **Monitor performance**: Target 35-45% WER for baseline

## Summary

The optimized BiLSTM model provides:
- **Correct dimensions** for your actual features (66) and vocabulary (1229)
- **Better training stability** through layer normalization and residual connections
- **Faster training** with 2x temporal subsampling for CTC
- **Lower memory usage** despite having more layers
- **Production-ready** architecture following best practices

Migrate to the optimized model for your Phase I baseline development.