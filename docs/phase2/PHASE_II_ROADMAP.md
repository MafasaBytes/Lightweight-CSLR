# Phase II Roadmap: Pushing to <45% WER

**Current Status**: Phase I Complete ✅
**Current Performance**: 48.41% WER (Test), 48.87% WER (Val)
**Phase II Goal**: <45% WER (3.4 percentage point improvement)

---

## 🎯 Phase I Achievement Summary

### Performance
- **Baseline**: 59.47% WER (108-dim features)
- **Enhanced**: 48.41% WER (512-dim PCA features)
- **Improvement**: 11.06 pp absolute (18.6% relative)
- **Status**: ✅ Target met (47-50% range)

### What Worked
1. ✅ Face landmarks: ~6.5 pp improvement (critical for grammar)
2. ✅ Temporal derivatives: ~2.5 pp improvement (motion dynamics)
3. ✅ Both hands: ~1.5 pp improvement (two-handed signs)
4. ✅ Normalization: ~0.6 pp improvement (signer-invariant)
5. ✅ Optimal dropout (0.4): Prevented overfitting while maintaining performance

### Model Specifications
- **Architecture**: 4-layer BiLSTM, 256 hidden units
- **Parameters**: 7.47M (28.5 MB)
- **Input**: 512-dim PCA features (99.998% variance)
- **Training**: 79 epochs with early stopping
- **Decoding**: Greedy (argmax)

---

## 📋 Phase II Strategy (3 Weeks)

### Week 1: Quick Win + Foundation (Days 1-7)
**Goal**: Get to 46.5% WER with beam search

#### Days 1-3: Beam Search Decoding
- **Implementation**: Use `ctcdecode` library
- **Configuration**:
  - Beam width: 10 (validate 5, 20)
  - 3-gram language model trained on training transcriptions
  - LM weight: 0.5, word bonus: 0.5
- **Expected**: 48.41% → 46.5% WER (1.9% gain)
- **Risk**: Low (standard technique)

#### Days 4-7: Teacher Model Training (Parallel)
- **Architecture**: 6-layer BiLSTM, 512 hidden units (~30M params)
- **Training**: 4 days on same dataset
- **Expected Teacher WER**: 43-44%
- **Deliverable**: Trained teacher checkpoint

---

### Week 2: High Impact (Days 8-14)
**Goal**: Get to 44.2% WER with distillation

#### Days 8-10: Knowledge Distillation
- **Strategy**: Hybrid loss (CTC + soft targets)
- **Configuration**:
  - Temperature: 5.0
  - Alpha (distillation weight): 0.7
  - Optional feature distillation: 0.1 weight
- **Training**: 2 days student training with frozen teacher
- **Expected**: 46.5% → 44.2% WER (2.3% gain)
- **Risk**: Medium (requires careful tuning)

#### Days 11-12: Lightweight Attention
- **Architecture**: Multi-head attention after final BiLSTM
- **Configuration**:
  - 4 attention heads
  - Applied only to high-variance temporal regions (30%)
  - Rotary positional embeddings
- **Parameters**: +0.5M (total 8M, still under constraint)
- **Expected**: Minimal additional gain (~0.3-0.5%)
- **Risk**: Medium (may not synergize with CTC)

#### Days 13-14: Integration Testing
- Test beam search + distilled model
- Evaluate attention contribution
- Measure inference speed (target: >22 FPS)

---

### Week 3: Optimization & Finalization (Days 15-21)

#### Days 15-17: Best Combination
- Combine most effective approaches
- Hyperparameter fine-tuning
- Ablation studies to isolate contributions

#### Days 18-19: Final Evaluation
- Comprehensive test set evaluation
- Error analysis by sign category
- Inference speed benchmarking
- Memory profiling

#### Days 20-21: Documentation
- Results documentation
- Reproducibility scripts
- Ablation study results
- Prepare for thesis/publication

---

## 🎯 Expected Outcomes

### Conservative Estimate (Likely)
| Improvement | Expected Gain | Cumulative WER |
|-------------|---------------|----------------|
| **Baseline** | - | 48.41% |
| + Beam Search | 1.9% | 46.5% ✓ |
| + Distillation | 2.3% | 44.2% ✓ |
| + Attention | 0.5% | **43.9%** ✓ |

**Final Target**: 43.9% WER (meets <45% goal)

### Optimistic Estimate (Possible)
- If synergies work well: 5.5-6% total gain
- **Final WER: 42.5-43%**

---

## 🔧 Implementation Details

### 1. Beam Search Decoding

**Files to Create**:
- `src/utils/beam_search.py` - Beam search decoder
- `src/baseline/evaluate_beam.py` - Evaluation script with beam search
- `configs/beam_search_config.yaml` - Beam search parameters

**Key Code**:
```python
from ctcdecode import CTCBeamDecoder

# Initialize decoder
decoder = CTCBeamDecoder(
    vocabulary,
    model_path='lm/3gram.arpa',  # Train on training transcriptions
    alpha=0.5,  # LM weight
    beta=0.5,   # Word insertion bonus
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=10,
    num_processes=4,
    blank_id=0
)

# Decode
beam_results, beam_scores, timesteps, out_lens = decoder.decode(
    log_probs.cpu(),  # (T, B, vocab_size)
    seq_lens.cpu()
)
```

**Training 3-gram LM**:
```bash
# Extract training transcriptions
python scripts/extract_transcriptions.py --split train --output lm/train_text.txt

# Train KenLM
lmplz -o 3 < lm/train_text.txt > lm/3gram.arpa
```

---

### 2. Teacher Model Training

**Configuration** (`configs/teacher_model_config.yaml`):
```yaml
model:
  architecture:
    input_dim: 512
    hidden_dim: 512      # Double student size
    num_layers: 6        # 50% more layers
    dropout: 0.3         # Lower dropout for teacher

training:
  batch_size: 16         # Smaller batch for larger model
  gradient_accumulation_steps: 4  # Effective batch = 64
  learning_rate: 0.0003  # Slightly lower LR
  num_epochs: 100
```

**Training Command**:
```bash
python src/baseline/train.py \
    --config configs/teacher_model_config.yaml \
    --output_dir models/teacher_bilstm \
    --experiment_name teacher_512h_6l
```

---

### 3. Knowledge Distillation

**Distillation Loss** (`src/utils/distillation_loss.py`):
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, student_logits, teacher_logits, targets,
                input_lengths, target_lengths):
        # Hard loss (CTC)
        hard_loss = self.ctc_loss(
            student_logits.log_softmax(2),
            targets,
            input_lengths,
            target_lengths
        )

        # Soft loss (distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=2)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=2)

        soft_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

**Training Script**:
```bash
python src/baseline/train_distillation.py \
    --student_config configs/bilstm_enhanced_config.yaml \
    --teacher_checkpoint models/teacher_bilstm/checkpoint_best.pt \
    --temperature 5.0 \
    --alpha 0.7 \
    --output_dir models/distilled_bilstm
```

---

### 4. Lightweight Attention

**Architecture** (`src/models/bilstm_attention.py`):
```python
class SelectiveAttention(nn.Module):
    def __init__(self, hidden_size=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # BiLSTM outputs
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x, lengths):
        # Apply attention only to top 30% variable frames
        variance = x.var(dim=1)
        top_k = int(0.3 * x.size(1))
        _, indices = torch.topk(variance, top_k, dim=1)

        # Selective attention
        x_selected = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        attn_out, _ = self.attention(x_selected, x_selected, x_selected)

        # Scatter back
        x_attended = x.clone()
        x_attended.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, x.size(2)), attn_out)

        return self.norm(x_attended + x)  # Residual connection
```

---

## 📊 Evaluation Protocol

### Metrics to Track
1. **Primary**: WER (Word Error Rate)
2. **Secondary**:
   - CER (Character Error Rate)
   - BLEU-4 score
   - Sign Error Rate (SER)
3. **Computational**:
   - Inference FPS
   - Memory usage (MB)
   - Latency (ms)
   - Model size (MB)

### Statistical Rigor
- Run each experiment 5 times with different seeds
- Report mean ± std
- Compute 95% confidence intervals
- Test significance with paired t-test (p<0.05)

### Ablation Studies
1. Beam width sensitivity (5, 10, 20, 50)
2. Temperature sensitivity (3, 5, 7, 10)
3. Distillation alpha (0.5, 0.7, 0.9)
4. Attention heads (1, 2, 4, 8)
5. Combined vs individual improvements

---

## ⚠️ Risk Management

### High-Priority Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Teacher overfits | Medium | High | Use ensemble of 3 smaller teachers |
| Distillation degrades | Low | High | Reduce temperature, adjust alpha |
| Beam search too slow | Low | Medium | Pre-compute prefix tree, GPU implementation |
| Attention doesn't help | Medium | Low | Try temporal convolution instead |
| OOM with larger models | Medium | Medium | Gradient checkpointing, smaller batch |

### Decision Checkpoints

**Day 3**: If beam search <1% gain → Skip to attention
**Day 10**: If distillation <2% gain → Try ensemble
**Day 14**: If combined >45% WER → Pivot to ensemble

---

## 📚 Research Contribution

### Novel Aspects
1. **Efficient distillation for CTC-SLR**: First comprehensive study
2. **Selective attention for landmarks**: Novel spatial-temporal focus
3. **Modern baseline ablations**: Comprehensive feature analysis

### Publication Strategy
- **Target**: WACV 2026 or ICASSP 2026
- **Title**: "Efficient Knowledge Distillation for Lightweight Continuous Sign Language Recognition"
- **Key Message**: 43-44% WER with 8M params, real-time capable

---

## 🚀 Getting Started

### Immediate Next Steps (Today)

1. **Start teacher training**:
```bash
python src/baseline/train.py --config configs/teacher_model_config.yaml
```

2. **Implement beam search decoder**:
```bash
python scripts/create_lm.py  # Extract transcriptions + train 3-gram
python src/utils/beam_search.py  # Implement decoder
```

3. **Set up experiment tracking**:
```bash
# Create experiment directory
mkdir -p experiments/phase2
mkdir -p experiments/phase2/beam_search
mkdir -p experiments/phase2/distillation
mkdir -p experiments/phase2/attention
```

---

## 📈 Success Criteria

### Minimum Viable (Required)
- ✅ Test WER < 45%
- ✅ Inference FPS > 20
- ✅ Model size < 100 MB

### Target (Expected)
- 🎯 Test WER: 43.5-44.5%
- 🎯 Inference FPS: 22-25
- 🎯 Clear ablation showing each contribution

### Stretch (Aspirational)
- 🌟 Test WER < 43%
- 🌟 Inference FPS > 25
- 🌟 Publishable novel contribution

---

## 📞 Support & Resources

### Documentation
- Phase I Results: `results/VISUALIZATION_SUMMARY.md`
- Training Guide: `TRAINING_READY.md`
- Model Architecture: `src/models/bilstm.py`

### Key Papers
- **Distillation**: Hinton et al. (2015) - Distilling Knowledge in Neural Networks
- **CTC Beam Search**: Graves et al. (2006) - Connectionist Temporal Classification
- **Attention for SLR**: Pu et al. (2020) - Boosting CSLR with Visual Context

### Codebase References
- **CTCDecode**: https://github.com/parlance/ctcdecode
- **KenLM**: https://github.com/kpu/kenlm
- **PyTorch Distillation**: torch.nn.functional.kl_div

---

**Last Updated**: 2025-10-23
**Status**: 🚀 Ready to Start Phase II
**Expected Completion**: 3 weeks (by 2025-11-13)

---

Good luck with Phase II! You're on track for an excellent thesis contribution. 🎉
