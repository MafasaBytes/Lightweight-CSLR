# Beam Search Implementation Guide

**Goal**: Improve WER from 48.41% to ~46.5% using beam search decoding
**Status**: Step 1 Complete ✅
**Estimated Time**: 2-3 hours

---

## Current Status

✅ **Step 1 Complete**: Training transcriptions extracted
- **File**: `lm/train_text.txt`
- **Sentences**: 5,672
- **Total words**: 65,227
- **Unique vocabulary**: 1,231 signs
- **Avg words/sentence**: 11.5

**Best Model**: `models/bilstm_baseline/checkpoint_best.pt`
- **Current WER**: 48.41% (greedy decoding)
- **Target WER**: 46.5% (beam search decoding)
- **Expected gain**: 1.9 percentage points

---

## Step-by-Step Implementation

### Step 2: Install KenLM (Language Model Toolkit)

KenLM is a fast toolkit for training n-gram language models.

**Option A: Windows (Recommended - Pre-compiled)**
```bash
# Install via pip (if available)
pip install kenlm

# OR download pre-built binaries from:
# https://github.com/kpu/kenlm/releases
```

**Option B: Build from Source (if pip fails)**
```bash
# Prerequisites: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Clone and build
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build
cd build
cmake ..
cmake --build . --config Release

# Executables will be in: kenlm\build\bin\Release\
```

**Verification**:
```bash
lmplz --help  # Should show usage information
```

---

### Step 3: Train 3-gram Language Model

Once KenLM is installed, train the language model on the extracted transcriptions.

**Command**:
```bash
# Create output directory
mkdir -p lm

# Train 3-gram model (takes ~1-2 minutes)
lmplz -o 3 --text lm/train_text.txt --arpa lm/3gram.arpa --discount_fallback

# Parameters:
#   -o 3              : Order 3 (trigram)
#   --text            : Input text file
#   --arpa            : Output ARPA format file
#   --discount_fallback : Handle low-count n-grams gracefully
```

**Expected Output**:
```
=== 1/5 Counting and sorting n-grams ===
Reading lm/train_text.txt
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
Unique 1-grams: 1231
Unique 2-grams: 15234
Unique 3-grams: 35678
=== 2/5 Calculating and sorting adjusted counts ===
...
```

**Verification**:
```bash
# Check file was created
ls -lh lm/3gram.arpa

# Should show file size ~5-10 MB
```

---

### Step 4: Install ctcdecode Library

ctcdecode provides fast beam search decoding with language model integration for CTC.

**Installation**:
```bash
# Activate virtual environment first
.\venv\Scripts\activate

# Install ctcdecode
pip install ctcdecode

# If above fails, try building from source:
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
```

**Common Issues**:
- **Error: "Microsoft Visual C++ required"**
  - Install Visual Studio Build Tools
  - Download: https://visualstudio.microsoft.com/downloads/

- **Error: "CUDA not found"**
  - ctcdecode will still work on CPU
  - Or install CUDA toolkit matching your PyTorch version

**Verification**:
```bash
python -c "from ctcdecode import CTCBeamDecoder; print('ctcdecode installed successfully!')"
```

---

### Step 5: Implement Beam Search Evaluation Script

Create `src/baseline/evaluate_beam.py` to evaluate the model with beam search.

**Key Components**:

1. **Load vocabulary** (same as training)
2. **Initialize CTCBeamDecoder** with language model
3. **Load trained model** (checkpoint_best.pt)
4. **Run inference** with beam search
5. **Calculate WER** and compare to greedy

**Script Structure**:
```python
from ctcdecode import CTCBeamDecoder
import torch
from pathlib import Path
from src.models.bilstm import OptimizedBiLSTMModel
from src.baseline.dataset import PhoenixDataset
from src.baseline.evaluate import calculate_wer

def create_beam_decoder(vocabulary_path, lm_path):
    """Initialize beam search decoder with language model."""

    # Load vocabulary
    with open(vocabulary_path, 'r', encoding='utf-8') as f:
        labels = [line.strip().split('\t')[0] for line in f]

    # Initialize decoder
    decoder = CTCBeamDecoder(
        labels,
        model_path=lm_path,          # Path to 3gram.arpa
        alpha=0.5,                   # LM weight (tune this!)
        beta=0.5,                    # Word insertion bonus
        cutoff_top_n=40,             # Vocabulary cutoff
        cutoff_prob=1.0,             # Probability cutoff
        beam_width=10,               # Beam size (try 5, 10, 20)
        num_processes=4,             # Parallel decoding
        blank_id=0,                  # <BLANK> token
        log_probs_input=True         # Expects log probabilities
    )

    return decoder, labels

def evaluate_with_beam_search(model, dataloader, decoder, labels, device):
    """Evaluate model using beam search decoding."""

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets']
            input_lengths = batch['input_lengths']

            # Get model outputs (log probabilities)
            log_probs = model(features, input_lengths)
            # Shape: (T, B, vocab_size)

            # Beam search decoding
            beam_results, beam_scores, timesteps, out_lens = decoder.decode(
                log_probs.cpu(),
                input_lengths.cpu()
            )

            # Extract best hypothesis (beam_results[:, 0, :])
            for i in range(beam_results.size(0)):
                # Get prediction tokens
                pred_tokens = beam_results[i, 0, :out_lens[i, 0]]
                pred_text = ' '.join([labels[t] for t in pred_tokens])

                # Get target text
                target_tokens = targets[i]
                target_text = ' '.join([labels[t] for t in target_tokens if t > 0])

                all_predictions.append(pred_text)
                all_targets.append(target_text)

    # Calculate WER
    wer = calculate_wer(all_predictions, all_targets)
    return wer, all_predictions, all_targets

def main():
    # Configuration
    checkpoint_path = "models/bilstm_baseline/checkpoint_best.pt"
    vocabulary_path = "data/baseline_vocabulary/vocabulary.txt"
    lm_path = "lm/3gram.arpa"
    features_dir = "data/features_enhanced"

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = OptimizedBiLSTMModel(
        input_dim=512,
        hidden_dim=256,
        num_layers=4,
        vocab_size=1229
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Create test dataloader
    test_dataset = PhoenixDataset(
        features_dir=features_dir,
        vocabulary_path=vocabulary_path,
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize beam decoder
    decoder, labels = create_beam_decoder(vocabulary_path, lm_path)

    # Evaluate
    print("Evaluating with beam search (beam_width=10)...")
    wer, predictions, targets = evaluate_with_beam_search(
        model, test_loader, decoder, labels, device
    )

    print(f"\nResults:")
    print(f"  Greedy WER: 48.41%")
    print(f"  Beam Search WER: {wer:.2f}%")
    print(f"  Improvement: {48.41 - wer:.2f} percentage points")

    # Save predictions
    output_path = Path("results/evaluation/beam_search_predictions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'wer': wer,
            'predictions': predictions,
            'targets': targets,
            'config': {
                'beam_width': 10,
                'alpha': 0.5,
                'beta': 0.5,
                'lm_order': 3
            }
        }, f, indent=2)

    print(f"\nPredictions saved to: {output_path}")

if __name__ == "__main__":
    main()
```

---

### Step 6: Run Beam Search Evaluation

```bash
# Make sure virtual environment is activated
.\venv\Scripts\activate

# Run evaluation
python src/baseline/evaluate_beam.py

# Expected output:
# Evaluating with beam search (beam_width=10)...
# Processing test set: 100%|████████████| 629/629
#
# Results:
#   Greedy WER: 48.41%
#   Beam Search WER: 46.52%
#   Improvement: 1.89 percentage points
```

---

## Hyperparameter Tuning (Optional)

If initial results are good, try tuning these parameters:

### Beam Width
```python
# Try different beam widths
for beam_width in [5, 10, 20, 50]:
    decoder = CTCBeamDecoder(..., beam_width=beam_width, ...)
    wer = evaluate(...)
    print(f"Beam width {beam_width}: {wer:.2f}% WER")
```

**Expected**:
- beam_width=5: ~47.0% WER (faster)
- beam_width=10: ~46.5% WER (balanced)
- beam_width=20: ~46.3% WER (diminishing returns)
- beam_width=50: ~46.2% WER (slow, minimal gain)

### Language Model Weight (alpha)
```python
# Try different LM weights
for alpha in [0.3, 0.5, 0.7, 1.0]:
    decoder = CTCBeamDecoder(..., alpha=alpha, ...)
    wer = evaluate(...)
    print(f"Alpha {alpha}: {wer:.2f}% WER")
```

**Expected**:
- alpha=0.3: ~47.2% WER (weak LM influence)
- alpha=0.5: ~46.5% WER (balanced)
- alpha=0.7: ~46.8% WER (too strong)
- alpha=1.0: ~47.5% WER (over-reliance on LM)

### Word Insertion Bonus (beta)
```python
# Try different word bonuses
for beta in [0.0, 0.5, 1.0, 1.5]:
    decoder = CTCBeamDecoder(..., beta=beta, ...)
    wer = evaluate(...)
    print(f"Beta {beta}: {wer:.2f}% WER")
```

---

## Troubleshooting

### Issue: "ctcdecode not found"
**Solution**: Build from source or use Python 3.7-3.9 (best compatibility)

### Issue: "Language model file not found"
**Solution**: Check `lm/3gram.arpa` exists and path is correct

### Issue: "CUDA out of memory"
**Solution**:
- Reduce batch size in evaluation
- Use CPU for decoding (slower but works)
- Reduce beam width

### Issue: "WER not improving"
**Solutions**:
1. Check vocabulary matches training
2. Verify language model trained correctly
3. Tune alpha/beta parameters
4. Try different beam widths

---

## Expected Timeline

| Step | Task | Time | Status |
|------|------|------|--------|
| 1 | Extract transcriptions | 2 min | ✅ Complete |
| 2 | Install KenLM | 10-30 min | ⏳ Next |
| 3 | Train 3-gram LM | 2 min | ⏳ Pending |
| 4 | Install ctcdecode | 5-15 min | ⏳ Pending |
| 5 | Implement evaluation | 30 min | ⏳ Pending |
| 6 | Run evaluation | 5-10 min | ⏳ Pending |
| 7 | Hyperparameter tuning | 30 min | ⏳ Optional |

**Total**: 1.5 - 2.5 hours

---

## Success Criteria

✅ **Minimum Viable**: WER < 47.5% (at least 0.9 pp gain)
🎯 **Target**: WER ~46.5% (1.9 pp gain as per roadmap)
🌟 **Stretch**: WER < 46.0% (2.4+ pp gain)

---

## Next Steps After Beam Search

Once beam search is working (expected: 46.5% WER), proceed to:

1. **Week 2: Knowledge Distillation**
   - Train larger teacher model (512 hidden, 6 layers)
   - Distill knowledge to student (target: 44.2% WER)
   - Expected gain: 2.3 pp

2. **Week 3: Lightweight Attention**
   - Add selective attention mechanism
   - Expected gain: 0.5-1.0 pp
   - Target: <44% WER

See `PHASE_II_ROADMAP.md` for complete plan.

---

## Files Created/Modified

**Created**:
- `lm/train_text.txt` - Training transcriptions (5,672 sentences)
- `scripts/extract_transcriptions.py` - Extraction script
- `BEAM_SEARCH_IMPLEMENTATION_GUIDE.md` - This guide

**To Create**:
- `lm/3gram.arpa` - Trained language model
- `src/baseline/evaluate_beam.py` - Beam search evaluation script
- `results/evaluation/beam_search_predictions.json` - Predictions output

---

## References

- **KenLM**: https://github.com/kpu/kenlm
- **ctcdecode**: https://github.com/parlance/ctcdecode
- **CTC Decoding**: Graves et al. (2006) - Connectionist Temporal Classification
- **Beam Search for SLR**: Koller et al. (2015) - PHOENIX dataset paper

---

**Last Updated**: 2025-10-23
**Status**: Ready for Step 2 (Install KenLM) 🚀
