# Continuous Sign Language Recognition

A lightweight continuous sign language recognition (CSLR) system using knowledge distillation from AdaptSign teacher features for efficient deployment on consumer hardware.

## Overview

This project implements a real-time CSLR pipeline targeting the RWTH-PHOENIX-Weather 2014 benchmark. The system leverages pre-extracted visual features from the AdaptSign framework, processed through a BiLSTM encoder with multi-head self-attention, trained with CTC loss and knowledge distillation.

### Feature Extraction

The pipeline uses learned visual features from the **AdaptSign** teacher model:
- AdaptSign pre-extracts frame-level visual representations using a pretrained visual encoder
- Features are stored as `.npz` files per video sequence
- This decouples feature extraction from sequence modeling, enabling efficient experimentation

### Architecture

```
[AdaptSign Teacher Features] → Temporal Subsampling → BiLSTM + MHSA → CTC Head → Beam Search Decoder
        (1024-dim)                 (Conv1d x2)          (3 layers)      (Linear)     (+ optional LM)
```

**Student Model: FeatureBiLSTMCTC**
- Input: Pre-extracted AdaptSign teacher features (1024-dim per frame)
- Temporal subsampling: 2x Conv1d (stride=2) for 4x downsampling
- Encoder: 3-layer BiLSTM (512 hidden) + 8-head self-attention
- Output: CTC log-probabilities over vocabulary (~1200 glosses)
- Parameters: ~8M (deployable weights ~16MB fp16)

### Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Dev WER | ~25-30% | Beam search (width=10) |
| Inference | 3000+ FPS | Feature frames on GPU |
| RTF | 0.008 | Real-time factor @ 25fps video |
| Model Size | ~31.6 MB | fp32 weights |

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ with CUDA support
- Linux/WSL recommended for training

```bash
# Clone repository
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation

1. Download RWTH-PHOENIX-Weather 2014 dataset
2. Obtain AdaptSign teacher features (pre-extracted visual representations):

```bash
# Features should be organized as:
data/teacher_features/adaptsign_official/
├── train/
│   ├── <video_id>.npz
│   └── ...
├── dev/
│   └── ...
└── test/
    └── ...
```

Each `.npz` file contains:
- `features`: (T, 1024) float32 array of frame-level visual features
- `gloss`: ground truth gloss sequence (space-separated)

## Usage

### Training

**Baseline CTC Training**
```bash
python -m src.training.train_visual_baseline_ctc \
    --features_dir data/teacher_features/adaptsign_official \
    --output_dir checkpoints/baseline \
    --epochs 100 \
    --batch_size 8 \
    --lr 3e-4
```

**Knowledge Distillation Training**
```bash
python -m src.training.train_distillation \
    --teacher_checkpoint checkpoints/teacher/best.pt \
    --features_dir data/teacher_features/adaptsign_official \
    --output_dir checkpoints/student_kd \
    --temperature 3.0 \
    --alpha 0.7
```

### Evaluation

```bash
# Greedy decoding
python -m src.evaluation.evaluate_student \
    --checkpoint checkpoints/student_kd/best.pt \
    --split test

# Beam search decoding
python -m src.evaluation.evaluate_student \
    --checkpoint checkpoints/student_kd/best.pt \
    --split test \
    --beam_search \
    --beam_width 10
```

### Inference Benchmarking

```bash
python scripts/phase3_benchmark.py \
    --checkpoint checkpoints/student_kd/best.pt \
    --features_dir data/teacher_features/adaptsign_official \
    --split dev \
    --decode beam_search \
    --beam_width 10
```

### Interactive Dashboard

```bash
# Requires WSL or Linux
streamlit run apps/edu_dashboard.py
```

### Streaming Demo

```bash
python scripts/phase3_streaming_demo.py \
    --checkpoint checkpoints/student_kd/best.pt \
    --features_dir data/teacher_features/adaptsign_official \
    --window_size 64 \
    --stride 16
```

## Project Structure

```
sign-language-recognition/
├── apps/
│   └── edu_dashboard.py          # Streamlit interactive dashboard
├── scripts/
│   ├── phase3_benchmark.py       # Throughput/latency benchmarking
│   ├── phase3_streaming_demo.py  # Sliding window streaming demo
│   └── visualize_student_attention.py  # Attention heatmap export
├── src/
│   ├── data/
│   │   ├── dataset.py            # Base dataset classes
│   │   └── sequence_feature_dataset.py  # Pre-extracted feature loader
│   ├── evaluation/
│   │   └── evaluate_student.py   # Comprehensive evaluation script
│   ├── lm/
│   │   ├── ngram_lm.py           # N-gram language model
│   │   └── neural_lm_fusion.py   # Neural LM fusion (experimental)
│   ├── models/
│   │   ├── baseline_ctc_bilstm.py    # FeatureBiLSTMCTC student model
│   │   ├── hybrid_ctc_encoder.py     # BiLSTMEncoder + MHSA
│   │   ├── mobilenet_v3.py           # MobileNetV3 feature extractor
│   │   └── i3d_teacher.py            # I3D teacher model
│   ├── protocols/
│   │   └── protocol_v1.py        # Evaluation protocol filtering
│   ├── training/
│   │   ├── ctc_trainer.py        # CTC training utilities
│   │   ├── train_distillation.py # Knowledge distillation
│   │   └── train_visual_baseline_ctc.py  # Baseline training
│   └── utils/
│       ├── ctc.py                # CTC decoding (greedy, beam search)
│       ├── metrics.py            # WER computation
│       └── augmentation.py       # SpecAugment, feature augmentation
├── checkpoints/                  # Model checkpoints
├── figures/                      # Generated visualizations
├── requirements.txt              # Python dependencies
└── README.md
```

## Technical Details

### CTC Decoding

The system supports multiple decoding strategies:

1. **Greedy Decoding**: Fast, baseline performance
2. **Beam Search**: Improved accuracy with configurable beam width
3. **Beam Search + LM**: N-gram language model shallow fusion

```python
from src.utils.ctc import ctc_greedy_decode_with_lengths, ctc_beam_search_decode

# Greedy
predictions = ctc_greedy_decode_with_lengths(log_probs, lengths, blank_idx=0)

# Beam search
predictions = ctc_beam_search_decode(log_probs, lengths, beam_width=10, blank_idx=0)
```

### Knowledge Distillation

Training uses soft label distillation from the AdaptSign teacher:

- **Teacher Features**: Pre-extracted from AdaptSign's visual encoder
- **Teacher Logits**: Soft targets from AdaptSign's sequence model (optional)
- **Temperature**: 3.0 (softens probability distributions)
- **Loss**: 0.7 * KL_divergence(soft) + 0.3 * CTC_loss(hard)
- **Student**: Lightweight BiLSTM+MHSA trained on teacher features

### Data Augmentation

- SpecAugment: Time and frequency masking
- Feature dropout: Input-level regularization
- Label smoothing: Implicit through temperature scaling

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{forster2014extensions,
  title={Extensions of the Sign Language Recognition and Translation Corpus RWTH-PHOENIX-Weather},
  author={Forster, Jens and Schmidt, Christoph and Koller, Oscar and Bellgardt, Martin and Ney, Hermann},
  booktitle={LREC},
  year={2014}
}
```

## License

This project is for research and educational purposes. See LICENSE for details.
