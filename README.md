# Sign Language Recognition System

A computationally efficient, real-time continuous sign language recognition system using MobileNetV3 and knowledge distillation, designed for educational accessibility.

## Research Alignment

This implementation follows the research proposal objectives:
- **Efficient Feature Extraction**: MediaPipe Holistic (500+ landmarks)
- **Lightweight Architecture**: MobileNetV3 + BiLSTM (< 100MB)
- **Knowledge Distillation**: From I3D/SlowFast teachers
- **Real-time Performance**: > 30 FPS on consumer hardware
- **Educational Focus**: < 25% WER for practical deployment

## Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt
```

### Project Structure
```
src/                 # Clean pipeline implementation
├── data/           # Data loading modules
├── models/         # Neural network architectures
├── training/       # Training scripts
└── utils/          # Utility functions
```

See [PIPELINE_STRUCTURE.md](PIPELINE_STRUCTURE.md) for detailed organization.

### Current Status

#### Completed
- MediaPipe feature extraction (6,516 dimensions)
- Dataset preparation (RWTH-PHOENIX-Weather 2014)
- Clean pipeline structure
- Archived old overfitted models

#### In Progress
- MobileNetV3 backbone implementation
- Modality-specific encoders
- Knowledge distillation framework

#### TODO
- [ ] Complete Phase I baseline (target: 40% WER)
- [ ] Implement Phase II distillation (target: < 25% WER)
- [ ] Phase III deployment optimization (> 30 FPS)

### Training Pipeline

#### Phase I: Baseline Training
```python
# Coming soon - MobileNetV3 implementation
python src/training/train.py --phase baseline
```

#### Phase II: Knowledge Distillation
```python
# Coming soon - Distillation from I3D/SlowFast
python src/training/distillation.py --teacher i3d --student mobilenet_v3
```

#### Phase III: Deployment
```python
# Coming soon - TensorRT optimization
python src/training/optimize.py --model checkpoint/best_model.pth
```

## Performance Targets

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| WER | < 25% | TBD | Phase II target |
| Model Size | < 100MB | N/A | MobileNetV3 design |
| Inference | > 30 FPS | TBD | Phase III target |
| Memory | < 8GB VRAM | ✓ | Achieved |

## Key Features

### 1. Comprehensive Feature Extraction
- **Pose**: 33 keypoints (body skeleton)
- **Hands**: 21 keypoints × 2 hands
- **Face**: 468 keypoints (expressions)
- **Temporal**: Velocity & acceleration
- **No PCA**: Preserves modality boundaries

### 2. Efficient Architecture
- **MobileNetV3-Small**: 3.2M parameters
- **BiLSTM**: 2 layers, 128 units
- **Cross-Modal Attention**: 4 heads
- **Total**: ~5M parameters (20MB)

### 3. Knowledge Distillation
- **Teachers**: I3D, SlowFast
- **Temperature**: 3.0
- **Loss**: 0.7×soft + 0.3×hard
- **Target**: 95% teacher accuracy

## Development

### Archive Structure
Old implementations are preserved in `.archive/`:
- `phase1_efficient_hybrid/`: Initial overfitted models

### Contributing
1. Follow the pipeline structure in `PIPELINE_STRUCTURE.md`
2. Align changes with `research-proposal.md`
3. Maintain < 100MB model constraint
4. Target < 25% WER

## Citation

```bibtex
@dataset{rwth_phoenix,
  title={RWTH-PHOENIX-Weather 2014},
  author={Forster et al.},
  year={2014}
}
```

## License

Research project for educational accessibility.

---

**Current Focus**: Implementing MobileNetV3 backbone to replace overfitted EfficientHybridModel (36M params) with lightweight architecture (5M params) aligned with research proposal.