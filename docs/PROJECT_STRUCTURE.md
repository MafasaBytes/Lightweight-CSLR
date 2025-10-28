# Project Structure Documentation

Last Updated: 2025-01-21

## Directory Organization

This document describes the organization of the Sign Language Recognition project repository.

## Root Directory

```
sign-language-recognition/
├── README.md                 # Project overview and quick start guide
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── CLAUDE.md                # Instructions for AI assistant
├── research-proposal.md     # Master's thesis proposal
├── PROJECT_STRUCTURE.md    # This file
├── REORGANIZATION_PLAN.md  # Documentation of reorganization
└── CHANGELOG.md            # Project change history
```

## Core Directories

### `/configs`
Configuration files for models and experiments.
- `bilstm_baseline_config.yaml` - Current BiLSTM training configuration

### `/data`
All data-related files. **Note**: Feature directories should not be modified during training.
```
data/
├── raw_data/               # Original PHOENIX dataset (53GB, DO NOT MODIFY)
│   └── phoenix-2014-multisigner/
├── features/              # Extracted 108-dim features (DO NOT MODIFY)
│   ├── train/            # 5,672 training features
│   ├── dev/              # 540 validation features
│   └── test/             # 629 test features
├── baseline_vocabulary/   # Vocabulary files
│   └── vocabulary.txt    # 1,229 glosses
└── metadata/             # Dataset splits and annotations
```

### `/src`
Source code for the project.
```
src/
├── models/               # Model architectures
│   ├── __init__.py
│   └── bilstm.py        # Optimized BiLSTM model
├── baseline/            # Phase I baseline implementation
│   ├── train.py         # Main training script
│   ├── dataset.py       # Dataset loader
│   ├── extract_features.py  # Feature extraction (108-dim)
│   ├── create_vocabulary.py # Vocabulary creation
│   ├── validate_vocabulary.py
│   ├── validate_baseline_data.py
│   └── diagnose_ctc.py # CTC debugging
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── ctc_decoder.py  # CTC decoding utilities
│   ├── metrics.py      # Evaluation metrics
│   └── paths.py        # Path management
├── evaluation/          # Evaluation scripts (to be added)
└── visualization/       # Plotting utilities (to be added)
```

### `/models`
Saved model checkpoints and training outputs.
```
models/
└── bilstm_baseline/
    ├── checkpoint_latest.pt  # Most recent checkpoint
    ├── checkpoint_best.pt    # Best performing model (59.47% WER)
    └── training_results.json # Final training metrics
```

### `/logs`
TensorBoard logs for training visualization.
```
logs/
└── bilstm_baseline_optimized/  # Current training run logs
```

### `/results`
Evaluation results and generated outputs.
```
results/
├── figures/             # Generated plots and visualizations
└── metrics/            # Performance metrics and analysis
```

### `/notebooks`
Jupyter notebooks for analysis and experimentation.
```
notebooks/
├── 01_dataset_validation_visualization.ipynb  # Dataset exploration
├── README_VISUALIZATION.md                    # Visualization guide
└── run_visualization.py                      # Script to run notebooks
```

### `/scripts`
Standalone helper scripts organized by purpose.
```
scripts/
├── extraction/         # Feature extraction scripts
│   ├── extract_all_features.bat
│   ├── extract_remaining_features.bat
│   └── test_extraction.py
├── preprocessing/      # Data preprocessing (to be added)
└── analysis/          # Analysis and comparison scripts
    └── compare_architectures.py
```

### `/docs`
Project documentation organized by type.
```
docs/
├── guides/            # How-to guides
│   ├── EXTRACTION_GUIDE.md
│   ├── PARALLEL_EXTRACTION_GUIDE.md
│   └── GPU_EXTRACTION_*.md
├── technical/         # Technical documentation
│   ├── CTC_LOSS_ANALYSIS.md
│   ├── CTC_QUICK_REFERENCE.md
│   └── FEATURE_EXTRACTION_VALIDATION_REPORT.md
└── planning/          # Planning and strategy documents
    ├── BILSTM_ARCHITECTURE_REVIEW.md
    ├── RESEARCH_ACTION_PLAN.md
    └── WEEK2_PLAN.md
```

### `/archive`
Organized storage for deprecated code and old experiments.
```
archive/
├── 2024-10-experiments/    # Experiments from October 2024
├── old_models/            # Previous model versions
│   ├── bilstm_baseline/   # Old 66-dim models
│   └── bilstm_normalized_lr001/
├── old_features/          # Previous feature extractions
│   ├── processed/
│   └── processed_holistic/
└── deprecated_scripts/    # Old/unused scripts
    ├── old_extraction_scripts/
    └── old_scripts/
```

### `/tests`
Unit tests for the project (to be implemented).

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Markdown docs**: `UPPER_CASE.md` for important docs, `lower-case.md` for general
- **Config files**: `descriptive_name_config.yaml`
- **Checkpoints**: `checkpoint_[best|latest|epoch_N].pt`
- **Features**: `video_name.npz`

## Data Flow

1. **Raw Videos** → `data/raw_data/`
2. **Feature Extraction** → `data/features/`
3. **Model Training** → `models/` and `logs/`
4. **Evaluation** → `results/`

## Important Notes

### Protected Directories
These directories should NOT be modified during active training:
- `data/raw_data/` - Original dataset
- `data/features/` - Extracted features being used
- `models/bilstm_baseline/` - Active checkpoints

### Active Development
Current focus is on Phase I (baseline) implementation:
- Main training script: `src/baseline/train.py`
- Model architecture: `src/models/bilstm.py`
- Configuration: `configs/bilstm_baseline_config.yaml`

### Git Workflow
- Use `git mv` when moving tracked files
- Large files (models, data) are in `.gitignore`
- Commit messages should reference the development phase

## Quick Navigation

- **Start training**: `python src/baseline/train.py`
- **Extract features**: `scripts/extraction/extract_all_features.bat`
- **View logs**: `tensorboard --logdir logs/`
- **Check results**: `models/bilstm_baseline/training_results.json`