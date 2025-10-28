# Documentation Index

This directory contains all project documentation for the sign language recognition system.

## Directory Structure

```
docs/
├── guides/          # How-to guides and tutorials
├── technical/       # Technical documentation and analysis
└── planning/        # Project planning and strategy documents
```

## Documentation Categories

### Guides (`guides/`)
Practical guides for using the system:

- **EXTRACTION_GUIDE.md** - Complete guide for feature extraction pipeline
- **PARALLEL_EXTRACTION_GUIDE.md** - Guide for parallel/distributed extraction
- **GPU_EXTRACTION_READY.md** - GPU-accelerated extraction setup
- **GPU_EXTRACTION_STRATEGY.md** - Strategy for GPU optimization
- **EXECUTE_OPTIMIZED_EXTRACTION.md** - Running optimized extraction

### Technical Documentation (`technical/`)
In-depth technical documentation:

- **CTC_LOSS_ANALYSIS.md** - Analysis of CTC loss behavior and optimization
- **CTC_QUICK_REFERENCE.md** - Quick reference for CTC implementation
- **FEATURE_EXTRACTION_VALIDATION_REPORT.md** - Validation of extracted features

### Planning Documents (`planning/`)
Project planning and strategy:

- **BILSTM_ARCHITECTURE_REVIEW.md** - Review of BiLSTM architecture choices
- **MIGRATION_GUIDE.md** - Guide for migrating between model versions
- **RESEARCH_ACTION_PLAN.md** - Research roadmap and action items
- **START_TRAINING.md** - Training initialization guide
- **VISUALIZATION_DELIVERABLES.md** - Planned visualizations
- **WEEK2_PLAN.md** - Week 2 development plan
- **RESEARCH_DECISION_MediaPipe_CPU.md** - Decision to use MediaPipe on CPU
- **FINAL_GPU_RECOMMENDATIONS.md** - GPU hardware recommendations
- **FPS_CORRECTIONS_SUMMARY.md** - Frame rate corrections
- **EXTRACTION_RUNNING.md** - Status of extraction processes
- **EXTRACTION_STATUS.md** - Current extraction status

## Quick Links

### For New Users
1. Start with the main [README.md](../README.md)
2. Read [EXTRACTION_GUIDE.md](guides/EXTRACTION_GUIDE.md) for feature extraction
3. Check [START_TRAINING.md](planning/START_TRAINING.md) for training

### For Developers
1. Review [BILSTM_ARCHITECTURE_REVIEW.md](planning/BILSTM_ARCHITECTURE_REVIEW.md)
2. Understand [CTC_LOSS_ANALYSIS.md](technical/CTC_LOSS_ANALYSIS.md)
3. Follow [RESEARCH_ACTION_PLAN.md](planning/RESEARCH_ACTION_PLAN.md)

### For Researchers
1. See [research-proposal.md](../research-proposal.md) for thesis details
2. Review technical documentation in `technical/`
3. Check planning documents for methodology

## Documentation Standards

When adding new documentation:
1. Use clear, descriptive filenames
2. Add to this index with a brief description
3. Include a header with purpose and last update date
4. Use consistent markdown formatting
5. Cross-reference related documents

## Related Documentation

- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Repository organization
- [CHANGELOG.md](../CHANGELOG.md) - Project change history
- [CLAUDE.md](../CLAUDE.md) - AI assistant instructions