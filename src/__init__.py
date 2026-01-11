"""Sign Language Recognition Pipeline - Core Implementation."""

__version__ = "0.2.0"  # Version 0.2.0: MobileNetV3 architecture alignment

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"