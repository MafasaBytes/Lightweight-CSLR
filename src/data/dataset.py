"""
MediaPipe Feature Dataset for Sign Language Recognition.
Loads pre-extracted MediaPipe features.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import json


class Vocabulary:
    """Vocabulary management for sign language."""

    def __init__(self):
        # CTC blank token at index 0
        self.word2idx = {"<blank>": 0}
        self.idx2word = {0: "<blank>"}
        self.blank_id = 0
        self.pad_id = 0  # Use same as blank for CTC

    def add_word(self, word: str) -> int:
        """Add a word to vocabulary and return its index."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word  # CRITICAL FIX: Add to idx2word mapping
        return self.word2idx[word]

    def words_to_indices(self, words: List[str]) -> List[int]:
        return [self.word2idx.get(w, self.word2idx["<blank>"]) for w in words]

    def indices_to_words(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(idx, "<blank>") for idx in indices]

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': {str(k): v for k, v in self.idx2word.items()}}, f)

    def load(self, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}


def build_vocabulary(annotation_file: Path) -> Vocabulary:
    """Build vocabulary from training annotations."""
    vocab = Vocabulary()

    # Patterns to EXCLUDE from vocabulary
    EXCLUDE_PATTERNS = [
        '__',       # Special markers like __ON__, __OFF__
        'loc-',     # Location markers
        'cl-',      # Classifier markers
        'lh-',      # Left hand markers
        'rh-',      # Right hand markers
        'IX',       # Pointing/deixis
        'WG',       # Unknown marker
        '$GEST',    # Gesture marker
        'PLUSPLUS', # Modifier
        'POS',      # Position marker
    ]

    df = pd.read_csv(annotation_file, sep='|', on_bad_lines='skip')
    if 'annotation' not in df.columns:
        df.columns = [col.strip() for col in df.columns]

    excluded_count = 0
    for annotation in df['annotation']:
        if isinstance(annotation, str):
            words = annotation.strip().split()
            for word in words:
                # Check if word should be excluded
                should_exclude = False
                for pattern in EXCLUDE_PATTERNS:
                    if pattern in word:
                        should_exclude = True
                        excluded_count += 1
                        break

                # Also exclude single letters (often fingerspelling markers)
                if len(word) == 1 and word.isalpha():
                    should_exclude = True
                    excluded_count += 1

                if not should_exclude:
                    vocab.add_word(word)

    logging.info(f"Vocabulary built with {len(vocab)} words (excluded {excluded_count} special tokens)")
    return vocab


class MediaPipeFeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted MediaPipe features.
    Features include pose, hands, face landmarks, and temporal dynamics.
    """

    def __init__(
        self,
        data_dir: Path,
        annotation_file: Path,
        vocabulary: Vocabulary,
        split: str = 'train',
        max_seq_length: Optional[int] = None,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing .npy feature files
            annotation_file: CSV file with annotations
            vocabulary: Vocabulary object
            split: Dataset split ('train', 'dev', 'test')
            max_seq_length: Maximum sequence length (for memory efficiency)
            augment: Apply data augmentation
            normalize: Apply feature normalization
        """
        self.data_dir = Path(data_dir)
        self.vocabulary = vocabulary
        self.split = split
        self.max_seq_length = max_seq_length
        self.augment = augment
        self.normalize = normalize

        # Load annotations
        self.annotations = self._load_annotations(annotation_file)

        # Precompute feature statistics for normalization
        if normalize:
            self._compute_feature_stats()

        logging.info(f"Loaded {len(self.annotations)} samples for {split}")

    def _load_annotations(self, annotation_file: Path) -> List[Dict]:
        """Load and parse annotation file."""
        annotations = []

        # Same exclusion patterns as vocabulary building
        EXCLUDE_PATTERNS = [
            '__',       # Special markers like __ON__, __OFF__, __PU__
            'loc-',     # Location markers
            'cl-',      # Classifier markers
            'lh-',      # Left hand markers
            'rh-',      # Right hand markers
            'IX',       # Pointing/deixis
            'WG',       # Unknown marker
            '$GEST',    # Gesture marker
            'PLUSPLUS', # Modifier
            'POS',      # Position marker
        ]

        df = pd.read_csv(annotation_file, sep='|', on_bad_lines='skip')
        if 'annotation' not in df.columns:
            df.columns = [col.strip() for col in df.columns]

        for idx, row in df.iterrows():
            video_id = row['name'] if 'name' in row else row['id']
            annotation = row['annotation'] if 'annotation' in row else ""

            if isinstance(annotation, str):
                raw_words = annotation.strip().split()
                
                # Filter out special tokens
                filtered_words = []
                for word in raw_words:
                    should_exclude = False
                    for pattern in EXCLUDE_PATTERNS:
                        if pattern in word:
                            should_exclude = True
                            break
                    # Also exclude single letters
                    if len(word) == 1 and word.isalpha():
                        should_exclude = True
                    
                    if not should_exclude:
                        filtered_words.append(word)
                
                # Use filtered words for both words and labels
                words = filtered_words
                labels = self.vocabulary.words_to_indices(words)

                # Check if feature file exists (try .npz first, then .npy)
                feature_path_npz = self.data_dir / self.split / f"{video_id}.npz"
                feature_path_npy = self.data_dir / f"{video_id}.npy"

                if feature_path_npz.exists():
                    feature_path = feature_path_npz
                elif feature_path_npy.exists():
                    feature_path = feature_path_npy
                else:
                    feature_path = None

                if feature_path:
                    annotations.append({
                        'video_id': video_id,
                        'feature_path': feature_path,
                        'words': words,
                        'labels': labels
                    })
                else:
                    logging.debug(f"Feature file not found: {feature_path}")

        return annotations

    def _compute_feature_stats(self):
        """Compute mean and std for feature normalization."""
        stats_file = self.data_dir.parent / "feature_stats.pkl"

        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
                self.feature_mean = stats['mean']
                self.feature_std = stats['std']
        else:
            logging.info("Computing feature statistics for normalization...")
            all_features = []

            # Sample subset for efficiency
            sample_size = min(100, len(self.annotations))
            if sample_size == 0:
                # If no annotations, use dummy stats
                self.feature_mean = np.zeros(6516)
                self.feature_std = np.ones(6516)
                return

            step = max(1, len(self.annotations) // sample_size)
            for i in range(0, len(self.annotations), step):
                if i < len(self.annotations):
                    feature_path = self.annotations[i]['feature_path']
                    if str(feature_path).endswith('.npz'):
                        data = np.load(feature_path)
                        if 'features' in data:
                            features = data['features']
                        elif 'feature' in data:
                            features = data['feature']
                        elif 'arr_0' in data:
                            features = data['arr_0']
                        else:
                            features = list(data.values())[0]
                    else:
                        features = np.load(feature_path)
                    all_features.append(features)

            all_features = np.concatenate(all_features, axis=0)
            self.feature_mean = np.mean(all_features, axis=0)
            self.feature_std = np.std(all_features, axis=0) + 1e-6

            # Save stats
            with open(stats_file, 'wb') as f:
                pickle.dump({
                    'mean': self.feature_mean,
                    'std': self.feature_std
                }, f)

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation to features."""
        if not self.augment or self.split != 'train':
            return features

        # Temporal augmentation - speed perturbation
        if np.random.random() < 0.5:
            speed_factor = np.random.uniform(0.9, 1.1)
            seq_len = features.shape[0]
            new_len = int(seq_len * speed_factor)
            indices = np.linspace(0, seq_len - 1, new_len).astype(int)
            features = features[indices]

        # Spatial augmentation - add noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        # Feature dropout
        if np.random.random() < 0.2:
            mask = np.random.binomial(1, 0.9, features.shape)
            features = features * mask

        return features

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get a single sample."""
        annotation = self.annotations[idx]

        # Load features (handle both .npy and .npz)
        feature_path = annotation['feature_path']
        if str(feature_path).endswith('.npz'):
            # Load from .npz file
            data = np.load(feature_path)
            # Try different keys that might contain the features
            if 'features' in data:
                features = data['features'].astype(np.float32)
            elif 'feature' in data:
                features = data['feature'].astype(np.float32)
            elif 'arr_0' in data:
                features = data['arr_0'].astype(np.float32)
            else:
                # Use the first array in the npz file
                features = list(data.values())[0].astype(np.float32)
        else:
            # Load from .npy file
            features = np.load(feature_path).astype(np.float32)

        # Apply augmentation
        if self.augment:
            features = self._augment_features(features)

        # Normalize features
        if self.normalize and hasattr(self, 'feature_mean'):
            features = (features - self.feature_mean) / self.feature_std

        # Uniform temporal subsampling
        if self.max_seq_length is not None and features.shape[0] > self.max_seq_length:
            # Uniformly sample frames across the entire sequence
            num_frames = features.shape[0]
            indices = np.linspace(0, num_frames - 1, self.max_seq_length, dtype=np.int64)
            features = features[indices]
        elif self.max_seq_length is not None and features.shape[0] < self.max_seq_length:
            # If sequence is shorter, we could pad or repeat last frame
            # For now, just use as-is (padding happens in collate_fn)
            pass

        # Convert to tensor
        features = torch.from_numpy(features)
        labels = torch.tensor(annotation['labels'], dtype=torch.long)

        return {
            'video_id': annotation['video_id'],
            'features': features,
            'labels': labels,
            'words': annotation['words']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors with padding
    """
    # Sort by sequence length (descending) for efficient packing
    batch = sorted(batch, key=lambda x: len(x['features']), reverse=True)

    video_ids = [item['video_id'] for item in batch]
    words = [item['words'] for item in batch]

    # Get dimensions
    batch_size = len(batch)
    max_seq_len = max(len(item['features']) for item in batch)
    feature_dim = batch[0]['features'].shape[-1]

    # Initialize tensors
    features = torch.zeros(batch_size, max_seq_len, feature_dim)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)

    # Fill features
    for i, item in enumerate(batch):
        seq_len = len(item['features'])
        features[i, :seq_len] = item['features']
        input_lengths[i] = seq_len

    # Concatenate labels
    labels = torch.cat([item['labels'] for item in batch])
    target_lengths = torch.tensor([len(item['labels']) for item in batch], dtype=torch.long)

    return {
        'video_ids': video_ids,
        'features': features,
        'labels': labels,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
        'words': words
    }


class DataAugmentation:
    """Additional data augmentation techniques for sign language."""

    @staticmethod
    def temporal_masking(features: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Randomly mask temporal frames."""
        seq_len = features.shape[0]
        num_mask = int(seq_len * mask_ratio)

        if num_mask > 0:
            mask_indices = torch.randperm(seq_len)[:num_mask]
            features[mask_indices] = 0

        return features

    @staticmethod
    def mixup(features1: torch.Tensor, features2: torch.Tensor,
             labels1: torch.Tensor, labels2: torch.Tensor,
             alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Mixup augmentation for two samples."""
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

        mixed_features = lam * features1 + (1 - lam) * features2

        return mixed_features, labels1, lam


if __name__ == "__main__":
    # Test the dataset
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data/features_enhanced")
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.corpus.csv")

    # Build vocabulary
    vocab = build_vocabulary(annotation_file)
    print(f"Vocabulary size: {len(vocab)}")

    # Create dataset
    dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Input lengths: {batch['input_lengths']}")
    print(f"Target lengths: {batch['target_lengths']}")