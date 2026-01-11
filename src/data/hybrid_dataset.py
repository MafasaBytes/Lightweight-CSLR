"""
Hybrid Feature Dataset for Sign Language Recognition

Combines MediaPipe pose features with MobileNetV3 CNN features
for improved recognition accuracy.

Feature dimensions:
- MediaPipe: 6,516 (pose + hands + face + temporal)
- MobileNetV3: 576 (CNN visual features)
- Combined: 7,092 or configurable fusion
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HybridFeatureDataset(Dataset):
    """
    Dataset combining MediaPipe and MobileNetV3 features.
    
    Supports multiple fusion strategies:
    - 'concat': Simple concatenation
    - 'weighted': Weighted combination
    - 'attention': Learned attention fusion (during training)
    """
    
    def __init__(
        self,
        mediapipe_dir: Path,
        mobilenet_dir: Path,
        annotation_file: Path,
        vocabulary,
        split: str = 'train',
        max_seq_length: int = 300,
        fusion: str = 'concat',
        mediapipe_weight: float = 0.5,
        augment: bool = False,
        normalize: bool = True
    ):
        self.mediapipe_dir = Path(mediapipe_dir) / split
        self.mobilenet_dir = Path(mobilenet_dir) / split
        self.vocabulary = vocabulary
        self.split = split
        self.max_seq_length = max_seq_length
        self.fusion = fusion
        self.mediapipe_weight = mediapipe_weight
        self.augment = augment and split == 'train'
        self.normalize = normalize
        
        self.samples = self._load_annotations(annotation_file)
        
        # Compute normalization stats if needed
        if self.normalize and split == 'train':
            self._compute_stats()
        else:
            self.mp_mean = None
            self.mp_std = None
            self.mn_mean = None
            self.mn_std = None
        
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_annotations(self, annotation_file: Path) -> List[Dict]:
        samples = []
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('|')
            if len(parts) >= 4:
                video_id = parts[0]
                annotation = parts[3]
                
                # Check both feature files exist
                mp_file = self.mediapipe_dir / f"{video_id}.npz"
                mn_file = self.mobilenet_dir / f"{video_id}.npz"
                
                if mp_file.exists() and mn_file.exists():
                    # Process annotation
                    words = []
                    for word in annotation.split():
                        word = word.strip()
                        if word and not word.startswith('__') and not word.startswith('loc-'):
                            if word != 'IX':
                                words.append(word)
                    
                    if words:
                        samples.append({
                            'video_id': video_id,
                            'mediapipe_file': mp_file,
                            'mobilenet_file': mn_file,
                            'annotation': words
                        })
        
        return samples
    
    def _compute_stats(self):
        """Compute mean/std for normalization from a subset of training data."""
        logger.info("Computing normalization statistics...")
        
        mp_features = []
        mn_features = []
        
        # Sample subset for efficiency
        indices = np.random.choice(len(self.samples), min(500, len(self.samples)), replace=False)
        
        for idx in indices:
            sample = self.samples[idx]
            try:
                mp_data = np.load(sample['mediapipe_file'])
                mn_data = np.load(sample['mobilenet_file'])
                
                mp_feat = mp_data['features'] if 'features' in mp_data else mp_data['arr_0']
                mn_feat = mn_data['features'] if 'features' in mn_data else mn_data['arr_0']
                
                mp_features.append(mp_feat.reshape(-1, mp_feat.shape[-1]))
                mn_features.append(mn_feat.reshape(-1, mn_feat.shape[-1]))
            except Exception as e:
                continue
        
        if mp_features and mn_features:
            mp_all = np.concatenate(mp_features, axis=0)
            mn_all = np.concatenate(mn_features, axis=0)
            
            self.mp_mean = np.nanmean(mp_all, axis=0)
            self.mp_std = np.nanstd(mp_all, axis=0) + 1e-8
            self.mn_mean = np.nanmean(mn_all, axis=0)
            self.mn_std = np.nanstd(mn_all, axis=0) + 1e-8
            
            logger.info(f"MediaPipe stats: mean={self.mp_mean.mean():.4f}, std={self.mp_std.mean():.4f}")
            logger.info(f"MobileNet stats: mean={self.mn_mean.mean():.4f}, std={self.mn_std.mean():.4f}")
        else:
            self.mp_mean = self.mp_std = self.mn_mean = self.mn_std = None
    
    def set_stats(self, mp_mean, mp_std, mn_mean, mn_std):
        """Set normalization stats from training set."""
        self.mp_mean = mp_mean
        self.mp_std = mp_std
        self.mn_mean = mn_mean
        self.mn_std = mn_std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load features
        mp_data = np.load(sample['mediapipe_file'])
        mn_data = np.load(sample['mobilenet_file'])
        
        mp_features = mp_data['features'] if 'features' in mp_data else mp_data['arr_0']
        mn_features = mn_data['features'] if 'features' in mn_data else mn_data['arr_0']
        
        # Handle NaN
        mp_features = np.nan_to_num(mp_features, nan=0.0)
        mn_features = np.nan_to_num(mn_features, nan=0.0)
        
        # Align sequence lengths (use minimum)
        min_len = min(len(mp_features), len(mn_features), self.max_seq_length)
        mp_features = mp_features[:min_len]
        mn_features = mn_features[:min_len]
        
        # Normalize
        if self.normalize and self.mp_mean is not None:
            mp_features = (mp_features - self.mp_mean) / self.mp_std
            mn_features = (mn_features - self.mn_mean) / self.mn_std
        
        # Fuse features
        if self.fusion == 'concat':
            features = np.concatenate([mp_features, mn_features], axis=-1)
        elif self.fusion == 'weighted':
            # Pad to same dimension for weighted sum
            max_dim = max(mp_features.shape[-1], mn_features.shape[-1])
            mp_padded = np.zeros((min_len, max_dim))
            mn_padded = np.zeros((min_len, max_dim))
            mp_padded[:, :mp_features.shape[-1]] = mp_features
            mn_padded[:, :mn_features.shape[-1]] = mn_features
            features = self.mediapipe_weight * mp_padded + (1 - self.mediapipe_weight) * mn_padded
        else:
            # Default to concat
            features = np.concatenate([mp_features, mn_features], axis=-1)
        
        # Data augmentation
        if self.augment:
            features = self._augment(features)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Process labels
        labels = [self.vocabulary.word2idx.get(w, self.vocabulary.word2idx.get('<unk>', 0)) 
                  for w in sample['annotation']]
        labels = torch.LongTensor(labels)
        
        return {
            'features': features,
            'labels': labels,
            'video_id': sample['video_id']
        }
    
    def _augment(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Time masking
        if np.random.random() < 0.3:
            T = features.shape[0]
            mask_len = int(T * np.random.uniform(0.05, 0.15))
            start = np.random.randint(0, max(1, T - mask_len))
            features[start:start + mask_len] = 0
        
        # Feature dropout
        if np.random.random() < 0.2:
            mask = np.random.random(features.shape) > 0.1
            features = features * mask
        
        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, features.shape)
            features = features + noise
        
        return features.astype(np.float32)
    
    @property
    def feature_dim(self) -> int:
        """Return the combined feature dimension."""
        if self.fusion == 'concat':
            return 6516 + 576  # MediaPipe + MobileNetV3
        else:
            return max(6516, 576)


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    # Sort by sequence length (descending)
    batch = sorted(batch, key=lambda x: len(x['features']), reverse=True)
    
    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    video_ids = [item['video_id'] for item in batch]
    
    # Pad features
    max_feat_len = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[-1]
    
    padded_features = torch.zeros(len(features), max_feat_len, feat_dim)
    input_lengths = torch.zeros(len(features), dtype=torch.long)
    
    for i, f in enumerate(features):
        padded_features[i, :len(f)] = f
        input_lengths[i] = len(f)
    
    # Pad labels for Seq2Seq (add EOS)
    max_label_len = max(len(l) for l in labels) + 1  # +1 for EOS
    padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
    label_lengths = torch.zeros(len(labels), dtype=torch.long)
    
    for i, l in enumerate(labels):
        padded_labels[i, :len(l)] = l
        padded_labels[i, len(l)] = 1  # EOS token
        label_lengths[i] = len(l) + 1
    
    return {
        'features': padded_features,
        'input_lengths': input_lengths,
        'targets': padded_labels,
        'target_lengths': label_lengths,
        'video_ids': video_ids
    }


# CTC-specific collate
def collate_fn_ctc(batch: List[Dict]) -> Dict:
    """Collate function for CTC training."""
    batch = sorted(batch, key=lambda x: len(x['features']), reverse=True)
    
    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_feat_len = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[-1]
    
    padded_features = torch.zeros(len(features), max_feat_len, feat_dim)
    input_lengths = torch.zeros(len(features), dtype=torch.long)
    
    for i, f in enumerate(features):
        padded_features[i, :len(f)] = f
        input_lengths[i] = len(f)
    
    # Concatenate labels for CTC
    concat_labels = torch.cat(labels)
    label_lengths = torch.LongTensor([len(l) for l in labels])
    
    return {
        'features': padded_features,
        'input_lengths': input_lengths,
        'labels': concat_labels,
        'label_lengths': label_lengths
    }

