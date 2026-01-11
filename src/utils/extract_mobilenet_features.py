"""
MobileNetV3 Visual Feature Extraction for Sign Language Recognition

Extracts CNN features from video frames using MobileNetV3-Small backbone.
These features complement MediaPipe pose features for improved recognition.

Usage:
    python src/utils/extract_mobilenet_features.py --split train
    python src/utils/extract_mobilenet_features.py --split dev
    python src/utils/extract_mobilenet_features.py --split test
    python src/utils/extract_mobilenet_features.py --all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MobileNetV3FeatureExtractor(nn.Module):
    """
    MobileNetV3-Small feature extractor.
    
    Extracts features from the penultimate layer (before classification).
    Output dimension: 576 (from MobileNetV3-Small)
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = mobilenet_v3_small(weights=weights)
            logger.info("Loaded pretrained MobileNetV3-Small (ImageNet)")
        else:
            self.model = mobilenet_v3_small(weights=None)
            logger.info("Initialized MobileNetV3-Small without pretrained weights")
        
        # Remove the classifier to get features
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        
        # Feature dimension after avgpool
        self.feature_dim = 576
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Features of shape (B, 576) or (B, T, 576)
        """
        if x.dim() == 5:
            # Video input: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self._extract(x)
            return features.view(B, T, -1)
        else:
            # Image input: (B, C, H, W)
            return self._extract(x)
    
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class VideoFrameDataset(Dataset):
    """Dataset for loading video frames from PHOENIX-2014."""
    
    def __init__(
        self,
        frames_dir: Path,
        annotation_file: Path,
        transform: Optional[transforms.Compose] = None,
        max_frames: int = 300
    ):
        self.frames_dir = Path(frames_dir)
        self.transform = transform
        self.max_frames = max_frames
        self.samples = []
        
        # Parse annotation file
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                video_id = parts[0]
                folder_pattern = parts[1]  # e.g., "video_id/1/*.png"
                
                # Get the folder path
                folder_path = self.frames_dir / video_id / "1"
                
                if folder_path.exists():
                    # Get frame files sorted by frame number
                    frame_files = sorted(
                        folder_path.glob("*.png"),
                        key=lambda x: self._extract_frame_number(x.name)
                    )
                    
                    if frame_files:
                        self.samples.append({
                            'video_id': video_id,
                            'frame_files': frame_files[:self.max_frames]
                        })
        
        logger.info(f"Loaded {len(self.samples)} video samples from {annotation_file}")
    
    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename like 'xxx_fn000001-0.png'"""
        try:
            # Try to find 'fn' pattern
            if '_fn' in filename:
                fn_part = filename.split('_fn')[1]
                num_str = fn_part.split('-')[0]
                return int(num_str)
            else:
                # Fallback: try to extract any number
                import re
                nums = re.findall(r'\d+', filename)
                return int(nums[-1]) if nums else 0
        except:
            return 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_id = sample['video_id']
        frame_files = sample['frame_files']
        
        # Load frames
        frames = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {frame_file}: {e}")
                continue
        
        if not frames:
            # Return dummy frame if all failed
            dummy = torch.zeros(3, 224, 224)
            frames = [dummy]
        
        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        
        return {
            'video_id': video_id,
            'frames': frames_tensor,
            'num_frames': len(frames)
        }


def get_transform() -> transforms.Compose:
    """Get image transformation pipeline for MobileNetV3."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_features_for_split(
    split: str,
    frames_base_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str = 'cuda'
):
    """
    Extract MobileNetV3 features for a dataset split.
    
    Args:
        split: 'train', 'dev', or 'test'
        frames_base_dir: Base directory containing video frames
        annotation_dir: Directory containing annotation CSV files
        output_dir: Output directory for extracted features
        batch_size: Batch size for processing
        num_workers: Number of data loading workers
        device: Device to use ('cuda' or 'cpu')
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting MobileNetV3 features for {split}")
    logger.info(f"{'='*60}")
    
    # Setup paths
    frames_dir = frames_base_dir / split
    annotation_file = annotation_dir / f"{split}.corpus.csv"
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    if not frames_dir.exists():
        logger.error(f"Frames directory not found: {frames_dir}")
        return
    
    if not annotation_file.exists():
        logger.error(f"Annotation file not found: {annotation_file}")
        return
    
    # Initialize model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MobileNetV3FeatureExtractor(pretrained=True).to(device)
    model.eval()
    
    # Create dataset
    transform = get_transform()
    dataset = VideoFrameDataset(
        frames_dir=frames_dir,
        annotation_file=annotation_file,
        transform=transform,
        max_frames=300
    )
    
    # Process samples
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split}"):
            sample = dataset[idx]
            video_id = sample['video_id']
            frames = sample['frames'].to(device)  # (T, C, H, W)
            num_frames = sample['num_frames']
            
            # Process in batches if too many frames
            max_batch = 64
            features_list = []
            
            for start in range(0, frames.size(0), max_batch):
                end = min(start + max_batch, frames.size(0))
                batch_frames = frames[start:end]
                batch_features = model(batch_frames)  # (batch, 576)
                features_list.append(batch_features.cpu().numpy())
            
            # Concatenate all features
            features = np.concatenate(features_list, axis=0)  # (T, 576)
            
            # Save features
            output_file = output_split_dir / f"{video_id}.npz"
            np.savez_compressed(
                output_file,
                features=features,
                num_frames=num_frames,
                feature_dim=model.feature_dim
            )
    
    logger.info(f"Saved {len(dataset)} feature files to {output_split_dir}")
    
    # Save metadata
    metadata = {
        'split': split,
        'num_samples': len(dataset),
        'feature_dim': model.feature_dim,
        'model': 'MobileNetV3-Small',
        'pretrained': 'ImageNet1K_V1'
    }
    
    with open(output_split_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Extract MobileNetV3 features from PHOENIX-2014 video frames'
    )
    parser.add_argument(
        '--split', 
        type=str, 
        choices=['train', 'dev', 'test'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Process all splits'
    )
    parser.add_argument(
        '--frames_dir',
        type=str,
        default='data/raw_data/phoenix-2014-multisigner/features/fullFrame-210x260px',
        help='Base directory containing video frames'
    )
    parser.add_argument(
        '--annotation_dir',
        type=str,
        default='data/raw_data/phoenix-2014-multisigner/annotations/manual',
        help='Directory containing annotation CSV files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/teacher_features/mobilenet_v3',
        help='Output directory for extracted features'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    frames_dir = Path(args.frames_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("MobileNetV3 Feature Extraction")
    logger.info(f"Frames directory: {frames_dir}")
    logger.info(f"Annotation directory: {annotation_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    if args.all:
        splits = ['train', 'dev', 'test']
    elif args.split:
        splits = [args.split]
    else:
        parser.error("Please specify --split or --all")
        return
    
    for split in splits:
        extract_features_for_split(
            split=split,
            frames_base_dir=frames_dir,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
    
    logger.info("\n" + "="*60)
    logger.info("Feature extraction complete!")
    logger.info("="*60)
    logger.info(f"\nMobileNetV3 features saved to: {output_dir}")
    logger.info(f"Feature dimension: 576 per frame")
    logger.info("\nNext step: Combine with MediaPipe features for training")


if __name__ == '__main__':
    main()

