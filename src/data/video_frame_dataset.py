"""
Video Frame Dataset for ViT-based models.

Loads raw RGB frames from Phoenix-2014 for end-to-end training with ViT backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _find_frame_dir(video_dir: Path) -> Optional[Path]:
    """Find the first subdirectory containing frames (e.g., '1/')."""
    if not video_dir.exists():
        return None
    subdirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
    return subdirs[0] if subdirs else None


def _extract_frame_number(filename: str) -> int:
    """Extract frame number from Phoenix filename like 'xxx_fn000123-0.png'."""
    try:
        if "_fn" in filename:
            fn_part = filename.split("_fn")[1]
            num_str = fn_part.split("-")[0]
            return int(num_str)
    except Exception:
        pass
    return 0


class PhoenixVideoDataset(Dataset):
    """
    Dataset for loading raw video frames from Phoenix-2014.

    Args:
        frames_root: Root directory containing frame folders (e.g., fullFrame-210x260px/{split}/)
        annotation_file: Path to annotation CSV file
        vocabulary: Vocabulary object with word2idx mapping
        split: 'train', 'dev', or 'test'
        max_frames: Maximum number of frames per video
        transform: Image transform (defaults to ViT-compatible transform)
    """

    def __init__(
        self,
        frames_root: Path,
        annotation_file: Path,
        vocabulary,
        split: str = "train",
        max_frames: int = 300,
        transform: Optional[T.Compose] = None,
    ):
        self.frames_root = Path(frames_root) / split
        self.vocabulary = vocabulary
        self.split = split
        self.max_frames = max_frames

        # Default transform for ViT-B/16 (224x224, ImageNet normalization)
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        self.samples = self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file: Path) -> List[Dict]:
        """Load and filter annotations."""
        samples = []

        df = pd.read_csv(annotation_file, sep="|", on_bad_lines="skip")
        if "id" not in df.columns:
            df.columns = [c.strip() for c in df.columns]

        for _, row in df.iterrows():
            video_id = str(row.get("id", ""))
            annotation = str(row.get("annotation", ""))

            if not video_id or not annotation:
                continue

            # Check frame directory exists
            video_dir = self.frames_root / video_id
            frame_dir = _find_frame_dir(video_dir)
            if frame_dir is None:
                continue

            # Process annotation: filter special tokens, keep only vocab words
            words = []
            for word in annotation.split():
                word = word.strip()
                if not word:
                    continue
                # Skip special tokens
                if word.startswith("__") or word.startswith("loc-"):
                    continue
                if word in {"IX", "S+H"}:
                    continue
                # Only include words in vocabulary (skip unknown)
                if word in self.vocabulary.word2idx:
                    words.append(word)

            if words:
                samples.append({
                    "video_id": video_id,
                    "frame_dir": frame_dir,
                    "annotation": words,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_id = sample["video_id"]
        frame_dir = sample["frame_dir"]
        annotation = sample["annotation"]

        # Load frame files
        frame_files = sorted(
            frame_dir.glob("*.png"),
            key=lambda p: _extract_frame_number(p.name),
        )[: self.max_frames]

        # Load and transform frames
        frames = []
        for f in frame_files:
            try:
                img = Image.open(f).convert("RGB")
                frames.append(self.transform(img))
            except Exception:
                continue

        if not frames:
            # Fallback: single black frame
            frames = [torch.zeros(3, 224, 224)]

        frames_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)

        # Encode labels (skip blank token index 0)
        labels = []
        for word in annotation:
            idx = self.vocabulary.word2idx.get(word, 0)
            if idx > 0:  # Don't include blank in targets
                labels.append(idx)

        labels_tensor = torch.LongTensor(labels) if labels else torch.LongTensor([1])

        return {
            "video_id": video_id,
            "frames": frames_tensor,
            "num_frames": len(frames),
            "labels": labels_tensor,
            "label_length": len(labels_tensor),
        }


def collate_fn_video_ctc(batch: List[Dict]) -> Dict:
    """
    Collate function for video CTC training.

    Returns:
        - frames: (B, T_max, C, H, W) padded frames
        - input_lengths: (B,) number of frames per sample
        - labels: (sum_L,) concatenated label sequences
        - label_lengths: (B,) length of each label sequence
        - video_ids: List[str]
    """
    # Sort by number of frames (descending) for potential packing
    batch = sorted(batch, key=lambda x: x["num_frames"], reverse=True)

    video_ids = [item["video_id"] for item in batch]
    num_frames = [item["num_frames"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    label_lengths = [item["label_length"] for item in batch]

    B = len(batch)
    T_max = max(num_frames)
    C, H, W = batch[0]["frames"].shape[1:]

    # Pad frames
    padded_frames = torch.zeros(B, T_max, C, H, W)
    for i, item in enumerate(batch):
        T_i = item["num_frames"]
        padded_frames[i, :T_i] = item["frames"]

    # Concatenate labels for CTC
    labels_concat = torch.cat(labels_list, dim=0)

    return {
        "video_ids": video_ids,
        "frames": padded_frames,
        "input_lengths": torch.LongTensor(num_frames),
        "labels": labels_concat,
        "label_lengths": torch.LongTensor(label_lengths),
    }

