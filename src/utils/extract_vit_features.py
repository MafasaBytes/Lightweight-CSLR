"""
ViT-B/16 Visual Feature Extraction for Sign Language Recognition (Phoenix PNG frames).

This is a drop-in replacement for MobileNetV3 feature extraction:
- Reads Phoenix frame sequences from `data/raw_data/.../features/fullFrame-210x260px/{split}/{video_id}/{cam}/*.png`
- Runs pretrained ViT-B/16 (ImageNet) per-frame
- Saves per-frame embeddings to .npz: (T, 768)

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ViT_B_16_Weights, vit_b_16
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ViTB16FeatureExtractor(nn.Module):
    """
    Extracts CLS embeddings from torchvision ViT-B/16.
    Output dim: 768
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.feature_dim = 768

        # We only need the encoder; keep full model for convenience but do not use classification head.
        self.vit.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float tensor
        Returns:
            (B, 768) CLS embeddings
        """
        # torchvision ViT internals:
        # - conv_proj -> (B, hidden_dim, n_h, n_w)
        # - flatten to tokens + prepend class token
        # - encoder -> take class token
        n = x.shape[0]
        x = self.vit._process_input(x)  # (B, num_patches, hidden_dim)
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1+num_patches, hidden_dim)
        x = self.vit.encoder(x)
        x = x[:, 0]  # CLS (B, hidden_dim)
        return x


def _find_frame_dir(video_dir: Path) -> Optional[Path]:
    if not video_dir.exists():
        return None
    subdirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
    return subdirs[0] if subdirs else None


def _extract_frame_number(filename: str) -> int:
    # '..._fn000123-0.png' -> 123
    try:
        if "_fn" in filename:
            fn_part = filename.split("_fn")[1]
            num_str = fn_part.split("-")[0]
            return int(num_str)
    except Exception:
        pass
    return 0


class PhoenixFrameSequenceDataset(Dataset):
    """
    Loads full frame sequences for each video_id (capped to max_frames).
    Returns:
      - frames: (T, C, H, W)
      - video_id: str
      - num_frames: int
    """

    def __init__(
        self,
        frames_root: Path,
        annotation_csv: Path,
        split: str,
        transform: T.Compose,
        max_frames: int = 300,
    ):
        self.frames_root = Path(frames_root)
        self.split = split
        self.transform = transform
        self.max_frames = max_frames

        df = pd.read_csv(annotation_csv, sep="|", on_bad_lines="skip")
        if "id" not in df.columns:
            df.columns = [c.strip() for c in df.columns]
        self.video_ids: List[str] = [str(x) for x in df["id"].tolist() if isinstance(x, str)]

        # Filter to those that actually exist on disk
        filtered: List[str] = []
        for vid in self.video_ids:
            vdir = self.frames_root / self.split / vid
            if _find_frame_dir(vdir) is not None:
                filtered.append(vid)
        self.video_ids = filtered
        logger.info(f"Loaded {len(self.video_ids)} videos for split={split}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict:
        video_id = self.video_ids[idx]
        video_dir = self.frames_root / self.split / video_id
        frame_dir = _find_frame_dir(video_dir)
        if frame_dir is None:
            raise FileNotFoundError(f"Missing frame directory for {video_id}: {video_dir}")

        frame_files = sorted(
            frame_dir.glob("*.png"),
            key=lambda p: _extract_frame_number(p.name),
        )[: self.max_frames]

        frames: List[torch.Tensor] = []
        for f in frame_files:
            img = Image.open(f).convert("RGB")
            frames.append(self.transform(img))

        if not frames:
            # Should not happen, but keep it robust.
            frames = [torch.zeros(3, 224, 224)]

        frames_tensor = torch.stack(frames, dim=0)  # (T,C,H,W)
        return {"video_id": video_id, "frames": frames_tensor, "num_frames": len(frames)}


def collate_fn(batch: List[Dict]) -> Dict:
    # Variable length sequences: keep as list for extraction loop (we chunk per video anyway).
    return batch


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser("Extract ViT-B/16 frame features to .npz")
    parser.add_argument("--frames_root", type=str, default="data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px")
    parser.add_argument("--annotations_root", type=str, default="data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev", "test"])
    parser.add_argument("--output_dir", type=str, default="data/teacher_features/vit_b16")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1, help="Videos per batch (keep 1; frames are chunked internally).")
    parser.add_argument("--frame_batch", type=int, default=64, help="Frames per forward pass through ViT.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    logger.info(f"Device: {device}")

    # Use official preprocessing for ViT weights.
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transform = weights.transforms()  # Resize/crop/normalize compatible with ViT

    ann_csv = Path(args.annotations_root) / f"{args.split}.SI5.corpus.csv"
    if not ann_csv.exists():
        # fallback (multisigner naming)
        alt = Path(args.annotations_root) / f"{args.split}.corpus.csv"
        if alt.exists():
            ann_csv = alt
        else:
            raise FileNotFoundError(f"Annotation CSV not found for split={args.split}: {ann_csv}")

    ds = PhoenixFrameSequenceDataset(
        frames_root=Path(args.frames_root),
        annotation_csv=ann_csv,
        split=args.split,
        transform=transform,
        max_frames=args.max_frames,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ViTB16FeatureExtractor(pretrained=True).to(device)
    model.eval()

    for batch in tqdm(dl, desc=f"Extract {args.split}"):
        # batch is a list of items because collate_fn returns list
        for item in batch:
            video_id = item["video_id"]
            frames: torch.Tensor = item["frames"]  # (T,C,H,W)
            T_len = frames.shape[0]

            out_path = out_dir / f"{video_id}.npz"
            if out_path.exists():
                continue

            feats = []
            # Chunk frames to avoid OOM.
            for s in range(0, T_len, args.frame_batch):
                x = frames[s : s + args.frame_batch].to(device, non_blocking=True)
                with torch.no_grad():
                    emb = model(x).detach().float().cpu().numpy()  # (Bf, 768)
                feats.append(emb)

            feats_np = np.concatenate(feats, axis=0)  # (T, 768)
            np.savez_compressed(out_path, features=feats_np, video_id=video_id)

    logger.info(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()


