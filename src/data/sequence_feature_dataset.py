"""
Generic feature-sequence dataset for CTC training.

Loads pre-extracted features from:
  <features_dir>/<split>/<video_id>.npz  (expects key 'features' or 'arr_0')

And labels from Phoenix corpus CSV:
  id|folder|signer|annotation

We intentionally filter tokens:
- remove special markers (__*), loc-*, IX (and other excluded patterns)
- keep only tokens that exist in the provided vocab (to avoid mapping OOV -> blank=0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


EXCLUDE_PATTERNS = [
    "__",
    "loc-",
    "cl-",
    "lh-",
    "rh-",
    "IX",
    "WG",
    "$GEST",
    "PLUSPLUS",
    "POS",
]


def _auto_split_csv(annotation_dir: Path, split: str) -> Path:
    # Prefer SI5 naming if present.
    p1 = annotation_dir / f"{split}.SI5.corpus.csv"
    if p1.exists():
        return p1
    p2 = annotation_dir / f"{split}.corpus.csv"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find annotation CSV for split={split} under {annotation_dir}")

def _load_adaptsign_info_npy(preprocess_dir: Path, dataset: str, split: str) -> list[dict]:
    """
    Load official AdaptSign split list from `adaptsign_repo/preprocess/<dataset>/<split>_info.npy`.

    Returns list of dicts with:
      - video_id: str (fileid)
      - annotation: str (label)
    """
    info_path = preprocess_dir / dataset / f"{split}_info.npy"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find AdaptSign split list: {info_path}")
    info = np.load(str(info_path), allow_pickle=True).item()
    keys = sorted([k for k in info.keys() if isinstance(k, int)])
    out: list[dict] = []
    for k in keys:
        v = info[k]
        fileid = v.get("fileid")
        label = v.get("label", "")
        if fileid and label:
            out.append({"video_id": str(fileid), "annotation": str(label).strip()})
    return out


def _load_npz_features(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "features" in data:
        x = data["features"]
    else:
        x = data["arr_0"]
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _filter_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for w in tokens:
        w = w.strip()
        if not w:
            continue
        # exclude patterns / markers
        if any(pat in w for pat in EXCLUDE_PATTERNS):
            continue
        # exclude single letters (often fingerspelling markers)
        if len(w) == 1 and w.isalpha():
            continue
        out.append(w)
    return out


class SequenceFeatureDataset(Dataset):
    """
    Pre-extracted feature dataset for CTC.

    Returns dict:
      - features: (T, D) float32
      - labels: (L,) int64 (no blanks)
      - video_id: str
    """

    def __init__(
        self,
        features_dir: Path,
        annotation_dir: Path,
        vocabulary,  # expects .word2idx and .idx2word (hybrid vocab)
        split: str,
        split_source: str = "si5",  # "si5" | "adaptsign_official"
        adaptsign_preprocess_dir: Path = Path("adaptsign_repo/preprocess"),
        adaptsign_dataset: str = "phoenix2014",
        teacher_logits_dir: Optional[Path] = None,
        max_seq_length: int = 300,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        self.features_dir = Path(features_dir) / split
        self.annotation_dir = Path(annotation_dir)
        self.vocab = vocabulary
        self.split = split
        self.split_source = split_source
        self.adaptsign_preprocess_dir = Path(adaptsign_preprocess_dir)
        self.adaptsign_dataset = adaptsign_dataset
        self.teacher_logits_dir = Path(teacher_logits_dir) if teacher_logits_dir is not None else None
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.mean = mean
        self.std = std

        samples: List[Dict] = []
        if split_source == "adaptsign_official":
            rows = _load_adaptsign_info_npy(self.adaptsign_preprocess_dir, self.adaptsign_dataset, split)
            for row in rows:
                video_id = row["video_id"]
                ann = row["annotation"]

                feat_path = self.features_dir / f"{video_id}.npz"
                if not feat_path.exists():
                    continue

                teacher_path = None
                if self.teacher_logits_dir is not None:
                    teacher_path = self.teacher_logits_dir / split / f"{video_id}.npz"
                    if not teacher_path.exists():
                        continue

                tokens = _filter_tokens(ann.split())
                kept: List[str] = [t for t in tokens if t in self.vocab.word2idx and int(self.vocab.word2idx[t]) != 0]
                if not kept:
                    continue

                label_ids = [int(self.vocab.word2idx[t]) for t in kept]
                if any(t == 0 for t in label_ids):
                    continue

                s = {"video_id": video_id, "feat_path": feat_path, "labels": label_ids}
                if teacher_path is not None:
                    s["teacher_path"] = teacher_path
                samples.append(s)
        else:
            csv_path = _auto_split_csv(self.annotation_dir, split)
            df = pd.read_csv(csv_path, sep="|", on_bad_lines="skip")
            if "annotation" not in df.columns:
                df.columns = [c.strip() for c in df.columns]

            for _, row in df.iterrows():
                video_id = row["id"] if "id" in row else row.get("name")
                ann = row.get("annotation", "")
                if not isinstance(video_id, str) or not isinstance(ann, str):
                    continue

                feat_path = self.features_dir / f"{video_id}.npz"
                if not feat_path.exists():
                    continue

                teacher_path = None
                if self.teacher_logits_dir is not None:
                    teacher_path = self.teacher_logits_dir / split / f"{video_id}.npz"
                    if not teacher_path.exists():
                        continue

                tokens = _filter_tokens(ann.split())
                kept: List[str] = [t for t in tokens if t in self.vocab.word2idx and int(self.vocab.word2idx[t]) != 0]
                if not kept:
                    continue

                label_ids = [int(self.vocab.word2idx[t]) for t in kept]
                if any(t == 0 for t in label_ids):
                    continue

                s = {"video_id": video_id, "feat_path": feat_path, "labels": label_ids}
                if teacher_path is not None:
                    s["teacher_path"] = teacher_path
                samples.append(s)

        self.samples = samples
        logger.info(
            f"Loaded {len(self.samples)} samples for split={split} "
            f"(split_source={split_source}) from {features_dir}"
        )

        # compute stats on train split if requested and not provided
        if self.split == "train" and self.normalize and (self.mean is None or self.std is None):
            self._compute_stats()

    def _compute_stats(self, max_files: int = 500) -> None:
        idx = np.random.choice(len(self.samples), min(max_files, len(self.samples)), replace=False)
        feats = []
        for i in idx:
            x = _load_npz_features(self.samples[int(i)]["feat_path"])
            x = x[: self.max_seq_length]
            feats.append(x.reshape(-1, x.shape[-1]))
        if feats:
            all_x = np.concatenate(feats, axis=0)
            self.mean = np.nanmean(all_x, axis=0)
            self.std = np.nanstd(all_x, axis=0) + 1e-8

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        x = _load_npz_features(s["feat_path"])
        x = x[: self.max_seq_length]

        if self.normalize and self.mean is not None:
            x = (x - self.mean) / self.std

        features = torch.from_numpy(x)  # (T,D)
        labels = torch.tensor(s["labels"], dtype=torch.long)  # (L,)

        out = {"features": features, "labels": labels, "video_id": s["video_id"]}
        if "teacher_path" in s:
            # Load lazily per-sample to let DataLoader workers overlap IO.
            t = np.load(s["teacher_path"], allow_pickle=False)
            # cache file format: logits (T_teacher,V), feat_len (scalar)
            logits = t["logits"]
            feat_len = int(np.asarray(t["feat_len"]).item())
            # keep as fp16 on CPU to reduce batch memory; cast later
            logits = logits[:feat_len].astype(np.float16, copy=False)
            out["teacher_logits"] = torch.from_numpy(logits)  # (T_teacher, V)
            out["teacher_len"] = int(feat_len)
        return out


def collate_fn_ctc_features(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Sort by T desc
    batch = sorted(batch, key=lambda b: b["features"].shape[0], reverse=True)
    feats = [b["features"] for b in batch]
    labels = [b["labels"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    has_teacher = all(("teacher_logits" in b and "teacher_len" in b) for b in batch)

    max_T = max(f.shape[0] for f in feats)
    D = feats[0].shape[1]
    B = len(batch)

    x = torch.zeros(B, max_T, D, dtype=torch.float32)
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f

    label_lengths = torch.tensor([int(l.numel()) for l in labels], dtype=torch.long)
    y = torch.cat(labels, dim=0).to(torch.long)  # (sum_L,)

    return {
        "features": x,
        "input_lengths": input_lengths,
        "labels": y,
        "label_lengths": label_lengths,
        "video_ids": video_ids,
        **(
            {
                # Keep as a list of tensors to avoid padding huge (B,T,V) in collate.
                "teacher_logits": [b["teacher_logits"] for b in batch],
                "teacher_lens": torch.tensor([int(b["teacher_len"]) for b in batch], dtype=torch.long),
            }
            if has_teacher
            else {}
        ),
    }


