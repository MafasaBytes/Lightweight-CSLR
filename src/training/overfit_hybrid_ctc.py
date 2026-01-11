"""
Overfit test for the Hybrid encoder+CTC architecture.

Moved from project root `overfit_hybrid_ctc.py` to keep entrypoints under `src/training/`.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.data.dataset import Vocabulary
from src.data.hybrid_dataset import HybridFeatureDataset, collate_fn_ctc
from src.models.hybrid_ctc_encoder import HybridSeq2Seq
from src.training.hybrid_ctc_trainer import validate


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_ctc_valid_indices(dataset: HybridFeatureDataset, num_samples: int, downsample_factor: int = 4) -> List[int]:
    chosen: List[int] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        in_len = int(sample["features"].shape[0] // downsample_factor)
        tgt_len = int(sample["labels"].shape[0])
        if in_len >= tgt_len and tgt_len > 0:
            chosen.append(i)
        if len(chosen) >= num_samples:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit test: Hybrid Encoder+CTC")
    parser.add_argument("--mediapipe_dir", type=str, default="data/teacher_features/mediapipe_full")
    parser.add_argument("--mobilenet_dir", type=str, default="data/teacher_features/mobilenet_v3")
    parser.add_argument("--subset_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_seq_length", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("overfit_hybrid_ctc")

    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    vocab_path = Path("checkpoints/vocabulary.json")
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary not found at checkpoints/vocabulary.json")
    vocab = Vocabulary()
    vocab.load(vocab_path)
    if vocab.word2idx.get("<blank>", None) != 0:
        raise ValueError("Expected vocabulary index 0 to be the CTC blank token (<blank>).")

    annotation_dir = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual")
    if not annotation_dir.exists():
        annotation_dir = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")

    train_dataset = HybridFeatureDataset(
        mediapipe_dir=Path(args.mediapipe_dir),
        mobilenet_dir=Path(args.mobilenet_dir),
        annotation_file=annotation_dir / "train.corpus.csv",
        vocabulary=vocab,
        split="train",
        max_seq_length=args.max_seq_length,
        augment=False,
        normalize=True,
    )

    indices = _pick_ctc_valid_indices(train_dataset, num_samples=args.subset_size, downsample_factor=4)
    if len(indices) < args.subset_size:
        raise RuntimeError(
            f"Could only find {len(indices)}/{args.subset_size} CTC-valid samples (after /4 downsampling)."
        )

    logger.info(f"Overfit subset indices: {indices}")
    subset = Subset(train_dataset, indices)

    batch_size = min(args.batch_size, len(subset))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_ctc,
        pin_memory=True,
    )

    model = HybridSeq2Seq(
        vocab_size=len(vocab),
        mediapipe_dim=6516,
        mobilenet_dim=576,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info("Starting overfit training (single subset)...")

    best_wer = float("inf")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["labels"].to(device)
            target_lengths = batch["label_lengths"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=device.type == "cuda"):
                log_probs, out_lengths = model(features, input_lengths)
                in_lens = out_lengths.cpu() if log_probs.is_cuda else out_lengths
                tgt_lens = target_lengths.cpu() if log_probs.is_cuda else target_lengths
                loss = criterion(log_probs.permute(1, 0, 2), targets, in_lens, tgt_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        val_loss, wer, _, _ = validate(model, loader, criterion, device, vocab.idx2word, logger)
        best_wer = min(best_wer, wer)

        if epoch == 1 or epoch % args.log_every == 0 or wer <= 1.0:
            logger.info(
                f"[{run_id}] epoch={epoch:03d} train_loss={avg_loss:.4f} subset_loss={val_loss:.4f} "
                f"subset_WER={wer:.2f}% (best={best_wer:.2f}%)"
            )
        if wer <= 1.0:
            logger.info("Overfit criterion reached (WER <= 1%). Stopping.")
            break


if __name__ == "__main__":
    main()


