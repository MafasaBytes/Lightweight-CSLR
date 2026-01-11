"""
Train ViT-B/16 CTC Teacher on Phoenix-2014 frames.

This produces a high-capacity video teacher checkpoint for:
- Direct decoding (teacher WER evaluation)
- Knowledge distillation / pseudo-labeling for lightweight students

Usage:
    python -m src.training.train_vit_teacher_ctc --epochs 100 --batch_size 4 --freeze_backbone

The teacher uses:
- ViT-B/16 backbone (ImageNet pretrained)
- Temporal Transformer encoder
- CTC loss
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import Vocabulary
from src.data.video_frame_dataset import PhoenixVideoDataset, collate_fn_video_ctc
from src.models.vit_ctc_teacher import ViTCTCTeacher
from src.utils.metrics import compute_wer


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def decode_ctc_greedy(
    log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int = 0
) -> List[List[int]]:
    """CTC greedy decoding."""
    pred_ids = log_probs.argmax(dim=-1)  # (B, T)
    B, T = pred_ids.shape
    decoded = []

    for b in range(B):
        L = min(int(lengths[b].item()), T)
        seq = pred_ids[b, :L].tolist()
        out = []
        prev = blank_id
        for tok in seq:
            if tok != blank_id and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)

    return decoded


def ids_to_string(ids: List[int], idx2word: Dict[int, str]) -> str:
    """Convert token IDs to space-separated string."""
    words = []
    for i in ids:
        w = idx2word.get(i, "")
        if w and w not in {"<blank>", "<pad>"}:
            words.append(w)
    return " ".join(words)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    scheduler: OneCycleLR,
    logger: logging.Logger,
    grad_accum_steps: int = 1,
) -> float:
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        with autocast("cuda", enabled=device.type == "cuda"):
            log_probs, out_lengths = model(frames, input_lengths)

        # CTC loss (compute in FP32)
        log_probs_ctc = log_probs.float().permute(1, 0, 2)  # (T', B, V)
        in_lens = out_lengths.cpu()
        tgt_lens = label_lengths.cpu()

        loss = criterion(log_probs_ctc, labels, in_lens, tgt_lens)
        loss = loss / grad_accum_steps  # Scale for accumulation

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss: {loss.item()}, skipping batch")
            continue

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    idx2word: Dict[int, str],
    logger: logging.Logger,
) -> Tuple[float, float, List[str], List[str]]:
    """Validate and compute WER."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Validation"):
        frames = batch["frames"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        log_probs, out_lengths = model(frames, input_lengths)

        # CTC loss
        log_probs_ctc = log_probs.float().permute(1, 0, 2)
        in_lens = out_lengths.cpu()
        tgt_lens = label_lengths.cpu()

        loss = criterion(log_probs_ctc, labels, in_lens, tgt_lens)
        if torch.isfinite(loss):
            total_loss += loss.item()

        # Decode predictions
        decoded = decode_ctc_greedy(log_probs, out_lengths, blank_id=0)

        # Split concatenated targets
        offset = 0
        for i, L in enumerate(label_lengths.tolist()):
            target_ids = labels[offset : offset + L].tolist()
            offset += L

            pred_str = ids_to_string(decoded[i], idx2word)
            target_str = ids_to_string(target_ids, idx2word)

            all_preds.append(pred_str if pred_str else "(empty)")
            all_targets.append(target_str if target_str else "(empty)")

    avg_loss = total_loss / max(1, len(dataloader))
    wer = compute_wer(all_preds, all_targets)

    return avg_loss, wer, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train ViT-B/16 CTC Teacher")
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px",
    )
    parser.add_argument(
        "--annotations_root",
        type=str,
        default="data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual",
    )
    parser.add_argument("--vocab_json", type=str, default="checkpoints/vocabulary.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=300)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze ViT backbone")
    parser.add_argument("--subsample_factor", type=int, default=4)

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vit_teacher")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"run_{timestamp}"
    logger = setup_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # Vocabulary
    vocab_path = Path(args.vocab_json)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    vocab = Vocabulary()
    vocab.load(vocab_path)
    logger.info(f"Loaded vocab: {len(vocab)} words (blank_id=0)")

    # Datasets
    frames_root = Path(args.frames_root)
    ann_root = Path(args.annotations_root)

    train_ann = ann_root / "train.SI5.corpus.csv"
    if not train_ann.exists():
        train_ann = ann_root / "train.corpus.csv"

    dev_ann = ann_root / "dev.SI5.corpus.csv"
    if not dev_ann.exists():
        dev_ann = ann_root / "dev.corpus.csv"

    train_dataset = PhoenixVideoDataset(
        frames_root=frames_root,
        annotation_file=train_ann,
        vocabulary=vocab,
        split="train",
        max_frames=args.max_frames,
    )

    val_dataset = PhoenixVideoDataset(
        frames_root=frames_root,
        annotation_file=dev_ann,
        vocabulary=vocab,
        split="dev",
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_video_ctc,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_video_ctc,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = ViTCTCTeacher(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        subsample_factor=args.subsample_factor,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Backbone frozen: {args.freeze_backbone}")

    # Training setup
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Only optimize trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    # Training loop
    best_wer = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            scheduler,
            logger,
            grad_accum_steps=args.grad_accum_steps,
        )

        val_loss, val_wer, preds, targets = validate(
            model, val_loader, criterion, device, vocab.idx2word, logger
        )

        # Log samples
        logger.info("Sample predictions:")
        for i in range(min(3, len(preds))):
            logger.info(f"  Target: '{targets[i]}'")
            logger.info(f"  Pred:   '{preds[i]}'")

        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Train - Loss: {train_loss:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, WER: {val_wer:.2f}% (lr={lr:.3e})")

        # Save best model
        if val_wer < best_wer:
            best_wer = val_wer
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_wer": best_wer,
                    "args": vars(args),
                },
                run_dir / "best_model.pt",
            )
            logger.info(f"New best WER: {best_wer:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info(f"\nTraining complete. Best WER: {best_wer:.2f}%")

    # Save results
    results = {
        "best_wer": best_wer,
        "epochs_trained": epoch,
        "args": vars(args),
        "model_size_mb": model_size_mb,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

