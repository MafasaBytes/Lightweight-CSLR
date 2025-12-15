"""
Train a visual-feature baseline: (features -> BiLSTM -> CTC).

Baseline per proposal: MobileNetV3 visual features (576-d) + temporal encoder + CTC.
We also allow swapping in ViT-B/16 features (768-d) with the same trainer.

This script intentionally reuses:
- `src/training/hybrid_ctc_trainer.py` for training/validation + WER
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.data.dataset import Vocabulary as HybridVocabulary
from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.models.feature_ctc_bilstm import FeatureBiLSTMCTC
from src.training.hybrid_ctc_trainer import train_epoch, validate
from src.utils.augmentation import FeatureAugmentation


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("visual_baseline")


def plot_training_curves(train_losses, val_losses, val_wers, lrs, out_dir: Path) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_losses, label="train")
    axes[0, 0].plot(epochs, val_losses, label="val")
    axes[0, 0].set_title("CTC loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, val_wers, color="red")
    axes[0, 1].set_title("WER (%)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, lrs, color="blue")
    axes[1, 0].set_title("LR")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(val_losses, val_wers, s=10, alpha=0.6)
    axes[1, 1].set_xlabel("val_loss")
    axes[1, 1].set_ylabel("val_WER")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Train visual feature baseline (CTC)")
    parser.add_argument("--features_dir", type=str, default="data/teacher_features/mobilenet_v3")
    parser.add_argument("--feature_dim", type=int, default=576, help="576 for MobileNetV3 features, 768 for ViT-B/16")
    parser.add_argument("--annotation_dir", type=str, default="data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")
    parser.add_argument("--vocab_json", type=str, default="checkpoints/vocabulary.json")

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=300)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_subsample", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.03)
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=1e4)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--use_augment", action="store_true", default=True)
    parser.add_argument("--time_mask_max", type=int, default=10)
    parser.add_argument("--freq_mask_max", type=int, default=100)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Entropy regularization weight")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/visual_baseline")
    args = parser.parse_args()

    run_dir = Path(args.checkpoint_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    vocab_path = Path(args.vocab_json)
    vocab = HybridVocabulary()
    vocab.load(vocab_path)
    logger.info(f"Loaded vocab: {len(vocab)} words (blank_id={vocab.blank_id})")

    train_ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path(args.annotation_dir),
        vocabulary=vocab,
        split="train",
        max_seq_length=args.max_seq_length,
        normalize=True,
    )
    dev_ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path(args.annotation_dir),
        vocabulary=vocab,
        split="dev",
        max_seq_length=args.max_seq_length,
        normalize=True,
        mean=train_ds.mean,
        std=train_ds.std,
    )
    dev_ds.set_stats(train_ds.mean, train_ds.std)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_ctc_features,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_ctc_features,
    )
    logger.info(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    model = FeatureBiLSTMCTC(
        vocab_size=len(vocab),
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        subsample=(not args.no_subsample),
    ).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_params:,}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=args.onecycle_pct_start,
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
    )

    augmenter = None
    if args.use_augment:
        augmenter = FeatureAugmentation(time_mask_max=args.time_mask_max, freq_mask_max=args.freq_mask_max)
        logger.info(f"SpecAugment enabled: time_mask={args.time_mask_max}, freq_mask={args.freq_mask_max}")

    best_wer = float("inf")
    bad_epochs = 0
    train_losses, val_losses, val_wers, lrs = [], [], [], []

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
            augmenter=augmenter,
            label_smoothing=args.label_smoothing,
        )
        val_loss, val_wer, _preds, _targets = validate(model, dev_loader, criterion, device, vocab.idx2word, logger)

        lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_wers.append(val_wer)
        lrs.append(lr)

        logger.info(f"Train - Loss: {train_loss:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, WER: {val_wer:.2f}% (lr={lr:.3e})")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_wer": best_wer,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last.pt")

        if val_wer < best_wer:
            best_wer = val_wer
            bad_epochs = 0
            ckpt["best_wer"] = best_wer
            torch.save(ckpt, run_dir / "best.pt")
            logger.info(f"New best WER: {best_wer:.2f}%")
        else:
            bad_epochs += 1

        plot_training_curves(train_losses, val_losses, val_wers, lrs, run_dir)
        if bad_epochs >= args.patience:
            logger.info(f"Early stopping (patience={args.patience}). Best WER: {best_wer:.2f}%")
            break


if __name__ == "__main__":
    main()


