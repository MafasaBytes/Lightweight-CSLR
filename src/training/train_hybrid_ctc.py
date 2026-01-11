"""
Hybrid Model Training (Encoder + CTC) for Sign Language Recognition.

Moved from project root `train_hybrid_model.py` to keep entrypoints under `src/training/`.

Run:
  .\\venv\\Scripts\\python -m src.training.train_hybrid_ctc  [args...]
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.data.dataset import Vocabulary
from src.data.hybrid_dataset import HybridFeatureDataset, collate_fn_ctc
from src.models.hybrid_ctc_encoder import HybridSeq2Seq
from src.models.hybrid_ctc_transformer import HybridTransformerCTC
from src.models.hybrid_late_fusion_ctc import HybridLateFusionCTC
from src.training.hybrid_ctc_trainer import train_epoch, validate
from src.utils.augmentation import FeatureAugmentation


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_hybrid_ctc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def plot_training_curves(
    train_losses,
    val_losses,
    val_wers,
    learning_rates,
    best_wer,
    output_dir: Path,
):
    """
    Plot training progress (adapted from src/training/train.py).

    Saves:
    - output_dir/training_curves.png
    - figures/hybrid_ctc/training_curves.png
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    epochs = range(1, len(train_losses) + 1)

    if sns is not None:
        try:
            plt.style.use("seaborn-v0_8-paper")
        except Exception:
            try:
                plt.style.use("seaborn-paper")
            except Exception:
                sns.set_style("whitegrid")
        sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hybrid Encoder+CTC Model - Training Progress", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2, alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CTC Loss")
    ax1.set_title("Loss", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)

    ax2 = axes[0, 1]
    ax2.plot(list(epochs), val_wers, "g-", linewidth=2.5, marker="o", markersize=5, alpha=0.8)
    ax2.axhline(y=best_wer, color="r", linestyle="--", label=f"Best WER: {best_wer:.2f}%", linewidth=2)
    ax2.axhline(y=25.0, color="orange", linestyle=":", label="Target: 25%", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("WER (%)")
    ax2.set_title("Validation WER", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)

    ax3 = axes[1, 0]
    if len(learning_rates) > 0:
        ax3.plot(list(epochs), learning_rates[: len(train_losses)], "m-", linewidth=2, marker="s", markersize=4, alpha=0.8)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate", fontweight="bold")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3, which="both")
        ax3.set_xlim(left=1)
    else:
        ax3.text(0.5, 0.5, "No LR data available", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Learning Rate", fontweight="bold")

    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2, alpha=0.7)
    line2 = ax4.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2, alpha=0.7)
    line3 = ax4_twin.plot(epochs, val_wers, "g-", label="Val WER (%)", linewidth=2.5, marker="o", markersize=5, alpha=0.8)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4_twin.set_ylabel("WER (%)", color="green")
    ax4_twin.tick_params(axis="y", labelcolor="green")
    ax4.set_title("Overview", fontweight="bold")
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=1)

    plt.tight_layout()

    figures_dir = Path("figures/hybrid_ctc")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_path_png = figures_dir / "training_curves.png"
    plt.savefig(plot_path_png, dpi=300, bbox_inches="tight", format="png")
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight", format="png")
    plt.close()
    return plot_path_png


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Encoder+CTC Model")
    parser.add_argument("--mediapipe_dir", type=str, default="data/teacher_features/mediapipe_full")
    parser.add_argument("--mobilenet_dir", type=str, default="data/teacher_features/mobilenet_v3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--encoder",
        type=str,
        default="late_fusion",
        choices=["bilstm", "transformer", "late_fusion"],
    )
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--modality_layers", type=int, default=2)
    parser.add_argument("--fusion_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--onecycle_pct_start", type=float, default=0.1)
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=1e4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/hybrid_model")

    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_augment", action="store_true", default=True)
    parser.add_argument("--time_mask_max", type=int, default=10)
    parser.add_argument("--freq_mask_max", type=int, default=100)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.checkpoint_dir) / f"run_{timestamp}"
    logger = setup_logging(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    if args.learning_rate > 1e-3:
        logger.warning(
            f"learning_rate={args.learning_rate} is quite high for this CTC setup. "
            "If you see NaNs, rerun with --learning_rate 3e-4 (or 1e-4..5e-4)."
        )

    mediapipe_dir = Path(args.mediapipe_dir)
    mobilenet_dir = Path(args.mobilenet_dir)

    annotation_dir = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual")
    if not annotation_dir.exists():
        annotation_dir = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")

    vocab_path = Path("checkpoints/vocabulary.json")
    if not vocab_path.exists():
        logger.error("Vocabulary not found. Please create vocabulary first.")
        return
    vocab = Vocabulary()
    vocab.load(vocab_path)
    logger.info(f"Loaded vocabulary: {len(vocab)} words")

    train_dataset = HybridFeatureDataset(
        mediapipe_dir=mediapipe_dir,
        mobilenet_dir=mobilenet_dir,
        annotation_file=annotation_dir / "train.corpus.csv",
        vocabulary=vocab,
        split="train",
        max_seq_length=300,
        augment=True,
        normalize=True,
    )
    val_dataset = HybridFeatureDataset(
        mediapipe_dir=mediapipe_dir,
        mobilenet_dir=mobilenet_dir,
        annotation_file=annotation_dir / "dev.corpus.csv",
        vocabulary=vocab,
        split="dev",
        max_seq_length=300,
        augment=False,
        normalize=True,
    )
    val_dataset.set_stats(train_dataset.mp_mean, train_dataset.mp_std, train_dataset.mn_mean, train_dataset.mn_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_ctc,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_ctc,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if args.encoder == "late_fusion":
        logger.info("Using Late Fusion: Encode(MP) + Encode(MN) → CrossAttn → Joint → CTC")
        model = HybridLateFusionCTC(
            vocab_size=len(vocab),
            mediapipe_dim=6516,
            mobilenet_dim=576,
            hidden_dim=args.hidden_dim,
            modality_layers=args.modality_layers,
            fusion_layers=args.fusion_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        ).to(device)
    elif args.encoder == "transformer":
        logger.info("Using Transformer Encoder (early fusion)")
        model = HybridTransformerCTC(
            vocab_size=len(vocab),
            mediapipe_dim=6516,
            mobilenet_dim=576,
            hidden_dim=args.hidden_dim,
            encoder_layers=args.encoder_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        logger.info("Using BiLSTM Encoder")
        model = HybridSeq2Seq(
            vocab_size=len(vocab),
            mediapipe_dim=6516,
            mobilenet_dim=576,
            hidden_dim=args.hidden_dim,
            encoder_layers=args.encoder_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.onecycle_pct_start,
        anneal_strategy="cos",
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
    )
    scaler = GradScaler()

    augmenter = None
    if args.use_augment:
        augmenter = FeatureAugmentation(
            time_mask_max=args.time_mask_max,
            time_mask_num=2,
            freq_mask_max=args.freq_mask_max,
            freq_mask_num=2,
            p=0.5,
        ).to(device)
        logger.info(f"SpecAugment enabled: time_mask={args.time_mask_max}, freq_mask={args.freq_mask_max}")

    best_wer = float("inf")
    patience_counter = 0

    train_losses, val_losses, val_wers, learning_rates = [], [], [], []

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
        val_loss, val_wer, _preds, _targets = validate(model, val_loader, criterion, device, vocab.idx2word, logger)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_wers.append(val_wer)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        logger.info(f"Train - Loss: {train_loss:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, WER: {val_wer:.1f}% (lr={optimizer.param_groups[0]['lr']:.3e})")

        try:
            if epoch == 1 or epoch % 5 == 0 or val_wer < best_wer:
                plot_training_curves(
                    train_losses,
                    val_losses,
                    val_wers,
                    learning_rates,
                    best_wer if best_wer < float("inf") else val_wer,
                    run_dir,
                )
        except Exception as e:
            logger.warning(f"Failed to save training curves: {e}")

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
    results = {
        "best_wer": best_wer,
        "epochs_trained": epoch,
        "args": vars(args),
        "model_size_mb": model_size_mb,
        "model_parameters": num_params,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


