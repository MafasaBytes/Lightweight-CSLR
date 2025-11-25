"""
Training script for I3D Teacher Model
This teacher will then be used to distill knowledge to the student
"""

import os
import sys
from pathlib import Path
from typing import Any

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import logging
import math

from src.models.i3d_teacher import I3DTeacher, create_i3d_teacher
from src.data.dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from src.utils.metrics import compute_wer, compute_ser
from src.utils.vocabulary import load_vocabulary_from_file
from src.utils.ctc import ctc_decode
from src.models.pretrained_loader import SignLanguagePretrainedLoader
import matplotlib.pyplot as plt
import seaborn as sns


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.current_epoch = 0
        # keep last lr for logging
        self._last_lr = base_lr

    def step(self, epoch_or_loss):
        """
        Accept either an integer epoch (preferred) or a numeric val loss (legacy).
        If a float is passed that looks like a loss, we increment epoch by 1.
        """
        # If a user accidentally passes a loss (float), treat as "advance one epoch"
        if isinstance(epoch_or_loss, (float,)) and epoch_or_loss >= 0 and epoch_or_loss < 10:
            # legacy call with val_loss -> increment epoch
            epoch = self.current_epoch + 1
        else:
            epoch = int(epoch_or_loss)

        self.current_epoch = epoch

        # Calculate learning rate based on current epoch
        if epoch <= 0:
            # Before training starts, use minimum learning rate
            lr = self.min_lr
        elif epoch < self.warmup_epochs:
            # Warmup phase: linearly increase from min_lr to base_lr
            # epoch is 1-indexed, so we use (epoch - 1) for proper warmup
            warmup_progress = (epoch - 1) / max(1, self.warmup_epochs - 1)
            warmup_progress = min(max(warmup_progress, 0.0), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * warmup_progress
        else:
            # Cosine annealing phase: decrease from base_lr to min_lr
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            progress = min(max(progress, 0.0), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))

        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self._last_lr = lr
        return lr

    def get_last_lr(self):
        return self._last_lr

    def state_dict(self):
        return {
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'current_epoch': self.current_epoch,
            '_last_lr': self._last_lr
        }

    def load_state_dict(self, d):
        self.warmup_epochs = int(d.get('warmup_epochs', self.warmup_epochs))
        self.total_epochs = int(d.get('total_epochs', self.total_epochs))
        self.base_lr = float(d.get('base_lr', self.base_lr))
        self.min_lr = float(d.get('min_lr', self.min_lr))
        self.current_epoch = int(d.get('current_epoch', self.current_epoch))
        self._last_lr = float(d.get('_last_lr', self._last_lr))


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / 'teacher_training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_dataloaders(data_dir, vocab, batch_size=2, num_workers=max(0, os.cpu_count() - 1)):
    """Create dataloaders for teacher training (smaller batch size due to model size)."""

    train_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"),
        vocabulary=vocab,
        split='train',
        augment=True,
        normalize=False,  # Disabled - using per-sample normalization in training loop
        max_seq_length=256   
    )

    val_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"),
        vocabulary=vocab,
        split='dev',
        augment=False,
        normalize=False,  # Disabled - using per-sample normalization in validation loop
        max_seq_length=256   
    )

    test_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv"),
        vocabulary=vocab,
        split='test',
        augment=False,
        normalize=False,  # Disabled - using per-sample normalization in validation loop
        max_seq_length=256   
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def plot_training_curves(
    train_losses,
    val_losses,
    val_wers,
    learning_rates,
    best_wer,
    output_dir: Path
):
    """Create comprehensive training visualization plots for teacher model."""
    epochs = range(1, len(train_losses) + 1)
    
    # Set style  plots
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        try:
            plt.style.use('seaborn-paper')
        except:
            sns.set_style("whitegrid")
    
    sns.set_palette("husl")
    
    # Create figure with subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('I3D Teacher Model - Training Progress', 
                 fontsize=16, fontweight='bold')
    
    # 1. Loss curves (train vs validation)
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('CTC Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)
    
    # 2. WER over epochs
    ax2 = axes[0, 1]
    ax2.plot(epochs, val_wers, 'g-', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax2.axhline(y=best_wer, color='r', linestyle='--', 
               label=f'Best WER: {best_wer:.2f}%', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Word Error Rate (%)', fontsize=12)
    ax2.set_title('Validation Word Error Rate', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    
    # 3. Learning rate schedule
    ax3 = axes[1, 0]
    if len(learning_rates) > 0:
        ax3.plot(epochs, learning_rates[:len(epochs)], 'm-', 
                linewidth=2, marker='s', markersize=4, alpha=0.8)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_xlim(left=1)
    else:
        ax3.text(0.5, 0.5, 'No LR data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    
    # 4. WER and Loss overview
    ax4 = axes[1, 1]
    ax4 = ax4.twinx()

    ax4.scatter(val_losses, val_wers, alpha=0.6, s=30)
    ax4.set_xlabel('Validation Loss', fontsize=12)
    ax4.set_ylabel('WER (%)', fontsize=12)
    ax4.set_title('Loss vs WER Correlation', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=1)
    ax4.set_ylim(bottom=0)
    ax4.set_ylim(top=100)
    
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures/teacher')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save high-resolution plots
    plot_path_pdf = figures_dir / 'training_curves.pdf'
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight', format='pdf')
    
    plot_path_png = figures_dir / 'training_curves.png'
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight', format='png')
    
    # Also save to checkpoint dir
    checkpoint_plot = output_dir / 'training_curves.png'
    plt.savefig(checkpoint_plot, dpi=300, bbox_inches='tight', format='png')
    
    plt.close()
    
    return plot_path_png


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, logger=None):
    """
    Train teacher for one epoch with stability improvements.
    
    """
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training Teacher')

    for batch_idx, batch in enumerate(progress_bar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Diagnostic logging for first batch of first epoch (before normalization)
        if epoch == 1 and batch_idx == 0 and logger:
            logger.info("First batch diagnostics:")
            logger.info(f"  Raw features (before norm): mean={features.mean():.4f}, std={features.std():.4f}")
            logger.info(f"  Raw features range: [{features.min():.4f}, {features.max():.4f}]")

        # Per-sample normalization (like overfit test - prevents double normalization)
        features_mean = features.mean(dim=(1, 2), keepdim=True)
        features_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        features = (features - features_mean) / features_std
        
        # Clip extreme values to prevent numerical instability (common in sign language features)
        features = torch.clamp(features, min=-10.0, max=10.0)

        # Diagnostic logging after normalization
        if epoch == 1 and batch_idx == 0 and logger:
            logger.info(f"  After per-sample norm: mean={features.mean():.4f}, std={features.std():.4f}")
            logger.info(f"  After per-sample norm range: [{features.min():.4f}, {features.max():.4f}]")

        # Check for NaN/Inf in input data after normalization
        if torch.isnan(features).any() or torch.isinf(features).any():
            if logger:
                logger.warning(f"NaN/Inf detected in normalized features at batch {batch_idx}")
            nan_count += 1
            continue

        # FP32 forward pass (disabled mixed precision for stability)
        # Mixed precision was causing numerical instability with teacher model
        logits = model(features, input_lengths)  # Model now returns raw logits

        # Check logits for NaN/Inf before loss computation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            if logger:
                logger.warning(f"NaN/Inf in model output at batch {batch_idx}, skipping")
            nan_count += 1
            if nan_count > 10:
                if logger:
                    logger.error("Too many NaN/Inf outputs, stopping epoch")
                break
            continue

        # Apply log_softmax for CTC loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        # Check loss validity
        if torch.isnan(loss) or torch.isinf(loss):
            if logger:
                logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            nan_count += 1
            if nan_count > 10:
                if logger:
                    logger.error("Too many NaN/Inf losses, stopping epoch")
                break
            continue

        # Backward pass (FP32, no gradient scaling)
        optimizer.zero_grad()  # Clear gradients before backward
        loss.backward()

        # Gradient clipping (same as overfitting test)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaN/Inf gradients (but don't skip for large values)
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break

        if has_nan_grad or math.isnan(grad_norm) or math.isinf(grad_norm):
            if logger:
                logger.warning(f"NaN/Inf gradient detected at batch {batch_idx}, skipping optimizer step")
            optimizer.zero_grad()
            nan_count += 1
            continue

        # Large gradients clipping
        if grad_norm > 200.0 and batch_idx % 100 == 0:
            if logger:
                logger.info(f"Large gradient norm: {grad_norm:.2f} at batch {batch_idx} (normal at start, clipped to 1.0)")

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'grad_norm': f'{grad_norm:.2f}'
        })

    if nan_count > 0 and logger:
        logger.warning(f"Epoch {epoch}: Encountered {nan_count} NaN/Inf batches")

    return {
        'train_loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'nan_count': nan_count
    }


@torch.no_grad()
def validate(model, dataloader, criterion, vocab, device):
    """Validate teacher model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Validation'):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Per-sample normalization (matching training loop)
        features_mean = features.mean(dim=(1, 2), keepdim=True)
        features_std = features.std(dim=(1, 2), keepdim=True) + 1e-6
        features = (features - features_mean) / features_std
        
        # Clip extreme values to prevent numerical instability (matching training)
        features = torch.clamp(features, min=-10.0, max=10.0)

        # FP32 forward pass (no autocast, matching overfitting test)
        logits = model(features, input_lengths)  # Model now returns raw logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

        # Decode predictions
        predictions = ctc_decode(
            log_probs=log_probs.cpu(),
            blank_idx=vocab.blank_id,
            method='greedy',
            beam_width=15,
        )

        for pred in predictions:
            # Convert list of word indices to string
            words = vocab.indices_to_words(pred)
            all_predictions.append(' '.join(words))

        # Convert targets
        labels_cpu = labels.cpu().numpy()
        target_lengths_cpu = target_lengths.cpu().numpy()
        start_idx = 0
        for length in target_lengths_cpu:
            target = labels_cpu[start_idx:start_idx+length]
            words = vocab.indices_to_words(target)
            all_targets.append(' '.join(words))
            start_idx += length

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    wer = compute_wer(all_targets, all_predictions) 
    ser = compute_ser(all_targets, all_predictions) 

    return {
        'val_loss': avg_loss,
        'wer': wer,
        'ser': ser,
        'predictions': all_predictions[:10],  # Save first 10 for inspection
        'targets': all_targets[:10]
    }


def main(args):
    """Main teacher training function."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / f"i3d_teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting I3D teacher training with args: {args}")
    logger.info(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = build_vocabulary(
        Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    )
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Create dataloaders (smaller batch size for teacher)
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Create teacher model with pre-training support
    logger.info("Creating I3D teacher model...")
    model = create_i3d_teacher(
        vocab_size=len(vocab),
        dropout=args.dropout,
    )
    
    model = model.to(device)
    logger.info(f"Teacher parameters: {model.count_parameters():,}")

    # Loss function
    criterion = nn.CTCLoss(blank=vocab.blank_id,
                           reduction='mean',
                           zero_infinity=True
                           )

    # Optimizer - 
    base_lr = args.learning_rate  

    # AdamW optimizer 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # Learning rate WarmupCosineScheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        base_lr=base_lr,
        min_lr=1e-6
    )
    
    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Save configuration
    config = {
        'model': 'I3DTeacher',
        'vocab_size': len(vocab),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'seed': args.seed,
        'model_params': model.count_parameters()
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training history tracking
    train_losses = []
    val_losses = []
    val_wers = []
    val_sers = []
    learning_rates = []
    
    # Training loop
    best_wer = float('inf')
    patience_counter = 0


    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Update learning rate scheduler BEFORE training (for current epoch)
        current_lr = scheduler.step(epoch + 1)
        logger.info(f"Learning rate: {current_lr:.2e}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, None, device, epoch+1, logger
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, vocab, device)

        # Track metrics for plotting
        train_losses.append(train_metrics['train_loss'])
        val_losses.append(val_metrics['val_loss'])
        val_wers.append(val_metrics['wer'])
        val_sers.append(val_metrics['ser'])
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"WER: {val_metrics['wer']:.2f}%")
        logger.info(f"SER: {val_metrics['ser']:.2f}%")

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/WER', val_metrics['wer'], epoch)
        writer.add_scalar('Metrics/SER', val_metrics['ser'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wer': best_wer,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_i3d.pth')
            logger.info(f"New best teacher saved with WER: {best_wer:.2f}%")

            # Show sample predictions
            logger.info("\nSample predictions:")
            for pred, target in zip(val_metrics['predictions'][:3], 
                                   val_metrics['targets'][:3]):
                logger.info(f"  Target: {target}")
                logger.info(f"  Pred:   {pred}")

            # Check if we reached target for teacher
            if best_wer < 30.0:
                logger.info(f"Teacher target achieved! WER < 30% ({best_wer:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered")
                break

        # Plot training curves every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            try:
                plot_path = plot_training_curves(
                    train_losses, val_losses, val_wers, 
                    learning_rates, best_wer, output_dir
                )
                logger.info(f"Training curves saved to figures/teacher/")
            except Exception as e:
                logger.warning(f"Failed to save training curves: {e}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_wers': val_wers,
            'val_sers': val_sers,
            'learning_rates': learning_rates,
            'best_wer': best_wer
        }
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wer': val_metrics['wer'],
                'config': config
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Final test evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Test Evaluation")
    logger.info("="*50)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_i3d.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, vocab, device)
    logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"Test WER: {test_metrics['wer']:.2f}%")
    logger.info(f"Test SER: {test_metrics['ser']:.2f}%")

    # Generate final training curves
    try:
        plot_path = plot_training_curves(
            train_losses, val_losses, val_wers, 
            learning_rates, best_wer, output_dir
        )
        logger.info(f"Final training curves saved to figures/teacher/training_curves.png")
    except Exception as e:
        logger.warning(f"Failed to save final training curves: {e}")

    # Save results
    results = {
        'best_val_wer': best_wer,
        'test_wer': test_metrics['wer'],
        'test_ser': test_metrics['ser'],
        'test_loss': test_metrics['val_loss'],
        'total_params': model.count_parameters(),
        'model_size_mb': model.count_parameters() * 4 / 1024 / 1024,
        'ready_for_distillation': best_wer < 35.0  
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTeacher training complete! Results saved to {output_dir}")
    logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
    logger.info(f"Best WER: {best_wer:.2f}%")
    logger.info(f"Training curves: figures/teacher/training_curves.png")

    # Copy best checkpoint to standard location for distillation
    import shutil
    teacher_checkpoint_dir = Path(args.output_dir).parent / 'teacher'
    teacher_checkpoint_dir.mkdir(exist_ok=True)
    shutil.copy(output_dir / 'best_i3d.pth', teacher_checkpoint_dir / 'best_i3d.pth')
    logger.info(f"Best teacher checkpoint copied to {teacher_checkpoint_dir / 'best_i3d.pth'}")

    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train I3D Teacher Model')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='data/teacher_features/mediapipe_full',
                        help='Path to features directory')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/hierarchical_teacher',
                        help='Output directory')

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate ')
    parser.add_argument('--warmup_epochs', type=int, default=30,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help='Early stopping patience')

    args = parser.parse_args()
    main(args)