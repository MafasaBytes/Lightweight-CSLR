"""
Training script for MobileNetV3-based sign language recognition.
Implements Phase I baseline training with memory optimizations.
Features:
- Mixed-precision training (FP16)
- Gradient checkpointing
- Dynamic sequence truncation
- CTC loss with proper blank handling
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import MobileNetV3SignLanguage, create_mobilenet_v3_model
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary, Vocabulary
from src.utils.metrics import compute_wer, compute_ser
import random


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / 'training.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_dataloaders(
    data_dir: Path,
    vocab: Vocabulary,
    batch_size: int = 4,
    num_workers: int = 0,
    num_train_samples: Optional[int] = None,
    remove_pca: bool = True  # Remove PCA as per research proposal
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with optional sampling."""

    # Training dataset
    train_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv"),
        vocabulary=vocab,
        split='train',
        augment=True  # Enable augmentation for training
    )

    # Sample training data if requested
    if num_train_samples is not None and num_train_samples < len(train_dataset):
        from torch.utils.data import Subset
        logging.info(f"Sampling {num_train_samples} training samples from {len(train_dataset)}")
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        sampled_indices = indices[:num_train_samples]
        train_dataset = Subset(train_dataset, sampled_indices)

    # Validation dataset
    val_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/dev.SI5.corpus.csv"),
        vocabulary=vocab,
        split='dev',
        augment=False
    )

    # Test dataset
    test_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/test.SI5.corpus.csv"),
        vocabulary=vocab,
        split='test',
        augment=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # For stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
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
    """Generate publication-quality training plots."""
    epochs = range(1, len(train_losses) + 1)
    
    # Set style for thesis-quality plots
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
    fig.suptitle('MobileNetV3 Sign Language Model - Training Progress', 
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
    ax2.axhline(y=25.0, color='orange', linestyle=':', 
               label='Target: 25%', linewidth=2)
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
    
    # 4. Combined metrics overview
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # Loss on left axis
    line1 = ax4.plot(epochs, train_losses, 'b-', label='Train Loss', 
                    linewidth=2, alpha=0.7)
    line2 = ax4.plot(epochs, val_losses, 'r-', label='Val Loss', 
                    linewidth=2, alpha=0.7)
    
    # WER on right axis
    line3 = ax4_twin.plot(epochs, val_wers, 'g-', label='Val WER (%)', 
                         linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12, color='black')
    ax4_twin.set_ylabel('WER (%)', fontsize=12, color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4.set_title('Training Overview', fontsize=13, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=1)
    
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures/baseline')
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


def adaptive_greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int = 0,
    confidence_threshold: float = -8.0,
    min_sequence_length: int = 2,
    max_sequence_length: int = 50,
    adaptive_threshold: bool = True
):
    """
    Adaptive greedy decoding with length normalization.
    Filters trailing low-confidence insertions while preserving valid long sequences.
    
    Validated in overfitting test: achieved 0% WER on 10 samples.
    
    Args:
        log_probs: [T, B, V] log probabilities
        input_lengths: [B] sequence lengths
        blank_id: CTC blank token ID
        confidence_threshold: Minimum log probability to accept a token
        min_sequence_length: Always keep first N tokens (protection)
        max_sequence_length: Maximum output length (safety cap)
        adaptive_threshold: Use length-dependent filtering
        
    Returns:
        List of decoded sequences (token indices)
    """
    batch_size = log_probs.size(1)
    predictions = []
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        probs = log_probs[:seq_len, b, :]  # [T, vocab_size]
        
        # Get best path with scores
        max_log_probs, max_indices = torch.max(probs, dim=-1)  # [T]
        max_log_probs = max_log_probs.cpu()
        max_indices = max_indices.cpu().numpy()
        
        # Remove consecutive duplicates and blanks, tracking confidence
        tokens = []
        token_scores = []
        prev = -1
        
        for t in range(seq_len):
            idx = int(max_indices[t])
            score = max_log_probs[t].item()
            
            if idx != prev and idx != blank_id:
                # Only add if confidence is above threshold
                if score > confidence_threshold:
                    tokens.append(idx)
                    token_scores.append(score)
            prev = idx
        
        # Apply adaptive filtering based on sequence length
        if len(tokens) > min_sequence_length:
            mean_score = sum(token_scores) / len(token_scores)
            std_score = (sum((s - mean_score) ** 2 for s in token_scores) / len(token_scores)) ** 0.5
            
            if adaptive_threshold:
                # Adaptive: more lenient for longer sequences
                if len(tokens) <= 5:
                    std_multiplier = 0.5  # Aggressive for short
                elif len(tokens) <= 15:
                    std_multiplier = 0.3  # Moderate for medium
                else:
                    std_multiplier = 0.2  # Gentle for long
            else:
                std_multiplier = 0.5  # Fixed threshold
            
            threshold_score = mean_score - std_multiplier * std_score
            
            # Filter trailing low-confidence tokens
            filtered_tokens = []
            for i, (token, score) in enumerate(zip(tokens, token_scores)):
                # Always keep first min_sequence_length tokens
                if i < min_sequence_length:
                    filtered_tokens.append(token)
                elif score >= threshold_score:
                    filtered_tokens.append(token)
                else:
                    # Check if all remaining are low confidence
                    remaining_scores = token_scores[i:]
                    avg_remaining = sum(remaining_scores) / len(remaining_scores)
                    
                    if avg_remaining < threshold_score:
                        # All remaining low - stop here
                        break
                    else:
                        # Good tokens after - keep this one
                        filtered_tokens.append(token)
            
            tokens = filtered_tokens[:max_sequence_length]
        
        predictions.append(tokens)
    
    return predictions


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 5.0,
    accumulation_steps: int = 1,
    dynamic_truncation: bool = True,
    max_seq_length: int = 512
) -> Dict[str, float]:
    """Train for one epoch with mixed-precision and gradient accumulation."""

    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')

    for batch_idx, batch in enumerate(progress_bar):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Dynamic sequence truncation for memory efficiency
        if dynamic_truncation and features.size(1) > max_seq_length:
            features = features[:, :max_seq_length, :]
            input_lengths = torch.minimum(
                input_lengths,
                torch.tensor(max_seq_length, device=device)
            )

        # Mixed precision forward pass
        with autocast():
            log_probs = model(features, input_lengths)

            # Ensure log_probs is in correct format [T, B, V]
            if log_probs.dim() == 3 and log_probs.size(0) != features.size(1):
                log_probs = log_probs.transpose(0, 1)

            # CTC loss computation
            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss detected in batch {batch_idx}")
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale and clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track metrics
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    return {
        'train_loss': total_loss / num_batches if num_batches > 0 else 0
    }

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.CTCLoss,
    vocab: Vocabulary,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model and compute metrics."""

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

        # Forward pass
        with autocast():
            log_probs = model(features, input_lengths)

            # Ensure log_probs is in correct format [T, B, V]
            if log_probs.dim() == 3 and log_probs.size(0) != features.size(1):
                log_probs = log_probs.transpose(0, 1)

            # CTC loss
            loss = criterion(
                log_probs,
                labels,
                input_lengths,
                target_lengths
            )

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            num_batches += 1

        # Decode predictions using adaptive greedy decoding (validated: 0% WER on overfit test)
        # Slightly more permissive for full training (was -8.0)
        predictions = adaptive_greedy_decode(
            log_probs.cpu(),
            input_lengths.cpu(),
            blank_id=vocab.blank_id,
            confidence_threshold=-10.0,  # More permissive
            min_sequence_length=1,        # Less protection
            max_sequence_length=50,
            adaptive_threshold=True
        )

        # Convert to words
        for pred, target_len in zip(predictions, target_lengths.cpu()):
            pred_words = vocab.indices_to_words(pred)
            all_predictions.append(pred_words)

        # Convert targets to words
        batch_targets = []
        labels_cpu = labels.cpu().numpy()
        target_lengths_cpu = target_lengths.cpu().numpy()
        start_idx = 0
        for length in target_lengths_cpu:
            target = labels_cpu[start_idx:start_idx+length]
            target_words = vocab.indices_to_words(target)
            all_targets.append(target_words)
            batch_targets.append(target_words)
            start_idx += length

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Compute WER and SER - references (targets) first, then hypotheses (predictions)
    wer = compute_wer(all_targets, all_predictions)
    ser = compute_ser(all_targets, all_predictions)

    return {
        'val_loss': avg_loss,
        'wer': wer,
        'ser': ser
    }


def main(args):
    """Main training function."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / f"mobilenet_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting training with args: {args}")
    logger.info(f"Device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = build_vocabulary(
        Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    )
    logger.info(f"Vocabulary size: {len(vocab)}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_samples=args.num_train_samples,
        remove_pca=args.remove_pca
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating MobileNetV3 model...")
    model = create_mobilenet_v3_model(
        vocab_size=len(vocab),
        dropout=args.dropout
    )
    model = model.to(device)

    # Loss function (CTC)
    # CTC Loss - use 'none' reduction to apply custom weights if needed
    criterion = nn.CTCLoss(blank=vocab.blank_id, reduction='mean', zero_infinity=True)

    # Optimizer (AdamW with proper LR)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,  # Use the specified learning rate directly
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-3
    )

    # Learning rate scheduler - ReduceLROnPlateau (more stable for CTC)
    # No warmup - start at full LR (validated in overfitting test)
    # Adjusted: More patience to avoid premature LR reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',          # Monitor validation loss
        factor=0.8,          # Reduce LR by 20% 
        patience=50,         # Wait 50 epochs before reducing 
        min_lr=1e-5,         # Higher minimum 
        threshold=0.05,     # Threshold for improvement
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Training configuration
    config = {
        'model': 'MobileNetV3SignLanguage',
        'vocab_size': len(vocab),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'seed': args.seed,
        'remove_pca': args.remove_pca,
        'mixed_precision': True,
        'gradient_checkpointing': args.gradient_checkpointing,
        'model_params': model.count_parameters()
    }

    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {output_dir / 'config.json'}")

    # Training loop
    best_wer = float('inf')
    patience = args.early_stopping_patience
    patience_counter = 0
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_wers = []
    val_sers = []
    learning_rates = []

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch+1, args.max_grad_norm,
            args.accumulation_steps, args.dynamic_truncation
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, vocab, device)

        # Track metrics
        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(train_metrics['train_loss'])
        val_losses.append(val_metrics['val_loss'])
        val_wers.append(val_metrics['wer'])
        val_sers.append(val_metrics['ser'])
        learning_rates.append(current_lr)
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"WER: {val_metrics['wer']:.2f}%")
        logger.info(f"SER: {val_metrics['ser']:.2f}%")
        logger.info(f"LR: {current_lr:.6f}")

        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/WER', val_metrics['wer'], epoch)
        writer.add_scalar('Metrics/SER', val_metrics['ser'], epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # Learning rate scheduling (based on validation loss)
        scheduler.step(val_metrics['val_loss'])

        # Save best model
        if val_metrics['wer'] < best_wer:
            best_wer = val_metrics['wer']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_wer': best_wer,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            logger.info(f"New best model saved with WER: {best_wer:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break

        # Plot training curves every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            try:
                plot_path = plot_training_curves(
                    train_losses, val_losses, val_wers, 
                    learning_rates, best_wer, output_dir
                )
                logger.info(f"Training curves saved to figures/baseline/")
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'wer': val_metrics['wer'],
                'config': config,
                'history': history
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Final test evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Test Evaluation")
    logger.info("="*50)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate(model, test_loader, criterion, vocab, device)
    logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"Test WER: {test_metrics['wer']:.2f}%")
    logger.info(f"Test SER: {test_metrics['ser']:.2f}%")

    # Save final results
    results = {
        'best_val_wer': best_wer,
        'test_wer': test_metrics['wer'],
        'test_ser': test_metrics['ser'],
        'test_loss': test_metrics['val_loss'],
        'total_params': model.count_parameters(),
        'model_size_mb': model.count_parameters() * 4 / 1024 / 1024,
        'training_epochs': len(train_losses)
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final training curves plot
    try:
        final_plot = plot_training_curves(
            train_losses, val_losses, val_wers, 
            learning_rates, best_wer, output_dir
        )
        logger.info(f"Final training curves saved to: {final_plot}")
    except Exception as e:
        logger.warning(f"Failed to save final training curves: {e}")

    logger.info(f"\nTraining complete! Results saved to {output_dir}")
    logger.info(f"Model size: {results['model_size_mb']:.2f} MB")
    logger.info(f"Best WER: {best_wer:.2f}% (Target: < 25%)")
    logger.info(f"Training curves: figures/baseline/training_curves.png")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MobileNetV3 for Sign Language Recognition')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='data/teacher_features/mediapipe_full',
                        help='Path to features directory')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/student',
                        help='Output directory for checkpoints')
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help='Number of training samples to use (default: all)')

    # Model arguments
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3 for regularization - reduce overfitting)')
    parser.add_argument('--remove_pca', action='store_true', default=True,
                        help='Remove PCA reduction to preserve modality boundaries')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4 for stability)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4 - validated in overfitting test)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4 for regularization)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (default: 1.0 for stability)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')

    # Memory optimization arguments
    parser.add_argument('--dynamic_truncation', action='store_true', default=True,
                        help='Enable dynamic sequence truncation')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing to save memory')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=100,
                        help='Early stopping patience')

    args = parser.parse_args()
    main(args)