"""
Knowledge Distillation Training Script
Trains MobileNetV3 student using I3D teacher model
Following research proposal: Loss = 0.7 * L_soft + 0.3 * L_hard
Temperature = 3.0
Target: <25% WER
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from src.models import MobileNetV3SignLanguage, create_mobilenet_v3_model
from src.models.i3d_teacher import I3DTeacher, create_i3d_teacher
from src.data.dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from src.utils.metrics import compute_wer, compute_ser


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    log_file = output_dir / 'distillation.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining soft and hard targets.
    Based on research proposal specifications.
    """

    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.7,
        blank_id: int = 0
    ):
        """
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for soft loss (0.7 * soft + 0.3 * hard)
            blank_id: CTC blank token ID
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.blank_id = blank_id
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model output [T, B, V]
            teacher_logits: Teacher model output [T, B, V]
            labels: Ground truth labels
            input_lengths: Input sequence lengths
            target_lengths: Target sequence lengths

        Returns:
            Total loss and component losses for logging
        """
        # Hard loss (CTC loss with ground truth)
        hard_loss = self.ctc_loss(student_logits, labels, input_lengths, target_lengths)

        # Soft loss (KL divergence between teacher and student)
        # Apply temperature and convert to probabilities
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # KL divergence loss
        # Note: We use mean reduction and scale by T^2 as in the original paper
        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        # Return total and components for logging
        loss_components = {
            'total_loss': total_loss.item(),
            'soft_loss': soft_loss.item(),
            'hard_loss': hard_loss.item()
        }

        return total_loss, loss_components


def load_teacher_model(
    checkpoint_path: Path,
    vocab_size: int,
    device: torch.device
) -> I3DTeacher:
    """Load pre-trained teacher model."""
    teacher = create_i3d_teacher(vocab_size)

    if checkpoint_path.exists():
        logging.info(f"Loading teacher from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        teacher.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Teacher loaded with WER: {checkpoint.get('best_wer', 'N/A')}%")
    else:
        logging.warning("No teacher checkpoint found. Using untrained teacher (not recommended)")

    teacher = teacher.to(device)
    teacher.eval()  # Teacher always in eval mode

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def train_distillation_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    criterion: DistillationLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 5.0
) -> Dict[str, float]:
    """Train one epoch with knowledge distillation."""

    student.train()
    teacher.eval()  # Teacher always in eval mode

    total_loss = 0
    total_soft_loss = 0
    total_hard_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} - Distillation')

    for batch in progress_bar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Get teacher predictions (no gradient needed)
        with torch.no_grad():
            teacher_logits = teacher(features, input_lengths)

        # Student forward pass with mixed precision
        with autocast():
            student_logits = student(features, input_lengths)

            # Compute distillation loss
            loss, loss_components = criterion(
                student_logits,
                teacher_logits,
                labels,
                input_lengths,
                target_lengths
            )

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss detected")
            continue

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Track metrics
        total_loss += loss_components['total_loss']
        total_soft_loss += loss_components['soft_loss']
        total_hard_loss += loss_components['hard_loss']
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{total_loss / num_batches:.4f}',
            'soft': f'{total_soft_loss / num_batches:.4f}',
            'hard': f'{total_hard_loss / num_batches:.4f}'
        })

    return {
        'total_loss': total_loss / num_batches if num_batches > 0 else 0,
        'soft_loss': total_soft_loss / num_batches if num_batches > 0 else 0,
        'hard_loss': total_hard_loss / num_batches if num_batches > 0 else 0
    }


@torch.no_grad()
def validate_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    vocab,
    device: torch.device
) -> Dict[str, float]:
    """Validate student and compare with teacher."""

    student.eval()
    teacher.eval()

    student_predictions = []
    teacher_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc='Validation'):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        # Get predictions from both models
        with autocast():
            student_logits = student(features, input_lengths)
            teacher_logits = teacher(features, input_lengths)

        # Decode predictions
        from src.models.bilstm_ctc import CTCDecoder
        decoder = CTCDecoder()

        student_preds = decoder.greedy_decode(
            student_logits.cpu(),
            input_lengths.cpu(),
            blank_id=vocab.blank_id
        )

        teacher_preds = decoder.greedy_decode(
            teacher_logits.cpu(),
            input_lengths.cpu(),
            blank_id=vocab.blank_id
        )

        # Convert to words
        for s_pred, t_pred in zip(student_preds, teacher_preds):
            student_predictions.append(vocab.indices_to_words(s_pred))
            teacher_predictions.append(vocab.indices_to_words(t_pred))

        # Convert targets
        labels_cpu = labels.cpu().numpy()
        target_lengths_cpu = target_lengths.cpu().numpy()
        start_idx = 0
        for length in target_lengths_cpu:
            target = labels_cpu[start_idx:start_idx+length]
            target_words = vocab.indices_to_words(target)
            all_targets.append(target_words)
            start_idx += length

    # Compute metrics
    student_wer = compute_wer(all_targets, student_predictions)  # Fixed: (references, hypotheses)
    teacher_wer = compute_wer(all_targets, teacher_predictions)  # Fixed: (references, hypotheses)
    student_ser = compute_ser(all_targets, student_predictions)  # Fixed: (references, hypotheses)
    teacher_ser = compute_ser(all_targets, teacher_predictions)  # Fixed: (references, hypotheses)

    # Agreement between teacher and student (both are hypotheses, order doesn't matter as much)
    agreement = compute_wer(teacher_predictions, student_predictions)

    return {
        'student_wer': student_wer,
        'teacher_wer': teacher_wer,
        'student_ser': student_ser,
        'teacher_ser': teacher_ser,
        'teacher_student_agreement': agreement
    }


def main(args):
    """Main distillation training function."""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / f"distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Starting knowledge distillation with args: {args}")
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

    # Create dataloaders
    logger.info("Creating dataloaders...")
    from src.training.train import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remove_pca=True
    )

    # Load teacher model
    logger.info("Loading teacher model...")
    teacher = load_teacher_model(
        Path(args.teacher_checkpoint),
        vocab_size=len(vocab),
        device=device
    )
    teacher_params = teacher.count_parameters()
    logger.info(f"Teacher parameters: {teacher_params:,}")

    # Create student model
    logger.info("Creating student model...")
    student = create_mobilenet_v3_model(
        vocab_size=len(vocab),
        dropout=args.dropout
    )

    # Load student checkpoint if provided (for fine-tuning)
    if args.student_checkpoint:
        logger.info(f"Loading student from {args.student_checkpoint}")
        checkpoint = torch.load(args.student_checkpoint, map_location=device)
        student.load_state_dict(checkpoint['model_state_dict'])

    student = student.to(device)
    student_params = student.count_parameters()
    logger.info(f"Student parameters: {student_params:,}")
    logger.info(f"Compression ratio: {teacher_params / student_params:.2f}x")

    # Loss function
    criterion = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        blank_id=vocab.blank_id
    )

    # Optimizer
    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # TensorBoard writer
    writer = SummaryWriter(output_dir / 'tensorboard')

    # Save configuration
    config = {
        'teacher_model': 'I3DTeacher',
        'student_model': 'MobileNetV3SignLanguage',
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': teacher_params / student_params,
        'temperature': args.temperature,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'seed': args.seed
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    best_wer = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_distillation_epoch(
            student, teacher, train_loader, criterion,
            optimizer, scaler, device, epoch+1, args.max_grad_norm
        )

        # Validate
        val_metrics = validate_distillation(
            student, teacher, val_loader, vocab, device
        )

        # Log metrics
        logger.info(f"Train - Total Loss: {train_metrics['total_loss']:.4f}, "
                   f"Soft: {train_metrics['soft_loss']:.4f}, "
                   f"Hard: {train_metrics['hard_loss']:.4f}")
        logger.info(f"Student WER: {val_metrics['student_wer']:.2f}% "
                   f"(Teacher: {val_metrics['teacher_wer']:.2f}%)")
        logger.info(f"Teacher-Student Agreement: {val_metrics['teacher_student_agreement']:.2f}%")

        # TensorBoard logging
        writer.add_scalar('Loss/Total', train_metrics['total_loss'], epoch)
        writer.add_scalar('Loss/Soft', train_metrics['soft_loss'], epoch)
        writer.add_scalar('Loss/Hard', train_metrics['hard_loss'], epoch)
        writer.add_scalar('WER/Student', val_metrics['student_wer'], epoch)
        writer.add_scalar('WER/Teacher', val_metrics['teacher_wer'], epoch)
        writer.add_scalar('WER/Agreement', val_metrics['teacher_student_agreement'], epoch)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if val_metrics['student_wer'] < best_wer:
            best_wer = val_metrics['student_wer']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_wer': best_wer,
                'teacher_wer': val_metrics['teacher_wer'],
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_student.pth')
            logger.info(f"New best student saved with WER: {best_wer:.2f}%")

            # Check if we achieved the target
            if best_wer < 25.0:
                logger.info(f"TARGET ACHIEVED! WER < 25% ({best_wer:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered")
                break

    # Final test evaluation
    logger.info("\n" + "="*50)
    logger.info("Final Test Evaluation")
    logger.info("="*50)

    # Load best model
    checkpoint = torch.load(output_dir / 'best_student.pth')
    student.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = validate_distillation(student, teacher, test_loader, vocab, device)
    logger.info(f"Test Student WER: {test_metrics['student_wer']:.2f}%")
    logger.info(f"Test Teacher WER: {test_metrics['teacher_wer']:.2f}%")
    logger.info(f"Test Student SER: {test_metrics['student_ser']:.2f}%")

    # Save final results
    results = {
        'best_val_wer': best_wer,
        'test_student_wer': test_metrics['student_wer'],
        'test_teacher_wer': test_metrics['teacher_wer'],
        'test_student_ser': test_metrics['student_ser'],
        'achieved_target': best_wer < 25.0,
        'compression_ratio': teacher_params / student_params
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDistillation complete! Results saved to {output_dir}")
    logger.info(f"Best WER: {best_wer:.2f}% (Target: < 25%)")
    logger.info(f"Target achieved: {'YES' if best_wer < 25.0 else 'NO'}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='data/teacher_features/mediapipe_full',
                        help='Path to features directory')
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/distillation',
                        help='Output directory')

    # Model arguments
    parser.add_argument('--teacher_checkpoint', type=str,
                        default='checkpoints/teacher/best_i3d.pth',
                        help='Path to teacher model checkpoint')
    parser.add_argument('--student_checkpoint', type=str, default=None,
                        help='Optional student checkpoint to fine-tune')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout rate for student')

    # Distillation arguments
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperature for distillation')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for soft loss (0.7 = 70% soft, 30% hard)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Max gradient norm')

    # Other arguments
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')

    args = parser.parse_args()
    main(args)