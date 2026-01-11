"""
CTC training utilities for the hybrid encoder model.

Provides:
- train_epoch: one epoch of CTC training (AMP + grad clipping preserved)
- validate: CTC loss + greedy decoding + WER reporting
- Label smoothing via entropy regularization
- SpecAugment-style feature augmentation
"""

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.metrics import compute_wer
from src.utils.augmentation import FeatureAugmentation


def _decode_ctc_greedy(
    pred_ids: torch.Tensor,
    out_lengths: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """
    CTC greedy decoding from argmax IDs.

    Args:
        pred_ids: (B, T') argmax token IDs
        out_lengths: (B,) valid lengths for each sequence (after subsampling)
        blank_id: CTC blank token ID

    Returns:
        List[List[int]] decoded token ID sequences (blanks removed, repeats collapsed).
    """
    B, T = pred_ids.shape
    decoded: List[List[int]] = []

    for b in range(B):
        L = int(min(out_lengths[b].item(), T))
        seq = pred_ids[b, :L].tolist()

        out_seq: List[int] = []
        prev = blank_id
        for tok in seq:
            if tok != blank_id and tok != prev:
                out_seq.append(tok)
            prev = tok

        decoded.append(out_seq)

    return decoded


def _ids_to_string(ids: List[int], idx2word: Dict[int, str]) -> str:
    """Convert token IDs to a space-separated string, skipping blank-like tokens."""
    words: List[str] = []
    for t in ids:
        w = idx2word.get(int(t), "")
        if not w:
            continue
        if w in {"<blank>", "<pad>", "<BLANK>", "<PAD>"}:
            continue
        words.append(w)
    return " ".join(words)


def _split_concatenated_targets(
    targets: torch.Tensor, target_lengths: torch.Tensor
) -> List[List[int]]:
    """
    Split concatenated CTC targets back into per-sample sequences.

    Args:
        targets: (sum_L,) concatenated targets
        target_lengths: (B,) lengths per sample
    """
    seqs: List[List[int]] = []
    offset = 0
    for L in target_lengths.tolist():
        L = int(L)
        seqs.append(targets[offset : offset + L].tolist())
        offset += L
    return seqs


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: GradScaler,
    scheduler,
    logger,
    augmenter: Optional[FeatureAugmentation] = None,
    label_smoothing: float = 0.0,
) -> float:
    """
    Train for one epoch with optional augmentation and label smoothing.
    
    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: CTC loss function
        device: Device to train on
        scaler: GradScaler for AMP
        scheduler: Learning rate scheduler (stepped per batch)
        logger: Logger
        augmenter: Optional FeatureAugmentation module
        label_smoothing: Entropy regularization weight (0 = disabled)
    """
    model.train()
    if augmenter is not None:
        augmenter.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)

        # Apply feature augmentation (SpecAugment-style)
        if augmenter is not None:
            features = augmenter(features)

        # CTC targets are concatenated 1D tensor with per-sample lengths
        targets = batch["labels"].to(device)
        target_lengths = batch["label_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward under autocast (AMP), but compute CTCLoss outside autocast in FP32.
        with autocast(enabled=device.type == "cuda"):
            log_probs, out_lengths = model(features, input_lengths)  # (B, T', V) fp32 from model, (B,)

        # CTC expects (T, B, V)
        # Note: some CUDA CTCLoss backends expect lengths on CPU.
        in_lens = out_lengths
        tgt_lens = target_lengths
        if log_probs.is_cuda:
            in_lens = in_lens.cpu()
            tgt_lens = tgt_lens.cpu()

        log_probs_ctc = log_probs.float().permute(1, 0, 2)  # (T', B, V) fp32
        loss = criterion(log_probs_ctc, targets, in_lens, tgt_lens)

        # Label smoothing via entropy regularization (discourage overconfident posteriors).
        if label_smoothing > 0:
            probs = log_probs.exp()  # (B, T', V), <= 1
            entropy = -(probs * log_probs.float()).sum(dim=-1)  # (B, T')

            T_prime = log_probs.size(1)
            mask = torch.arange(T_prime, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)
            masked_entropy = (entropy * mask.float()).sum() / mask.sum().clamp(min=1)
            loss = loss - label_smoothing * masked_entropy

        # Hard guard: never backprop NaN/Inf loss (it will corrupt weights and all future eval).
        if not torch.isfinite(loss):
            logger.warning(
                f"Non-finite loss encountered (loss={loss.item()}). "
                "Skipping optimizer step this batch. "
                "This is often caused by an overly large learning rate or numerical overflow."
            )
            # IMPORTANT: Do NOT call scaler.update() here.
            # GradScaler.update() requires that an inf/nan check was recorded via scaler.step(...).
            # Since we skip backward/step entirely, there is nothing to update and calling update()
            # will raise: "No inf checks were recorded prior to update."
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            # OneCycleLR is designed to step per optimizer update (i.e., per batch).
            scheduler.step()

        total_loss += float(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device: torch.device,
    idx2word: Dict[int, str],
    logger,
) -> Tuple[float, float, List[str], List[str]]:
    model.eval()

    total_loss = 0.0
    all_preds: List[str] = []
    all_targets: List[str] = []

    for batch in tqdm(dataloader, desc="Validation"):
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)

        targets = batch["labels"].to(device)
        target_lengths = batch["label_lengths"].to(device)

        log_probs, out_lengths = model(features, input_lengths)  # (B, T', V)

        in_lens = out_lengths
        tgt_lens = target_lengths
        if log_probs.is_cuda:
            in_lens = in_lens.cpu()
            tgt_lens = tgt_lens.cpu()
        loss = criterion(
            log_probs.float().permute(1, 0, 2),  # (T', B, V)
            targets,
            in_lens,
            tgt_lens,
        )
        total_loss += float(loss.item())

        # CTC greedy decoding:
        # 1) per-timestep argmax
        pred_ids = log_probs.argmax(dim=-1)  # (B, T')
        # 2) collapse repeats + remove blanks using out_lengths
        pred_seqs = _decode_ctc_greedy(pred_ids, out_lengths, blank_id=0)

        # Reconstruct per-sample target sequences from concatenated targets
        tgt_seqs = _split_concatenated_targets(targets, target_lengths)

        for pred_ids_i, tgt_ids_i in zip(pred_seqs, tgt_seqs):
            all_preds.append(_ids_to_string(pred_ids_i, idx2word))
            all_targets.append(_ids_to_string(tgt_ids_i, idx2word))

    avg_loss = total_loss / max(1, len(dataloader))
    wer = compute_wer(all_targets, all_preds)

    logger.info("Sample predictions:")
    for i in range(min(3, len(all_preds))):
        logger.info(f"  Target: '{all_targets[i]}'")
        logger.info(f"  Pred:   '{all_preds[i] if all_preds[i] else '(empty)'}'")

    return avg_loss, wer, all_preds, all_targets


