"""
Canonical CTC training utilities (AMP + grad clipping preserved).

This file is the **single source of truth** for the CTC training/validation loops.
"""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.augmentation import FeatureAugmentation
from src.utils.ctc import (
    ctc_greedy_decode_with_lengths,
    ctc_beam_search_decode,
    ids_to_string,
    split_concatenated_targets,
)
from src.utils.metrics import compute_wer

# Suppress false-positive warning about scheduler.step() order
warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
)


class EMA:
    """
    Exponential Moving Average of model weights.
    Evaluating on EMA weights often improves generalization and reduces overfitting symptoms.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


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
    ema: Optional[EMA] = None,
) -> float:
    model.train()
    if augmenter is not None:
        augmenter.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)

        if augmenter is not None:
            features = augmenter(features)

        targets = batch["labels"].to(device)
        target_lengths = batch["label_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            log_probs, out_lengths = model(features, input_lengths)  # (B,T',V), (B,)

        in_lens = out_lengths
        tgt_lens = target_lengths
        if log_probs.is_cuda:
            in_lens = in_lens.cpu()
            tgt_lens = tgt_lens.cpu()

        log_probs_ctc = log_probs.float().permute(1, 0, 2)  # (T',B,V)
        loss = criterion(log_probs_ctc, targets, in_lens, tgt_lens)

        # Entropy regularization (anti-overconfidence)
        if label_smoothing > 0:
            probs = log_probs.exp()
            entropy = -(probs * log_probs.float()).sum(dim=-1)  # (B,T')
            T_prime = log_probs.size(1)
            mask = torch.arange(T_prime, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)
            masked_entropy = (entropy * mask.float()).sum() / mask.sum().clamp(min=1)
            loss = loss - label_smoothing * masked_entropy

        if not torch.isfinite(loss):
            logger.warning(
                f"Non-finite loss encountered (loss={loss.item()}). Skipping optimizer step this batch."
            )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

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
    ema: Optional[EMA] = None,
    decode_method: str = "greedy",
    beam_width: int = 10,
    string_filter_fn=None,
) -> Tuple[float, float, List[str], List[str]]:
    """
    Validate model with greedy or beam search decoding.
    
    Args:
        decode_method: "greedy" or "beam_search"
        beam_width: Beam width for beam search (default: 10)
    """
    def _run_eval() -> Tuple[float, float, List[str], List[str]]:
        model.eval()
        total_loss = 0.0
        all_preds: List[str] = []
        all_targets: List[str] = []

        for batch in tqdm(dataloader, desc="Validation"):
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["labels"].to(device)
            target_lengths = batch["label_lengths"].to(device)

            log_probs, out_lengths = model(features, input_lengths)

            in_lens = out_lengths
            tgt_lens = target_lengths
            if log_probs.is_cuda:
                in_lens = in_lens.cpu()
                tgt_lens = tgt_lens.cpu()

            loss = criterion(log_probs.float().permute(1, 0, 2), targets, in_lens, tgt_lens)
            total_loss += float(loss.item())

            # Decode based on method
            if decode_method == "beam_search":
                # ctc_beam_search_decode expects (T, B, V)
                log_probs_tbv = log_probs.permute(1, 0, 2)  # (B,T,V) -> (T,B,V)
                pred_seqs = ctc_beam_search_decode(log_probs_tbv, lengths=out_lengths, blank_idx=0, beam_width=beam_width)
            else:
                # Greedy decode
                pred_ids = log_probs.argmax(dim=-1)  # (B,T')
                pred_seqs = ctc_greedy_decode_with_lengths(pred_ids, out_lengths, blank_idx=0)
            
            tgt_seqs = split_concatenated_targets(targets, target_lengths)

            for pred_ids_i, tgt_ids_i in zip(pred_seqs, tgt_seqs):
                all_preds.append(ids_to_string(pred_ids_i, idx2word))
                all_targets.append(ids_to_string(tgt_ids_i, idx2word))

        avg_loss = total_loss / max(1, len(dataloader))
        if string_filter_fn is not None:
            all_targets_f = [string_filter_fn(s) for s in all_targets]
            all_preds_f = [string_filter_fn(s) for s in all_preds]
            wer = compute_wer(all_targets_f, all_preds_f)
            all_targets = all_targets_f
            all_preds = all_preds_f
        else:
            wer = compute_wer(all_targets, all_preds)

        logger.info(f"Sample predictions (decode={decode_method}):")
        for i in range(min(3, len(all_preds))):
            logger.info(f"  Target: '{all_targets[i]}'")
            logger.info(f"  Pred:   '{all_preds[i] if all_preds[i] else '(empty)'}'")

        return avg_loss, wer, all_preds, all_targets

    if ema is None:
        return _run_eval()

    with ema.apply_to(model):
        return _run_eval()


__all__ = ["EMA", "train_epoch", "validate"]


