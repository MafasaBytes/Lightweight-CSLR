"""CTC (Connectionist Temporal Classification) utilities."""

import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import os
from itertools import groupby

_HAS_CTCDECODE = False
try:
    import ctcdecode  # type: ignore

    _HAS_CTCDECODE = True
except Exception:
    _HAS_CTCDECODE = False


def _make_ctcdecode_vocab(num_classes: int) -> list[str]:
    # Same trick as AdaptSign: map class IDs to a stable unicode range.
    return [chr(x) for x in range(20000, 20000 + int(num_classes))]


_CTCDECODE_DECODERS: dict[tuple[int, int, int, int], "ctcdecode.CTCBeamDecoder"] = {}


def _get_ctcdecode_decoder(num_classes: int, beam_width: int, blank_id: int, num_processes: int):
    key = (int(num_classes), int(beam_width), int(blank_id), int(num_processes))
    if key in _CTCDECODE_DECODERS:
        return _CTCDECODE_DECODERS[key]
    vocab = _make_ctcdecode_vocab(num_classes)
    dec = ctcdecode.CTCBeamDecoder(
        vocab,
        beam_width=int(beam_width),
        blank_id=int(blank_id),
        num_processes=int(num_processes),
    )
    _CTCDECODE_DECODERS[key] = dec
    return dec

class CTCLoss(nn.Module):
    """CTC Loss wrapper with blank penalty to prevent blank collapse."""
    
    def __init__(self, blank_idx: int = 0, reduction: str = 'mean', blank_penalty: float = 0.0,
                 repetition_penalty: float = 0.0):
        """
        Initialize CTC Loss with optional blank and repetition penalties.
        
        Args:
            blank_idx: Index of blank token
            reduction: 'mean' or 'sum'
            blank_penalty: Penalty weight for blank predictions (0.0 = no penalty)
                          Higher values (0.1-0.5) discourage blank-heavy predictions
            repetition_penalty: Penalty weight for consecutive same-token predictions (0.0 = no penalty)
                               Higher values (0.1-0.5) discourage repetitive predictions
        """
        super().__init__()
        self.blank_idx = blank_idx
        self.blank_penalty = blank_penalty
        self.repetition_penalty = repetition_penalty
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, 
                input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CTC loss with optional blank penalty.
        
        Args:
            log_probs: [T, N, C] log probabilities (T=time, N=batch, C=vocab)
            targets: [N*S] target sequence (concatenated)
            input_lengths: [N] actual sequence lengths (can be on GPU, will be moved to CPU only for CTC loss)
            target_lengths: [N] target sequence lengths (can be on GPU, will be moved to CPU only for CTC loss)
        
        Returns:
            CTC loss value (with blank penalty if enabled)
        """
        # Keep lengths on GPU for penalty computations
        # Only move to CPU right before CTC loss call
        device = log_probs.device
        T = log_probs.shape[0]
        
        # Ensure input_lengths don't exceed sequence length (on GPU)
        input_lengths_gpu = input_lengths.long().clamp(max=T)
        
        # Compute penalties on GPU before moving lengths to CPU
        blank_penalty_term = None
        repetition_penalty_term = None
        
        # Add mild blank penalty to discourage excessive blank predictions
        # SIMPLIFIED: Only use blank penalty for overfit test, no other penalties
        # Research advisor: "regularize last, not first" - avoid penalty stacking during early training
        if self.blank_penalty > 0.0:
            # Compute average blank probability across all timesteps (all on GPU)
            blank_log_probs = log_probs[:, :, self.blank_idx]  # [T, N]
            blank_probs = torch.exp(blank_log_probs)  # [T, N]

            # Vectorized computation: mask by sequence lengths and compute weighted mean
            batch_size = blank_probs.shape[1]
            # Create mask: [T, N] where True indicates valid timesteps
            seq_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
            length_mask = seq_indices < input_lengths_gpu.unsqueeze(0)  # [T, N]
            
            # Compute mean blank probability per sequence (vectorized)
            masked_blank_probs = blank_probs * length_mask.float()  # [T, N]
            seq_blank_probs = masked_blank_probs.sum(dim=0) / input_lengths_gpu.float().clamp(min=1)  # [N]
            blank_penalty_term = seq_blank_probs.mean()  # Scalar
        
        # Add repetition penalty to discourage consecutive same-token predictions
        # This addresses CTC's tendency to allow many-to-one alignment where same token
        # is predicted at all timesteps, which CTC collapses to single token
        if self.repetition_penalty > 0.0:
            probs = torch.exp(log_probs)  # [T, N, C]
            batch_size = probs.shape[1]
            
            # Get argmax predictions (all on GPU)
            argmax_preds = torch.argmax(probs, dim=2)  # [T, N]
            
            # Vectorized computation: find consecutive same tokens
            # Shift predictions by 1 timestep to compare with previous
            prev_preds = torch.cat([argmax_preds[0:1, :], argmax_preds[:-1, :]], dim=0)  # [T, N]
            
            # Find where current == previous and both are non-blank
            same_token = (argmax_preds == prev_preds)  # [T, N]
            non_blank = (argmax_preds != self.blank_idx)  # [T, N]
            consecutive_mask = same_token & non_blank  # [T, N]
            
            # Mask by sequence lengths
            seq_indices = torch.arange(T, device=device).unsqueeze(1)  # [T, 1]
            length_mask = seq_indices < input_lengths_gpu.unsqueeze(0)  # [T, N]
            # Don't count first timestep (no previous token)
            length_mask[0, :] = False
            
            # Count consecutive same tokens per sequence
            masked_consecutive = consecutive_mask & length_mask
            consecutive_counts = masked_consecutive.sum(dim=0).float()  # [N]
            valid_lengths = (input_lengths_gpu - 1).float().clamp(min=1)  # [N], exclude first timestep
            repetition_ratios = consecutive_counts / valid_lengths  # [N]
            repetition_penalty_term = repetition_ratios.mean()  # Scalar
        
        # Now move lengths to CPU only for CTC loss call
        input_lengths_cpu = input_lengths_gpu.cpu()
        target_lengths_cpu = target_lengths.cpu().long()
        
        # Standard CTC loss (requires CPU lengths)
        ctc_loss_value = self.ctc_loss(log_probs, targets, input_lengths_cpu, target_lengths_cpu)
        
        # Apply penalties computed on GPU
        if blank_penalty_term is not None:
            ctc_loss_value = ctc_loss_value + self.blank_penalty * (blank_penalty_term ** 2)
        
        if repetition_penalty_term is not None:
            ctc_loss_value = ctc_loss_value + self.repetition_penalty * (repetition_penalty_term ** 2)
        
        return ctc_loss_value


def ctc_decode(log_probs: torch.Tensor, blank_idx: int = 0, method: str = 'greedy',
                beam_width: int = 10) -> List[List[int]]:
    """
    Decode CTC output.

    Args:
        log_probs: [T, N, C] log probabilities
        blank_idx: Index of blank token
        method: 'greedy' or 'beam_search'
        beam_width: Beam width for beam search (default: 10)
        use_gpu: Use GPU-accelerated beam search if available (default: True)

    Returns:
        List of decoded sequences
    """
    if method == 'greedy':
        return ctc_greedy_decode(log_probs, blank_idx)

    elif method == 'beam_search':
        return ctc_beam_search_decode(log_probs, blank_idx, beam_width)

    else:
        raise ValueError(f"Unknown decode method: {method}")


def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
    """
    Greedy CTC decoding (GPU-accelerated).
    
    Args:
        log_probs: [T, N, C] log probabilities
        blank_idx: Index of blank token
    
    Returns:
        List of decoded sequences
    """
    # Handle both [T, N, C] and [N, T, C] formats
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")
    
    # Assume [T, N, C] format as per docstring
    T, N, C = log_probs.shape
    
    device = log_probs.device
    probs = torch.exp(log_probs)  # [T, N, C]
    predictions = torch.argmax(probs, dim=2)  # [T, N] - all on GPU
    
    # Vectorized CTC decoding on GPU
    # Remove blanks and collapse repeats
    # Strategy: shift predictions and compare to find transitions
    
    # Create mask for non-blank tokens
    non_blank_mask = (predictions != blank_idx)  # [T, N]
    
    # Find transitions: where token changes (or first non-blank)
    # Shift predictions by 1 to compare with previous
    prev_predictions = torch.cat([
        torch.full((1, N), blank_idx, device=device, dtype=predictions.dtype),
        predictions[:-1, :]
    ], dim=0)  # [T, N]
    
    # Token changes when: (current != previous) OR (current is first non-blank after blank)
    token_changes = (predictions != prev_predictions)  # [T, N]
    
    # Keep only non-blank tokens that represent transitions
    keep_mask = non_blank_mask & token_changes  # [T, N]
    
    # For each sequence, collect kept tokens
    decoded_sequences = []
    for n in range(N):
        # Get indices where we keep tokens
        kept_tokens = predictions[:, n][keep_mask[:, n]]  # [num_kept]
        # Convert to list (only move to CPU at the very end)
        decoded_sequences.append(kept_tokens.cpu().tolist())
    
    return decoded_sequences


def ctc_beam_search_decode(
    log_probs: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
    beam_width: int = 10,
    prefer_ctcdecode: bool = True,
    num_processes: Optional[int] = None,
) -> List[List[int]]:
    """
    Beam search CTC decoding with proper prefix beam search algorithm.
    
    Args:
        log_probs: [T, N, C] log probabilities
        lengths: Optional [N] lengths. Required for ctcdecode backend; if None, uses full T.
        blank_idx: Index of blank token
        beam_width: Beam width for search
        prefer_ctcdecode: If True and ctcdecode is available, use it for speed.
        num_processes: ctcdecode worker processes (defaults to min(4, cpu_count)).
    
    Returns:
        List of decoded sequences
    """
    # Fast path: use ctcdecode if available (much faster than Python prefix beam).
    if prefer_ctcdecode and _HAS_CTCDECODE:
        if log_probs.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")
        T, N, C = log_probs.shape
        if lengths is None:
            lengths = torch.full((N,), T, dtype=torch.long, device=log_probs.device)
        lengths = lengths.to(torch.long).clamp(min=1, max=T)

        # ctcdecode expects probs (B,T,C) on CPU.
        probs_btc = log_probs.permute(1, 0, 2).softmax(dim=-1).cpu()
        lens_cpu = lengths.cpu()

        if num_processes is None:
            num_processes = max(1, min(4, (os.cpu_count() or 1)))
        dec = _get_ctcdecode_decoder(num_classes=C, beam_width=beam_width, blank_id=blank_idx, num_processes=num_processes)
        beam_result, _beam_scores, _timesteps, out_seq_len = dec.decode(probs_btc, lens_cpu)

        decoded_sequences: List[List[int]] = []
        for b in range(N):
            seq = beam_result[b][0][: out_seq_len[b][0]].tolist()
            # Collapse repeats and remove blank (mirror AdaptSign).
            seq = [x for x, _ in groupby(seq)]
            seq = [x for x in seq if x != blank_idx]
            decoded_sequences.append(seq)
        return decoded_sequences

    # Handle both [T, N, C] and [N, T, C] formats
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")
    
    # Assume [T, N, C] format as per docstring
    T, N, C = log_probs.shape
    
    device = log_probs.device
    # Pre-compute log probabilities on GPU (more numerically stable than exp then log)
    # Add small epsilon to avoid log(0)
    log_prob_dist_all = log_probs  # Already log probabilities, [T, N, C]
    
    decoded_sequences = []
    
    for n in range(N):
        # Prefix beam search for sequence n
        # We maintain prefixes as (sequence, last_token, log_prob)
        # Use log probabilities to avoid numerical issues
        prefixes = {tuple(): (blank_idx, 0.0)}  # Start with empty prefix
        
        for t in range(T):
            # Get log probabilities for this timestep (keep on GPU)
            log_prob_dist = log_prob_dist_all[t, n, :]  # [C] - on GPU
            
            new_prefixes = {}
            
            for prefix_seq, (last_token, prefix_log_prob) in prefixes.items():
                # Batch compute all token log probabilities on GPU
                # Add prefix_log_prob to all token log probs at once
                token_log_probs = log_prob_dist + prefix_log_prob  # [C] - all on GPU
                
                # Process blank token first (most common case)
                blank_log_prob = token_log_probs[blank_idx].item()
                if prefix_seq not in new_prefixes:
                    new_prefixes[prefix_seq] = (last_token, blank_log_prob)
                else:
                    # Merge probabilities (log-sum-exp approximation: use max for speed)
                    old_log_prob = new_prefixes[prefix_seq][1]
                    if blank_log_prob > old_log_prob:
                        new_prefixes[prefix_seq] = (last_token, blank_log_prob)
                
                # Process non-blank tokens
                # Get all non-blank token log probs at once (vectorized on GPU)
                non_blank_mask = torch.arange(C, device=device) != blank_idx
                non_blank_log_probs = token_log_probs[non_blank_mask]  # [C-1] - on GPU
                non_blank_indices = torch.arange(C, device=device)[non_blank_mask]  # [C-1] - on GPU
                
                # Batch transfer to CPU (single transfer instead of C-1 individual transfers)
                non_blank_log_probs_cpu = non_blank_log_probs.cpu()
                non_blank_indices_cpu = non_blank_indices.cpu()
                
                # Iterate over non-blank tokens (dictionary operations require CPU)
                for idx in range(len(non_blank_indices_cpu)):
                    token_idx = non_blank_indices_cpu[idx].item()
                    new_log_prob = non_blank_log_probs_cpu[idx].item()
                    
                    if token_idx == last_token:
                        # Same as last: extend without adding (CTC collapse)
                        if prefix_seq not in new_prefixes:
                            new_prefixes[prefix_seq] = (token_idx, new_log_prob)
                        else:
                            old_log_prob = new_prefixes[prefix_seq][1]
                            if new_log_prob > old_log_prob:
                                new_prefixes[prefix_seq] = (token_idx, new_log_prob)
                    else:
                        # Different token: extend by adding new token
                        new_seq = prefix_seq + (token_idx,)
                        if new_seq not in new_prefixes:
                            new_prefixes[new_seq] = (token_idx, new_log_prob)
                        else:
                            old_log_prob = new_prefixes[new_seq][1]
                            if new_log_prob > old_log_prob:
                                new_prefixes[new_seq] = (token_idx, new_log_prob)
            
            # Keep top beam_width prefixes
            sorted_prefixes = sorted(new_prefixes.items(), key=lambda x: x[1][1], reverse=True)
            prefixes = dict(sorted_prefixes[:beam_width])
        
        # Get best sequence
        if prefixes:
            best_prefix = max(prefixes.items(), key=lambda x: x[1][1])
            decoded_sequences.append(list(best_prefix[0]))
        else:
            decoded_sequences.append([])
    
    return decoded_sequences


def ctc_beam_search_lm_decode(
    log_probs: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
    beam_width: int = 10,
    lm = None,
    lm_weight: float = 0.3,
    idx2word: Optional[Dict[int, str]] = None,
) -> List[List[int]]:
    """
    Beam search CTC decoding with Language Model shallow fusion.
    
    Score = log_ctc + lm_weight * log_lm
    
    Args:
        log_probs: [T, N, C] log probabilities from CTC model
        blank_idx: Index of blank token
        beam_width: Beam width for search
        lm: Language model (must have log_prob(word, context) method)
        lm_weight: Weight for LM score (0 = no LM, 1 = equal weight)
        idx2word: Mapping from token index to word string (required if lm is provided)
    
    Returns:
        List of decoded sequences (token indices)
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {log_probs.dim()}D")

    T, N, C = log_probs.shape
    if lengths is None:
        lengths = torch.full((N,), T, dtype=torch.long, device=log_probs.device)
    lengths = lengths.to(torch.long).clamp(min=1, max=T)

    # Fast path: use ctcdecode to generate beams, then re-rank with LM.
    # This is a practical shallow-fusion approximation (re-ranking top beams).
    if _HAS_CTCDECODE and lm is not None and idx2word is not None:
        probs_btc = log_probs.permute(1, 0, 2).softmax(dim=-1).cpu()
        lens_cpu = lengths.cpu()
        dec = _get_ctcdecode_decoder(num_classes=C, beam_width=beam_width, blank_id=blank_idx, num_processes=max(1, min(4, (os.cpu_count() or 1))))
        beam_result, beam_scores, _timesteps, out_seq_len = dec.decode(probs_btc, lens_cpu)

        decoded_sequences: List[List[int]] = []
        for b in range(N):
            best_seq: List[int] = []
            best_score = -1e30
            for k in range(min(beam_width, beam_result.shape[1])):
                raw = beam_result[b][k][: out_seq_len[b][k]].tolist()
                raw = [x for x, _ in groupby(raw)]
                raw = [x for x in raw if x != blank_idx]
                # CTC score: ctcdecode returns negative log-likelihood; higher is better => negate.
                ctc_log = -float(beam_scores[b][k].item())
                # LM score over token strings
                words = [idx2word.get(int(t), "<unk>") for t in raw]
                lm_log = float(lm.score_sequence(words)) if hasattr(lm, "score_sequence") else 0.0
                score = ctc_log + float(lm_weight) * lm_log
                if score > best_score:
                    best_score = score
                    best_seq = raw
            decoded_sequences.append(best_seq)
        return decoded_sequences

    # Fallback: very slow pure-Python prefix beam with LM fusion (kept for completeness).
    # Note: With vocab ~1296, this is not recommended.
    device = log_probs.device
    decoded_sequences = []
    for n in range(N):
        prefixes = {tuple(): (blank_idx, 0.0, 0.0, 0.0)}
        for t in range(int(lengths[n].item())):
            log_prob_dist = log_probs[t, n, :].cpu()
            new_prefixes = {}
            for prefix_seq, (last_token, ctc_log_prob, lm_log_prob, _combined) in prefixes.items():
                blank_ctc = ctc_log_prob + log_prob_dist[blank_idx].item()
                combined_blank = blank_ctc + lm_weight * lm_log_prob
                if prefix_seq not in new_prefixes or combined_blank > new_prefixes[prefix_seq][3]:
                    new_prefixes[prefix_seq] = (last_token, blank_ctc, lm_log_prob, combined_blank)
                for token_idx in range(C):
                    if token_idx == blank_idx:
                        continue
                    token_ctc = ctc_log_prob + log_prob_dist[token_idx].item()
                    if token_idx == last_token:
                        combined = token_ctc + lm_weight * lm_log_prob
                        if prefix_seq not in new_prefixes or combined > new_prefixes[prefix_seq][3]:
                            new_prefixes[prefix_seq] = (token_idx, token_ctc, lm_log_prob, combined)
                    else:
                        new_seq = prefix_seq + (token_idx,)
                        new_lm_log_prob = lm_log_prob
                        if lm is not None and idx2word is not None:
                            word = idx2word.get(token_idx, "<unk>")
                            context_words = tuple(
                                idx2word.get(idx, "<unk>") for idx in prefix_seq[-(lm.n - 1):]
                            ) if prefix_seq else ()
                            new_lm_log_prob = lm_log_prob + lm.log_prob(word, context_words)
                        combined = token_ctc + lm_weight * new_lm_log_prob
                        if new_seq not in new_prefixes or combined > new_prefixes[new_seq][3]:
                            new_prefixes[new_seq] = (token_idx, token_ctc, new_lm_log_prob, combined)
            sorted_prefixes = sorted(new_prefixes.items(), key=lambda x: x[1][3], reverse=True)
            prefixes = {
                seq: (last, ctc, lm_score, combined)
                for seq, (last, ctc, lm_score, combined) in sorted_prefixes[:beam_width]
            }
        if prefixes:
            best_prefix = max(prefixes.items(), key=lambda x: x[1][3])
            decoded_sequences.append(list(best_prefix[0]))
        else:
            decoded_sequences.append([])
    return decoded_sequences


def prepare_ctc_targets(targets: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare targets for CTC loss.
    
    Args:
        targets: List of target sequences
        device: Device to place tensors on
    
    Returns:
        (targets_tensor, target_lengths)
        - targets_tensor: [N*S] concatenated targets
        - target_lengths: [N] target lengths
    """
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long, device=device)
    targets_flat = [token for seq in targets for token in seq]
    targets_tensor = torch.tensor(targets_flat, dtype=torch.long, device=device)
    
    return targets_tensor, target_lengths


def split_concatenated_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
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


def ids_to_string(
    ids: Iterable[int],
    idx2word: Dict[int, str],
    skip_tokens: Optional[Set[str]] = None,
) -> str:
    """
    Convert token IDs to a space-separated string.

    Args:
        ids: token id sequence
        idx2word: mapping int -> token string
        skip_tokens: token strings to skip (defaults to common blank/pad tokens)
    """
    if skip_tokens is None:
        skip_tokens = {"<blank>", "<pad>", "<BLANK>", "<PAD>"}
    words: List[str] = []
    for t in ids:
        w = idx2word.get(int(t), "")
        if not w or w in skip_tokens:
            continue
        words.append(w)
    return " ".join(words)


def ctc_greedy_decode_with_lengths(
    pred_ids: torch.Tensor,
    lengths: torch.Tensor,
    blank_idx: int = 0,
) -> List[List[int]]:
    """
    Greedy CTC decode from argmax IDs, honoring per-sample valid lengths.

    Args:
        pred_ids: (B, T) argmax token IDs
        lengths: (B,) valid lengths for each sequence
        blank_idx: blank token ID
    """
    if pred_ids.dim() != 2:
        raise ValueError(f"Expected pred_ids to be (B,T), got {tuple(pred_ids.shape)}")
    B, T = pred_ids.shape
    decoded: List[List[int]] = []
    for b in range(B):
        L = int(min(int(lengths[b].item()), T))
        seq = pred_ids[b, :L].tolist()
        out: List[int] = []
        prev = blank_idx
        for tok in seq:
            if tok != blank_idx and tok != prev:
                out.append(tok)
            prev = tok
        decoded.append(out)
    return decoded

