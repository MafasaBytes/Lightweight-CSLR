"""
Feature-level augmentation for CTC training.

SpecAugment-style augmentation applied to pre-extracted features.
Proven to help generalization in ASR/SLR with CTC.
"""

import torch
import torch.nn as nn
import random


class FeatureAugmentation(nn.Module):
    """
    SpecAugment-style augmentation for pre-extracted features.
    
    Applies:
    - Time masking: mask random contiguous time steps
    - Feature masking: mask random feature dimensions
    - Time warping: slight temporal distortion (optional)
    """
    
    def __init__(
        self,
        time_mask_max: int = 10,
        time_mask_num: int = 2,
        freq_mask_max: int = 50,
        freq_mask_num: int = 2,
        p: float = 0.5,
    ):
        """
        Args:
            time_mask_max: Maximum number of consecutive time steps to mask
            time_mask_num: Number of time masks to apply
            freq_mask_max: Maximum number of consecutive features to mask
            freq_mask_num: Number of frequency masks to apply
            p: Probability of applying augmentation
        """
        super().__init__()
        self.time_mask_max = time_mask_max
        self.time_mask_num = time_mask_num
        self.freq_mask_max = freq_mask_max
        self.freq_mask_num = freq_mask_num
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to features.
        
        Args:
            x: (B, T, D) features
            
        Returns:
            Augmented features (B, T, D)
        """
        if not self.training or random.random() > self.p:
            return x
        
        x = x.clone()
        B, T, D = x.shape
        
        # Time masking
        for _ in range(self.time_mask_num):
            t = random.randint(0, min(self.time_mask_max, T - 1))
            t0 = random.randint(0, max(0, T - t))
            x[:, t0:t0 + t, :] = 0
        
        # Feature masking
        for _ in range(self.freq_mask_num):
            f = random.randint(0, min(self.freq_mask_max, D - 1))
            f0 = random.randint(0, max(0, D - f))
            x[:, :, f0:f0 + f] = 0
        
        return x


class CTCLabelSmoothing:
    """
    Label smoothing for CTC loss.
    
    Instead of hard targets, use soft targets that redistribute
    some probability mass to non-target classes.
    
    This prevents overconfidence and improves generalization.
    """
    
    def __init__(self, smoothing: float = 0.1, blank_idx: int = 0):
        """
        Args:
            smoothing: Amount of probability to redistribute (0.0 = no smoothing)
            blank_idx: Index of CTC blank token
        """
        self.smoothing = smoothing
        self.blank_idx = blank_idx
    
    def __call__(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_loss_fn: nn.CTCLoss,
    ) -> torch.Tensor:
        """
        Compute CTC loss with label smoothing via confidence penalty.
        
        Args:
            log_probs: (T, B, V) log probabilities
            targets: (sum(target_lengths),) concatenated targets
            input_lengths: (B,)
            target_lengths: (B,)
            ctc_loss_fn: nn.CTCLoss instance
            
        Returns:
            Smoothed loss
        """
        # Standard CTC loss
        ctc_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        
        if self.smoothing == 0:
            return ctc_loss
        
        # Confidence penalty: penalize confident predictions
        # This encourages the model to be less certain, reducing overfitting
        # KL divergence from uniform distribution (entropy regularization)
        T, B, V = log_probs.shape
        
        # Compute mean log probability (negative entropy proxy)
        # Higher confidence = lower entropy = higher penalty
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # (T, B)
        
        # Mask padded positions
        mask = torch.arange(T, device=log_probs.device).unsqueeze(1) < input_lengths.unsqueeze(0)
        entropy = entropy * mask.float()
        
        # Mean entropy (we want to maximize entropy = minimize negative entropy)
        mean_entropy = entropy.sum() / mask.sum()
        
        # Combine: CTC loss - smoothing * entropy (maximize entropy)
        smoothed_loss = ctc_loss - self.smoothing * mean_entropy
        
        return smoothed_loss


def mixup_features(
    features: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    alpha: float = 0.2,
) -> tuple:
    """
    MixUp augmentation for features (adapted for CTC).
    
    Note: MixUp for CTC is tricky because targets are sequences.
    This is a simplified version that only mixes features.
    
    Args:
        features: (B, T, D) features
        targets: concatenated targets
        target_lengths: (B,)
        alpha: MixUp interpolation parameter
        
    Returns:
        Mixed features (targets unchanged for CTC)
    """
    if alpha <= 0:
        return features, targets, target_lengths
    
    B = features.size(0)
    
    # Sample lambda from Beta distribution
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    
    # Random permutation
    idx = torch.randperm(B, device=features.device)
    
    # Mix features only (CTC loss will still use original targets)
    mixed_features = lam * features + (1 - lam) * features[idx]
    
    return mixed_features, targets, target_lengths

