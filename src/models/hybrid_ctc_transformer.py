"""
Hybrid Transformer Encoder + CTC model for sign language recognition.

Replaces BiLSTM with Transformer Encoder for better long-range modeling
and reduced overfitting through attention-based regularization.
"""

from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            x + positional encoding: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeatureFusion(nn.Module):
    """Learnable feature fusion module (same as BiLSTM version)."""

    def __init__(
        self,
        mediapipe_dim: int = 6516,
        mobilenet_dim: int = 576,
        output_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project each modality to same dimension
        self.mp_proj = nn.Sequential(
            nn.Linear(mediapipe_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mn_proj = nn.Sequential(
            nn.Linear(mobilenet_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention-based fusion
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, 7092) concatenated MediaPipe + MobileNet features
        Returns:
            fused: (B, T, output_dim)
        """
        # Split features
        mp_feat = features[:, :, :6516]
        mn_feat = features[:, :, 6516:]

        # Project
        mp_proj = self.mp_proj(mp_feat)  # (B, T, D)
        mn_proj = self.mn_proj(mn_feat)  # (B, T, D)

        # Compute attention weights
        concat = torch.cat([mp_proj, mn_proj], dim=-1)  # (B, T, 2D)
        weights = self.attention(concat)  # (B, T, 2)

        # Weighted fusion
        fused = weights[:, :, 0:1] * mp_proj + weights[:, :, 1:2] * mn_proj  # (B, T, D)

        return self.output_proj(fused)


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for sequence modeling.
    
    Replaces BiLSTM with:
    - Positional encoding
    - nn.TransformerEncoder (multi-layer self-attention)
    
    Benefits over BiLSTM:
    - Global receptive field (all positions attend to all)
    - Better gradient flow (no vanishing gradients through recurrence)
    - Natural regularization through attention dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection (if needed)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,) valid sequence lengths

        Returns:
            outputs: (B, T, hidden_dim)
            mask: (B, T) boolean mask (True = valid timestep)
        """
        B, T, _ = x.shape

        # Create padding mask (True = masked/invalid position for PyTorch Transformer)
        mask = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)
        src_key_padding_mask = ~mask  # PyTorch convention: True = ignore

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, T, hidden_dim)

        # Transformer forward
        output = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, hidden_dim)

        return output, mask


class HybridTransformerCTC(nn.Module):
    """
    Hybrid Transformer Encoder + CTC model.

    Architecture:
    1. Feature fusion (MediaPipe + MobileNet)
    2. Temporal downsampling (Conv1D x2, stride=2)
    3. Transformer Encoder
    4. CTC projection head

    Compared to BiLSTM version:
    - Better long-range dependency modeling
    - More stable training with pre-norm
    - Natural attention-based regularization
    """

    def __init__(
        self,
        vocab_size: int,
        mediapipe_dim: int = 6516,
        mobilenet_dim: int = 576,
        hidden_dim: int = 512,
        encoder_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Feature fusion
        self.fusion = FeatureFusion(
            mediapipe_dim=mediapipe_dim,
            mobilenet_dim=mobilenet_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Temporal downsampling after fusion (B, T, D) -> (B, T/4, D)
        self.subsample = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # CTC head
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, features: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (non-autoregressive).

        Args:
            features: (B, T, 7092) hybrid features
            input_lengths: (B,) original sequence lengths

        Returns:
            log_probs: (B, T', V) log-probabilities for CTC
            out_lengths: (B,) output lengths after temporal downsampling
        """
        # Clean input
        features = torch.nan_to_num(features, nan=0.0).clamp(-100, 100)

        # Fuse features
        fused = self.fusion(features)  # (B, T, D)

        # Temporal subsampling (Conv1D expects (B, D, T))
        x = fused.transpose(1, 2)  # (B, D, T)
        x = self.subsample(x)  # (B, D, T')
        x = x.transpose(1, 2)  # (B, T', D)

        # Lengths after two stride-2 convolutions
        # For Conv1d(kernel=3, stride=2, pad=1): L_out = floor((L+1)/2) = (L+1)//2
        # Applied twice => ( (L+1)//2 + 1 )//2 = (L+3)//4
        out_lengths = (input_lengths + 3) // 4
        out_lengths = out_lengths.clamp(min=1)

        # Encode with Transformer
        enc_out, _ = self.encoder(x, out_lengths)  # (B, T', D)

        # CTC projection
        logits = self.ctc_head(enc_out)  # (B, T', V)

        # Force log_softmax in fp32 for numerical stability under AMP
        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T', V) fp32

        return log_probs, out_lengths

