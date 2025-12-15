"""
Hybrid encoder + CTC model for sign language recognition.

Keeps the existing fusion + BiLSTM encoder design, removes the autoregressive decoder,
and adds temporal downsampling + CTC projection head.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """Learnable feature fusion module."""

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


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder with multi-head self-attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 8,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM layers (bidirectional => output dim = hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,)
        Returns:
            outputs: (B, T, hidden_dim)
            mask: (B, T) boolean mask (True = valid timestep)
        """
        B, T, _ = x.shape

        # Create mask
        mask = torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)

        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Pack for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=T)

        # Self-attention with residual
        key_padding_mask = ~mask
        attn_out, _ = self.self_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask
        )
        output = self.layer_norm(lstm_out + self.dropout(attn_out))  # (B, T, hidden_dim)

        return output, mask


class HybridSeq2Seq(nn.Module):
    """
    Hybrid encoder + CTC model combining MediaPipe and MobileNetV3 features.

    Keeps:
    - FeatureFusion
    - BiLSTMEncoder

    Adds:
    - Temporal downsampling (Conv1D x2, stride=2) before the encoder
    - CTC projection head + log_softmax
    """

    def __init__(
        self,
        vocab_size: int,
        mediapipe_dim: int = 6516,
        mobilenet_dim: int = 576,
        hidden_dim: int = 512,
        encoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
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

        # Encoder
        self.encoder = BiLSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            num_heads=num_heads,
        )

        # CTC head
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (non-autoregressive).

        Args:
            features: (B, T, 7092) hybrid features
            input_lengths: (B,) original sequence lengths

        Returns:
            log_probs: (B, T', V) log-probabilities for CTC
            out_lengths: (B,) output lengths after temporal downsampling
        """
        # Clean input (keep identical behavior to previous script)
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

        # Encode
        enc_out, _ = self.encoder(x, out_lengths)  # (B, T', D)

        # CTC projection
        logits = self.ctc_head(enc_out)  # (B, T', V)

        # IMPORTANT (numerical stability under AMP):
        # When this forward runs under autocast, logits may be fp16/bf16. Computing log_softmax
        # in reduced precision can produce slightly positive "log-probabilities" near 0,
        # which can make CTCLoss go negative. Force log_softmax to run in fp32.
        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, T', V) fp32

        return log_probs, out_lengths


