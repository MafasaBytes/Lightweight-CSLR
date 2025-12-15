"""
ViT-B/16 CTC Teacher for Sign Language Recognition.

Architecture:
    Raw frames (B, T, C, H, W)
    → ViT-B/16 backbone (per-frame) → (B, T, 768)
    → Temporal Transformer encoder
    → CTC head → log_probs (B, T', vocab_size)

The backbone can be frozen (feature extractor) or fine-tuned.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ViTCTCTeacher(nn.Module):
    """
    ViT-B/16 backbone + Temporal Transformer + CTC head.

    Args:
        vocab_size: Number of output classes (including CTC blank at index 0).
        hidden_dim: Dimension for temporal encoder (projects from 768 if different).
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feedforward dimension in Transformer.
        dropout: Dropout rate.
        freeze_backbone: If True, ViT weights are frozen (faster, less memory).
        subsample_factor: Temporal downsampling factor (1 = no downsampling).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        subsample_factor: int = 4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        self.subsample_factor = subsample_factor

        # ---- ViT-B/16 backbone ----
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)
        self.vit_dim = 768  # ViT-B/16 hidden dimension

        # Remove classification head (we only need features)
        self.vit.heads = nn.Identity()

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # ---- Projection to hidden_dim ----
        self.input_proj = nn.Sequential(
            nn.Linear(self.vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ---- Temporal downsampling (Conv1D) ----
        if subsample_factor > 1:
            # Two Conv1D layers with stride=2 each → factor of 4
            self.subsample = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
        else:
            self.subsample = None

        # ---- Positional encoding ----
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=1000, dropout=dropout)

        # ---- Temporal Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- CTC head ----
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)

    def _extract_vit_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame ViT features.

        Args:
            frames: (B, T, C, H, W) video frames

        Returns:
            features: (B, T, 768) ViT CLS embeddings per frame
        """
        B, T, C, H, W = frames.shape

        # Reshape to (B*T, C, H, W) for batch processing through ViT
        frames_flat = frames.view(B * T, C, H, W)

        if self.freeze_backbone:
            with torch.no_grad():
                # ViT forward (use internal processing)
                x = self.vit._process_input(frames_flat)  # (B*T, num_patches, 768)
                n = x.shape[0]
                cls_token = self.vit.class_token.expand(n, -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = self.vit.encoder(x)
                features_flat = x[:, 0]  # CLS token: (B*T, 768)
        else:
            x = self.vit._process_input(frames_flat)
            n = x.shape[0]
            cls_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.vit.encoder(x)
            features_flat = x[:, 0]

        # Reshape back to (B, T, 768)
        features = features_flat.view(B, T, self.vit_dim)
        return features

    def forward(
        self, frames: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            frames: (B, T, C, H, W) input video frames (normalized for ViT)
            input_lengths: (B,) number of valid frames per sample

        Returns:
            log_probs: (B, T', vocab_size) log probabilities
            out_lengths: (B,) output sequence lengths after downsampling
        """
        B, T, C, H, W = frames.shape

        # 1) Extract ViT features: (B, T, 768)
        features = self._extract_vit_features(frames)

        # 2) Project to hidden_dim: (B, T, hidden_dim)
        x = self.input_proj(features)

        # 3) Temporal downsampling
        if self.subsample is not None:
            x = x.transpose(1, 2)  # (B, hidden_dim, T)
            x = self.subsample(x)  # (B, hidden_dim, T')
            x = x.transpose(1, 2)  # (B, T', hidden_dim)
            # Update lengths: T' = ceil((T + 2*pad - kernel) / stride + 1) applied twice
            # With kernel=3, stride=2, padding=1: T' ≈ T // 4
            out_lengths = (input_lengths + 3) // 4
        else:
            out_lengths = input_lengths.clone()

        T_prime = x.size(1)

        # 4) Positional encoding
        x = self.pos_enc(x)

        # 5) Create attention mask for padding
        mask = torch.arange(T_prime, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        # 6) Temporal Transformer encoder
        x = self.temporal_encoder(x, src_key_padding_mask=mask)

        # 7) CTC head
        logits = self.ctc_head(x)  # (B, T', vocab_size)

        # 8) Log softmax (compute in FP32 for numerical stability)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        return log_probs, out_lengths

    def train(self, mode: bool = True):
        """Override to keep backbone frozen if specified."""
        super().train(mode)
        if self.freeze_backbone:
            self.vit.eval()
        return self

