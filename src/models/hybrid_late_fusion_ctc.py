"""
Late Fusion Transformer CTC model for sign language recognition.

Architecture:
  Encode(MediaPipe) → h_mp   (modality-specific temporal encoding)
  Encode(MobileNet) → h_mn   (modality-specific temporal encoding)
  Fuse(h_mp, h_mn) → h_fused
  CTC Head → log_probs

Key insight: Each modality may have different temporal dynamics.
- MediaPipe: Pose/hand keypoints (sparse, geometric)
- MobileNet: Appearance features (dense, visual)

Early fusion forces a single encoder to learn both patterns.
Late fusion lets each encoder specialize, then combines.
"""

from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ModalityEncoder(nn.Module):
    """
    Transformer encoder for a single modality.
    
    Lightweight: fewer layers than the joint encoder would have.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,) valid lengths
        Returns:
            encoded: (B, T, hidden_dim)
        """
        B, T, _ = x.shape

        # Padding mask
        mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)

        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Encode
        x = self.transformer(x, src_key_padding_mask=mask)

        return x


class CrossModalFusion(nn.Module):
    """
    Cross-modal attention fusion.
    
    Each modality attends to the other, then combines.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()

        # Cross-attention: MP attends to MN
        self.mp_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: MN attends to MP
        self.mn_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, h_mp: torch.Tensor, h_mn: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_mp: (B, T, D) MediaPipe encoded
            h_mn: (B, T, D) MobileNet encoded
            mask: (B, T) padding mask (True = valid)
        Returns:
            fused: (B, T, D)
        """
        key_padding_mask = ~mask  # PyTorch convention

        # Cross-attention with residual
        mp_attended, _ = self.mp_cross_attn(
            h_mp, h_mn, h_mn, key_padding_mask=key_padding_mask
        )
        h_mp = self.norm1(h_mp + mp_attended)

        mn_attended, _ = self.mn_cross_attn(
            h_mn, h_mp, h_mp, key_padding_mask=key_padding_mask
        )
        h_mn = self.norm2(h_mn + mn_attended)

        # Concatenate and project
        concat = torch.cat([h_mp, h_mn], dim=-1)  # (B, T, 2D)
        fused = self.fusion_proj(concat)  # (B, T, D)

        return fused


class HybridLateFusionCTC(nn.Module):
    """
    Late Fusion Transformer CTC model.

    Architecture:
    1. Temporal downsampling (shared, before encoding)
    2. Modality-specific encoders (separate Transformers)
    3. Cross-modal attention fusion
    4. Joint encoder (optional refinement)
    5. CTC head

    This separates modality-specific temporal patterns before fusion,
    which should help when MediaPipe and MobileNet have different dynamics.
    """

    def __init__(
        self,
        vocab_size: int,
        mediapipe_dim: int = 6516,
        mobilenet_dim: int = 576,
        hidden_dim: int = 256,
        modality_layers: int = 2,
        fusion_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Temporal downsampling (shared, applied to raw features)
        # We do this before splitting to ensure aligned timesteps
        self.mp_subsample = nn.Sequential(
            nn.Linear(mediapipe_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mn_subsample = nn.Sequential(
            nn.Linear(mobilenet_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Shared temporal downsampling conv
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Modality-specific encoders
        self.mp_encoder = ModalityEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=modality_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.mn_encoder = ModalityEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=modality_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Joint encoder (refines fused representation)
        joint_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_encoder = nn.TransformerEncoder(
            joint_layer,
            num_layers=fusion_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # CTC head
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, features: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (B, T, 7092) hybrid features [MediaPipe 6516 | MobileNet 576]
            input_lengths: (B,) original sequence lengths

        Returns:
            log_probs: (B, T', V)
            out_lengths: (B,)
        """
        # Clean input
        features = torch.nan_to_num(features, nan=0.0).clamp(-100, 100)

        # Split modalities
        mp_feat = features[:, :, :6516]  # (B, T, 6516)
        mn_feat = features[:, :, 6516:]  # (B, T, 576)

        # Project each modality
        mp_proj = self.mp_subsample(mp_feat)  # (B, T, D)
        mn_proj = self.mn_subsample(mn_feat)  # (B, T, D)

        # Temporal downsampling (applied separately but identically)
        mp_ds = self.temporal_conv(mp_proj.transpose(1, 2)).transpose(1, 2)  # (B, T', D)
        mn_ds = self.temporal_conv(mn_proj.transpose(1, 2)).transpose(1, 2)  # (B, T', D)

        # Update lengths
        out_lengths = (input_lengths + 3) // 4
        out_lengths = out_lengths.clamp(min=1)

        B, T_prime, _ = mp_ds.shape

        # Create mask
        mask = torch.arange(T_prime, device=features.device).expand(B, T_prime) < out_lengths.unsqueeze(1)

        # Encode each modality separately
        h_mp = self.mp_encoder(mp_ds, out_lengths)  # (B, T', D)
        h_mn = self.mn_encoder(mn_ds, out_lengths)  # (B, T', D)

        # Cross-modal fusion
        fused = self.fusion(h_mp, h_mn, mask)  # (B, T', D)

        # Joint refinement
        key_padding_mask = ~mask
        refined = self.joint_encoder(fused, src_key_padding_mask=key_padding_mask)  # (B, T', D)

        # CTC head
        logits = self.ctc_head(refined)  # (B, T', V)

        # Force fp32 for numerical stability
        log_probs = F.log_softmax(logits.float(), dim=-1)

        return log_probs, out_lengths

