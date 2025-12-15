"""
Baseline: BiLSTM + CTC over pre-extracted visual features (MobileNetV3 / ViT-B/16).

This is deliberately simple:
- optional temporal subsampling (Conv1d x2, stride=2) to speed up CTC and help alignment
- BiLSTM encoder + MHSA (reuses BiLSTMEncoder from hybrid_ctc_encoder.py)
- Linear CTC head
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hybrid_ctc_encoder import BiLSTMEncoder


class FeatureBiLSTMCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        hidden_dim: int = 512,
        encoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        subsample: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_subsample = subsample

        if self.use_subsample:
            self.subsample = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        self.encoder = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            num_heads=num_heads,
        )
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, D)
            input_lengths: (B,)
        Returns:
            log_probs: (B, T', V)
            out_lengths: (B,)
        """
        x = features
        out_lengths = input_lengths

        if self.use_subsample:
            # (B,T,D) -> (B,D,T)
            x = x.transpose(1, 2)
            x = self.subsample(x)
            x = x.transpose(1, 2)  # (B,T',D)
            # Two stride-2 convs with k=3,p=1 => ceil_div4
            out_lengths = (out_lengths + 3) // 4

        enc_out, _mask = self.encoder(x, out_lengths)
        logits = self.ctc_head(enc_out)  # (B,T',V)
        log_probs = F.log_softmax(logits.float(), dim=-1)  # fp32 stability
        return log_probs, out_lengths


