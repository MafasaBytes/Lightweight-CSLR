"""
AdaptSign-style CTC model with self-distillation.

Key innovations from AdaptSign:
1. Dual-branch architecture (Conv + BiLSTM)
2. Self-distillation: LSTM branch teaches Conv branch
3. Weight-normalized classifier
4. Multi-loss training (SeqCTC + ConvCTC + Distillation)

This should help break the val_loss ~2.5 plateau!
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinear(nn.Module):
    """Weight-normalized linear layer (from AdaptSign)."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize weights along input dimension
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class SeqKD(nn.Module):
    """
    Sequence-level Knowledge Distillation loss (from AdaptSign).
    
    The student (conv branch) learns from teacher (LSTM branch).
    Temperature softens the probability distribution.
    
    Note: Using T=4 instead of T=8 for better numerical stability with mixed precision.
    """
    
    def __init__(self, T: float = 4.0):  # Reduced from 8 for stability
        super().__init__()
        self.T = T
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor,  # Conv branch output
        teacher_logits: torch.Tensor,  # LSTM branch output (detached)
        use_blank: bool = False
    ) -> torch.Tensor:
        """
        Args:
            student_logits: (T, B, V) from conv branch
            teacher_logits: (T, B, V) from LSTM branch
            use_blank: whether to include blank token in distillation
        """
        start_idx = 0 if use_blank else 1
        
        # Cast to float32 for numerical stability
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()
        
        # Temperature-scaled softmax with clamping for stability
        student_scaled = student_logits[:, :, start_idx:] / self.T
        teacher_scaled = teacher_logits[:, :, start_idx:] / self.T
        
        # Clamp to prevent overflow
        student_scaled = student_scaled.clamp(-50, 50)
        teacher_scaled = teacher_scaled.clamp(-50, 50)
        
        student_log_probs = F.log_softmax(student_scaled, dim=-1).view(-1, student_logits.shape[2] - start_idx)
        teacher_probs = F.softmax(teacher_scaled, dim=-1).view(-1, teacher_logits.shape[2] - start_idx)
        
        # KL divergence scaled by T^2 (standard KD practice)
        loss = self.kl_loss(student_log_probs, teacher_probs) * (self.T ** 2)
        return loss


class TemporalConv(nn.Module):
    """
    Temporal convolution module (from AdaptSign).
    
    Conv type 2: K5 -> P2 -> K5 -> P2
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_classes: int,
        use_bn: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # K5 -> P2 -> K5 -> P2 (conv_type=2 from AdaptSign)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=5, padding=0),
            nn.BatchNorm1d(hidden_size) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=0),
            nn.BatchNorm1d(hidden_size) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, ceil_mode=False),
        )
        
        # Conv branch classifier (will be shared with LSTM branch)
        self.fc = None  # Set externally to share with main classifier
    
    def compute_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Compute output length after temporal conv."""
        # K5 reduces by 4, P2 halves
        # K5 -> P2 -> K5 -> P2
        length = input_length
        length = length - 4  # K5
        length = length // 2  # P2
        length = length - 4  # K5
        length = length // 2  # P2
        return length.clamp(min=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, D, T) - features in channel-first format
            lengths: (B,) - sequence lengths
        """
        visual_feat = self.temporal_conv(x)  # (B, H, T')
        out_lengths = self.compute_output_length(lengths)
        
        # Conv branch logits
        if self.fc is not None:
            # (B, H, T') -> (B, T', H) -> (B, T', V) -> (T', B, V)
            conv_logits = self.fc(visual_feat.transpose(1, 2)).permute(1, 0, 2)
        else:
            conv_logits = None
        
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),  # (T', B, H)
            "conv_logits": conv_logits,
            "feat_len": out_lengths,
        }


class BiLSTMLayer(nn.Module):
    """BiLSTM temporal model (from AdaptSign)."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=False  # (T, B, D)
        )
    
    def forward(
        self, 
        src_feats: torch.Tensor, 
        src_lens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            src_feats: (T, B, D)
            src_lens: (B,)
        """
        # Pack for variable length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            src_feats, src_lens.cpu(), enforce_sorted=False
        )
        packed_out, hidden = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        
        return {"predictions": outputs, "hidden": hidden}


class AdaptSignStyleCTC(nn.Module):
    """
    AdaptSign-style CTC model with self-distillation.
    
    Architecture:
        Features -> TemporalConv -> BiLSTM -> Classifier
                         |                      |
                         v                      v
                    ConvCTC loss         SeqCTC loss
                         |                      |
                         +--- Distillation loss -+
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int = 512,
        hidden_size: int = 1024,
        use_bn: bool = True,
        weight_norm: bool = True,
        share_classifier: bool = True,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Loss weights (from AdaptSign baseline.yaml)
        self.loss_weights = {
            'SeqCTC': 1.0,
            'ConvCTC': 1.0,
            'Dist': 25.0,  # This is the key!
        }
        
        # Temporal convolution (produces intermediate features + conv logits)
        self.conv1d = TemporalConv(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            use_bn=use_bn,
        )
        
        # BiLSTM temporal model
        self.temporal_model = BiLSTMLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        
        # Classifier (weight-normalized)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Share classifier between conv and lstm branches
        if share_classifier:
            self.conv1d.fc = self.classifier
        else:
            if weight_norm:
                self.conv1d.fc = NormLinear(hidden_size, num_classes)
            else:
                self.conv1d.fc = nn.Linear(hidden_size, num_classes)
        
        # Loss functions
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=False)
        self.distillation_loss = SeqKD(T=8)
        
        # Gradient hook to handle NaN gradients
        self.register_backward_hook(self._backward_hook)
    
    @staticmethod
    def _backward_hook(module, grad_input, grad_output):
        """Replace NaN gradients with zeros."""
        for g in grad_input:
            if g is not None:
                g[g != g] = 0
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
        labels: torch.Tensor = None,
        label_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, D) - input features
            lengths: (B,) - input lengths
            labels: (sum(label_lengths),) - concatenated labels for CTC
            label_lengths: (B,) - label lengths
        """
        batch_size = x.shape[0]
        
        # (B, T, D) -> (B, D, T) for conv
        x_conv = x.transpose(1, 2)
        
        # Temporal conv branch
        conv_outputs = self.conv1d(x_conv, lengths)
        conv_logits = conv_outputs['conv_logits']  # (T', B, V)
        feat_len = conv_outputs['feat_len']  # (B,)
        visual_feat = conv_outputs['visual_feat']  # (T', B, H)
        
        # BiLSTM branch
        lstm_outputs = self.temporal_model(visual_feat, feat_len)
        lstm_feat = lstm_outputs['predictions']  # (T', B, H)
        
        # Sequence logits from LSTM branch
        seq_logits = self.classifier(lstm_feat.transpose(0, 1)).permute(1, 0, 2)  # (T', B, V)
        
        # For inference: return log probs
        if not self.training:
            log_probs = F.log_softmax(seq_logits, dim=-1).permute(1, 0, 2)  # (B, T', V)
            return {
                "log_probs": log_probs,
                "feat_len": feat_len,
                "conv_logits": conv_logits,
                "sequence_logits": seq_logits,
            }
        
        # For training: compute multi-loss
        return {
            "conv_logits": conv_logits,
            "sequence_logits": seq_logits,
            "feat_len": feat_len,
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-loss: SeqCTC + ConvCTC + Distillation.
        
        Returns:
            total_loss: combined loss for backprop
            loss_dict: individual loss values for logging
        """
        conv_logits = outputs['conv_logits']  # (T', B, V)
        seq_logits = outputs['sequence_logits']  # (T', B, V)
        feat_len = outputs['feat_len']  # (B,)
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Sequence CTC loss (LSTM branch)
        seq_log_probs = F.log_softmax(seq_logits, dim=-1)
        seq_ctc = self.ctc_loss(
            seq_log_probs,
            labels.cpu().int(),
            feat_len.cpu().int(),
            label_lengths.cpu().int()
        ).mean()
        loss_dict['SeqCTC'] = seq_ctc.item()
        total_loss += self.loss_weights['SeqCTC'] * seq_ctc
        
        # 2. Conv CTC loss (auxiliary supervision)
        conv_log_probs = F.log_softmax(conv_logits, dim=-1)
        conv_ctc = self.ctc_loss(
            conv_log_probs,
            labels.cpu().int(),
            feat_len.cpu().int(),
            label_lengths.cpu().int()
        ).mean()
        loss_dict['ConvCTC'] = conv_ctc.item()
        total_loss += self.loss_weights['ConvCTC'] * conv_ctc
        
        # 3. Self-distillation loss (LSTM teaches Conv)
        # IMPORTANT: detach teacher (LSTM) to prevent gradients flowing back
        dist_loss = self.distillation_loss(
            conv_logits,
            seq_logits.detach(),
            use_blank=False
        )
        loss_dict['Dist'] = dist_loss.item()
        total_loss += self.loss_weights['Dist'] * dist_loss
        
        loss_dict['Total'] = total_loss.item()
        
        return total_loss, loss_dict

