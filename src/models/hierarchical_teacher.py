"""
Hierarchical Teacher Model for Sign Language Recognition

This architecture addresses overfitting through:
1. Multi-scale temporal modeling (frame → sign → sentence)
2. Hierarchical attention (local → global)
3. Progressive feature abstraction
4. Better regularization through hierarchical structure

Aligns with research proposal:
- Still uses I3D-inspired features (compatible with knowledge distillation)
- Maintains BiLSTM + attention (as proposed)
- Can serve as teacher for MobileNetV3 student
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class HierarchicalTemporalEncoder(nn.Module):
    """
    Hierarchical temporal encoder that processes features at multiple scales.
    
    Architecture:
    - Level 1: Frame-level features (fine-grained, short-term)
    - Level 2: Sign-level features (medium-term, word-level)
    - Level 3: Sentence-level features (coarse-grained, long-term)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_levels: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Level 1: Frame-level (short-term, 1-3 frames)
        self.level1_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)  # Less dropout at fine level
        )
        
        # Level 2: Sign-level (medium-term, 5-15 frames)
        self.level2_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7)
        )
        
        # Level 3: Sentence-level (long-term, 20-50 frames)
        self.level3_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=15, padding=7),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] input features
        Returns:
            [B, T, hidden_dim] hierarchical features
        """
        B, T, C = x.shape
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Extract features at different scales
        level1 = self.level1_conv(x)  # Fine-grained
        level2 = self.level2_conv(x)  # Medium-grained
        level3 = self.level3_conv(x)  # Coarse-grained
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([level1, level2, level3], dim=1)  # [B, 3*C, T]
        
        # Fuse multi-scale features
        fused = self.fusion(multi_scale)  # [B, hidden_dim, T]
        
        return fused.transpose(1, 2)  # [B, T, hidden_dim]


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism:
    1. Local attention (within sign boundaries)
    2. Global attention (across entire sequence)
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Local attention (short-range dependencies)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Global attention (long-range dependencies)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating mechanism to balance local vs global
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, hidden_dim]
            mask: Optional attention mask
        Returns:
            [B, T, hidden_dim] attended features
        """
        # Local attention (within local window)
        local_out, _ = self.local_attention(x, x, x, attn_mask=mask)
        local_out = self.norm1(x + self.dropout(local_out))
        
        # Global attention (across entire sequence)
        global_out, _ = self.global_attention(local_out, local_out, local_out)
        global_out = self.norm2(local_out + self.dropout(global_out))
        
        # Gated fusion
        combined = torch.cat([local_out, global_out], dim=-1)
        gate_weights = self.gate(combined)
        output = gate_weights * local_out + (1 - gate_weights) * global_out
        
        return output


class HierarchicalBiLSTM(nn.Module):
    """
    Hierarchical BiLSTM that processes features at multiple temporal scales.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Bottom layer: Frame-level LSTM
        self.frame_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout between layers
        )
        
        # Middle layer: Sign-level LSTM (processes frame-level outputs)
        self.sign_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        # Top layer: Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim,
            num_layers=num_layers - 2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 2 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
            lengths: Optional sequence lengths
        Returns:
            [B, T, hidden_dim * 2] hierarchical LSTM outputs
        """
        # Frame-level processing
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            frame_out, _ = self.frame_lstm(x_packed)
            frame_out, _ = nn.utils.rnn.pad_packed_sequence(
                frame_out, batch_first=True
            )
        else:
            frame_out, _ = self.frame_lstm(x)
        
        frame_out = self.dropout(frame_out)
        
        # Sign-level processing
        if lengths is not None:
            sign_packed = nn.utils.rnn.pack_padded_sequence(
                frame_out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            sign_out, _ = self.sign_lstm(sign_packed)
            sign_out, _ = nn.utils.rnn.pad_packed_sequence(
                sign_out, batch_first=True
            )
        else:
            sign_out, _ = self.sign_lstm(frame_out)
        
        sign_out = self.dropout(sign_out)
        
        # Sentence-level processing
        if lengths is not None:
            sent_packed = nn.utils.rnn.pack_padded_sequence(
                sign_out, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            sent_out, _ = self.sentence_lstm(sent_packed)
            sent_out, _ = nn.utils.rnn.pad_packed_sequence(
                sent_out, batch_first=True
            )
        else:
            sent_out, _ = self.sentence_lstm(sign_out)
        
        return sent_out


class HierarchicalTeacher(nn.Module):
    """
    Hierarchical Teacher Model for Sign Language Recognition.
    
    Architecture:
    1. Modality fusion (same as I3D teacher)
    2. Hierarchical temporal encoder (multi-scale features)
    3. Hierarchical attention (local + global)
    4. Hierarchical BiLSTM (frame → sign → sentence)
    5. Final classifier
    
    Benefits:
    - Better generalization through multi-scale features
    - Reduced overfitting via hierarchical regularization
    - Improved temporal modeling
    """
    
    def __init__(
        self,
        vocab_size: int,
        pose_dim: int = 99,
        hands_dim: int = 126,
        face_dim: int = 1404,
        temporal_dim: int = 4887,
        hidden_dim: int = 512,
        dropout: float = 0.5,  # Higher default dropout for regularization
        use_pretrained: Optional[str] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Import modality fusion from I3D teacher 
        from src.models.i3d_teacher import SignLanguageModalityFusion
        
        # Modality fusion (same as I3D teacher)
        self.modality_fusion = SignLanguageModalityFusion(
            pose_dim, hands_dim, face_dim, temporal_dim,
            output_dim=hidden_dim
        )
        
        # Hierarchical temporal encoder
        self.temporal_encoder = HierarchicalTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_levels=3,
            dropout=dropout
        )
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Hierarchical BiLSTM
        self.hierarchical_lstm = HierarchicalBiLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=dropout
        )
        
        # Final classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def extract_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract modality features from MediaPipe features.
        Same as I3D teacher for compatibility.
        
        Args:
            features: [B, T, 6516] MediaPipe features
        
        Returns:
            Tuple of (pose, hands, face, temporal) tensors
        """
        # MediaPipe feature structure:
        # [0:99] = pose (33 keypoints × 3 coords)
        # [99:225] = hands (63 left + 63 right = 126)
        # [225:1629] = face (468 keypoints × 3 coords = 1404)
        # [1629:] = temporal features (velocities, accelerations, etc.)
        pose = features[:, :, :99]
        hands = features[:, :, 99:225]
        face = features[:, :, 225:1629]
        temporal = features[:, :, 1629:]
        return pose, hands, face, temporal
    
    def _initialize_weights(self):
        """Initialize weights for hierarchical architecture."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical architecture.
        
        Args:
            features: [B, T, feature_dim] MediaPipe features
            input_lengths: Optional sequence lengths
            return_features: If True, return intermediate features
        
        Returns:
            [T, B, vocab_size] log probabilities for CTC
        """
        B, T, _ = features.shape
        
        # Clean NaN/Inf values 
        features = torch.nan_to_num(features,
                                    nan=0.0, posinf=10.0, neginf=-10.0,
                                    nan_policy='clip', posinf_policy='clip', neginf_policy='clip'
                                    )
        features = torch.clamp(features, min=-100.0, max=100.0)
        
        # Extract modality features
        pose, hands, face, temporal = self.extract_features(features)
        
        # Modality fusion
        fused = self.modality_fusion(pose, hands, face, temporal)  # [B, T, hidden_dim]
        
        # Hierarchical temporal encoding
        temporal_features = self.temporal_encoder(fused)  # [B, T, hidden_dim]
        
        # Hierarchical attention
        attended = self.hierarchical_attention(temporal_features)  # [B, T, hidden_dim]
        
        # Hierarchical BiLSTM
        lstm_out = self.hierarchical_lstm(attended, input_lengths)  # [B, T, hidden_dim * 2]
        
        # Final classification
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # [B, T, vocab_size]
        
        # Apply log_softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC: [T, B, vocab_size]
        output = log_probs.transpose(0, 1)
        
        if return_features:
            return output, {
                'fused': fused,
                'temporal': temporal_features,
                'attended': attended,
                'lstm': lstm_out
            }
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hierarchical_teacher(
    vocab_size: int,
    dropout: float = 0.5,
    use_pretrained: Optional[str] = None,
    **kwargs
) -> HierarchicalTeacher:
    """
    Create hierarchical teacher model.
    
    Args:
        vocab_size: Vocabulary size
        dropout: Dropout rate (default 0.5 for strong regularization)
        use_pretrained: Optional pretrained model path
        **kwargs: Additional arguments
    
    Returns:
        HierarchicalTeacher model
    """
    model = HierarchicalTeacher(
        vocab_size=vocab_size,
        dropout=dropout,
        use_pretrained=use_pretrained,
        **kwargs
    )
    
    return model

