"""
I3D Teacher Model for Sign Language Recognition
with sign language specific pre-training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SignLanguageModalityFusion(nn.Module):
    """
    Modality fusion specifically for sign language.
    Gives higher importance to hands and face for sign language.
    """

    def __init__(
        self,
        pose_dim: int = 99,
        hands_dim: int = 126,
        face_dim: int = 1404,
        temporal_dim: int = 4887,
        output_dim: int = 512,
        hand_weight: float = 0.4,  # Hands are most important
        face_weight: float = 0.3,  # Face for grammar
        pose_weight: float = 0.2,  # Body for context
        temporal_weight: float = 0.1  # Temporal features
    ):
        super().__init__()

        self.weights = {
            'hands': hand_weight,
            'face': face_weight,
            'pose': pose_weight,
            'temporal': temporal_weight
        }

        # Input normalization layers for each modality
        self.pose_norm = nn.BatchNorm1d(pose_dim)
        self.hands_norm = nn.BatchNorm1d(hands_dim)
        self.face_norm = nn.BatchNorm1d(face_dim)
        self.temporal_norm = nn.BatchNorm1d(temporal_dim)

        # Adaptive dimensions based on importance
        hand_hidden = int(output_dim * hand_weight)
        face_hidden = int(output_dim * face_weight)
        pose_hidden = int(output_dim * pose_weight)
        temporal_hidden = int(output_dim * temporal_weight)

        # Ensure dimensions sum correctly
        diff = output_dim - (hand_hidden + face_hidden + pose_hidden + temporal_hidden)
        hand_hidden += diff  # Add remainder to most important modality
        
        # Modality-specific encoders with LayerNorm
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden * 2),
            nn.LayerNorm(pose_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(pose_hidden * 2, pose_hidden)
        )
        
        self.hands_encoder = nn.Sequential(
            nn.Linear(hands_dim, hand_hidden * 2),
            nn.LayerNorm(hand_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hand_hidden * 2, hand_hidden)
        )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(face_dim, face_hidden * 2),
            nn.LayerNorm(face_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(face_hidden * 2, face_hidden)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, temporal_hidden * 2),
            nn.LayerNorm(temporal_hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(temporal_hidden * 2, temporal_hidden)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(output_dim)
        
        # Learnable modality importance weights
        self.modality_gates = nn.Parameter(torch.ones(4))
        
    def forward(
        self,
        pose: torch.Tensor,
        hands: torch.Tensor,
        face: torch.Tensor,
        temporal: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with modality-aware fusion.

        Args:
            pose: [B, T, 99] pose features
            hands: [B, T, 126] hand features
            face: [B, T, 1404] face features
            temporal: [B, T, 4887] temporal features

        Returns:
            [B, T, output_dim] fused features
        """
        B, T = pose.shape[:2]

        # Normalize each modality independently
        # Reshape for BatchNorm1d: [B*T, C]
        pose = self.pose_norm(pose.reshape(B*T, -1)).reshape(B, T, -1)
        hands = self.hands_norm(hands.reshape(B*T, -1)).reshape(B, T, -1)
        face = self.face_norm(face.reshape(B*T, -1)).reshape(B, T, -1)
        temporal = self.temporal_norm(temporal.reshape(B*T, -1)).reshape(B, T, -1)

        # Encode each modality
        pose_feat = self.pose_encoder(pose)
        hands_feat = self.hands_encoder(hands)
        face_feat = self.face_encoder(face)
        temporal_feat = self.temporal_encoder(temporal)
        
        # Apply learnable gates (sigmoid for 0-1 range)
        gates = torch.sigmoid(self.modality_gates)
        pose_feat = pose_feat * gates[0]
        hands_feat = hands_feat * gates[1]
        face_feat = face_feat * gates[2]
        temporal_feat = temporal_feat * gates[3]
        
        # Concatenate features
        fused = torch.cat([hands_feat, face_feat, pose_feat, temporal_feat], dim=-1)
        
        # Cross-modal attention
        attended, _ = self.cross_attention(fused, fused, fused)
        
        # Residual connection and normalization
        output = self.output_norm(fused + attended)
        
        return output


class InceptionModule(nn.Module):
    """
    Inception module for sign language features.
    Optimized for 1D temporal convolutions with better gradient flow.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: dict,
        stride: int = 1,
        use_batch_norm: bool = True,
        sign_language_mode: bool = True
    ):
        super().__init__()
        
        self.sign_language_mode = sign_language_mode
        
        # Branch 1: 1x1 conv
        self.branch1 = self._make_branch(
            in_channels, out_channels['1x1'], 
            kernel_size=1, stride=stride, use_bn=use_batch_norm
        )
        
        # Branch 2: 1x1 -> 3x3 (captures short-term patterns)
        self.branch2_reduce = self._make_branch(
            in_channels, out_channels['3x3_reduce'],
            kernel_size=1, use_bn=use_batch_norm
        )
        self.branch2 = self._make_branch(
            out_channels['3x3_reduce'], out_channels['3x3'],
            kernel_size=3, stride=stride, padding=1, use_bn=use_batch_norm
        )
        
        # Branch 3: 1x1 -> 5x5 (captures longer patterns)
        self.branch3_reduce = self._make_branch(
            in_channels, out_channels['5x5_reduce'],
            kernel_size=1, use_bn=use_batch_norm
        )
        
        if sign_language_mode:
            # For sign language, use 7x7 instead of 5x5 for longer temporal context
            self.branch3 = self._make_branch(
                out_channels['5x5_reduce'], out_channels['5x5'],
                kernel_size=7, stride=stride, padding=3, use_bn=use_batch_norm
            )
        else:
            self.branch3 = self._make_branch(
                out_channels['5x5_reduce'], out_channels['5x5'],
                kernel_size=5, stride=stride, padding=2, use_bn=use_batch_norm
            )
        
        # Branch 4: pooling branch
        if stride == 1:
            self.branch4 = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                self._make_branch(
                    in_channels, out_channels['pool'],
                    kernel_size=1, use_bn=use_batch_norm
                )
            )
        else:
            self.branch4 = self._make_branch(
                in_channels, out_channels['pool'],
                kernel_size=1, stride=stride, use_bn=use_batch_norm
            )
    
    def _make_branch(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_bn: bool = True
    ) -> nn.Sequential:
        """Create a branch of the inception module."""
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        ]
        
        if use_bn:
            # Use GroupNorm for stability with small batches
            num_groups = min(32, out_channels // 2)
            num_groups = max(1, num_groups)  # Ensure at least 1 group
        
            while num_groups > 1 and out_channels % num_groups != 0:
                num_groups -= 1
            layers.append(nn.GroupNorm(num_groups, out_channels))

        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through inception module."""
        x1 = self.branch1(x)
        
        x2 = self.branch2_reduce(x)
        x2 = self.branch2(x2)
        
        x3 = self.branch3_reduce(x)
        x3 = self.branch3(x3)
        
        x4 = self.branch4(x)
        
        return torch.cat([x1, x2, x3, x4], dim=1)


class I3DTeacher(nn.Module):
    """
    I3D Teacher model for sign language recognition.
    
    Key improvements:
    - Sign language specific modality fusion
    - Better weight initialization for sign language
    - Support for multiple pre-trained checkpoints
    - Improved gradient flow with residual connections
    """
    
    def __init__(
        self,
        vocab_size: int,
        pose_dim: int = 99,
        hands_dim: int = 126,
        face_dim: int = 1404,
        temporal_dim: int = 4887,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_pretrained: str = None,  # 'bsl1k', 'adaptsign', etc.
        sign_language_mode: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.sign_language_mode = sign_language_mode
        
        # Enhanced modality fusion for sign language
        self.modality_fusion = SignLanguageModalityFusion(
            pose_dim, hands_dim, face_dim, temporal_dim,
            output_dim=hidden_dim
        )
        
        # Stem network with residual
        self.stem = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual connection for stem
        self.stem_residual = nn.Conv1d(hidden_dim, 64, kernel_size=1, stride=4)
        
        # Inception blocks with sign language optimization
        self.mixed_3b = InceptionModule(64, {
            '1x1': 64,
            '3x3_reduce': 96, '3x3': 128,
            '5x5_reduce': 16, '5x5': 32,
            'pool': 32
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_3c = InceptionModule(256, {
            '1x1': 128,
            '3x3_reduce': 128, '3x3': 192,
            '5x5_reduce': 32, '5x5': 96,
            'pool': 64
        }, sign_language_mode=sign_language_mode)
        
        self.maxpool_4a = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual for pooling
        self.pool_residual_4a = nn.Conv1d(480, 480, kernel_size=1, stride=2)
        
        self.mixed_4b = InceptionModule(480, {
            '1x1': 192,
            '3x3_reduce': 96, '3x3': 208,
            '5x5_reduce': 16, '5x5': 48,
            'pool': 64
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_4c = InceptionModule(512, {
            '1x1': 160,
            '3x3_reduce': 112, '3x3': 224,
            '5x5_reduce': 24, '5x5': 64,
            'pool': 64
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_4d = InceptionModule(512, {
            '1x1': 128,
            '3x3_reduce': 128, '3x3': 256,
            '5x5_reduce': 24, '5x5': 64,
            'pool': 64
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_4e = InceptionModule(512, {
            '1x1': 112,
            '3x3_reduce': 144, '3x3': 288,
            '5x5_reduce': 32, '5x5': 64,
            'pool': 64
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_4f = InceptionModule(528, {
            '1x1': 256,
            '3x3_reduce': 160, '3x3': 320,
            '5x5_reduce': 32, '5x5': 128,
            'pool': 128
        }, sign_language_mode=sign_language_mode)
        
        self.maxpool_5a = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool_residual_5a = nn.Conv1d(832, 832, kernel_size=1, stride=2)
        
        self.mixed_5b = InceptionModule(832, {
            '1x1': 256,
            '3x3_reduce': 160, '3x3': 320,
            '5x5_reduce': 32, '5x5': 128,
            'pool': 128
        }, sign_language_mode=sign_language_mode)
        
        self.mixed_5c = InceptionModule(832, {
            '1x1': 384,
            '3x3_reduce': 192, '3x3': 384,
            '5x5_reduce': 48, '5x5': 128,
            'pool': 128
        }, sign_language_mode=sign_language_mode)
        
        # Enhanced BiLSTM for sign language sequences
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            num_layers=3,  # Deeper temporal modeling
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Attention over LSTM outputs
        self.temporal_attention = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Final classifier with dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, vocab_size)
        
        # Initialize weights based on pre-training choice
        self._initialize_weights(use_pretrained)
    
    def _initialize_weights(self, pretrained_type: Optional[str] = None):
        """Initialize weights for sign language recognition."""

        # Sign language specific initialization
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                if 'residual' in name:
                    # Residual connections - very small initialization
                    nn.init.normal_(module.weight, std=0.001)
                else:
                    # Regular convolutions - Xavier/He initialization
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                if 'classifier' in name:
                    # Proper initialization for CTC output layer - needs enough variance
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)  # Start neutral
                elif 'attention' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                elif 'encoder' in name:
                    # Encoder layers in modality fusion
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)

                if module.bias is not None and 'classifier' not in name:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LSTM):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param, gain=1.0)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in param_name:
                        nn.init.constant_(param, 0)
                        # Forget gate bias = 1 for better gradient flow
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)

            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        if pretrained_type:
            logger.info(f"Applied base initialization, pretrained '{pretrained_type}' will override some weights")
    
    def extract_features(self, features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Split input features by modality."""
        pose = features[:, :, :99]
        hands = features[:, :, 99:225]
        face = features[:, :, 225:1629]
        temporal = features[:, :, 1629:]
        return pose, hands, face, temporal
    
    def forward(
        self,
        features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            features: [B, T, 6516] MediaPipe features
            input_lengths: [B] sequence lengths
            return_features: Return intermediate features for distillation

        Returns:
            [T, B, vocab_size] log probabilities for CTC loss
        """
        B, T, _ = features.shape

        # Clean NaN/Inf values but DON'T normalize here - let BatchNorm handle it
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = torch.clamp(features, min=-100.0, max=100.0)  # Wider range for raw features
        
        # Extract modality features
        pose, hands, face, temporal = self.extract_features(features)
        
        # Modality fusion
        fused = self.modality_fusion(pose, hands, face, temporal)  # [B, T, 512]
        
        # Reshape for Conv1d [B, C, T]
        x = fused.transpose(1, 2)
        
        # Store features for distillation
        intermediate_features = []
        
        # Stem with residual
        identity = self.stem_residual(x)
        x = self.stem(x)
        x = x + identity  # Residual connection
        
        if return_features:
            intermediate_features.append(x)
        
        # Inception blocks
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        
        if return_features:
            intermediate_features.append(x)
        
        # Pooling with residual
        identity = self.pool_residual_4a(x)
        x = self.maxpool_4a(x)
        x = x + identity
        
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        
        if return_features:
            intermediate_features.append(x)
        
        # Pooling with residual
        identity = self.pool_residual_5a(x)
        x = self.maxpool_5a(x)
        x = x + identity
        
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        
        if return_features:
            intermediate_features.append(x)
        
        # Align temporal resolution for the LSTM
        conv_time = x.shape[-1]
        if conv_time != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
        
        x = x.transpose(1, 2)  # [B, T, 1024]
        
        # BiLSTM with packing for efficiency
        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, _ = self.lstm(x)
        
        if input_lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
                
        # Classifier with dropout
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)  # [B, T, vocab_size]
        
        # Apply log_softmax for CTC loss (CTC expects log probabilities, not logits)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC loss [T, B, vocab_size]
        output = log_probs.transpose(0, 1)
        
        if return_features:
            return output, intermediate_features
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_i3d_teacher(
    vocab_size: int,
    dropout: float = 0.3,
    use_pretrained: str = None,
    freeze_backbone: bool = False,
    **kwargs
) -> I3DTeacher:
    """
    Create I3D teacher model with sign language optimizations.
    
    Args:
        vocab_size: Size of vocabulary
        dropout: Dropout rate
        use_pretrained: Pre-trained model name ('bsl1k', 'adaptsign', etc.)
        freeze_backbone: Whether to freeze early layers
        **kwargs: Additional arguments
        
    Returns:
        I3DTeacher model
    """
    model = I3DTeacher(
        vocab_size=vocab_size,
        dropout=dropout,
        use_pretrained=use_pretrained,
        sign_language_mode=True,
        **kwargs
    )
    
    def _freeze_backbone_layers(target_model: I3DTeacher, freeze_until: Optional[str]):
        if not freeze_backbone:
            return
        frozen = []
        for name, param in target_model.named_parameters():
            if freeze_until and freeze_until in name:
                break
            if 'classifier' in name or 'fusion' in name:
                continue
            param.requires_grad = False
            frozen.append(name)
        logger.info(f"Froze {len(frozen)} parameter tensors until {freeze_until}")
    
    # Load pre-trained weights if specified
    if use_pretrained:
        freeze_until = 'mixed_4f' if freeze_backbone else None
        candidate_path = Path(use_pretrained)
        
        if candidate_path.is_file():
            logger.info(f"Loading local checkpoint from {candidate_path}")
            try:
                checkpoint = torch.load(candidate_path, map_location='cpu')
            except pickle.UnpicklingError as exc:
                logger.warning(
                    "Secure weights-only load failed for %s (%s). "
                    "Retrying with weights_only=False; only do this for trusted checkpoints.",
                    candidate_path,
                    exc
                )
                checkpoint = torch.load(candidate_path, map_location='cpu', weights_only=False)
            preferred_keys = ['state_dict', 'model_state_dict', 'model', 'weights']
            state_dict = None
            for key in preferred_keys:
                maybe_state = checkpoint.get(key) if isinstance(checkpoint, dict) else None
                if isinstance(maybe_state, dict):
                    state_dict = maybe_state
                    logger.info(f"Loaded weights from checkpoint key '{key}'")
                    break
            if state_dict is None:
                state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            
            model_state = model.state_dict()
            filtered_state = {}
            mismatched = []
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_state[key] = value
                else:
                    mismatched.append(key)
            load_result = model.load_state_dict(filtered_state, strict=False)
            missing_keys = getattr(load_result, "missing_keys", [])
            unexpected_keys = getattr(load_result, "unexpected_keys", [])
            if missing_keys:
                preview = missing_keys[:10]
                logger.warning(f"Missing keys when loading {candidate_path}: {preview}{'...' if len(missing_keys) > 10 else ''}")
            if unexpected_keys:
                preview = unexpected_keys[:10]
                logger.warning(f"Unexpected keys when loading {candidate_path}: {preview}{'...' if len(unexpected_keys) > 10 else ''}")
            if mismatched:
                preview = mismatched[:10]
                logger.warning(f"Skipped {len(mismatched)} mismatched tensors (e.g., {preview})")
            _freeze_backbone_layers(model, freeze_until)
        else:
            try:
                from src.models.pretrained_loader import load_sign_language_pretrained
            except ImportError:
                from pretrained_loader import load_sign_language_pretrained
            
            model = load_sign_language_pretrained(
                model,
                model_name=use_pretrained,
                freeze_backbone=freeze_backbone,
                freeze_until=freeze_until
            )
    
    # Log model info
    total_params = model.count_parameters()
    model_size_mb = total_params * 4 / 1024 / 1024
    
    logger.info(f"I3D Teacher Model created!")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    logger.info(f"Using pretrained: {use_pretrained}")
    logger.info(f"Sign language mode: True")
    
    return model


if __name__ == "__main__":
    # Test the  model
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Local imports for demonstration
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.vocabulary import load_vocabulary_from_file

    # Load vocabulary from the file generated by create_vocabulary.py
    vocab_path = Path("data/clean_vocabulary/vocabulary.txt")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Please run src/utils/create_vocabulary.py first.")
        
    vocab = load_vocabulary_from_file(vocab_path)
    vocab_size = len(vocab)
    
    # Test with sign language pre-training
    model = create_i3d_teacher(
        vocab_size,
        use_pretrained='adaptsign_base',
        freeze_backbone=True
    )
    
    # Test forward pass
    batch_size = 2
    seq_length = 100
    feature_dim = 6516
    
    dummy_input = torch.randn(batch_size, seq_length, feature_dim)
    dummy_lengths = torch.tensor([100, 90])
    
    output = model(dummy_input, dummy_lengths, return_features=False)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [T={seq_length}, B={batch_size}, V={vocab_size}]")
    
    print("\nI3D teacher model ready for training!")