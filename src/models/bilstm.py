"""
Optimized BiLSTM-CTC model for sign language recognition (Phase I baseline).

Architecture improvements:
- Input projection layer for gradual dimension increase
- Layer normalization for training stability
- Temporal subsampling to reduce CTC sequence length
- Residual connections for better gradient flow
- Optimized hyperparameters for the PHOENIX dataset

Target WER: 35-45% baseline performance with improved training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class TemporalConvSubsampling(nn.Module):
    """
    Temporal subsampling using 1D convolutions.
    Reduces sequence length by factor of 2 while preserving temporal information.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, features)
            lengths: (batch,) actual sequence lengths
        Returns:
            x: (batch, time//2, features)
            lengths: (batch,) subsampled lengths
        """
        # Transpose for Conv1d: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)  # Back to (B, T, F)
        x = self.norm(x)
        x = self.activation(x)

        # Update lengths after subsampling
        lengths = (lengths + 1) // 2  # Ceiling division

        return x, lengths


class BiLSTMLayer(nn.Module):
    """
    Single BiLSTM layer with layer normalization and residual connection.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # We handle dropout separately
            bidirectional=True
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.residual_proj = None
        if input_dim != hidden_dim * 2:
            self.residual_proj = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Pack sequences for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward
        packed_output, _ = self.lstm(packed_input)

        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        # Residual connection
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        # Add residual and normalize
        lstm_out = self.layer_norm(lstm_out + residual)
        lstm_out = self.dropout(lstm_out)

        return lstm_out


class OptimizedBiLSTMModel(nn.Module):
    """
    Optimized Bidirectional LSTM with CTC loss for continuous sign language recognition.

    Key improvements:
    - Input projection for gradual dimension scaling
    - Temporal subsampling to reduce CTC computation
    - Layer normalization for training stability
    - Residual connections for better gradient flow
    - Optimized for 512-dim enhanced features (pose + hands + face + temporal with PCA)

    Args:
        input_dim: Input feature dimensionality (default: 512 for enhanced PCA features)
        hidden_dim: LSTM hidden state size (default: 256 - increased for richer features)
        num_layers: Number of stacked BiLSTM layers (default: 4 - deeper for complex patterns)
        vocab_size: Output vocabulary size (default: 1229)
        dropout: Dropout probability (default: 0.3)
        projection_dim: Dimension of input projection (default: 256)
        subsample_factor: Temporal subsampling factor (default: 2)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 4,
        vocab_size: int = 1229,
        dropout: float = 0.3,
        projection_dim: int = 256,
        subsample_factor: int = 2
    ):
        super(OptimizedBiLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.subsample_factor = subsample_factor

        # Input projection layer for gradual dimension increase
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Light dropout on input
        )

        # Optional temporal subsampling to reduce sequence length
        self.temporal_subsample = None
        if subsample_factor > 1:
            self.temporal_subsample = TemporalConvSubsampling(
                projection_dim, hidden_dim * 2
            )
            first_lstm_input = hidden_dim * 2
        else:
            first_lstm_input = projection_dim

        # Stack of BiLSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = first_lstm_input if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                BiLSTMLayer(input_size, hidden_dim, dropout)
            )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # LSTM input weights - Xavier
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # LSTM hidden weights - Orthogonal
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases - Zero with forget gate bias = 1
                nn.init.constant_(param, 0.0)
                # Forget gate bias trick for LSTM
                if 'lstm' in name:
                    # Forget gate is second quarter of bias vector
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            elif 'weight' in name and 'norm' not in name:
                # Linear layers - proper initialization based on dimensions
                if param.dim() >= 2:  # Only for 2D+ tensors
                    if 'output_projection' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'conv' in name:
                        # Convolutional layers
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        # Other linear layers
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through optimized BiLSTM-CTC model.

        Args:
            x: (batch_size, max_seq_len, input_dim) input features
            lengths: (batch_size,) actual sequence lengths

        Returns:
            log_probs: (max_seq_len, batch_size, vocab_size) for CTC loss
            output_lengths: (batch_size,) output sequence lengths after subsampling
        """
        batch_size, max_seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Temporal subsampling (if enabled)
        if self.temporal_subsample is not None:
            x, lengths = self.temporal_subsample(x, lengths)
            max_seq_len = x.size(1)

        # Forward through stacked BiLSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, lengths)

        # Output projection to vocabulary
        logits = self.output_projection(x)

        # Log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to time-first for CTC: (B, T, V) -> (T, B, V)
        log_probs = log_probs.transpose(0, 1).contiguous()

        return log_probs, lengths

    def forward_with_hidden_states(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_layer_indices: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with intermediate hidden state extraction for knowledge distillation.

        Args:
            x: (batch_size, max_seq_len, input_dim) input features
            lengths: (batch_size,) actual sequence lengths
            return_layer_indices: List of LSTM layer indices to return hidden states from.
                                 If None, returns all layers.

        Returns:
            log_probs: (max_seq_len, batch_size, vocab_size) for CTC loss
            output_lengths: (batch_size,) output sequence lengths after subsampling
            hidden_states: Dict mapping layer index to hidden state tensor
                          (batch_size, max_seq_len, hidden_dim * 2)
        """
        batch_size, max_seq_len, _ = x.shape
        hidden_states = {}

        # Input projection
        x = self.input_projection(x)

        # Temporal subsampling (if enabled)
        if self.temporal_subsample is not None:
            x, lengths = self.temporal_subsample(x, lengths)
            max_seq_len = x.size(1)

        # Forward through stacked BiLSTM layers, capturing hidden states
        for layer_idx, lstm_layer in enumerate(self.lstm_layers):
            x = lstm_layer(x, lengths)

            # Store hidden state if requested
            if return_layer_indices is None or layer_idx in return_layer_indices:
                hidden_states[layer_idx] = x.clone()

        # Output projection to vocabulary
        logits = self.output_projection(x)

        # Log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to time-first for CTC: (B, T, V) -> (T, B, V)
        log_probs = log_probs.transpose(0, 1).contiguous()

        return log_probs, lengths, hidden_states

    def predict(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference with greedy decoding.

        Args:
            x: (batch_size, max_seq_len, input_dim) input features
            lengths: (batch_size,) actual sequence lengths

        Returns:
            predictions: (batch_size, max_seq_len) predicted token indices
            scores: (batch_size,) confidence scores
        """
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(x, lengths)

            # Greedy decoding: take argmax at each timestep
            predictions = log_probs.argmax(dim=-1).transpose(0, 1)

            # Calculate confidence scores (sum of max log probs)
            scores = log_probs.max(dim=-1)[0].sum(dim=0)

        return predictions, scores

    def beam_search(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        beam_width: int = 5,
        blank_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Beam search decoding for better accuracy (optional, for evaluation).

        Args:
            x: (batch_size, max_seq_len, input_dim) input features
            lengths: (batch_size,) actual sequence lengths
            beam_width: Number of beams to maintain
            blank_id: Index of CTC blank token

        Returns:
            predictions: (batch_size, max_seq_len) best predicted sequences
            scores: (batch_size,) sequence scores
        """
        # Implementation placeholder - can be added if needed
        # For now, fallback to greedy decoding
        return self.predict(x, lengths)

    def get_num_params(self) -> Dict[str, int]:
        """Calculate detailed parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Detailed breakdown
        details = {}
        details['input_projection'] = sum(
            p.numel() for name, p in self.named_parameters()
            if 'input_projection' in name
        )
        details['temporal_subsample'] = sum(
            p.numel() for name, p in self.named_parameters()
            if 'temporal_subsample' in name
        )
        details['lstm_layers'] = sum(
            p.numel() for name, p in self.named_parameters()
            if 'lstm_layers' in name
        )
        details['output_projection'] = sum(
            p.numel() for name, p in self.named_parameters()
            if 'output_projection' in name
        )

        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable,
            'details': details
        }

    def get_model_size_mb(self) -> float:
        """Estimate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def get_flops_per_frame(self) -> int:
        """
        Estimate FLOPs per input frame for efficiency analysis.

        Returns approximate FLOPs for processing one frame.
        """
        flops = 0

        # Input projection
        flops += 2 * self.input_dim * self.projection_dim

        # LSTM layers (approximate: 8 * hidden * (input + hidden) per timestep)
        for i in range(self.num_layers):
            input_size = self.projection_dim if i == 0 else self.hidden_dim * 2
            flops += 8 * self.hidden_dim * (input_size + self.hidden_dim)

        # Output projection
        flops += 2 * self.hidden_dim * 2 * self.vocab_size

        # Account for subsampling
        if self.subsample_factor > 1:
            flops = flops // self.subsample_factor

        return flops


def create_model(
    input_dim: int = 512,
    hidden_dim: int = 256,
    num_layers: int = 4,
    vocab_size: int = 1229,
    dropout: float = 0.3,
    projection_dim: int = 256,
    subsample_factor: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> OptimizedBiLSTMModel:
    """
    Factory function to create optimized BiLSTM model with detailed configuration.

    Default configuration is optimized for:
    - 512-dimensional enhanced features (pose + hands + face + temporal) with PCA
    - 1229 vocabulary size (including special tokens)
    - 8GB VRAM training constraint
    - ~47-50% WER target with enhanced features
    """
    model = OptimizedBiLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dropout=dropout,
        projection_dim=projection_dim,
        subsample_factor=subsample_factor
    )
    model = model.to(device)

    # Calculate model statistics
    params = model.get_num_params()
    size_mb = model.get_model_size_mb()
    flops_per_frame = model.get_flops_per_frame()

    # Print detailed model summary
    print(f"\n{'='*70}")
    print("Optimized BiLSTM Model Summary")
    print(f"{'='*70}")
    print(f"Architecture Configuration:")
    print(f"  Input dimension:      {input_dim}")
    print(f"  Projection dimension: {projection_dim}")
    print(f"  Hidden dimension:     {hidden_dim}")
    print(f"  Number of layers:     {num_layers}")
    print(f"  Vocabulary size:      {vocab_size}")
    print(f"  Dropout rate:         {dropout}")
    print(f"  Subsample factor:     {subsample_factor}")
    print(f"  Bidirectional:        True")

    print(f"\nParameter Breakdown:")
    print(f"  Total parameters:     {params['total']:,}")
    print(f"  Trainable:           {params['trainable']:,}")
    print(f"  Non-trainable:       {params['non_trainable']:,}")

    if 'details' in params:
        print(f"\n  Component breakdown:")
        for component, count in params['details'].items():
            if count > 0:
                print(f"    {component:20s}: {count:,}")

    print(f"\nEfficiency Metrics:")
    print(f"  Model size:          {size_mb:.2f} MB")
    print(f"  FLOPs per frame:     ~{flops_per_frame/1e6:.1f}M")
    print(f"  Device:              {device}")

    # Memory estimation for batch processing
    batch_size = 32
    seq_length = 150
    input_memory = batch_size * seq_length * input_dim * 4 / (1024**2)  # Float32
    activation_memory_estimate = batch_size * seq_length * hidden_dim * 2 * 4 * num_layers / (1024**2)

    print(f"\nMemory Estimates (batch_size={batch_size}, seq_len={seq_length}):")
    print(f"  Input tensor:        {input_memory:.1f} MB")
    print(f"  Activations (est):   {activation_memory_estimate:.1f} MB")
    print(f"  Total (est):         {size_mb + input_memory + activation_memory_estimate:.1f} MB")
    print(f"{'='*70}\n")

    return model


if __name__ == "__main__":
    """Test the optimized model with realistic input."""
    import time

    print("Testing Optimized BiLSTM Model...")
    print("-" * 70)

    # Create model with actual parameters
    model = create_optimized_bilstm_model(
        input_dim=66,      # Actual pose features dimension
        hidden_dim=192,    # Optimized hidden size
        num_layers=3,      # Deeper for better temporal modeling
        vocab_size=1229,   # Actual vocabulary size
        dropout=0.3,
        projection_dim=128,
        subsample_factor=2
    )

    # Test with realistic batch
    batch_size = 8
    max_seq_len = 150  # Average video length
    x = torch.randn(batch_size, max_seq_len, 66)
    lengths = torch.tensor([150, 140, 130, 120, 110, 100, 90, 80])

    # Move to device
    device = next(model.parameters()).device
    x, lengths = x.to(device), lengths.to(device)

    # Test forward pass
    print(f"\nInput shape: {x.shape}")
    print(f"Input lengths: {lengths.tolist()}")

    # Warmup
    for _ in range(3):
        _ = model(x, lengths)

    # Time the forward pass
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()

    log_probs, output_lengths = model(x, lengths)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    forward_time = time.time() - start_time

    print(f"\nOutput shape: {log_probs.shape} (T, B, V format for CTC)")
    print(f"Output lengths: {output_lengths.tolist()}")
    print(f"Forward pass time: {forward_time*1000:.2f} ms")
    print(f"FPS capability: {batch_size/forward_time:.1f} sequences/sec")

    # Test prediction
    predictions, scores = model.predict(x, lengths)
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Score shape: {scores.shape}")

    # Verify CTC compatibility
    assert log_probs.dim() == 3, "Output should be 3D"
    assert log_probs.size(0) <= max_seq_len, "Time dimension should be first"
    assert log_probs.size(1) == batch_size, "Batch dimension should be second"
    assert log_probs.size(2) == 1229, "Vocabulary dimension should match"

    print("\n[PASSED] All tests passed!")
    print("Model is ready for training with CTC loss.")