"""
BiLSTM + CTC module for sign language recognition.
Provides temporal modeling and CTC loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BiLSTMCTC(nn.Module):
    """
    Bidirectional LSTM with CTC output layer for sequence modeling.
    Used as the temporal modeling component in the sign language pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        vocab_size: int = 1232,
        dropout: float = 0.5,
    ):
        """
        Initialize BiLSTM-CTC module.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            vocab_size: Size of output vocabulary
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Input projection (optional, for dimension matching)
        self.input_proj = None
        if input_dim != hidden_dim:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Output projection for vocabulary
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize LSTM and linear weights."""
        # LSTM initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.)

        # Linear layer initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BiLSTM-CTC.

        Args:
            x: Input features [B, T, input_dim]
            lengths: Sequence lengths [B]

        Returns:
            Log probabilities [T, B, vocab_size] for CTC loss
        """
        B, T, _ = x.shape

        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Pack sequences for efficiency (if lengths provided)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True
            )

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Project to vocabulary size
        output = self.output_proj(lstm_out)

        # Apply log_softmax for CTC loss
        output = F.log_softmax(output, dim=-1)

        # Transpose for CTC loss format [T, B, vocab_size]
        output = output.transpose(0, 1)

        return output

    def get_hidden_states(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hidden states from BiLSTM (useful for feature extraction).

        Args:
            x: Input features [B, T, input_dim]
            lengths: Sequence lengths [B]

        Returns:
            lstm_out: BiLSTM outputs [B, T, hidden_dim*2]
            final_hidden: Final hidden states [B, hidden_dim*2]
        """
        B, T, _ = x.shape

        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)

        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )

        # BiLSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Unpack if needed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True
            )

        # Get final hidden states (concatenate forward and backward)
        # h_n shape: [num_layers*2, B, hidden_dim]
        final_hidden = torch.cat([
            h_n[-2, :, :],  # Last layer forward
            h_n[-1, :, :]   # Last layer backward
        ], dim=1)  # [B, hidden_dim*2]

        return lstm_out, final_hidden


class CTCDecoder:
    """
    CTC decoding utilities for converting model outputs to predictions.
    Supports both greedy and beam search decoding.
    """

    @staticmethod
    def greedy_decode(
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        blank_id: int = 0
    ) -> list:
        """
        Greedy decoding for CTC output.

        Args:
            log_probs: Log probabilities [T, B, vocab_size]
            lengths: Sequence lengths [B]
            blank_id: ID of blank token

        Returns:
            List of decoded sequences
        """
        T, B, V = log_probs.shape
        predictions = []

        for b in range(B):
            seq_len = min(lengths[b].item(), T)
            # Get most probable tokens
            tokens = log_probs[:seq_len, b, :].argmax(dim=-1)

            # Remove blanks and repeated tokens
            decoded = []
            prev_token = blank_id

            for token in tokens:
                token = token.item()
                if token != blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token

            predictions.append(decoded)

        return predictions

    @staticmethod
    def beam_search_decode(
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
        beam_width: int = 10,
        blank_id: int = 0,
        lm_weight: float = 0.0,
        word_bonus: float = 0.0
    ) -> list:
        """
        Beam search decoding for CTC output.

        Args:
            log_probs: Log probabilities [T, B, vocab_size]
            lengths: Sequence lengths [B]
            beam_width: Beam search width
            blank_id: ID of blank token
            lm_weight: Language model weight
            word_bonus: Word insertion bonus

        Returns:
            List of decoded sequences
        """
        # Simplified beam search implementation
        # For production, use libraries like ctcdecode or flashlight
        T, B, V = log_probs.shape
        predictions = []

        for b in range(B):
            seq_len = min(lengths[b].item(), T)
            frame_probs = log_probs[:seq_len, b, :]

            # Initialize beam with blank
            beam = [([], 0.0)]  # (prefix, score)

            for t in range(seq_len):
                new_beam = {}

                for prefix, score in beam:
                    # Extend with blank
                    new_beam[tuple(prefix)] = new_beam.get(
                        tuple(prefix), float('-inf')
                    )
                    blank_score = score + frame_probs[t, blank_id].item()
                    new_beam[tuple(prefix)] = max(
                        new_beam[tuple(prefix)], blank_score
                    )

                    # Extend with non-blank tokens
                    for v in range(V):
                        if v != blank_id:
                            new_prefix = prefix + [v]
                            new_beam[tuple(new_prefix)] = new_beam.get(
                                tuple(new_prefix), float('-inf')
                            )
                            token_score = score + frame_probs[t, v].item()
                            new_beam[tuple(new_prefix)] = max(
                                new_beam[tuple(new_prefix)], token_score
                            )

                # Keep top beam_width candidates
                beam = sorted(
                    [(list(k), v) for k, v in new_beam.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:beam_width]

            # Return best hypothesis
            if beam:
                predictions.append(beam[0][0])
            else:
                predictions.append([])

        return predictions


def create_bilstm_ctc(
    input_dim: int,
    vocab_size: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.5
) -> BiLSTMCTC:
    """
    Factory function to create BiLSTM-CTC module.

    Args:
        input_dim: Input feature dimension
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate

    Returns:
        BiLSTMCTC module
    """
    return BiLSTMCTC(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        vocab_size=vocab_size,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the BiLSTM-CTC module
    input_dim = 384  # From MobileNetV3 encoder output
    vocab_size = 1232
    batch_size = 4
    seq_length = 100

    model = create_bilstm_ctc(input_dim, vocab_size)
    print(f"BiLSTM-CTC Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(batch_size, seq_length, input_dim)
    dummy_lengths = torch.tensor([100, 90, 80, 70])

    output = model(dummy_input, dummy_lengths)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test greedy decoding
    decoder = CTCDecoder()
    predictions = decoder.greedy_decode(output, dummy_lengths)
    print(f"Decoded sequences: {len(predictions)} sequences")

    # Test beam search decoding
    beam_predictions = decoder.beam_search_decode(
        output, dummy_lengths, beam_width=5
    )
    print(f"Beam search sequences: {len(beam_predictions)} sequences")

    print("\nBiLSTM-CTC module working correctly!")