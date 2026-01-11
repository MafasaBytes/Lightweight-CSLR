"""
Neural Language Model Fusion for Sign Language Recognition

Uses an LSTM-based neural LM instead of n-gram for better 
long-range dependency modeling and smoother probability estimates.

Steps:
1. Train a neural LM on training transcriptions
2. Use shallow fusion during beam search
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import Counter
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MediaPipeFeatureDataset, Vocabulary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============== Neural Language Model ==============

class LMDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(self, sentences: List[List[int]], seq_length: int = 32):
        self.seq_length = seq_length
        
        # Flatten all sentences with BOS/EOS tokens
        # BOS = 2 (after blank=0, eos=1)
        self.data = []
        for sent in sentences:
            # Add BOS at start, EOS at end
            self.data.extend([2] + sent + [1])
        
        self.data = torch.tensor(self.data, dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_length - 1)
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


class NeuralLM(nn.Module):
    """LSTM-based Neural Language Model."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights
        if embed_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        Args:
            x: [B, T] input token ids
            hidden: Optional (h, c) tuple
        Returns:
            logits: [B, T, vocab_size]
            hidden: (h, c) tuple
        """
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    @torch.no_grad()
    def get_next_token_probs(self, token_id: int, hidden: Optional[Tuple] = None, device=None):
        """Get probability distribution for next token given current token and hidden state."""
        if device is None:
            device = next(self.parameters()).device
        
        x = torch.tensor([[token_id]], device=device)
        logits, new_hidden = self.forward(x, hidden)
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        return log_probs, new_hidden


def train_neural_lm(lm: NeuralLM, train_loader: DataLoader, device: torch.device,
                    epochs: int = 20, lr: float = 1e-3):
    """Train the neural language model."""
    lm.train()
    optimizer = AdamW(lm.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"LM Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, _ = lm(x)
            
            loss = criterion(logits.view(-1, lm.vocab_size), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            optimizer.step()
            
            mask = y != 0
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(min(avg_loss, 20))
        logger.info(f"LM Epoch {epoch}: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")
    
    return lm


# ============== Seq2Seq Model Components ==============

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4, 
                 num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=500, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        if mask is not None:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.output_norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None):
        query = query.unsqueeze(1)
        scores = self.V(torch.tanh(self.W_q(query) + self.W_k(keys)))
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def init_hidden(self, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = encoder_outputs.size(0)
        encoder_mean = encoder_outputs.mean(dim=1)
        h_0 = encoder_mean.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0
        
    def forward_step(self, input_token: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, encoder_mask: Optional[torch.Tensor] = None):
        embedded = self.embedding(input_token)
        h_n = hidden[0][-1]
        context, attn_weights = self.attention(h_n, encoder_outputs, encoder_mask)
        lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        lstm_output, hidden = self.lstm(lstm_input, hidden)
        lstm_output = lstm_output.squeeze(1)
        output = self.output_proj(torch.cat([lstm_output, context], dim=-1))
        return output, hidden, attn_weights


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int = 6516, hidden_dim: int = 512,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 2,
                 num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_encoder_layers, num_heads, ff_dim, dropout)
        self.decoder = Decoder(vocab_size, hidden_dim, num_decoder_layers, dropout)
        
    def encode(self, features: torch.Tensor, input_lengths: torch.Tensor):
        features = torch.nan_to_num(features, nan=0.0)
        features = torch.clamp(features, -100, 100)
        B, T, _ = features.shape
        encoder_mask = torch.arange(T, device=features.device).expand(B, T) < input_lengths.unsqueeze(1)
        encoder_outputs = self.encoder(features, encoder_mask)
        return encoder_outputs, encoder_mask
    
    def beam_search_with_neural_lm(self, encoder_outputs: torch.Tensor, encoder_mask: torch.Tensor,
                                    neural_lm: NeuralLM, lm_weight: float = 0.3,
                                    beam_size: int = 10, max_len: int = 50, 
                                    length_penalty: float = 0.6) -> List[List[int]]:
        """
        Beam search with neural LM shallow fusion.
        """
        B = encoder_outputs.size(0)
        device = encoder_outputs.device
        vocab_size = self.decoder.vocab_size
        
        all_results = []
        
        for b in range(B):
            enc_out = encoder_outputs[b:b+1]
            enc_mask = encoder_mask[b:b+1]
            
            hidden = self.decoder.init_hidden(enc_out)
            
            # Initialize LM hidden state
            lm_hidden = neural_lm.init_hidden(1, device)
            
            # Beam: (score, tokens, decoder_hidden, lm_hidden)
            # Start with BOS token (2) for LM
            beams = [(0.0, [0], hidden, lm_hidden)]
            completed = []
            
            for step in range(max_len):
                all_candidates = []
                
                for score, tokens, dec_hidden, lm_hid in beams:
                    last_token = torch.tensor([tokens[-1]], device=device)
                    
                    # Decoder step
                    output, new_dec_hidden, _ = self.decoder.forward_step(
                        last_token, dec_hidden, enc_out, enc_mask
                    )
                    dec_log_probs = F.log_softmax(output[0], dim=-1)
                    
                    # LM step - use previous token (or BOS=2 if first step)
                    lm_input_token = tokens[-1] if tokens[-1] > 0 else 2
                    lm_log_probs, new_lm_hidden = neural_lm.get_next_token_probs(
                        lm_input_token, lm_hid, device
                    )
                    
                    # Get top candidates
                    topk_probs, topk_ids = dec_log_probs.topk(beam_size * 2)
                    
                    for i in range(min(beam_size * 2, len(topk_ids))):
                        token_id = topk_ids[i].item()
                        model_score = topk_probs[i].item()
                        
                        # Get LM score for this token
                        lm_score = lm_log_probs[token_id].item()
                        
                        # Combined score
                        combined_score = score + model_score + lm_weight * lm_score
                        
                        new_tokens = tokens + [token_id]
                        
                        if token_id == 1:  # EOS
                            normalized_score = combined_score / (len(new_tokens) ** length_penalty)
                            completed.append((normalized_score, new_tokens))
                        else:
                            all_candidates.append((combined_score, new_tokens, new_dec_hidden, new_lm_hidden))
                
                if not all_candidates:
                    break
                
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_size]
                
                if len(completed) >= beam_size:
                    break
            
            for score, tokens, _, _ in beams:
                normalized_score = score / (len(tokens) ** length_penalty)
                completed.append((normalized_score, tokens))
            
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_tokens = completed[0][1]
            else:
                best_tokens = [0]
            
            all_results.append(best_tokens)
        
        return all_results


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x['features']), reverse=True)
    features = [item['features'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_feat_len = max(f.shape[0] for f in features)
    feat_dim = features[0].shape[-1]
    padded_features = torch.zeros(len(features), max_feat_len, feat_dim)
    input_lengths = torch.zeros(len(features), dtype=torch.long)
    
    for i, f in enumerate(features):
        padded_features[i, :len(f)] = f
        input_lengths[i] = len(f)
    
    max_tgt_len = max(len(l) for l in labels) + 1
    padded_targets = torch.zeros(len(labels), max_tgt_len, dtype=torch.long)
    target_lengths = torch.zeros(len(labels), dtype=torch.long)
    
    for i, l in enumerate(labels):
        padded_targets[i, :len(l)] = l
        padded_targets[i, len(l)] = 1
        target_lengths[i] = len(l) + 1
    
    return {
        'features': padded_features,
        'input_lengths': input_lengths,
        'targets': padded_targets,
        'target_lengths': target_lengths
    }


def decode_tokens(token_ids: List[int], idx2word: Dict[int, str]) -> str:
    words = []
    for token in token_ids:
        if token == 1:
            break
        if token > 1:
            word = idx2word.get(token, '<unk>')
            if word not in ['<blank>', '<eos>']:
                words.append(word)
    return ' '.join(words)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    total_errors = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split() if pred else []
        ref_tokens = ref.split() if ref else []
        
        if not ref_tokens:
            continue
        
        m, n = len(ref_tokens), len(pred_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == pred_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_errors += dp[m][n]
        total_words += len(ref_tokens)
    
    return (total_errors / total_words * 100) if total_words > 0 else 100.0


def load_training_sentences(annotation_dir: Path, vocab: Vocabulary) -> List[List[int]]:
    """Load and tokenize training sentences."""
    EXCLUDE = ['__', 'loc-', 'cl-', 'lh-', 'rh-', 'IX', 'WG', '$GEST', 'PLUSPLUS', 'POS']
    
    sentences = []
    anno_file = annotation_dir / 'train.corpus.csv'
    
    if anno_file.exists():
        df = pd.read_csv(anno_file, sep='|', on_bad_lines='skip')
        if 'annotation' not in df.columns:
            df.columns = [col.strip() for col in df.columns]
        
        for annotation in df['annotation']:
            if isinstance(annotation, str):
                tokens = []
                for word in annotation.strip().split():
                    if not any(p in word for p in EXCLUDE) and not (len(word) == 1 and word.isalpha()):
                        if word in vocab.word2idx:
                            tokens.append(vocab.word2idx[word])
                if tokens:
                    sentences.append(tokens)
    
    return sentences


@torch.no_grad()
def evaluate_with_neural_lm(model, dataloader, device, idx2word, neural_lm: NeuralLM,
                            lm_weight: float, beam_size: int = 10):
    """Evaluate with neural LM fusion."""
    model.eval()
    neural_lm.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating (λ={lm_weight})", leave=False):
        features = batch['features'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        targets = batch['targets']
        target_lengths = batch['target_lengths']
        
        encoder_outputs, encoder_mask = model.encode(features, input_lengths)
        
        predictions = model.beam_search_with_neural_lm(
            encoder_outputs, encoder_mask, neural_lm,
            lm_weight=lm_weight, beam_size=beam_size
        )
        
        B = features.size(0)
        for i in range(B):
            pred_str = decode_tokens(predictions[i], idx2word)
            all_preds.append(pred_str)
            
            tgt_tokens = []
            for t in range(target_lengths[i].item() - 1):
                token = targets[i, t].item()
                if token > 1:
                    word = idx2word.get(token, '<unk>')
                    if word not in ['<blank>', '<eos>']:
                        tgt_tokens.append(word)
            all_targets.append(' '.join(tgt_tokens))
    
    wer = compute_wer(all_preds, all_targets)
    
    return {
        'wer': wer,
        'predictions': all_preds[:5],
        'targets': all_targets[:5]
    }


def main():
    parser = argparse.ArgumentParser(description='Neural LM Fusion Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/teacher_features/mediapipe_full')
    parser.add_argument('--lm_weights', type=str, default='0.0,0.1,0.2,0.3,0.4', help='LM weights to try')
    parser.add_argument('--lm_epochs', type=int, default=30, help='LM training epochs')
    parser.add_argument('--lm_hidden', type=int, default=512, help='LM hidden dimension')
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Load vocabulary
    vocab_path = checkpoint_path.parent / 'vocab.json'
    vocab = Vocabulary()
    vocab.load(vocab_path)
    idx2word = vocab.idx2word
    vocab_size = len(vocab)
    
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Get annotation dir
    annotation_dir = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual")
    if not annotation_dir.exists():
        annotation_dir = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")
    
    # Train or load Neural LM
    lm_path = checkpoint_path.parent / 'neural_lm.pt'
    
    neural_lm = NeuralLM(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=args.lm_hidden,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    if lm_path.exists():
        logger.info(f"Loading neural LM from {lm_path}")
        neural_lm.load_state_dict(torch.load(lm_path, map_location=device))
    else:
        logger.info("Training neural language model...")
        sentences = load_training_sentences(annotation_dir, vocab)
        logger.info(f"Loaded {len(sentences)} training sentences for LM")
        
        lm_dataset = LMDataset(sentences, seq_length=32)
        lm_loader = DataLoader(lm_dataset, batch_size=64, shuffle=True, num_workers=0)
        
        neural_lm = train_neural_lm(neural_lm, lm_loader, device, epochs=args.lm_epochs)
        torch.save(neural_lm.state_dict(), lm_path)
        logger.info(f"Saved neural LM to {lm_path}")
    
    neural_lm.eval()
    
    # Build seq2seq model
    model_args = checkpoint.get('args', {})
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        input_dim=model_args.get('input_dim', 6516),
        hidden_dim=model_args.get('hidden_dim', 512),
        num_encoder_layers=model_args.get('num_encoder_layers', 4),
        num_decoder_layers=model_args.get('num_decoder_layers', 2),
        num_heads=model_args.get('num_heads', 8),
        ff_dim=model_args.get('ff_dim', 1024),
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Seq2Seq model loaded")
    
    # Load validation data
    data_dir = Path(args.data_dir)
    val_dataset = MediaPipeFeatureDataset(
        data_dir=data_dir,
        annotation_file=annotation_dir / 'dev.corpus.csv',
        vocabulary=vocab,
        split='dev',
        max_seq_length=300,
        augment=False,
        normalize=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate with different LM weights
    lm_weights = [float(w) for w in args.lm_weights.split(',')]
    
    logger.info("\n" + "=" * 60)
    logger.info("NEURAL LM FUSION EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    for lm_weight in lm_weights:
        logger.info(f"\n--- LM Weight: {lm_weight} ---")
        
        metrics = evaluate_with_neural_lm(
            model, val_loader, device, idx2word, neural_lm,
            lm_weight=lm_weight, beam_size=args.beam_size
        )
        
        results[lm_weight] = metrics['wer']
        logger.info(f"WER: {metrics['wer']:.2f}%")
        
        if lm_weight in [0.0, 0.2, 0.3]:
            logger.info("Sample predictions:")
            for i in range(min(3, len(metrics['predictions']))):
                logger.info(f"  Target: '{metrics['targets'][i]}'")
                logger.info(f"  Pred:   '{metrics['predictions'][i] if metrics['predictions'][i] else '(empty)'}'")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for lm_weight, wer in sorted(results.items()):
        marker = " <<<" if wer == min(results.values()) else ""
        logger.info(f"λ={lm_weight:.1f}: {wer:.2f}% WER{marker}")
    
    best_weight = min(results, key=results.get)
    logger.info(f"\nBest LM weight: {best_weight} with WER: {results[best_weight]:.2f}%")
    
    improvement = results[0.0] - results[best_weight]
    logger.info(f"Improvement over no LM: {improvement:.2f}% absolute")


if __name__ == '__main__':
    main()

