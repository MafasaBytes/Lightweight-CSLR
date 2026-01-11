"""
Train an n-gram Language Model on Phoenix-2014 gloss sequences.

Usage:
    python -m src.lm.train_ngram_lm --n 3 --output models/lm/phoenix_3gram.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import math

import numpy as np
import pandas as pd
from src.protocols.protocol_v1 import filter_sentence


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    return logging.getLogger("train_lm")


# Special tokens
BOS = "<s>"  # Beginning of sentence
EOS = "</s>"  # End of sentence
UNK = "<unk>"  # Unknown token


class NGramLM:
    """
    Simple n-gram Language Model with Kneser-Ney smoothing.
    
    For CTC decoding, we use this to score candidate sequences.
    """
    
    def __init__(self, n: int = 3, discount: float = 0.75):
        """
        Args:
            n: Order of the n-gram model (3 = trigram)
            discount: Discount factor for Kneser-Ney smoothing
        """
        self.n = n
        self.discount = discount
        self.vocab: set = set()
        
        # Counts: ngram_counts[n][context][word] = count
        # context is a tuple of (n-1) words
        self.ngram_counts: Dict[int, Dict[Tuple, Dict[str, int]]] = {
            i: defaultdict(lambda: defaultdict(int)) for i in range(1, n + 1)
        }
        
        # Context counts for normalization
        self.context_counts: Dict[int, Dict[Tuple, int]] = {
            i: defaultdict(int) for i in range(1, n + 1)
        }
        
        # For Kneser-Ney: continuation counts
        self.continuation_counts: Dict[str, int] = defaultdict(int)
        self.total_continuations = 0
        
        self.trained = False
    
    def train(self, sentences: List[List[str]]) -> None:
        """
        Train the n-gram model on a list of tokenized sentences.
        
        Args:
            sentences: List of sentences, each sentence is a list of tokens
        """
        logger = logging.getLogger("train_lm")
        logger.info(f"Training {self.n}-gram LM on {len(sentences)} sentences...")
        
        # Build vocabulary
        for sent in sentences:
            self.vocab.update(sent)
        self.vocab.add(BOS)
        self.vocab.add(EOS)
        self.vocab.add(UNK)
        
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        
        # Count n-grams
        for sent in sentences:
            # Add BOS and EOS tokens
            padded = [BOS] * (self.n - 1) + sent + [EOS]
            
            for i in range(len(padded) - self.n + 1):
                for order in range(1, self.n + 1):
                    # Extract n-gram of this order
                    start = i + (self.n - order)
                    context = tuple(padded[start:start + order - 1]) if order > 1 else ()
                    word = padded[start + order - 1]
                    
                    self.ngram_counts[order][context][word] += 1
                    self.context_counts[order][context] += 1
            
            # Track continuation counts for Kneser-Ney
            for i in range(len(padded) - 1):
                bigram = (padded[i], padded[i + 1])
                if self.continuation_counts[padded[i + 1]] == 0 or bigram not in self._seen_bigrams:
                    self.continuation_counts[padded[i + 1]] += 1
                    self.total_continuations += 1
                    if not hasattr(self, '_seen_bigrams'):
                        self._seen_bigrams = set()
                    self._seen_bigrams.add(bigram)
        
        self.trained = True
        
        # Log statistics
        for order in range(1, self.n + 1):
            num_ngrams = sum(len(words) for words in self.ngram_counts[order].values())
            logger.info(f"  {order}-grams: {num_ngrams}")
    
    def log_prob(self, word: str, context: Tuple[str, ...] = ()) -> float:
        """
        Compute log probability of word given context using interpolated Kneser-Ney.
        
        Args:
            word: The word to compute probability for
            context: Tuple of preceding words (up to n-1)
        
        Returns:
            Log probability (base e)
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Handle unknown words
        if word not in self.vocab:
            word = UNK
        
        # Truncate context to n-1 words
        context = context[-(self.n - 1):] if len(context) >= self.n - 1 else context
        
        # Interpolated Kneser-Ney smoothing
        return self._kneser_ney_log_prob(word, context)
    
    def _kneser_ney_log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute log prob using interpolated Kneser-Ney smoothing."""
        order = len(context) + 1
        
        if order == 1:
            # Unigram: use continuation probability
            if self.total_continuations > 0:
                prob = self.continuation_counts.get(word, 1) / self.total_continuations
            else:
                prob = 1.0 / len(self.vocab)
            return math.log(max(prob, 1e-10))
        
        # Higher order
        count = self.ngram_counts[order][context].get(word, 0)
        context_count = self.context_counts[order][context]
        
        if context_count == 0:
            # Backoff to lower order
            return self._kneser_ney_log_prob(word, context[1:])
        
        # Number of unique words following this context
        num_unique = len(self.ngram_counts[order][context])
        
        # Discounted probability
        discounted_prob = max(count - self.discount, 0) / context_count
        
        # Interpolation weight
        lambda_weight = (self.discount * num_unique) / context_count
        
        # Backoff probability
        backoff_log_prob = self._kneser_ney_log_prob(word, context[1:])
        backoff_prob = math.exp(backoff_log_prob)
        
        # Interpolated probability
        prob = discounted_prob + lambda_weight * backoff_prob
        
        return math.log(max(prob, 1e-10))
    
    def score_sequence(self, tokens: List[str]) -> float:
        """
        Compute log probability of a full sequence.
        
        Args:
            tokens: List of tokens in the sequence
        
        Returns:
            Total log probability (sum of log probs)
        """
        # Add BOS padding
        padded = [BOS] * (self.n - 1) + tokens + [EOS]
        
        total_log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            total_log_prob += self.log_prob(word, context)
        
        return total_log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """
        Compute perplexity on a set of sentences.
        
        Args:
            sentences: List of tokenized sentences
        
        Returns:
            Perplexity score (lower is better)
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        for sent in sentences:
            total_log_prob += self.score_sequence(sent)
            total_tokens += len(sent) + 1  # +1 for EOS
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def save(self, path: Path) -> None:
        """Save model to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert nested defaultdicts to regular dicts for pickling
        ngram_counts_dict = {}
        for order, contexts in self.ngram_counts.items():
            ngram_counts_dict[order] = {
                ctx: dict(words) for ctx, words in contexts.items()
            }
        
        context_counts_dict = {}
        for order, contexts in self.context_counts.items():
            context_counts_dict[order] = dict(contexts)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'discount': self.discount,
                'vocab': self.vocab,
                'ngram_counts': ngram_counts_dict,
                'context_counts': context_counts_dict,
                'continuation_counts': dict(self.continuation_counts),
                'total_continuations': self.total_continuations,
                'trained': self.trained,
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> "NGramLM":
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'], discount=data['discount'])
        model.vocab = data['vocab']
        model.ngram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for order, contexts in data['ngram_counts'].items():
            for context, words in contexts.items():
                model.ngram_counts[order][context] = defaultdict(int, words)
        model.context_counts = defaultdict(lambda: defaultdict(int))
        for order, contexts in data['context_counts'].items():
            model.context_counts[order] = defaultdict(int, contexts)
        model.continuation_counts = defaultdict(int, data['continuation_counts'])
        model.total_continuations = data['total_continuations']
        model.trained = data['trained']
        
        return model


def load_phoenix_annotations(annotation_dir: Path, split: str = "train") -> List[List[str]]:
    """Load and tokenize Phoenix-2014 annotations."""
    # Try SI5 naming first
    csv_path = annotation_dir / f"{split}.SI5.corpus.csv"
    if not csv_path.exists():
        csv_path = annotation_dir / f"{split}.corpus.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, sep="|", on_bad_lines="skip")
    if "annotation" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
    
    sentences = []
    for _, row in df.iterrows():
        ann = row.get("annotation", "")
        if isinstance(ann, str) and ann.strip():
            # Tokenize by whitespace, filter empty
            tokens = [t.strip() for t in ann.split() if t.strip()]
            if tokens:
                sentences.append(tokens)
    
    return sentences


def load_adaptsign_official_sentences(
    adaptsign_preprocess_dir: Path,
    dataset: str,
    split: str,
) -> List[List[str]]:
    """
    Load official AdaptSign split list from `adaptsign_repo/preprocess/<dataset>/<split>_info.npy`
    and return tokenized + Protocol-v1-filtered sentences.
    """
    info_path = Path(adaptsign_preprocess_dir) / dataset / f"{split}_info.npy"
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find AdaptSign split list: {info_path}")
    info = np.load(str(info_path), allow_pickle=True).item()
    keys = sorted([k for k in info.keys() if isinstance(k, int)])
    sentences: List[List[str]] = []
    for k in keys:
        v = info[k]
        label = v.get("label", "")
        if not isinstance(label, str) or not label.strip():
            continue
        # Match Protocol v1 (same filtering used for student WER reporting).
        filtered = filter_sentence(label.strip())
        toks = [t.strip() for t in filtered.split() if t.strip()]
        if toks:
            sentences.append(toks)
    return sentences


def load_vocab_from_npy(npy_path: Path) -> Set[str]:
    """Load vocabulary from AdaptSign's gloss_dict.npy format.
    
    Format: {word: [index, count]}
    """
    gloss_dict = np.load(npy_path, allow_pickle=True).item()
    # gloss_dict: {word: [index, count]} - indices start from 1
    vocab = set(gloss_dict.keys())
    return vocab


def load_vocab_from_json(json_path: Path) -> Set[str]:
    """Load vocabulary from vocabulary.json format."""
    with open(json_path) as f:
        vocab_data = json.load(f)
    return set(vocab_data['word2idx'].keys())


def main():
    parser = argparse.ArgumentParser("Train n-gram Language Model")
    parser.add_argument("--n", type=int, default=3, help="N-gram order (default: 3)")
    parser.add_argument("--discount", type=float, default=0.75, help="Kneser-Ney discount")
    parser.add_argument("--annotation_dir", type=str,
                        default="data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")
    parser.add_argument("--split_source", type=str, default="si5", choices=["si5", "adaptsign_official"])
    parser.add_argument("--adaptsign_preprocess_dir", type=str, default="adaptsign_repo/preprocess")
    parser.add_argument("--adaptsign_dataset", type=str, default="phoenix2014")
    parser.add_argument("--output", type=str, default="models/lm/phoenix_3gram.pkl")
    parser.add_argument("--vocab_filter", type=str, default=None,
                        help="Optional: path to vocabulary.json to filter tokens")
    parser.add_argument("--vocab_npy", type=str, default=None,
                        help="Optional: path to gloss_dict.npy (AdaptSign format) to filter tokens")
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"Training {args.n}-gram Language Model")
    
    # Load training annotations
    annotation_dir = Path(args.annotation_dir)
    if args.split_source == "adaptsign_official":
        train_sentences = load_adaptsign_official_sentences(
            adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
            dataset=str(args.adaptsign_dataset),
            split="train",
        )
    else:
        train_sentences = load_phoenix_annotations(annotation_dir, "train")
    logger.info(f"Loaded {len(train_sentences)} training sentences")
    
    # Optional: filter to vocabulary (support both JSON and NPY formats)
    allowed_vocab = None
    if args.vocab_npy:
        allowed_vocab = load_vocab_from_npy(Path(args.vocab_npy))
        logger.info(f"Loaded vocab from NPY: {len(allowed_vocab)} words")
    elif args.vocab_filter:
        allowed_vocab = load_vocab_from_json(Path(args.vocab_filter))
        logger.info(f"Loaded vocab from JSON: {len(allowed_vocab)} words")
    
    if allowed_vocab:
        logger.info(f"Filtering to {len(allowed_vocab)} vocabulary words")
        filtered_sentences = []
        for sent in train_sentences:
            filtered = [t for t in sent if t in allowed_vocab]
            if filtered:
                filtered_sentences.append(filtered)
        train_sentences = filtered_sentences
        logger.info(f"After filtering: {len(train_sentences)} sentences")
    
    # Train model
    lm = NGramLM(n=args.n, discount=args.discount)
    lm.train(train_sentences)
    
    # Evaluate on dev set
    if args.split_source == "adaptsign_official":
        dev_sentences = load_adaptsign_official_sentences(
            adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
            dataset=str(args.adaptsign_dataset),
            split="dev",
        )
    else:
        dev_sentences = load_phoenix_annotations(annotation_dir, "dev")
    if allowed_vocab:
        dev_sentences = [
            [t for t in sent if t in allowed_vocab]
            for sent in dev_sentences
        ]
        dev_sentences = [s for s in dev_sentences if s]
    
    perplexity = lm.perplexity(dev_sentences)
    logger.info(f"Dev perplexity: {perplexity:.2f}")
    
    # Save model
    output_path = Path(args.output)
    lm.save(output_path)
    logger.info(f"Model saved to: {output_path}")
    
    # Test scoring
    logger.info("\nTest scoring:")
    test_sentences = dev_sentences[:3]
    for sent in test_sentences:
        score = lm.score_sequence(sent)
        logger.info(f"  '{' '.join(sent[:5])}...' -> log_prob={score:.2f}")


if __name__ == "__main__":
    main()

