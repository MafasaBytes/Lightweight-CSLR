"""Vocabulary handling utilities for sign language recognition.

This module provides the Vocabulary class that loads and uses vocabulary files
created by create_vocabulary.py.

RELATIONSHIP WITH create_vocabulary.py:
- create_vocabulary.py: Preprocessing script that generates vocabulary.txt
- vocabulary.py: Runtime class that loads vocabulary.txt for training/inference

WORKFLOW:
1. Run create_vocabulary.py to generate data/clean_vocabulary/vocabulary.txt
2. Load vocabulary in training/inference:
   from utils.vocabulary import load_vocabulary_from_file
   vocab = load_vocabulary_from_file(Path("data/clean_vocabulary/vocabulary.txt"))
3. Use vocab.encode() to convert text to indices, vocab.decode() for reverse
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Patterns to EXCLUDE from annotations (same as in create_vocabulary.py)
EXCLUDE_PATTERNS = [
    '__',       # Special markers like __ON__, __OFF__
    'loc-',     # Location markers
    'cl-',      # Classifier markers
    'lh-',      # Left hand markers
    'rh-',      # Right hand markers
    'IX',       # Pointing/deixis
    'IX-',      # More pointing
    'WG',       # Unknown marker
    '$GEST',    # Gesture marker
    'PLUSPLUS', # Modifier
    'POS',      # Position marker
    'NICHTALP', # Negation marker
    'qu-',      # Question marker
    'poss-',    # Possessive marker
]


def should_exclude_token(word: str) -> bool:
    """Check if a word should be excluded (same logic as create_vocabulary.py)."""
    # Check each exclusion pattern
    for pattern in EXCLUDE_PATTERNS:
        if pattern in word:
            return True
    
    # Also exclude single letters (often fingerspelling markers)
    if len(word) == 1 and word.isalpha():
        return True
    
    return False


def filter_annotation(annotation: str) -> str:
    """
    Filter out excluded tokens from an annotation string.
    
    This removes tokens like __ON__, loc-NORD, IX, etc. that were excluded
    from the vocabulary. Use this to clean annotations before encoding.
    
    Args:
        annotation: Raw annotation string
        
    Returns:
        Cleaned annotation string with excluded tokens removed
    """
    words = annotation.split()
    filtered_words = [word for word in words if not should_exclude_token(word)]
    return ' '.join(filtered_words)


class Vocabulary:
    """Vocabulary manager for sign language tokens."""
    
    def __init__(self, vocab_file: Optional[Path] = None, annotations: Optional[List[str]] = None):
        """
        Initialize vocabulary from file or annotations.
        
        Args:
            vocab_file: Path to vocabulary file (format: WORD COUNT)
            annotations: List of annotation strings to build vocabulary from
        """
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        
        if vocab_file and vocab_file.exists():
            self._load_from_file(vocab_file)
        elif annotations:
            self._build_from_annotations(annotations)
        else:
            # Default vocabulary with special tokens
            self._init_default()
    
    def _init_default(self):
        """Initialize with default special tokens."""
        special_tokens = ['<BLANK>', '<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for idx, token in enumerate(special_tokens):
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            self.word_counts[token] = 0
    
    def _load_from_file(self, vocab_file: Path):
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 1:
                    word = parts[0]
                    count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                    
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    self.word_counts[word] = count
    
    def _build_from_annotations(self, annotations: List[str]):
        """Build vocabulary from annotation strings."""
        from collections import Counter
        
        # Special tokens first
        special_tokens = ['<BLANK>', '<PAD>', '<UNK>', '<SOS>', '<EOS>']
        for idx, token in enumerate(special_tokens):
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            self.word_counts[token] = 0
        
        # Count all tokens
        all_tokens = []
        for ann in annotations:
            tokens = ann.split()
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        # Add tokens to vocabulary
        for token, count in token_counts.most_common():
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
                self.word_counts[token] = count
    
    def encode(self, text: str, filter_excluded: bool = True) -> List[int]:
        """Encode text to indices.
        
        Args:
            text: Annotation string to encode
            filter_excluded: If True, filter out excluded tokens (loc-, cl-, IX, etc.)
                           before encoding. If False, they will map to <UNK>.
        
        Returns:
            List of token indices
        """
        # Optionally filter excluded tokens before encoding
        if filter_excluded:
            text = filter_annotation(text)
        
        tokens = text.split()
        indices = []
        unk_idx = self.word_to_idx.get('<UNK>', None)
        if unk_idx is None:
            raise ValueError("<UNK> token not found in vocabulary! Vocabulary must include <UNK> at index 2.")
        
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                # Map unknown tokens to <UNK> (shouldn't happen if filter_excluded=True)
                indices.append(unk_idx)
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                if token not in ['<BLANK>', '<PAD>', '<SOS>', '<EOS>']:
                    tokens.append(token)
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_idx)
    
    @property
    def blank_idx(self) -> int:
        """Get blank token index for CTC."""
        return self.word_to_idx.get('<BLANK>', 0)
    
    @property
    def pad_idx(self) -> int:
        """Get padding token index."""
        return self.word_to_idx.get('<PAD>', 1)
    
    @property
    def unk_idx(self) -> int:
        """Get unknown token index."""
        return self.word_to_idx.get('<UNK>', 2)


def load_vocabulary_from_annotations(annotations_file: Path) -> Vocabulary:
    """Load vocabulary from annotation CSV file."""
    df = pd.read_csv(annotations_file, sep='|')
    annotations = df['annotation'].tolist()
    return Vocabulary(annotations=annotations)


def load_vocabulary_from_file(vocab_file: Path) -> Vocabulary:
    """Load vocabulary from vocabulary file."""
    return Vocabulary(vocab_file=vocab_file)

