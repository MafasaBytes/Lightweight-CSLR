"""
Protocol v1 (thesis): filtered-token CSLR evaluation.

Motivation
----------
Phoenix annotations and AdaptSign vocab include special / meta tokens like:
  - __ON__ / __LEFTHAND__ (markers)
  - loc-* (location markers)
  - IX / WG / etc. (deixis / meta markers)

Our current training dataset (`SequenceFeatureDataset`) *filters these tokens out*
of the target sequences. Therefore, allowing the model to emit them during
decoding creates a systematic mismatch: references will never contain them, so
they inflate WER and can create misleading loss/metric behavior.

Protocol v1 defines:
  - How we filter reference/hypothesis strings for WER reporting
  - How we identify excluded tokens from token strings

This module is intentionally small and deterministic so it can be cited in the thesis.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set

from src.utils.vocabulary import filter_annotation, should_exclude_token


def filter_tokens(tokens: Iterable[str]) -> List[str]:
    """Filter excluded tokens from a token list using the project-wide rules."""
    return [t for t in tokens if t and (not should_exclude_token(t))]


def filter_sentence(sentence: str) -> str:
    """
    Filter excluded tokens from a space-delimited string.

    Note: we reuse `filter_annotation` to stay consistent with training data.
    """
    return filter_annotation(sentence or "")


def filter_pairwise(
    references: List[str],
    hypotheses: List[str],
) -> tuple[list[str], list[str]]:
    """Apply protocol filtering to parallel ref/hyp lists."""
    return [filter_sentence(r) for r in references], [filter_sentence(h) for h in hypotheses]


def excluded_token_ids(idx2word: Dict[int, str], keep_blank_id: int = 0) -> Set[int]:
    """
    Compute the set of token IDs that should be excluded under Protocol v1.

    Args:
        idx2word: mapping from id -> token string
        keep_blank_id: blank id to always keep (default: 0)
    """
    out: Set[int] = set()
    for i, w in idx2word.items():
        if int(i) == int(keep_blank_id):
            continue
        if w and should_exclude_token(w):
            out.add(int(i))
    return out


def make_string_filter() -> Callable[[str], str]:
    """Convenience hook for functions that accept a callable token filter."""
    return filter_sentence


