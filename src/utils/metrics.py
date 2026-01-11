"""Evaluation metrics for sign language recognition."""

import numpy as np
from typing import List, Tuple
from collections import Counter


def edit_distance(s1: List[str], s2: List[str]) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
    
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def word_error_rate(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    Compute Word Error Rate (WER).

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total words in reference

    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences

    Returns:
        Word Error Rate (0.0 to 1.0, where 1.0 = 100% error)
    """
    if not references:
        return 0.0

    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        # Skip empty references (no ground truth to compare against)
        if len(ref) == 0:
            continue

        # If hypothesis is empty but reference has words, all words are deletions
        # If hypothesis has words, compute the edit distance
        errors = edit_distance(ref, hyp)
        total_errors += errors
        total_words += len(ref)

    # If all references were empty (no ground truth), return 0
    if total_words == 0:
        return 0.0

    # WER can be > 1.0 if there are many insertions
    # But typically we cap it at 1.0 for reporting
    wer = total_errors / total_words
    return min(wer, 1.0)


def sign_error_rate(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """
    Compute Sign Error Rate (SER) - same as WER but for isolated signs.
    
    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
    
    Returns:
        Sign Error Rate
    """
    return word_error_rate(references, hypotheses)


def compute_wer(references, hypotheses):
    """
    Compute WER from string or list sequences.

    Args:
        references: List of reference strings or list of lists
        hypotheses: List of hypothesis strings or list of lists

    Returns:
        WER percentage
    """
    # Convert to token lists if needed
    if references and isinstance(references[0], str):
        ref_tokens = [ref.split() for ref in references]
    else:
        ref_tokens = references

    if hypotheses and isinstance(hypotheses[0], str):
        hyp_tokens = [hyp.split() for hyp in hypotheses]
    else:
        hyp_tokens = hypotheses
    
    wer = word_error_rate(ref_tokens, hyp_tokens)

    # Return WER as percentage
    return wer * 100.0


def compute_ser(references, hypotheses):
    """
    Compute SER from string or list sequences.

    Args:
        references: List of reference strings or list of lists
        hypotheses: List of hypothesis strings or list of lists

    Returns:
        SER percentage
    """
    return compute_wer(references, hypotheses)  # SER is same as WER for sign language


def bleu_score(references: List[List[str]], hypotheses: List[List[str]], n: int = 4) -> float:
    """
    Compute BLEU score (simplified version).
    
    Args:
        references: List of reference sequences
        hypotheses: List of hypothesis sequences
        n: Maximum n-gram order
    
    Returns:
        BLEU score
    """
    # Simplified BLEU - just for reference
    # Full BLEU implementation would require more sophisticated n-gram matching
    if len(references) != len(hypotheses):
        return 0.0
    
    matches = 0
    total = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_set = set(ref)
        hyp_set = set(hyp)
        matches += len(ref_set & hyp_set)
        total += len(ref_set)
    
    return matches / total if total > 0 else 0.0

