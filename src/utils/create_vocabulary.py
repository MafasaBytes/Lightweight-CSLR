"""
Create Vocabulary
Filter out all special tokens, location markers, and other non-gloss tokens.

This script generates a clean vocabulary file that is then loaded by vocabulary.py.

WORKFLOW:
1. This script (create_vocabulary.py):
   - Reads raw annotation files from Phoenix dataset
   - Filters out unwanted tokens (loc-, cl-, IX, etc.)
   - Creates vocabulary.txt file with clean tokens

2. vocabulary.py:
   - Loads the vocabulary.txt file created by this script
   - Provides Vocabulary class for encoding/decoding text during training/inference

USAGE:
    python utils/create_vocabulary.py
    # Creates: data/clean_vocabulary/vocabulary.txt
    
    Then in your training code:
    from utils.vocabulary import load_vocabulary_from_file
    vocab = load_vocabulary_from_file(Path("data/clean_vocabulary/vocabulary.txt"))
"""

from pathlib import Path
import csv
from collections import Counter

print("="*70)
print("CREATING CLEAN VOCABULARY")
print("="*70)

# Patterns to EXCLUDE from vocabulary
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

def should_exclude(word):
    """Check if a word should be excluded from vocabulary."""
    # Check each exclusion pattern
    for pattern in EXCLUDE_PATTERNS:
        if pattern in word:
            return True

    # Also exclude single letters (often fingerspelling markers)
    if len(word) == 1 and word.isalpha():
        return True

    return False

# Load all annotations to build vocabulary
annotations_path = Path("data/raw_data/phoenix-2014-multisigner/annotations/manual")
all_words = Counter()
excluded_words = Counter()

for split in ['train', 'dev', 'test']:
    corpus_file = annotations_path / f"{split}.corpus.csv"

    if not corpus_file.exists():
        print(f"Warning: {corpus_file} not found")
        continue

    print(f"\nProcessing {split} split...")

    with open(corpus_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')

        for row in reader:
            annotation = row.get('annotation', '')
            words = annotation.split()

            for word in words:
                # Skip empty
                if not word:
                    continue

                # Check if should exclude
                if should_exclude(word):
                    excluded_words[word] += 1
                else:
                    # This is a clean gloss
                    all_words[word] += 1

print(f"\n" + "="*70)
print("VOCABULARY STATISTICS:")
print(f"Total clean glosses: {len(all_words)}")
print(f"Total excluded tokens: {len(excluded_words)}")
print(f"Total gloss occurrences: {sum(all_words.values())}")
print(f"Total excluded occurrences: {sum(excluded_words.values())}")

# Show most common clean glosses
print(f"\nMost common CLEAN glosses:")
for word, count in all_words.most_common(20):
    print(f"  {word:20s} : {count:6d}")

# Show most common excluded tokens
print(f"\nMost common EXCLUDED tokens:")
for word, count in excluded_words.most_common(20):
    print(f"  {word:20s} : {count:6d}")

# Create clean vocabulary file
output_path = Path("data/clean_vocabulary")
output_path.mkdir(parents=True, exist_ok=True)

vocab_file = output_path / "vocabulary.txt"

# Sort by frequency
# IMPORTANT: Special tokens must be in this exact order:
# - <BLANK> (index 0): CTC blank token for alignment
# - <PAD> (index 1): Padding token for batching
# - <UNK> (index 2): Unknown token for out-of-vocabulary words
# Note: Excluded tokens (loc-, cl-, IX, etc.) will map to <UNK> if they appear in data
vocab_items = [('<BLANK>', 0), ('<PAD>', 0), ('<UNK>', 0)]  # Keep special tokens
vocab_items.extend(sorted(all_words.items(), key=lambda x: (-x[1], x[0])))

print(f"\n" + "="*70)
print("Writing File Clean Vocabulary...")
print(f"Output: {vocab_file}")

with open(vocab_file, 'w', encoding='utf-8') as f:
    for word, count in vocab_items:
        f.write(f"{word}\t{count}\n")

print(f"Wrote {len(vocab_items)} vocabulary items")

# Verify the clean vocabulary
print(f"\n" + "="*70)
print("Verification:")

# Check first 50 items
print("\nFirst 50 vocabulary items (after special tokens):")
for i in range(3, min(53, len(vocab_items))):
    word, count = vocab_items[i]
    print(f"  {i:4d}: {word:20s} (count: {count})")

# Double-check for problematic tokens
problematic = []
for i, (word, _) in enumerate(vocab_items[3:], start=3):  # Skip special tokens
    if should_exclude(word):
        problematic.append((i, word))

if problematic:
    print(f"\nWARNING: Found {len(problematic)} problematic tokens that slipped through!")
    for idx, word in problematic[:10]:
        print(f"  Index {idx}: {word}")
else:
    print("\nSUCCESS: Vocabulary is clean!")
    print("No location markers, deixis, or special tokens found.")

print("\n" + "="*70)
print("IMPACT ANALYSIS:")
original_vocab_size = 1229
new_vocab_size = len(vocab_items)
reduction = original_vocab_size - new_vocab_size

print(f"Original vocabulary size: {original_vocab_size}")
print(f"Clean vocabulary size: {new_vocab_size}")
print(f"Removed: {reduction} tokens ({reduction/original_vocab_size*100:.1f}%)")

if reduction > 50:
    print("\nSignificant cleanup achieved!")
    print("This should help the model focus on actual sign glosses.")
else:
    print("\nMinimal cleanup - vocabulary might have been partially clean.")

print("\n" + "="*70)
print("IMPORTANT NOTES:")
print("="*70)
print(f"Excluded tokens found in data: {sum(excluded_words.values())} occurrences")
print(f"These tokens (loc-, cl-, IX, etc.) are NOT in vocabulary.")
print(f"If they appear in training/inference data, they will map to <UNK>.")
print(f"\nTo prevent <UNK> mapping, you should:")
print(f"  1. Filter excluded tokens from data during preprocessing, OR")
print(f"  2. Accept that excluded tokens map to <UNK> (current behavior)")
print("="*70)