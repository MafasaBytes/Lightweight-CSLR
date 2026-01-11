"""
Diagnostic script to identify why I3D Teacher isn't learning on full dataset
when it succeeded on overfitting test.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.i3d_teacher import create_i3d_teacher
from src.data.mediapipe_dataset import MediaPipeFeatureDataset, collate_fn, build_vocabulary
from torch.utils.data import DataLoader
import numpy as np


def check_model_outputs(model, dataloader, device, vocab):
    """Check if model is producing sensible outputs."""
    print("\n" + "="*70)
    print("CHECKING MODEL OUTPUTS")
    print("="*70)
    
    model.eval()
    batch = next(iter(dataloader))
    
    features = batch['features'].to(device)
    feature_lengths = batch['input_lengths'].to(device)
    
    with torch.no_grad():
        log_probs = model(features, feature_lengths)
    
    print(f"\n1. Output Shape: {log_probs.shape}")
    print(f"   Expected: [T, B, V] where T=seq_len, B=batch, V=vocab_size")
    
    print(f"\n2. Log Probability Statistics:")
    print(f"   Min: {log_probs.min().item():.4f}")
    print(f"   Max: {log_probs.max().item():.4f}")
    print(f"   Mean: {log_probs.mean().item():.4f}")
    print(f"   Std: {log_probs.std().item():.4f}")
    
    # Check if outputs are valid log probabilities
    is_negative = log_probs.max().item() <= 0.0
    print(f"\n3. Valid log probs (all negative)? {is_negative}")
    
    # Check if softmax sums to 1
    probs = torch.exp(log_probs)
    prob_sums = probs.sum(dim=-1)
    print(f"\n4. Probability sums (should be ~1.0):")
    print(f"   Min: {prob_sums.min().item():.4f}")
    print(f"   Max: {prob_sums.max().item():.4f}")
    print(f"   Mean: {prob_sums.mean().item():.4f}")
    
    # Check entropy (measure of uncertainty)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    max_entropy = np.log(len(vocab))
    print(f"\n5. Prediction Entropy:")
    print(f"   Current: {entropy.item():.4f}")
    print(f"   Maximum (uniform): {max_entropy:.4f}")
    print(f"   Ratio: {entropy.item() / max_entropy:.2%}")
    
    if entropy.item() / max_entropy > 0.95:
        print("   ⚠️  WARNING: Model is predicting almost uniformly (not learning!)")
    elif entropy.item() / max_entropy < 0.3:
        print("   ✓ Good: Model is making confident predictions")
    else:
        print("   ~ Moderate: Model is somewhat confident")
    
    # Check most predicted tokens
    print(f"\n6. Most Predicted Tokens:")
    max_probs, max_indices = log_probs.max(dim=-1)
    token_counts = {}
    for t in range(log_probs.shape[0]):
        for b in range(log_probs.shape[1]):
            idx = max_indices[t, b].item()
            token_counts[idx] = token_counts.get(idx, 0) + 1
    
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for idx, count in sorted_tokens:
        if idx == vocab.blank_id:
            token_name = "<BLANK>"
        elif idx in vocab.idx2word:
            token_name = vocab.idx2word[idx]
        else:
            token_name = f"<UNK_{idx}>"
        percentage = count / (log_probs.shape[0] * log_probs.shape[1]) * 100
        print(f"   {token_name}: {count} ({percentage:.1f}%)")
    
    # Check if model is predicting mostly blanks
    blank_percentage = token_counts.get(vocab.blank_id, 0) / (log_probs.shape[0] * log_probs.shape[1]) * 100
    if blank_percentage > 90:
        print(f"\n   ⚠️  WARNING: Model is predicting blank {blank_percentage:.1f}% of the time!")
        print("   This suggests the model hasn't learned to predict actual words yet.")
    
    # Check gradient flow
    print(f"\n7. Gradient Flow Check:")
    model.train()
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    feature_lengths = batch['input_lengths'].to(device)
    label_lengths = batch['target_lengths'].to(device)
    
    criterion = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    log_probs = model(features, feature_lengths)
    loss = criterion(log_probs, labels, feature_lengths, label_lengths)
    
    print(f"   Loss value: {loss.item():.4f}")
    
    loss.backward()
    
    # Check gradient norms
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "classifier" in name:
                grad_norms["classifier"] = grad_norms.get("classifier", []) + [grad_norm]
            elif "lstm" in name:
                grad_norms["lstm"] = grad_norms.get("lstm", []) + [grad_norm]
            elif "modality_fusion" in name:
                grad_norms["fusion"] = grad_norms.get("fusion", []) + [grad_norm]
            elif "inception" in name or "mixed" in name:
                grad_norms["inception"] = grad_norms.get("inception", []) + [grad_norm]
    
    for component, norms in grad_norms.items():
        avg_norm = np.mean(norms)
        max_norm = np.max(norms)
        print(f"   {component}: avg={avg_norm:.6f}, max={max_norm:.6f}")
        if avg_norm < 1e-7:
            print(f"      ⚠️  WARNING: Very small gradients in {component}!")
        if max_norm > 100:
            print(f"      ⚠️  WARNING: Very large gradients in {component}!")
    
    return {
        'is_valid': is_negative,
        'entropy_ratio': entropy.item() / max_entropy,
        'blank_percentage': blank_percentage,
        'loss': loss.item()
    }


def test_decoding_strategy():
    """Test if decoding strategy is working correctly."""
    print("\n" + "="*70)
    print("TESTING DECODING STRATEGY")
    print("="*70)
    
    # Create dummy vocabulary
    class DummyVocab:
        def __init__(self):
            self.blank_id = 0
            self.idx2word = {1: 'WORD1', 2: 'WORD2', 3: 'WORD3', 4: 'WORD4', 5: 'WORD5'}
    
    vocab = DummyVocab()
    
    # Test case 1: Clear predictions
    print("\nTest 1: Clear predictions (should decode to words)")
    T, B, V = 10, 1, 10
    log_probs = torch.full((T, B, V), -10.0)  # Low probability for all
    log_probs[:, 0, 1] = -0.1  # High probability for word 1
    log_probs[5:, 0, 2] = -0.1  # Then word 2
    
    from train_teacher import decode_predictions
    decoded = decode_predictions(log_probs, vocab)
    print(f"   Result: '{decoded[0]}'")
    print(f"   Expected: 'WORD1 WORD2' or similar")
    
    # Test case 2: Mostly blanks
    print("\nTest 2: Mostly blanks (should decode to few/no words)")
    log_probs = torch.full((T, B, V), -10.0)
    log_probs[:, 0, 0] = -0.1  # High probability for blank
    log_probs[4, 0, 1] = -0.1  # One word
    
    decoded = decode_predictions(log_probs, vocab)
    print(f"   Result: '{decoded[0]}'")
    print(f"   Expected: 'WORD1' or similar")
    
    # Test case 3: Uniform (uncertain)
    print("\nTest 3: Uniform predictions (model uncertain)")
    log_probs = torch.full((T, B, V), np.log(1/V))  # Uniform distribution
    
    decoded = decode_predictions(log_probs, vocab)
    print(f"   Result: '{decoded[0]}'")
    print(f"   Expected: Empty or very few words (model is guessing)")
    
    # Test case 4: All blanks
    print("\nTest 4: All blanks (model predicting nothing)")
    log_probs = torch.full((T, B, V), -10.0)
    log_probs[:, 0, 0] = -0.1  # All blanks
    
    decoded = decode_predictions(log_probs, vocab)
    print(f"   Result: '{decoded[0]}'")
    print(f"   Expected: Empty string")
    if decoded[0] == '':
        print("   ✓ Correct: Empty prediction for all blanks")
    else:
        print("   ⚠️  WARNING: Should be empty but got words!")


def check_dataset():
    """Check if dataset is loaded correctly."""
    print("\n" + "="*70)
    print("CHECKING DATASET")
    print("="*70)
    
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    
    dataset = MediaPipeFeatureDataset(
        data_dir=Path("data/teacher_features/mediapipe_full"),
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False
    )
    
    print(f"\n1. Dataset size: {len(dataset)}")
    print(f"2. Vocabulary size: {len(vocab)}")
    print(f"3. Blank ID: {vocab.blank_id}")
    
    # Check a few samples
    print(f"\n4. Checking first 5 samples:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n   Sample {i}:")
        print(f"     Features shape: {sample['features'].shape}")
        print(f"     Num labels: {len(sample['labels'])}")
        print(f"     Num words: {len(sample['words'])}")
        print(f"     Words: {' '.join(sample['words'][:10])}{'...' if len(sample['words']) > 10 else ''}")
        
        # Check for NaN/Inf in features
        has_nan = torch.isnan(sample['features']).any()
        has_inf = torch.isinf(sample['features']).any()
        if has_nan or has_inf:
            print(f"     ⚠️  WARNING: Features contain NaN={has_nan}, Inf={has_inf}")
        
        # Check feature statistics
        print(f"     Feature mean: {sample['features'].mean():.4f}")
        print(f"     Feature std: {sample['features'].std():.4f}")
        print(f"     Feature min: {sample['features'].min():.4f}")
        print(f"     Feature max: {sample['features'].max():.4f}")


def main():
    """Run all diagnostics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test decoding strategy first
    test_decoding_strategy()
    
    # Check dataset
    check_dataset()
    
    # Build vocabulary and create dataset
    annotation_file = Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
    vocab = build_vocabulary(annotation_file)
    
    dataset = MediaPipeFeatureDataset(
        data_dir=Path("data/teacher_features/mediapipe_full"),
        annotation_file=annotation_file,
        vocabulary=vocab,
        split='train',
        augment=False
    )
    
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    model = create_i3d_teacher(vocab_size=len(vocab), dropout=0.3)
    model = model.to(device)
    print(f"Model created: {model.count_parameters():,} parameters")
    
    # Check model outputs
    results = check_model_outputs(model, dataloader, device, vocab)
    
    # Final diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues = []
    if not results['is_valid']:
        issues.append("❌ Model outputs are not valid log probabilities")
    if results['entropy_ratio'] > 0.95:
        issues.append("❌ Model is predicting uniformly (not learning)")
    if results['blank_percentage'] > 90:
        issues.append("❌ Model is predicting blanks > 90% of the time")
    if results['loss'] > 10:
        issues.append(f"❌ Loss is very high ({results['loss']:.2f})")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\nLIKELY CAUSES:")
        if results['entropy_ratio'] > 0.95:
            print("  1. Model hasn't started learning yet (needs more epochs)")
            print("  2. Learning rate might be too low")
            print("  3. Early stopping triggered too early")
        if results['blank_percentage'] > 90:
            print("  4. Decoding strategy might be filtering out predictions")
            print("  5. Model needs simpler/greedy decoding initially")
    else:
        print("\n✓ No major issues found - model outputs look reasonable")
    
    print("\nRECOMMENDATIONS:")
    print("  1. Disable early stopping or increase patience to 30+")
    print("  2. Use simpler greedy decoding instead of adaptive filtering")
    print("  3. Train for at least 100 epochs before evaluating")
    print("  4. Check if overfitting test used same decoding strategy")
    print("="*70)


if __name__ == "__main__":
    main()

