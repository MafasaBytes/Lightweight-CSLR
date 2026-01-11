"""
Comprehensive evaluation script for trained student models.

Supports:
- Greedy vs Beam Search decoding comparison
- Multiple checkpoints comparison
- Detailed WER breakdown
- Ready for LM integration

Usage:
    python -m src.evaluation.evaluate_student --checkpoint checkpoints/student_adaptsign_features/run_XXX/best.pt --split test
    python -m src.evaluation.evaluate_student --checkpoint checkpoints/student_adaptsign_features/run_XXX/best.pt --split test --beam_search --beam_width 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import Vocabulary
from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.models.baseline_ctc_bilstm import FeatureBiLSTMCTC
from src.utils.ctc import (
    ctc_greedy_decode_with_lengths,
    ctc_beam_search_decode,
    ctc_beam_search_lm_decode,
    ids_to_string,
    split_concatenated_targets,
)
from src.utils.metrics import compute_wer
from src.lm import NGramLM
from src.protocols.protocol_v1 import make_string_filter


class AdaptSignVocab:
    """Vocabulary compatible with AdaptSign's gloss_dict.npy format."""
    def __init__(self, word2idx: dict, idx2word: dict):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.blank_id = 0
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, text: str):
        tokens = text.strip().split()
        return [self.word2idx.get(t, 0) for t in tokens if t in self.word2idx]


def load_adaptsign_vocab(npy_path: Path) -> AdaptSignVocab:
    """Load vocabulary from AdaptSign's gloss_dict.npy format."""
    gloss_dict = np.load(npy_path, allow_pickle=True).item()
    # gloss_dict: {word: [index, count]} - indices start from 1
    # We need word2idx with blank=0
    word2idx = {"<blank>": 0}
    for word, (idx, _count) in gloss_dict.items():
        word2idx[word] = idx  # Keep original indices (1-based)
    
    idx2word = {v: k for k, v in word2idx.items()}
    return AdaptSignVocab(word2idx, idx2word)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    return logging.getLogger("evaluate")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    
    # Get model parameters from checkpoint
    feature_dim = args.get("feature_dim", 512)
    hidden_dim = args.get("hidden_dim", 512)
    encoder_layers = args.get("encoder_layers", 3)
    num_heads = args.get("num_heads", 8)
    dropout = args.get("dropout", 0.3)
    subsample = not args.get("no_subsample", False)
    
    # Load vocabulary - support both JSON and NPY formats
    vocab_npy = args.get("vocab_npy")
    vocab_json = args.get("vocab_json", "checkpoints/vocabulary.json")
    
    if vocab_npy and Path(vocab_npy).exists():
        vocab = load_adaptsign_vocab(Path(vocab_npy))
        print(f"Loaded vocab from NPY: {len(vocab)} words")
    else:
        vocab = Vocabulary()
        vocab.load(Path(vocab_json))
        print(f"Loaded vocab from JSON: {len(vocab)} words")
    
    # Create model
    model = FeatureBiLSTMCTC(
        vocab_size=len(vocab),
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        encoder_layers=encoder_layers,
        num_heads=num_heads,
        dropout=dropout,
        subsample=subsample,
    ).to(device)
    
    # Load weights (try EMA first, then regular)
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded regular weights")
    
    model.eval()
    meta = {
        "best_wer": ckpt.get("best_wer", None),
        "epoch": ckpt.get("epoch", None),
    }
    return model, vocab, args, meta


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: torch.device,
    idx2word: dict,
    decode_method: str = "greedy",
    beam_width: int = 10,
    lm = None,
    lm_weight: float = 0.3,
    string_filter_fn=None,
):
    """
    Evaluate model on dataloader.
    
    Args:
        lm: Language model (NGramLM instance or None)
        lm_weight: Weight for LM in beam search (0 = CTC only)
    
    Returns:
        dict with WER, predictions, timing info
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_video_ids = []
    total_frames = 0
    total_time = 0.0
    
    for batch in dataloader:
        features = batch["features"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["labels"].to(device)
        target_lengths = batch["label_lengths"].to(device)
        video_ids = batch.get("video_ids", ["unknown"] * features.shape[0])
        
        # Time the forward pass
        start_time = time.time()
        log_probs, out_lengths = model(features, input_lengths)
        
        # Decode
        if decode_method == "beam_search_lm" and lm is not None:
            log_probs_tbv = log_probs.permute(1, 0, 2)
            pred_seqs = ctc_beam_search_lm_decode(
                log_probs_tbv, lengths=out_lengths, blank_idx=0, beam_width=beam_width,
                lm=lm, lm_weight=lm_weight, idx2word=idx2word
            )
        elif decode_method == "beam_search":
            log_probs_tbv = log_probs.permute(1, 0, 2)
            pred_seqs = ctc_beam_search_decode(log_probs_tbv, lengths=out_lengths, blank_idx=0, beam_width=beam_width)
        else:
            pred_ids = log_probs.argmax(dim=-1)
            pred_seqs = ctc_greedy_decode_with_lengths(pred_ids, out_lengths, blank_idx=0)
        
        total_time += time.time() - start_time
        total_frames += int(input_lengths.sum().item())
        
        # Convert to strings
        tgt_seqs = split_concatenated_targets(targets, target_lengths)
        
        for pred_ids_i, tgt_ids_i, vid in zip(pred_seqs, tgt_seqs, video_ids):
            pred_str = ids_to_string(pred_ids_i, idx2word)
            tgt_str = ids_to_string(tgt_ids_i, idx2word)
            all_preds.append(pred_str)
            all_targets.append(tgt_str)
            all_video_ids.append(vid)
    
    # Compute WER
    if string_filter_fn is not None:
        all_targets = [string_filter_fn(s) for s in all_targets]
        all_preds = [string_filter_fn(s) for s in all_preds]
    wer = compute_wer(all_targets, all_preds)
    
    # Compute per-sample WER for analysis
    sample_wers = []
    for ref, hyp in zip(all_targets, all_preds):
        sample_wer = compute_wer([ref], [hyp])
        sample_wers.append(sample_wer)
    
    fps = total_frames / total_time if total_time > 0 else 0
    
    return {
        "wer": wer,
        "predictions": all_preds,
        "targets": all_targets,
        "video_ids": all_video_ids,
        "sample_wers": sample_wers,
        "total_frames": total_frames,
        "total_time": total_time,
        "fps": fps,
        "decode_method": decode_method,
        "beam_width": beam_width if decode_method == "beam_search" else None,
    }


def print_results(results: dict, logger, num_samples: int = 10):
    """Print evaluation results."""
    logger.info("=" * 60)
    logger.info(f"Decode method: {results['decode_method']}")
    if results['beam_width']:
        logger.info(f"Beam width: {results['beam_width']}")
    logger.info(f"WER: {results['wer']:.2f}%")
    logger.info(f"Total frames: {results['total_frames']}")
    logger.info(f"Inference time: {results['total_time']:.2f}s")
    logger.info(f"FPS: {results['fps']:.1f}")
    logger.info("=" * 60)
    
    # Show samples
    logger.info(f"\nSample predictions (showing {num_samples}):")
    indices = list(range(min(num_samples, len(results['predictions']))))
    
    for i in indices:
        wer_i = results['sample_wers'][i]
        logger.info(f"\n[{i+1}] WER: {wer_i:.1f}%")
        logger.info(f"  REF: {results['targets'][i]}")
        logger.info(f"  HYP: {results['predictions'][i] or '(empty)'}")


def main():
    parser = argparse.ArgumentParser("Evaluate trained student model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--features_dir", type=str, default=None, 
                        help="Override features dir from checkpoint")
    parser.add_argument("--annotation_dir", type=str,
                        default="data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual")
    parser.add_argument("--split_source", type=str, default="si5", choices=["si5", "adaptsign_official"])
    parser.add_argument("--adaptsign_preprocess_dir", type=str, default="adaptsign_repo/preprocess")
    parser.add_argument("--adaptsign_dataset", type=str, default="phoenix2014")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Decoding options
    parser.add_argument("--beam_search", action="store_true", help="Use beam search instead of greedy")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--lm_path", type=str, default=None,
                        help="Path to trained n-gram LM (e.g., models/lm/phoenix_3gram.pkl)")
    parser.add_argument("--lm_weight", type=float, default=0.3, 
                        help="Language model weight for beam search (0=CTC only, 0.3=default)")
    
    # Output
    parser.add_argument("--output_json", type=str, default=None, 
                        help="Save detailed results to JSON")
    parser.add_argument("--compare_greedy", action="store_true",
                        help="Also run greedy decoding for comparison")
    args = parser.parse_args()
    
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, vocab, ckpt_args, ckpt_meta = load_model_from_checkpoint(checkpoint_path, device)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"Best WER in training (from checkpoint): {ckpt_meta.get('best_wer', 'N/A')}")
    if ckpt_meta.get("epoch") is not None:
        logger.info(f"Checkpoint epoch: {ckpt_meta.get('epoch')}")
    
    # Get features dir
    features_dir = args.features_dir or ckpt_args.get("features_dir", "data/teacher_features/adaptsign")
    logger.info(f"Features dir: {features_dir}")
    
    # Load dataset
    # Need to get train stats for normalization
    train_ds = SequenceFeatureDataset(
        features_dir=Path(features_dir),
        annotation_dir=Path(args.annotation_dir),
        vocabulary=vocab,
        split="train",
        split_source=args.split_source,
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=args.adaptsign_dataset,
        max_seq_length=ckpt_args.get("max_seq_length", 300),
        normalize=True,
    )
    
    eval_ds = SequenceFeatureDataset(
        features_dir=Path(features_dir),
        annotation_dir=Path(args.annotation_dir),
        vocabulary=vocab,
        split=args.split,
        split_source=args.split_source,
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=args.adaptsign_dataset,
        max_seq_length=ckpt_args.get("max_seq_length", 300),
        normalize=True,
        mean=train_ds.mean,
        std=train_ds.std,
    )
    
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_ctc_features,
    )
    logger.info(f"Loaded {len(eval_ds)} {args.split} samples")

    # Match training Protocol v1 filtering for thesis-consistent WERs.
    string_filter_fn = make_string_filter()
    
    # Load Language Model if specified
    lm = None
    if args.lm_path:
        lm_path = Path(args.lm_path)
        if lm_path.exists():
            lm = NGramLM.load(lm_path)
            logger.info(f"Loaded {lm.n}-gram LM from {lm_path}")
            logger.info(f"LM vocab size: {len(lm.vocab)}, LM weight: {args.lm_weight}")
        else:
            logger.warning(f"LM file not found: {lm_path}")
    
    # Run evaluation
    if args.beam_search and lm is not None:
        decode_method = "beam_search_lm"
    elif args.beam_search:
        decode_method = "beam_search"
    else:
        decode_method = "greedy"
    
    results = evaluate(
        model, eval_loader, device, vocab.idx2word,
        decode_method=decode_method,
        beam_width=args.beam_width,
        lm=lm,
        lm_weight=args.lm_weight,
        string_filter_fn=string_filter_fn,
    )
    
    print_results(results, logger)
    
    # Compare with greedy if requested
    if args.compare_greedy and args.beam_search:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: Running greedy decode...")
        greedy_results = evaluate(
            model, eval_loader, device, vocab.idx2word,
            decode_method="greedy",
            lm=None,  # No LM for greedy baseline
        )
        lm_note = " (with LM)" if lm is not None else ""
        logger.info(f"\nGreedy WER: {greedy_results['wer']:.2f}%")
        logger.info(f"Beam Search{lm_note} WER: {results['wer']:.2f}%")
        logger.info(f"Improvement: {greedy_results['wer'] - results['wer']:.2f}%")
    
    # Save results
    if args.output_json:
        output_data = {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "wer": results["wer"],
            "fps": results["fps"],
            "decode_method": results["decode_method"],
            "beam_width": results["beam_width"],
            "num_samples": len(results["predictions"]),
            "predictions": [
                {"video_id": vid, "ref": ref, "hyp": hyp, "wer": wer}
                for vid, ref, hyp, wer in zip(
                    results["video_ids"], results["targets"], 
                    results["predictions"], results["sample_wers"]
                )
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to: {args.output_json}")


if __name__ == "__main__":
    main()

