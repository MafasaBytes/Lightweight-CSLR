"""
Phase 3 (Deployment): sliding-window "streaming" CSLR demo over pre-extracted features.

This simulates continuous recognition by:
- scanning a feature sequence with a fixed window (W) and stride (S)
- decoding each window (beam or greedy)
- showing a stable prefix across steps (simple longest-common-prefix heuristic)
- reporting per-step latency

It is not a full online CTC decoder, but is appropriate for a thesis Phase 3 prototype.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.evaluation.evaluate_student import load_model_from_checkpoint
from src.protocols.protocol_v1 import make_string_filter
from src.utils.ctc import ctc_beam_search_decode, ctc_greedy_decode_with_lengths, ids_to_string


def lcp(a: List[str], b: List[str]) -> List[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def decode_window(
    model,
    vocab,
    feats: torch.Tensor,
    lengths: torch.Tensor,
    decode: str,
    beam_width: int,
    string_filter_fn,
) -> List[str]:
    with torch.no_grad():
        log_probs, out_lengths = model(feats, lengths)
    if decode == "beam_search":
        tbv = log_probs.permute(1, 0, 2)
        seqs = ctc_beam_search_decode(tbv, lengths=out_lengths, blank_idx=0, beam_width=beam_width)
    else:
        pred_ids = log_probs.argmax(dim=-1)
        seqs = ctc_greedy_decode_with_lengths(pred_ids, out_lengths, blank_idx=0)
    s = ids_to_string(seqs[0], vocab.idx2word)
    s = string_filter_fn(s) if string_filter_fn is not None else s
    return [t for t in s.split() if t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--features_dir", type=str, default="data/teacher_features/adaptsign_official")
    ap.add_argument("--split_source", type=str, default="adaptsign_official", choices=["si5", "adaptsign_official"])
    ap.add_argument("--adaptsign_preprocess_dir", type=str, default="adaptsign_repo/preprocess")
    ap.add_argument("--adaptsign_dataset", type=str, default="phoenix2014")
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    ap.add_argument("--video_id", type=str, default="", help="If empty, pick a random sample.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--window", type=int, default=64, help="Window size in input frames.")
    ap.add_argument("--stride", type=int, default=8, help="Stride in input frames.")
    ap.add_argument("--decode", type=str, default="beam_search", choices=["greedy", "beam_search"])
    ap.add_argument("--beam_width", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, ckpt_args, _meta = load_model_from_checkpoint(Path(args.checkpoint), device)
    model.eval()

    # Normalize using train stats (same as evaluation).
    train_ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
        vocabulary=vocab,
        split="train",
        split_source=str(args.split_source),
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=str(args.adaptsign_dataset),
        max_seq_length=int(ckpt_args.get("max_seq_length", 300)),
        normalize=True,
    )
    ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
        vocabulary=vocab,
        split=str(args.split),
        split_source=str(args.split_source),
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=str(args.adaptsign_dataset),
        max_seq_length=int(ckpt_args.get("max_seq_length", 300)),
        normalize=True,
        mean=train_ds.mean,
        std=train_ds.std,
    )

    # Select sample
    if args.video_id:
        idx = next((i for i, s in enumerate(ds.samples) if s["video_id"] == args.video_id), None)
        if idx is None:
            raise SystemExit(f"video_id not found in split={args.split}: {args.video_id}")
    else:
        idx = int(np.random.randint(0, len(ds)))

    sample = ds[idx]
    x_full = sample["features"]  # torch.Tensor (T,D) on CPU
    if isinstance(x_full, torch.Tensor):
        x_np = x_full.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_full)
    T = int(x_np.shape[0])

    string_filter_fn = make_string_filter()

    print(f"Device: {device}")
    print(f"Sample: split={args.split} video_id={sample['video_id']} T={T} D={x_np.shape[1]}")
    print(f"Streaming: window={args.window}, stride={args.stride}, decode={args.decode}, beam={args.beam_width}")
    print("-" * 80)

    stable: List[str] = []
    prev: List[str] = []
    latencies: List[float] = []

    # Warmup (important for CUDA + ctcdecode one-time initialization).
    warm_chunk = x_np[: min(T, int(args.window))]
    warm_feats = torch.from_numpy(warm_chunk).unsqueeze(0).to(torch.float32).to(device)
    warm_lens = torch.tensor([warm_chunk.shape[0]], dtype=torch.long, device=device)
    _ = decode_window(model, vocab, warm_feats, warm_lens, str(args.decode), int(args.beam_width), string_filter_fn)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Sliding window across the full sequence, including the tail.
    for t0 in range(0, T, int(args.stride)):
        chunk = x_np[t0 : min(T, t0 + int(args.window))]
        feats = torch.from_numpy(chunk).unsqueeze(0).to(torch.float32).to(device)
        in_lens = torch.tensor([chunk.shape[0]], dtype=torch.long, device=device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        st = time.time()
        hyp = decode_window(model, vocab, feats, in_lens, str(args.decode), int(args.beam_width), string_filter_fn)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - st
        latencies.append(float(dt))

        if not prev:
            stable = hyp
        else:
            stable = lcp(prev, hyp)
        prev = hyp

        stable_str = " ".join(stable)
        hyp_str = " ".join(hyp)
        print(f"t={t0:4d}..{t0+int(args.window):4d}  latency={dt*1000:6.1f}ms  stable='{stable_str}'  hyp='{hyp_str}'")

    if latencies:
        print("-" * 80)
        print(f"Mean step latency: {np.mean(latencies)*1000:.1f}ms | p95: {np.quantile(latencies, 0.95)*1000:.1f}ms")


if __name__ == "__main__":
    main()


