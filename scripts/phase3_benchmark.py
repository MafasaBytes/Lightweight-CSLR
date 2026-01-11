"""
Phase 3 (Deployment): benchmark student inference throughput/latency.

Measures:
- Feature IO time (npz load)
- Model forward time
- Decode time (greedy vs ctcdecode beam)
- End-to-end throughput (FPS in feature-frames / sec) and Real-Time Factor (RTF)

This script is designed for thesis reporting on consumer-grade hardware.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.evaluation.evaluate_student import load_model_from_checkpoint
from src.protocols.protocol_v1 import make_string_filter
from src.utils.ctc import ctc_beam_search_decode, ctc_greedy_decode_with_lengths, ids_to_string
from src.utils.metrics import compute_wer


@dataclass
class BenchRow:
    split: str
    video_id: str
    T_in: int
    T_out: int
    load_s: float
    forward_s: float
    decode_s: float
    total_s: float
    wer: float


def _decode(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    idx2word: Dict[int, str],
    decode: str,
    beam_width: int,
    string_filter_fn,
) -> List[str]:
    if decode == "beam_search":
        # (B,T,V) -> (T,B,V)
        tbv = log_probs.permute(1, 0, 2)
        seqs = ctc_beam_search_decode(tbv, lengths=out_lengths, blank_idx=0, beam_width=beam_width)
    else:
        pred_ids = log_probs.argmax(dim=-1)  # (B,T)
        seqs = ctc_greedy_decode_with_lengths(pred_ids, out_lengths, blank_idx=0)
    out = [ids_to_string(s, idx2word) for s in seqs]
    if string_filter_fn is not None:
        out = [string_filter_fn(x) for x in out]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--features_dir", type=str, default="data/teacher_features/adaptsign_official")
    ap.add_argument("--split_source", type=str, default="adaptsign_official", choices=["si5", "adaptsign_official"])
    ap.add_argument("--adaptsign_preprocess_dir", type=str, default="adaptsign_repo/preprocess")
    ap.add_argument("--adaptsign_dataset", type=str, default="phoenix2014")
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--batch_size", type=int, default=1, help="Keep 1 for per-sample latency; >1 for throughput.")
    ap.add_argument("--decode", type=str, default="beam_search", choices=["greedy", "beam_search"])
    ap.add_argument("--beam_width", type=int, default=10)
    ap.add_argument("--output_json", type=str, default="figures/phase3_benchmark.json")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    model, vocab, ckpt_args, _meta = load_model_from_checkpoint(ckpt_path, device)
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

    # Deterministic subset
    rng = np.random.RandomState(int(args.seed))
    n = min(int(args.num_samples), len(ds))
    idxs = rng.choice(len(ds), size=n, replace=False).tolist()

    string_filter_fn = make_string_filter()

    rows: List[BenchRow] = []
    total_frames_in = 0
    total_time = 0.0
    total_forward = 0.0
    total_decode = 0.0
    total_load = 0.0

    # Mini-batching (still records per-batch latency as "row"; good enough for throughput)
    for start in range(0, len(idxs), int(args.batch_size)):
        batch_idxs = idxs[start : start + int(args.batch_size)]

        t0 = time.time()
        samples = [ds[i] for i in batch_idxs]
        load_s = time.time() - t0

        batch = collate_fn_ctc_features(samples)
        feats = batch["features"].to(device)
        in_lens = batch["input_lengths"].to(device)

        t1 = time.time()
        with torch.no_grad():
            log_probs, out_lens = model(feats, in_lens)
        forward_s = time.time() - t1

        t2 = time.time()
        preds = _decode(
            log_probs=log_probs,
            out_lengths=out_lens,
            idx2word=vocab.idx2word,
            decode=str(args.decode),
            beam_width=int(args.beam_width),
            string_filter_fn=string_filter_fn,
        )
        # references from labels
        targets = []
        for s in samples:
            ref = " ".join([vocab.idx2word[int(t)] for t in s["labels"].tolist()])
            targets.append(string_filter_fn(ref))
        decode_s = time.time() - t2

        wer = float(compute_wer(targets, preds))
        total_s = load_s + forward_s + decode_s

        # aggregate
        total_load += load_s
        total_forward += forward_s
        total_decode += decode_s
        total_time += total_s
        total_frames_in += int(in_lens.sum().item())

        # representative row (first sample id)
        vid0 = samples[0]["video_id"]
        rows.append(
            BenchRow(
                split=str(args.split),
                video_id=str(vid0),
                T_in=int(in_lens[0].item()),
                T_out=int(out_lens[0].item()),
                load_s=float(load_s),
                forward_s=float(forward_s),
                decode_s=float(decode_s),
                total_s=float(total_s),
                wer=float(wer),
            )
        )

    fps = float(total_frames_in / total_time) if total_time > 0 else 0.0
    # Without true video seconds, we approximate RTF with feature-frames at 25 FPS (Phoenix default).
    assumed_fps = 25.0
    rtf = float((total_time) / (total_frames_in / assumed_fps)) if total_frames_in > 0 else 0.0

    summary = {
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "features_dir": str(args.features_dir),
        "split": str(args.split),
        "split_source": str(args.split_source),
        "decode": str(args.decode),
        "beam_width": int(args.beam_width),
        "num_samples": int(n),
        "batch_size": int(args.batch_size),
        "assumed_video_fps": assumed_fps,
        "total_frames_in": int(total_frames_in),
        "total_time_s": float(total_time),
        "total_load_s": float(total_load),
        "total_forward_s": float(total_forward),
        "total_decode_s": float(total_decode),
        "fps_feature_frames_per_s": float(fps),
        "rtf_assuming_25fps": float(rtf),
        "mean_row_total_s": float(np.mean([r.total_s for r in rows])) if rows else 0.0,
        "rows": [asdict(r) for r in rows],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved benchmark to: {out_path}")
    print(f"Device: {device} | decode={args.decode} beam={args.beam_width}")
    print(f"Total frames: {total_frames_in} | total time: {total_time:0.2f}s | FPS(frames): {fps:0.1f}")
    print(f"RTF (assume 25 FPS video): {rtf:0.3f} (lower is better; <1 is real-time)")


if __name__ == "__main__":
    main()


