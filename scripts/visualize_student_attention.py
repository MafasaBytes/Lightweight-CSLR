"""
Export academic-quality self-attention visualizations from a trained FeatureBiLSTMCTC checkpoint.

What it produces (per selected sample):
- Attention heatmap(s): time x time, optionally per-head
- Predicted gloss segmentation overlay (from greedy CTC argmax collapse)
- Reference gloss string (Protocol v1 filtered, from official split list)

This is meant for thesis figures: interpretability + alignment behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.evaluation.evaluate_student import load_adaptsign_vocab
from src.models.baseline_ctc_bilstm import FeatureBiLSTMCTC
from src.protocols.protocol_v1 import filter_sentence, make_string_filter


@dataclass
class Segment:
    token_id: int
    token: str
    start: int
    end: int


def _greedy_segments(log_probs_btv: torch.Tensor, out_len: int, idx2word: dict, blank_id: int = 0) -> List[Segment]:
    """
    Convert per-timestep argmax to collapsed segments (CTC-style), returning spans in T' space.
    """
    ids = log_probs_btv[0, :out_len].argmax(dim=-1).tolist()  # batch=1
    segs: List[Segment] = []
    cur = None
    cur_start = 0
    for t, tok in enumerate(ids + [None]):
        if cur is None:
            cur = tok
            cur_start = t
            continue
        if tok != cur:
            if cur is not None and int(cur) != int(blank_id):
                w = idx2word.get(int(cur), "<unk>")
                segs.append(Segment(int(cur), str(w), int(cur_start), int(t)))
            cur = tok
            cur_start = t
    return segs


def _plot_attention(
    attn: torch.Tensor,
    segments: List[Segment],
    title: str,
    out_path: Path,
    per_head: bool,
    max_heads: int = 8,
) -> None:
    """
    attn:
      - if per_head: (H, T, T)
      - else: (T, T)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if per_head:
        H = int(attn.shape[0])
        H = min(H, int(max_heads))
        ncols = min(4, H)
        nrows = int(np.ceil(H / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
        for i in range(nrows * ncols):
            ax = axes[i // ncols][i % ncols]
            ax.axis("off")
        for h in range(H):
            ax = axes[h // ncols][h % ncols]
            ax.axis("on")
            m = attn[h].detach().cpu().numpy()
            im = ax.imshow(m, cmap="viridis", aspect="auto", vmin=0.0, vmax=float(np.max(m) + 1e-6))
            ax.set_title(f"Head {h}")
            ax.set_xlabel("Key time")
            ax.set_ylabel("Query time")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _overlay_segments(ax, segments)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return

    m = attn.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2))
    im = ax.imshow(m, cmap="viridis", aspect="auto", vmin=0.0, vmax=float(np.max(m) + 1e-6))
    ax.set_title(title)
    ax.set_xlabel("Key time")
    ax.set_ylabel("Query time")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _overlay_segments(ax, segments)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _overlay_segments(ax, segments: List[Segment]) -> None:
    # Draw segment boundaries on both axes (academic, minimal styling)
    for s in segments:
        ax.axvline(s.start, color="white", linewidth=0.6, alpha=0.7)
        ax.axhline(s.start, color="white", linewidth=0.6, alpha=0.7)
    # Label a few segments along x-axis (top)
    for s in segments[:12]:
        mid = (s.start + s.end) / 2.0
        ax.text(mid, -2, s.token, ha="center", va="bottom", fontsize=7, rotation=90, color="black")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--features_dir", type=str, default="data/teacher_features/adaptsign_official")
    ap.add_argument("--split_source", type=str, default="adaptsign_official", choices=["si5", "adaptsign_official"])
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    ap.add_argument("--adaptsign_preprocess_dir", type=str, default="adaptsign_repo/preprocess")
    ap.add_argument("--adaptsign_dataset", type=str, default="phoenix2014")
    ap.add_argument("--vocab_npy", type=str, default="adaptsign_repo/preprocess/phoenix2014/gloss_dict.npy")
    ap.add_argument("--feature_dim", type=int, default=512)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--encoder_layers", type=int, default=3)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no_subsample", action="store_true")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", type=str, default="figures/attention_student")
    ap.add_argument("--per_head", action="store_true", help="Save per-head attention grids (slower, larger figures).")
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    vocab = load_adaptsign_vocab(Path(args.vocab_npy))
    model = FeatureBiLSTMCTC(
        vocab_size=len(vocab),
        input_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        encoder_layers=int(args.encoder_layers),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
        subsample=(not bool(args.no_subsample)),
    ).to(device)

    # Prefer EMA weights if present (matches your evaluation practice).
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Dataset (official split uses *_info.npy labels; we use it via SequenceFeatureDataset).
    ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
        vocabulary=vocab,
        split=str(args.split),
        split_source=str(args.split_source),
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=str(args.adaptsign_dataset),
        max_seq_length=300,
        normalize=True,
    )

    # Recompute train stats for consistent normalization (small overhead; thesis reproducibility).
    train_ds = SequenceFeatureDataset(
        features_dir=Path(args.features_dir),
        annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
        vocabulary=vocab,
        split="train",
        split_source=str(args.split_source),
        adaptsign_preprocess_dir=Path(args.adaptsign_preprocess_dir),
        adaptsign_dataset=str(args.adaptsign_dataset),
        max_seq_length=300,
        normalize=True,
    )
    ds.set_stats(train_ds.mean, train_ds.std)

    rng = np.random.RandomState(int(args.seed))
    idxs = rng.choice(len(ds), size=min(int(args.num_samples), len(ds)), replace=False).tolist()

    string_filter_fn = make_string_filter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for j, idx in enumerate(idxs):
        sample = ds[idx]
        # Build a batch of size 1 via collate for correct lengths.
        batch = collate_fn_ctc_features([sample])
        feats = batch["features"].to(device)
        lengths = batch["input_lengths"].to(device)

        with torch.no_grad():
            log_probs, out_lengths, attn = model(feats, lengths, return_attn=True)

        Tprime = int(out_lengths[0].item())
        segments = _greedy_segments(log_probs, Tprime, vocab.idx2word, blank_id=0)

        # Protocol v1 reference (already filtered in dataset labels; for thesis text, show filtered string)
        ref = " ".join([vocab.idx2word[int(t)] for t in sample["labels"].tolist()])
        ref = filter_sentence(ref)

        hyp = " ".join([s.token for s in segments])
        hyp = string_filter_fn(hyp)

        title = f"{args.split} sample {j+1}/{len(idxs)} | vid={sample['video_id']}\nREF: {ref}\nHYP: {hyp}"

        # attn is (B,H,T,T) in our encoder when return_attn=True
        attn_b = attn[0, :, :Tprime, :Tprime]
        if args.per_head:
            out_path = out_dir / f"{args.split}_{j:02d}_{sample['video_id']}_attn_heads.png"
            _plot_attention(attn_b, segments, title, out_path, per_head=True)
        else:
            # Average over heads for a single, clean figure
            out_path = out_dir / f"{args.split}_{j:02d}_{sample['video_id']}_attn_avg.png"
            _plot_attention(attn_b.mean(dim=0), segments, title, out_path, per_head=False)

    print(f"Saved attention figures to: {out_dir}")


if __name__ == "__main__":
    main()


