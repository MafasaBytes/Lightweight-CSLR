"""
Phase 3 (Deployment): Educational-style dashboard (Streamlit).

Features:
- Select a dataset sample (dev/test) and run student inference (greedy/beam)
- Show REF vs HYP + per-sample WER
- Show confidence timeline (CTC entropy / top-1 prob over time)
- Show attention heatmap (avg over heads) with predicted segment markers

Run (WSL):
  streamlit run apps/edu_dashboard.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sequence_feature_dataset import SequenceFeatureDataset, collate_fn_ctc_features
from src.evaluation.evaluate_student import load_model_from_checkpoint
from src.protocols.protocol_v1 import make_string_filter
from src.utils.ctc import ctc_beam_search_decode, ctc_greedy_decode_with_lengths, ids_to_string
from src.utils.metrics import compute_wer


def greedy_segments_from_ids(ids: List[int], idx2word: dict, blank_id: int = 0) -> List[Tuple[str, int, int]]:
    segs = []
    cur = None
    cur_start = 0
    for t, tok in enumerate(ids + [None]):
        if cur is None:
            cur = tok
            cur_start = t
            continue
        if tok != cur:
            if cur is not None and int(cur) != int(blank_id):
                segs.append((idx2word.get(int(cur), "<unk>"), int(cur_start), int(t)))
            cur = tok
            cur_start = t
    return segs


def _make_export_figure(
    out: dict,
    show_segments: bool = True,
    max_side: int = 256,
    include_segments_table: bool = True,
) -> plt.Figure:
    """
    Create a single, publication-friendly composite figure for screenshots/exports.
    Includes: (a) REF vs HYP + sample WER/ACC, (b) confidence over time, (c) attention heatmap.
    """
    ref = str(out["ref"])
    hyp = str(out["hyp"])
    wer_pct = float(out["wer"])
    acc_pct = float(max(0.0, 100.0 - wer_pct))

    entropy = np.asarray(out["entropy"])
    top1 = np.asarray(out["top1"])
    attn = np.asarray(out["attn_avg"])
    segs = list(out.get("segments", []))
    vid = str(out.get("video_id", ""))
    decode_method = str(out.get("decode_method", ""))
    beam_width = out.get("beam_width", None)

    T = int(attn.shape[0])
    if T > max_side:
        step = int(np.ceil(T / max_side))
        attn_view = attn[::step, ::step]
        segs_view = [(tok, s // step, e // step) for tok, s, e in segs]
        title = f"Self-attention (avg heads) — overview (downsample x{step})"
    else:
        attn_view = attn
        segs_view = segs
        title = "Self-attention (avg heads) — overview"

    # Thesis-ready styling: clean white background + slightly larger typography.
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(11.2, 6.6), dpi=170, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.55], height_ratios=[1.0, 1.2])

    # Left column: (a) REF/HYP summary + (d) predicted segments table (dashboard-style)
    left_gs = gs[:, 0].subgridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.05)

    # (a) Text summary: REF/HYP
    ax_text = fig.add_subplot(left_gs[0, 0])
    ax_text.axis("off")
    decode_str = (
        f"{decode_method}, beam={beam_width}" if decode_method and beam_width is not None else decode_method
    ).strip(", ")
    subtitle = f" | {decode_str}" if decode_str else ""
    ax_text.set_title(
        f"Sample: {vid}{subtitle}\nWER={wer_pct:.1f}% | Word accuracy proxy={acc_pct:.1f}% (100 − WER)",
        loc="left",
    )
    text = (
        "REF:\n"
        f"{ref}\n\n"
        "HYP:\n"
        f"{hyp}\n"
    )
    ax_text.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        family="DejaVu Sans",
        fontsize=10.4,
        wrap=True,
        transform=ax_text.transAxes,
    )

    # (b) Confidence over time
    ax_conf = fig.add_subplot(gs[0, 1])
    ax_conf.plot(entropy, label="entropy", linewidth=1.8, color="#1f77b4")
    ax_conf.plot(top1, label="top1_prob", linewidth=1.8, color="#ff7f0e")
    ax_conf.set_title("Confidence over time")
    ax_conf.set_xlabel("time index (T')")
    ax_conf.grid(True, alpha=0.25)
    ax_conf.legend(loc="upper right", fontsize=8)

    # (c) Attention heatmap
    ax_attn = fig.add_subplot(gs[1, 1])
    im = ax_attn.imshow(attn_view, cmap="viridis", aspect="auto")
    ax_attn.set_title(title)
    ax_attn.set_xlabel("key time")
    ax_attn.set_ylabel("query time")
    if show_segments:
        for _tok, s, _e in segs_view:
            if 0 <= int(s) < attn_view.shape[0]:
                ax_attn.axvline(int(s), color="white", linewidth=0.6, alpha=0.7)
                ax_attn.axhline(int(s), color="white", linewidth=0.6, alpha=0.7)
    fig.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.02)

    # (d) Predicted segments table (dashboard-style)
    if include_segments_table:
        ax_seg = fig.add_subplot(left_gs[1, 0])
        ax_seg.axis("off")
        ax_seg.set_title("Predicted segments (greedy, for visualization)", loc="left")

        seg_rows = [(tok, int(s), int(e)) for (tok, s, e) in segs[:14]]
        if len(segs) > 14:
            seg_rows.append(("…", -1, -1))

        if len(seg_rows) == 0:
            ax_seg.text(0.0, 0.8, "No non-blank segments found.", ha="left", va="top")
        else:
            tbl = ax_seg.table(
                cellText=seg_rows,
                colLabels=["token", "start_t", "end_t"],
                loc="upper left",
                cellLoc="left",
                colLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9.4)
            tbl.scale(1.0, 1.12)
            # Light borders to match the dashboard's clean table look.
            for key, cell in tbl.get_celld().items():
                cell.set_edgecolor("#DDDDDD")
                if key[0] == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#F6F6F6")

    return fig


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_resource
def load_model_and_data(
    checkpoint: str,
    features_dir: str,
    split_source: str,
    adaptsign_preprocess_dir: str,
    adaptsign_dataset: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, ckpt_args, _meta = load_model_from_checkpoint(Path(checkpoint), device)
    model.eval()

    # Model size stats (for <100MB constraint reporting)
    ckpt_path = Path(checkpoint)
    ckpt_mb = float(ckpt_path.stat().st_size / (1024**2)) if ckpt_path.exists() else float("nan")
    num_params = int(sum(p.numel() for p in model.parameters()))
    fp32_mb = float(num_params * 4 / (1024**2))
    fp16_mb = float(num_params * 2 / (1024**2))
    model_stats = {
        "checkpoint_mb": ckpt_mb,
        "num_params": num_params,
        "param_fp32_mb": fp32_mb,
        "param_fp16_mb": fp16_mb,
    }

    # Normalize with train stats
    train_ds = SequenceFeatureDataset(
        features_dir=Path(features_dir),
        annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
        vocabulary=vocab,
        split="train",
        split_source=split_source,
        adaptsign_preprocess_dir=Path(adaptsign_preprocess_dir),
        adaptsign_dataset=adaptsign_dataset,
        max_seq_length=int(ckpt_args.get("max_seq_length", 300)),
        normalize=True,
    )

    def make_split(split: str):
        return SequenceFeatureDataset(
            features_dir=Path(features_dir),
            annotation_dir=Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual"),
            vocabulary=vocab,
            split=split,
            split_source=split_source,
            adaptsign_preprocess_dir=Path(adaptsign_preprocess_dir),
            adaptsign_dataset=adaptsign_dataset,
            max_seq_length=int(ckpt_args.get("max_seq_length", 300)),
            normalize=True,
            mean=train_ds.mean,
            std=train_ds.std,
        )

    return device, model, vocab, make_split, model_stats


def decode(
    model,
    vocab,
    device,
    sample,
    decode_method: str,
    beam_width: int,
):
    batch = collate_fn_ctc_features([sample])
    feats = batch["features"].to(device)
    in_lens = batch["input_lengths"].to(device)

    with torch.no_grad():
        # return_attn support for interpretability
        log_probs, out_lens, attn = model(feats, in_lens, return_attn=True)  # type: ignore[arg-type]

    # Decode
    if decode_method == "beam_search":
        tbv = log_probs.permute(1, 0, 2)
        seqs = ctc_beam_search_decode(tbv, lengths=out_lens, blank_idx=0, beam_width=beam_width)
    else:
        pred_ids = log_probs.argmax(dim=-1)
        seqs = ctc_greedy_decode_with_lengths(pred_ids, out_lens, blank_idx=0)

    hyp = ids_to_string(seqs[0], vocab.idx2word)
    # Ref from labels
    ref = " ".join([vocab.idx2word[int(t)] for t in sample["labels"].tolist()])

    string_filter_fn = make_string_filter()
    hyp_f = string_filter_fn(hyp)
    ref_f = string_filter_fn(ref)

    # Confidence timeline
    lp = log_probs[0, : int(out_lens[0].item())].detach().cpu()
    p = lp.exp().clamp_min(1e-8)
    entropy = (-p * lp).sum(dim=-1).numpy()
    top1 = p.max(dim=-1).values.numpy()

    # Attention avg
    Tprime = int(out_lens[0].item())
    attn_avg = attn[0, :, :Tprime, :Tprime].mean(dim=0).detach().cpu().numpy()

    # Segment markers from greedy argmax (for visualization)
    greedy_ids = log_probs[0, :Tprime].argmax(dim=-1).detach().cpu().tolist()
    segs = greedy_segments_from_ids(greedy_ids, vocab.idx2word, blank_id=0)

    return {
        "ref": ref_f,
        "hyp": hyp_f,
        "wer": float(compute_wer([ref_f], [hyp_f])),
        "entropy": entropy,
        "top1": top1,
        "attn_avg": attn_avg,
        "segments": segs,
        "video_id": sample["video_id"],
        "decode_method": decode_method,
        "beam_width": int(beam_width),
    }


def main():
    st.set_page_config(page_title="CSLR Educational Dashboard", layout="wide")
    st.title("Continuous Sign Language Recognition: Educational Dashboard")
    st.caption("Student Model: FeatureBiLSTM-CTC - Inference Demo")

    # Make the page more screenshot-friendly (less padding, tighter spacing).
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
          div[data-testid="stSidebar"] { min-width: 320px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model & Data")
        checkpoint = st.text_input(
            "Checkpoint",
            value="checkpoints/student_featurebilstmctc_adaptsign_official_kd_pv1mask/run_20251225_181935/best.pt",
        )
        features_dir = st.text_input("Features dir", value="data/teacher_features/adaptsign_official")
        split_source = st.selectbox("Split source", ["adaptsign_official", "si5"], index=0)
        adaptsign_preprocess_dir = st.text_input("AdaptSign preprocess dir", value="adaptsign_repo/preprocess")
        adaptsign_dataset = st.text_input("AdaptSign dataset", value="phoenix2014")

        st.header("Decoding")
        decode_method = st.selectbox("Decode", ["beam_search", "greedy"], index=0)
        beam_width = int(st.number_input("Beam width", min_value=1, max_value=100, value=10, step=1))

        st.header("Sample")
        split = st.selectbox("Split", ["dev", "test"], index=0)
        max_show = int(st.number_input("Max samples to list", min_value=50, max_value=2000, value=200, step=50))

    device, model, vocab, make_split, model_stats = load_model_and_data(
        checkpoint, features_dir, split_source, adaptsign_preprocess_dir, adaptsign_dataset
    )
    ds = make_split(split)

    # Dashboard header: model card + dataset
    h1, h2, h3, h4 = st.columns([1.2, 1.2, 1.2, 1.4])
    h1.metric("Device", str(device))
    h2.metric("Model params", f"{model_stats['num_params']:,}")
    h3.metric("Model size (fp32)", f"{model_stats['param_fp32_mb']:.1f} MB", help="Parameter tensors only (not optimizer/EMA).")
    h4.metric(
        "Training ckpt size",
        f"{model_stats['checkpoint_mb']:.1f} MB",
        help="On-disk training checkpoint (includes EMA + optimizer state).",
    )

    st.caption(
        f"Split: **{split}** | Samples: **{len(ds)}** | "
        f"Deployable fp16 weights estimate: **{model_stats['param_fp16_mb']:.1f} MB** | "
        f"Target constraint: **< 100MB**"
    )

    # Build selector
    ids = [s["video_id"] for s in ds.samples[:max_show]]
    vid = st.selectbox("Select a video_id (subset list)", ids)
    idx = next(i for i, s in enumerate(ds.samples) if s["video_id"] == vid)
    sample = ds[idx]

    if st.button("Run inference", type="primary"):
        out = decode(model, vocab, device, sample, decode_method, beam_width)

        # Export block (single screenshot-friendly composite)
        with st.expander("Export (screenshot-friendly)", expanded=True):
            col_a, col_b, col_c = st.columns([1.1, 1.0, 1.0])
            with col_a:
                st.write("**Tip**: Use the downloads below to avoid multiple screenshots.")
            with col_b:
                export_side = int(st.selectbox("Export attention max side", [192, 256, 384], index=1))
            with col_c:
                export_segments = st.checkbox("Export with segment boundaries", value=True)

            include_seg_table = st.checkbox("Include predicted segments table", value=True)

            export_fig = _make_export_figure(
                out,
                show_segments=export_segments,
                max_side=export_side,
                include_segments_table=include_seg_table,
            )
            export_png = _fig_to_png_bytes(export_fig)
            fname = f"{out['video_id']}_dashboard_export.png"
            st.download_button(
                "Download composite PNG (REF/HYP + confidence + attention)",
                data=export_png,
                file_name=fname,
                mime="image/png",
                use_container_width=True,
            )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Prediction")
            st.write(f"**video_id**: `{out['video_id']}`")
            # `compute_wer` returns percentage (0..100). Define ACC as (100 - WER) for dashboard reporting.
            wer_pct = float(out["wer"])
            acc_pct = float(max(0.0, 100.0 - wer_pct))
            m1, m2 = st.columns(2)
            m1.metric("WER (sample)", f"{wer_pct:.1f}%")
            m2.metric("ACC (sample)", f"{acc_pct:.1f}%", help="Word accuracy proxy: ACC = 100 − WER (capped at 0).")
            st.write("**REF**")
            st.code(out["ref"])
            st.write("**HYP**")
            st.code(out["hyp"])

            seg_df = pd.DataFrame(out["segments"], columns=["token", "start_t", "end_t"])
            st.write("**Predicted segments (greedy, for visualization)**")
            st.dataframe(seg_df, use_container_width=True, height=220)

        with c2:
            st.subheader("Confidence over time")
            conf_df = pd.DataFrame({"entropy": out["entropy"], "top1_prob": out["top1"]})
            st.line_chart(conf_df, height=260)

            st.subheader("Attention (avg over heads)")
            attn = out["attn_avg"]
            T = int(attn.shape[0])

            # Controls: overview downsampling + zoom window
            show_segments = st.checkbox("Overlay predicted segment boundaries", value=True)
            max_side = int(st.selectbox("Overview size (downsample)", [128, 192, 256, 384], index=2))
            zoom = st.checkbox("Zoom into a time window", value=False)

            if zoom:
                start, end = st.slider(
                    "Zoom range (time indices in T')",
                    min_value=0,
                    max_value=max(1, T - 1),
                    value=(0, min(T - 1, 96)),
                    step=1,
                )
                end = max(start + 1, end)
                attn_view = attn[start:end, start:end]
                segs = [(tok, s - start, e - start) for tok, s, e in out["segments"] if start <= s < end]
                title = f"Self-attention (avg heads) — zoom [{start}:{end}] / T'={T}"
                fig_size = (6.8, 6.0)
            else:
                # Downsample for overview so the entire map is visible at once.
                if T > max_side:
                    step = int(np.ceil(T / max_side))
                    attn_view = attn[::step, ::step]
                    segs = [(tok, s // step, e // step) for tok, s, e in out["segments"]]
                    title = f"Self-attention (avg heads) — overview (downsample x{step})"
                else:
                    attn_view = attn
                    segs = out["segments"]
                    title = "Self-attention (avg heads) — overview"
                fig_size = (6.8, 5.6)

            # Render with matplotlib for reliable display + publication-ready colormap.
            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            im = ax.imshow(attn_view, cmap="viridis", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("Key time")
            ax.set_ylabel("Query time")

            if show_segments:
                for _tok, s, _e in segs:
                    if 0 <= s < attn_view.shape[0]:
                        ax.axvline(s, color="white", linewidth=0.6, alpha=0.7)
                        ax.axhline(s, color="white", linewidth=0.6, alpha=0.7)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig, clear_figure=True)
            st.caption(
                f"Raw shape: {attn.shape} | view shape: {attn_view.shape} | "
                f"min={float(attn_view.min()):.4g}, max={float(attn_view.max()):.4g}"
            )


if __name__ == "__main__":
    main()


