"""
Dashboard for Continuous Sign Language Recognition (CSLR).

Design goals:
- clean, thesis-friendly UI (no icons/emojis)
- lightweight visualizations for benchmarking + qualitative inspection
- attention heatmap viewer (loads precomputed artifacts under figures/)
"""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List

# Page config
st.set_page_config(
    page_title="Sign Language Recognition Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: bold;
        margin: 0.25rem 0 1.25rem 0;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.04);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load model from checkpoint (placeholder - implement based on your model loading)."""
    try:
        # This is a placeholder - implement based on your actual model loading code
        # Example: from src.evaluation.evaluate_student import load_model_from_checkpoint
        # model, vocab, args, meta = load_model_from_checkpoint(checkpoint_path, device)
        st.warning("Model loading not yet implemented. Please implement based on your model structure.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def main():
    st.markdown('<div class="main-header">Sign Language Recognition Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Inspection", "Attention", "Inference", "Metrics & Analysis"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Model Inspection":
        show_model_inspection()
    elif page == "Attention":
        show_attention()
    elif page == "Inference":
        show_inference_demo()
    elif page == "Metrics & Analysis":
        show_metrics_analysis()


def _list_pngs(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob("*.png") if p.is_file()])


def _extract_video_id_from_filename(p: Path) -> str:
    # expected: dev_00_<video_id>_attn_avg.png OR dev_00_<video_id>_attn_heads.png
    name = p.stem
    if "_attn_" in name:
        return name.split("_attn_")[0]
    return name


def show_attention():
    """Attention artifact viewer (precomputed PNGs)."""
    st.header("Attention")

    base = Path("figures")
    avg_dir = base / "attention_kd_best_avg"
    heads_dir = base / "attention_kd_best"

    colL, colR = st.columns([1, 2])
    with colL:
        mode = st.selectbox(
            "Artifact type",
            ["Average attention (single heatmap)", "Per-head attention (grid)"],
        )
        query = st.text_input("Filter (substring)", value="", help="Filter by filename/video_id substring.")

        dir_path = avg_dir if mode.startswith("Average") else heads_dir
        files = _list_pngs(dir_path)
        if query.strip():
            q = query.strip().lower()
            files = [p for p in files if q in p.name.lower()]

        if not files:
            st.warning(f"No attention PNGs found in `{dir_path.as_posix()}`.")
            st.caption("Expected files like: dev_00_<video_id>_attn_avg.png")
            return

        labels = [f"{_extract_video_id_from_filename(p)}" for p in files]
        idx = st.selectbox("Sample", list(range(len(files))), format_func=lambda i: labels[i])
        selected = files[int(idx)]
        st.caption(f"File: `{selected.as_posix()}`")

    with colR:
        st.image(str(selected), use_container_width=True)
        st.caption("Precomputed attention visualization loaded from disk.")


def show_overview():
    """Overview page with project information."""
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target WER", "< 25%", help="Word Error Rate target for deployment")
    
    with col2:
        st.metric("Model Size", "< 100MB", help="Target model size constraint")
    
    with col3:
        st.metric("Inference Speed", "> 30 FPS", help="Real-time performance target")
    
    st.markdown("---")
    
    st.subheader("About This System")
    st.markdown("""
    This is a **computationally efficient, real-time continuous sign language recognition (CSLR)** system
    designed for educational accessibility.
    
    **Key Features:**
    - Efficient feature extraction from video sequences
    - Lightweight neural network architecture (BiLSTM + CTC)
    - Knowledge distillation from teacher models
    - Real-time inference capabilities
    
    **Dataset:** RWTH-PHOENIX-Weather 2014
    """)
    
    st.markdown("---")
    
    st.subheader("Quick Start")
    st.code("""
# Run the streaming demo
python -m scripts.phase3_streaming_demo \\
    --checkpoint checkpoints/your_model/best.pt \\
    --features_dir data/teacher_features/adaptsign_official \\
    --split dev

# Run benchmark
python -m scripts.phase3_benchmark \\
    --checkpoint checkpoints/your_model/best.pt \\
    --output_json figures/benchmark.json
    """, language="bash")


def show_model_inspection():
    """Model inspection page."""
    st.header("Model Inspection")
    
    checkpoint_dir = st.text_input(
        "Checkpoint Directory",
        value="checkpoints/",
        help="Path to checkpoint directory"
    )
    
    if st.button("Scan for Checkpoints"):
        checkpoints_dir = Path(checkpoint_dir)
        if checkpoints_dir.exists():
            # Find all .pt files
            checkpoint_files = list(checkpoints_dir.rglob("*.pt"))
            checkpoint_files = [str(f) for f in checkpoint_files if "best.pt" in str(f) or "checkpoint" in str(f).lower()]
            
            if checkpoint_files:
                st.success(f"Found {len(checkpoint_files)} checkpoint(s)")
                selected_checkpoint = st.selectbox("Select Checkpoint", checkpoint_files)
                
                if st.button("Load Checkpoint"):
                    with st.spinner("Loading checkpoint..."):
                        try:
                            ckpt = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
                            
                            st.subheader("Checkpoint Information")
                            
                            # Display checkpoint metadata
                            if "args" in ckpt:
                                st.json(ckpt["args"])
                            
                            if "best_wer" in ckpt:
                                st.metric("Best WER", f"{ckpt['best_wer']:.2f}%")
                            
                            if "epoch" in ckpt:
                                st.metric("Epoch", ckpt["epoch"])
                            
                            # Model state dict info
                            if "model_state_dict" in ckpt:
                                st.success("Model weights found")
                                num_params = sum(p.numel() for p in ckpt["model_state_dict"].values())
                                st.metric("Total Parameters", f"{num_params:,}")
                            
                            if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
                                st.info("EMA weights available")
                                
                        except Exception as e:
                            st.error(f"Error loading checkpoint: {e}")
            else:
                st.warning("No checkpoint files found. Looking for *.pt files with 'best' or 'checkpoint' in name.")
        else:
            st.error(f"Directory not found: {checkpoint_dir}")


def show_inference_demo():
    """Inference demonstration page."""
    st.header("Inference")
    
    st.info("""
    This page allows you to run inference on sample data.
    **Note:** Full implementation requires model loading and feature extraction setup.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Selection")
        input_type = st.radio(
            "Input Type",
            ["Pre-extracted Features"],
            help="Select input type for inference"
        )
        
        if input_type == "Pre-extracted Features":
            features_dir = st.text_input(
                "Features Directory",
                value="data/teacher_features/adaptsign_official",
                help="Directory containing pre-extracted feature files (.npz)"
            )
    
    with col2:
        st.subheader("Model Selection")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoints/student_featurebilstmctc_adaptsign_official_kd_pv1mask/run_20251225_181935/best.pt",
            help="Path to model checkpoint"
        )
        
        decode_method = st.selectbox(
            "Decode Method",
            ["greedy", "beam_search"],
            help="CTC decoding method"
        )
        
        if decode_method == "beam_search":
            beam_width = st.slider("Beam Width", 1, 20, 10)
    
    if st.button("Run Inference", type="primary"):
        st.warning(
            "Inference wiring is not implemented in this rebuilt dashboard yet. "
            "If you want it to match your previous UI (WER/ACC + REF/HYP + confidence-over-time + segments), "
            "tell me where those outputs are saved (JSON/CTM/log), and I’ll hook the dashboard to them."
        )


def show_metrics_analysis():
    """Metrics and analysis page."""
    st.header("Metrics & Analysis")
    
    # Load benchmark results if available
    benchmark_file = st.text_input(
        "Benchmark JSON File",
        value="figures/phase3_benchmark_dev.json",
        help="Path to benchmark results JSON file"
    )
    
    if st.button("Load Benchmark Results"):
        benchmark_path = Path(benchmark_file)
        if benchmark_path.exists():
            try:
                with open(benchmark_path, 'r') as f:
                    benchmark_data = json.load(f)
                
                st.success("Benchmark data loaded successfully!")
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if "rtf_assuming_25fps" in benchmark_data:
                        st.metric("RTF", f"{benchmark_data['rtf_assuming_25fps']:.3f}")
                
                with col2:
                    if "fps_feature_frames_per_s" in benchmark_data:
                        st.metric("FPS", f"{benchmark_data['fps_feature_frames_per_s']:.1f}")
                
                with col3:
                    if "total_time_s" in benchmark_data:
                        st.metric("Total Time", f"{benchmark_data['total_time_s']:.2f}s")
                
                with col4:
                    # Calculate average WER if rows available
                    if "rows" in benchmark_data and benchmark_data["rows"]:
                        avg_wer = np.mean([r.get("wer", 0) for r in benchmark_data["rows"]])
                        st.metric("Avg WER", f"{avg_wer:.2f}%")
                
                # Timing breakdown
                st.subheader("Timing Breakdown")
                if "rows" in benchmark_data:
                    df = pd.DataFrame(benchmark_data["rows"])
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    # Load time distribution
                    axes[0].hist(df["load_s"], bins=20, edgecolor='black')
                    axes[0].set_title("Feature Load Time (s)")
                    axes[0].set_xlabel("Time (s)")
                    axes[0].set_ylabel("Frequency")
                    
                    # Forward time distribution
                    axes[1].hist(df["forward_s"], bins=20, edgecolor='black', color='orange')
                    axes[1].set_title("Model Forward Time (s)")
                    axes[1].set_xlabel("Time (s)")
                    axes[1].set_ylabel("Frequency")
                    
                    # Decode time distribution
                    axes[2].hist(df["decode_s"], bins=20, edgecolor='black', color='green')
                    axes[2].set_title("Decode Time (s)")
                    axes[2].set_xlabel("Time (s)")
                    axes[2].set_ylabel("Frequency")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # WER distribution
                    st.subheader("WER Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(df["wer"], bins=30, edgecolor='black', color='red', alpha=0.7)
                    ax.axvline(df["wer"].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["wer"].mean():.2f}%')
                    ax.set_xlabel("WER (%)")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Word Error Rate Distribution")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error loading benchmark data: {e}")
        else:
            st.warning(f"Benchmark file not found: {benchmark_file}")
            st.info("Run `scripts/phase3_benchmark.py` to generate benchmark results.")


if __name__ == "__main__":
    main()

