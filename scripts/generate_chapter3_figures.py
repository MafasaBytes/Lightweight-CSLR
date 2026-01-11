"""
Generate thesis-ready comparison figures for Chapter 3:
- Figure 3.4: Feature Value Distribution Comparison (MediaPipe vs AdaptSign)
- Figure 3.5: Per-Dimension Variance Distribution (MediaPipe vs AdaptSign)

This script loads feature data from both sources and creates publication-quality comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Create output directory
output_dir = project_root / 'figures' / 'thesis_chapter3'
output_dir.mkdir(parents=True, exist_ok=True)


def load_mediapipe_features(data_root: Path, num_samples: int = 200):
    """Load MediaPipe features for analysis."""
    mediapipe_dir = data_root / 'teacher_features' / 'mediapipe_full' / 'train'
    feature_files = list(mediapipe_dir.glob('*.npz'))[:num_samples]
    
    all_features = []
    for feature_file in feature_files:
        data = np.load(feature_file, allow_pickle=True)
        features = data['features']
        all_features.append(features)
    
    return np.concatenate(all_features, axis=0)  # (total_frames, 6516)


def load_adaptsign_features(data_root: Path, num_samples: int = 500):
    """Load AdaptSign features for analysis."""
    adaptsign_dir = data_root / 'teacher_features' / 'adaptsign_official' / 'train'
    feature_files = list(adaptsign_dir.glob('*.npz'))[:num_samples]
    
    all_features = []
    for feature_file in feature_files:
        data = np.load(feature_file, allow_pickle=False)
        x = data['features'] if 'features' in data.files else data['arr_0']
        x = x.astype(np.float32, copy=False)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(x)
    
    return np.concatenate(all_features, axis=0)  # (total_frames, 512)


def compute_feature_statistics(features: np.ndarray):
    """Compute per-dimension statistics."""
    feature_dim = features.shape[1]
    
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    vars_ = np.var(features, axis=0)
    
    return {
        'means': means,
        'stds': stds,
        'vars': vars_,
        'all_features': features,
    }


def generate_figure_3_4(mp_features: np.ndarray, as_features: np.ndarray, output_path: Path):
    """
    Figure 3.4: Feature Value Distribution Comparison
    Histogram comparing MediaPipe vs AdaptSign feature value distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample for visualization (to avoid memory issues)
    mp_sample = mp_features[:20000].flatten()
    as_sample = as_features[:20000].flatten()
    
    # Left: MediaPipe
    axes[0].hist(mp_sample, bins=100, edgecolor='black', alpha=0.75, color='#3498db')
    axes[0].set_title('MediaPipe (6,516-D) Feature Value Distribution', fontweight='bold')
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    mp_mean = float(np.mean(mp_sample))
    mp_std = float(np.std(mp_sample))
    axes[0].text(0.05, 0.95, f'Mean: {mp_mean:.4f}\nStd: {mp_std:.4f}', 
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: AdaptSign
    axes[1].hist(as_sample, bins=100, edgecolor='black', alpha=0.75, color='#e74c3c')
    axes[1].set_title('AdaptSign (512-D) Feature Value Distribution', fontweight='bold')
    axes[1].set_xlabel('Feature Value')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    as_mean = float(np.mean(as_sample))
    as_std = float(np.std(as_sample))
    axes[1].text(0.05, 0.95, f'Mean: {as_mean:.4f}\nStd: {as_std:.4f}', 
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved Figure 3.4 to: {output_path}")
    plt.close()


def generate_figure_3_5(mp_stats: dict, as_stats: dict, output_path: Path):
    """
    Figure 3.5: Per-Dimension Variance Distribution
    Histogram comparing variance distribution for MediaPipe vs AdaptSign features.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: MediaPipe
    mp_vars = mp_stats['vars']
    axes[0].hist(mp_vars, bins=80, edgecolor='black', alpha=0.75, color='#3498db')
    axes[0].set_yscale('log')
    axes[0].set_title('MediaPipe (6,516-D) Per-Dimension Variance', fontweight='bold')
    axes[0].set_xlabel('Variance')
    axes[0].set_ylabel('Count (log scale)')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mp_mean_var = float(np.mean(mp_vars))
    mp_median_var = float(np.median(mp_vars))
    mp_low_var_count = int(np.sum(mp_vars < 0.001))
    axes[0].text(0.05, 0.95, 
                 f'Mean: {mp_mean_var:.6f}\nMedian: {mp_median_var:.6f}\nLow var (<0.001): {mp_low_var_count}',
                 transform=axes[0].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right: AdaptSign
    as_vars = as_stats['vars']
    axes[1].hist(as_vars, bins=80, edgecolor='black', alpha=0.75, color='#e74c3c')
    axes[1].set_yscale('log')
    axes[1].set_title('AdaptSign (512-D) Per-Dimension Variance', fontweight='bold')
    axes[1].set_xlabel('Variance')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    as_mean_var = float(np.mean(as_vars))
    as_median_var = float(np.median(as_vars))
    as_low_var_count = int(np.sum(as_vars < 0.001))
    axes[1].text(0.05, 0.95, 
                 f'Mean: {as_mean_var:.6f}\nMedian: {as_median_var:.6f}\nLow var (<0.001): {as_low_var_count}',
                 transform=axes[1].transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved Figure 3.5 to: {output_path}")
    plt.close()


def main():
    data_root = project_root / 'data'
    
    print("Loading MediaPipe features...")
    try:
        mp_features = load_mediapipe_features(data_root, num_samples=200)
        print(f"Loaded MediaPipe: {mp_features.shape}")
        mp_stats = compute_feature_statistics(mp_features)
    except Exception as e:
        print(f"Warning: Could not load MediaPipe features: {e}")
        print("MediaPipe features may not be available. Skipping MediaPipe analysis.")
        mp_features = None
        mp_stats = None
    
    print("Loading AdaptSign features...")
    as_features = load_adaptsign_features(data_root, num_samples=500)
    print(f"Loaded AdaptSign: {as_features.shape}")
    as_stats = compute_feature_statistics(as_features)
    
    # Generate figures
    if mp_features is not None and mp_stats is not None:
        print("\nGenerating Figure 3.4: Feature Value Distribution Comparison...")
        generate_figure_3_4(mp_features, as_features, output_dir / 'figure_3_4_feature_value_distribution.png')
        
        print("\nGenerating Figure 3.5: Per-Dimension Variance Distribution...")
        generate_figure_3_5(mp_stats, as_stats, output_dir / 'figure_3_5_variance_distribution.png')
    else:
        print("\nSkipping comparison figures (MediaPipe data not available).")
        print("Generating AdaptSign-only figures for reference...")
        
        # Generate AdaptSign-only versions
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        as_sample = as_features[:20000].flatten()
        ax.hist(as_sample, bins=100, edgecolor='black', alpha=0.75, color='#e74c3c')
        ax.set_title('AdaptSign (512-D) Feature Value Distribution', fontweight='bold')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        as_mean = float(np.mean(as_sample))
        as_std = float(np.std(as_sample))
        ax.text(0.05, 0.95, f'Mean: {as_mean:.4f}\nStd: {as_std:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(output_dir / 'adaptsign_feature_value_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        as_vars = as_stats['vars']
        ax.hist(as_vars, bins=80, edgecolor='black', alpha=0.75, color='#e74c3c')
        ax.set_yscale('log')
        ax.set_title('AdaptSign (512-D) Per-Dimension Variance', fontweight='bold')
        ax.set_xlabel('Variance')
        ax.set_ylabel('Count (log scale)')
        ax.grid(True, alpha=0.3, axis='y')
        as_mean_var = float(np.mean(as_vars))
        as_median_var = float(np.median(as_vars))
        ax.text(0.05, 0.95, f'Mean: {as_mean_var:.6f}\nMedian: {as_median_var:.6f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(output_dir / 'adaptsign_variance_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

