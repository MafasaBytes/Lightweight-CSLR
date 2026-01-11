"""
PCA Transform Utility for MediaPipe Features

Based on analysis from notebooks/analyze_features.ipynb:
- MediaPipe features: 6516 dimensions
- Massive redundancy: 99.0% variance captured by 100 components
- Compression ratios:
  - 100 components: 65.2x compression, 99.0% variance
  - 200 components: 32.6x compression, 99.8% variance
  - 300 components: 21.7x compression, 99.9% variance

This utility provides:
1. Fit PCA on training data
2. Transform features (both NPZ files and in-memory)
3. Save/load PCA models
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict
import json


class MediaPipePCATransform:
    """PCA transformation for MediaPipe features."""

    def __init__(self, n_components: int = 1024, whiten: bool = True):
        """
        Initialize PCA transform.

        Args:
            n_components: Number of PCA components to keep
            whiten: Whether to whiten the components (recommended)
        """
        self.n_components = n_components
        self.whiten = whiten
        self.pca = None
        self.scaler = None
        self.fitted = False

        # Statistics
        self.input_dim = None
        self.explained_variance_ratio = None
        self.total_variance_explained = None
        self.compression_ratio = None

    def fit(self, features: np.ndarray, verbose: bool = True):
        """
        Fit PCA on training features.

        Args:
            features: Training features (N_frames, 6516)
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Fitting PCA on {features.shape[0]} frames, {features.shape[1]} features...")

        self.input_dim = features.shape[1]

        # Standardize features first
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        if verbose:
            print(f"Features standardized. Mean: {np.mean(features_scaled):.4f}, Std: {np.std(features_scaled):.4f}")

        # Fit PCA
        self.pca = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca.fit(features_scaled)

        # Store statistics
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.total_variance_explained = np.sum(self.explained_variance_ratio)
        self.compression_ratio = self.input_dim / self.n_components
        self.fitted = True

        if verbose:
            print(f"\nPCA Fit Complete:")
            print(f"  Input dimensions: {self.input_dim}")
            print(f"  Output dimensions: {self.n_components}")
            print(f"  Compression ratio: {self.compression_ratio:.2f}x")
            print(f"  Total variance explained: {self.total_variance_explained * 100:.2f}%")
            print(f"  Top 10 components variance: {np.sum(self.explained_variance_ratio[:10]) * 100:.2f}%")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted PCA.

        Args:
            features: Input features (N_frames, 6516)

        Returns:
            Transformed features (N_frames, n_components)
        """
        if not self.fitted:
            raise ValueError("PCA not fitted yet. Call fit() first.")

        # Standardize and transform
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)

        return features_pca

    def fit_transform(self, features: np.ndarray, verbose: bool = True) -> np.ndarray:
        """
        Fit PCA and transform in one step.

        Args:
            features: Training features (N_frames, 6516)
            verbose: Whether to print progress

        Returns:
            Transformed features (N_frames, n_components)
        """
        self.fit(features, verbose=verbose)
        return self.transform(features)

    def save(self, save_path: str):
        """
        Save fitted PCA model.

        Args:
            save_path: Path to save the model (will save as .pkl)
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted PCA model.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'n_components': self.n_components,
            'whiten': self.whiten,
            'input_dim': self.input_dim,
            'explained_variance_ratio': self.explained_variance_ratio,
            'total_variance_explained': self.total_variance_explained,
            'compression_ratio': self.compression_ratio
        }

        joblib.dump(model_data, save_path)
        print(f"PCA model saved to: {save_path}")

        # Also save metadata as JSON
        metadata = {
            'n_components': int(self.n_components),
            'whiten': bool(self.whiten),
            'input_dim': int(self.input_dim),
            'total_variance_explained': float(self.total_variance_explained),
            'compression_ratio': float(self.compression_ratio)
        }

        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {json_path}")

    @classmethod
    def load(cls, load_path: str) -> 'MediaPipePCATransform':
        """
        Load fitted PCA model.

        Args:
            load_path: Path to the saved model (.pkl)

        Returns:
            Loaded MediaPipePCATransform instance
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        model_data = joblib.load(load_path)

        # Create instance
        instance = cls(
            n_components=model_data['n_components'],
            whiten=model_data['whiten']
        )

        # Restore fitted components
        instance.pca = model_data['pca']
        instance.scaler = model_data['scaler']
        instance.input_dim = model_data['input_dim']
        instance.explained_variance_ratio = model_data['explained_variance_ratio']
        instance.total_variance_explained = model_data['total_variance_explained']
        instance.compression_ratio = model_data['compression_ratio']
        instance.fitted = True

        print(f"PCA model loaded from: {load_path}")
        print(f"  Components: {instance.n_components}")
        print(f"  Compression: {instance.compression_ratio:.2f}x")
        print(f"  Variance explained: {instance.total_variance_explained * 100:.2f}%")

        return instance


def fit_pca_from_npz_directory(
    npz_dir: str,
    n_components: int = 1024,
    max_samples: int = 500,
    save_path: Optional[str] = None
) -> MediaPipePCATransform:
    """
    Fit PCA on MediaPipe features from NPZ directory.

    Args:
        npz_dir: Directory containing .npz files
        n_components: Number of PCA components
        max_samples: Maximum number of samples to use for fitting
        save_path: Path to save the fitted model (optional)

    Returns:
        Fitted MediaPipePCATransform instance
    """
    npz_dir = Path(npz_dir)
    npz_files = list(npz_dir.glob('*.npz'))

    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    print(f"Found {len(npz_files)} NPZ files in {npz_dir}")
    print(f"Loading up to {max_samples} samples for PCA fitting...")

    # Load features from NPZ files
    all_features = []
    for i, npz_file in enumerate(npz_files[:max_samples]):
        if i % 100 == 0:
            print(f"  Loading sample {i}/{min(max_samples, len(npz_files))}...")

        data = np.load(npz_file, allow_pickle=True)
        features = data['features']
        all_features.append(features)

    # Concatenate all frames
    all_features = np.concatenate(all_features, axis=0)
    print(f"\nTotal frames collected: {all_features.shape[0]}")
    print(f"Feature dimensions: {all_features.shape[1]}")

    # Fit PCA
    pca_transform = MediaPipePCATransform(n_components=n_components, whiten=True)
    pca_transform.fit(all_features, verbose=True)

    # Save if requested
    if save_path:
        pca_transform.save(save_path)

    return pca_transform


def transform_npz_file(
    npz_file: str,
    pca_transform: MediaPipePCATransform,
    output_file: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Transform a single NPZ file using PCA.

    Args:
        npz_file: Input NPZ file path
        pca_transform: Fitted PCA transform
        output_file: Output NPZ file path (optional, will overwrite input if None)

    Returns:
        Dictionary with transformed features
    """
    npz_file = Path(npz_file)

    # Load original data
    data = np.load(npz_file, allow_pickle=True)
    features = data['features']

    # Transform features
    features_pca = pca_transform.transform(features)

    # Create new data dictionary
    new_data = {
        'features': features_pca,
        'detection_masks': data['detection_masks'] if 'detection_masks' in data else None,
        'metadata': data['metadata'] if 'metadata' in data else None,
        'video_id': data['video_id'] if 'video_id' in data else npz_file.stem
    }

    # Remove None values
    new_data = {k: v for k, v in new_data.items() if v is not None}

    # Save if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_file, **new_data)

    return new_data


def transform_npz_directory(
    input_dir: str,
    output_dir: str,
    pca_transform: MediaPipePCATransform,
    verbose: bool = True
):
    """
    Transform all NPZ files in a directory using PCA.

    Args:
        input_dir: Input directory with original NPZ files
        output_dir: Output directory for transformed NPZ files
        pca_transform: Fitted PCA transform
        verbose: Whether to print progress
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list(input_dir.glob('*.npz'))

    if verbose:
        print(f"Transforming {len(npz_files)} NPZ files...")
        print(f"  Input: {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Compression: {pca_transform.compression_ratio:.2f}x")

    for i, npz_file in enumerate(npz_files):
        if verbose and i % 100 == 0:
            print(f"  Processing {i}/{len(npz_files)}...")

        output_file = output_dir / npz_file.name
        transform_npz_file(npz_file, pca_transform, output_file)

    if verbose:
        print(f"\nTransformation complete! {len(npz_files)} files processed.")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fit and apply PCA to MediaPipe features')
    parser.add_argument('--fit', action='store_true', help='Fit PCA on training data')
    parser.add_argument('--transform', action='store_true', help='Transform NPZ files')
    parser.add_argument('--train-dir', type=str, default='data/teacher_features/mediapipe_full/train',
                        help='Training NPZ directory')
    parser.add_argument('--input-dir', type=str, help='Input NPZ directory to transform')
    parser.add_argument('--output-dir', type=str, help='Output NPZ directory')
    parser.add_argument('--n-components', type=int, default=1024,
                        help='Number of PCA components (default: 1024)')
    parser.add_argument('--max-samples', type=int, default=500,
                        help='Max samples for fitting (default: 500)')
    parser.add_argument('--model-path', type=str, default='models/mediapipe_pca_1024.pkl',
                        help='Path to save/load PCA model')

    args = parser.parse_args()

    if args.fit:
        print("="*80)
        print("FITTING PCA ON MEDIAPIPE FEATURES")
        print("="*80)

        pca_transform = fit_pca_from_npz_directory(
            npz_dir=args.train_dir,
            n_components=args.n_components,
            max_samples=args.max_samples,
            save_path=args.model_path
        )

        print("\n" + "="*80)
        print("PCA MODEL FITTED AND SAVED")
        print("="*80)
        print(f"\nTo transform data, run:")
        print(f"python utils/pca_transform.py --transform \\")
        print(f"  --input-dir <input_dir> \\")
        print(f"  --output-dir <output_dir> \\")
        print(f"  --model-path {args.model_path}")

    elif args.transform:
        if not args.input_dir or not args.output_dir:
            raise ValueError("--input-dir and --output-dir required for transform")

        print("="*80)
        print("TRANSFORMING MEDIAPIPE FEATURES WITH PCA")
        print("="*80)

        # Load PCA model
        pca_transform = MediaPipePCATransform.load(args.model_path)

        # Transform directory
        transform_npz_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pca_transform=pca_transform,
            verbose=True
        )

        print("\n" + "="*80)
        print("TRANSFORMATION COMPLETE")
        print("="*80)

    else:
        print("Usage:")
        print("  Fit PCA: python utils/pca_transform.py --fit")
        print("  Transform: python utils/pca_transform.py --transform --input-dir <dir> --output-dir <dir>")
        print("\nFor help: python utils/pca_transform.py --help")
