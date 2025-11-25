"""
MediaPipe Feature Extraction Script
Extracts pose, hand, and face landmarks from sign language videos
Computes temporal features (velocity, acceleration) and spatial relationships
Saves features as .npy files for training
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import sys
import os




# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """Extract all landmarks from MediaPipe results."""

    # Pose landmarks (33 keypoints x 3 coordinates)
    pose = np.zeros(33 * 3)
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            pose[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Face landmarks (468 keypoints x 3 coordinates)
    face = np.zeros(468 * 3)
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            face[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Left hand landmarks (21 keypoints x 3 coordinates)
    left_hand = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        for i, landmark in enumerate(results.left_hand_landmarks.landmark):
            left_hand[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Right hand landmarks (21 keypoints x 3 coordinates)
    right_hand = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        for i, landmark in enumerate(results.right_hand_landmarks.landmark):
            right_hand[i*3:(i+1)*3] = [landmark.x, landmark.y, landmark.z]

    # Concatenate all landmarks
    # Total: 33 + 468 + 21 + 21 = 543 keypoints x 3 = 1,629 spatial features
    spatial_features = np.concatenate([pose, face, left_hand, right_hand])

    return spatial_features


def compute_temporal_features(features_sequence):
    """
    Compute velocity and acceleration from spatial features.

    Args:
        features_sequence: numpy array of shape (T, D) where T is frames and D is features

    Returns:
        enhanced_features: numpy array with spatial, velocity, and acceleration features
    """
    T, D = features_sequence.shape

    # Initialize arrays
    velocities = np.zeros_like(features_sequence)
    accelerations = np.zeros_like(features_sequence)

    # Compute velocities (first-order derivatives)
    if T > 1:
        velocities[1:] = features_sequence[1:] - features_sequence[:-1]
        velocities[0] = velocities[1]  # Replicate first velocity

    # Compute accelerations (second-order derivatives)
    if T > 2:
        accelerations[2:] = velocities[2:] - velocities[1:-1]
        accelerations[0] = accelerations[2]
        accelerations[1] = accelerations[2]

    # Concatenate all features: spatial + velocity + acceleration
    # Total: 1,629 * 3 = 4,887 features + original 1,629 = 6,516 features per frame
    enhanced_features = np.concatenate([
        features_sequence,  # Original spatial features (1,629)
        velocities,         # Velocity features (1,629)
        accelerations,      # Acceleration features (1,629)
        features_sequence   # Duplicate for compatibility (1,629)
    ], axis=1)

    return enhanced_features


def process_video(video_path, holistic):
    """
    Process a single video to extract MediaPipe features.

    Args:
        video_path: Path to video file
        holistic: MediaPipe Holistic instance

    Returns:
        features: numpy array of shape (T, 6516) with all features
    """
    cap = cv2.VideoCapture(str(video_path))
    features_list = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = holistic.process(rgb_frame)

        # Extract landmarks
        spatial_features = extract_landmarks(results)
        features_list.append(spatial_features)

        frame_count += 1

    cap.release()

    if len(features_list) == 0:
        logger.warning(f"No frames extracted from {video_path}")
        return None

    # Convert to numpy array
    features_sequence = np.array(features_list, dtype=np.float32)

    # Compute temporal features
    enhanced_features = compute_temporal_features(features_sequence)

    logger.info(f"Processed {video_path.name}: {frame_count} frames, shape {enhanced_features.shape}")

    return enhanced_features


def extract_features_from_videos(video_dir, output_dir, annotation_file=None):
    """
    Extract MediaPipe features from all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_dir: Directory to save extracted features
        annotation_file: Optional CSV file with video annotations
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'*{ext}')))
        video_files.extend(list(video_dir.glob(f'**/*{ext}')))  # Recursive search

    if len(video_files) == 0:
        logger.error(f"No video files found in {video_dir}")

        # Check if the directory structure is different
        logger.info("Checking for alternative directory structures...")
        subdirs = ['train', 'dev', 'test', 'val']
        for subdir in subdirs:
            subdir_path = video_dir / subdir
            if subdir_path.exists():
                for ext in video_extensions:
                    sub_videos = list(subdir_path.glob(f'*{ext}'))
                    if sub_videos:
                        logger.info(f"Found {len(sub_videos)} videos in {subdir_path}")
                        video_files.extend(sub_videos)

    if len(video_files) == 0:
        logger.error("No video files found. Please ensure videos are downloaded.")
        logger.info("\nTo download RWTH-PHOENIX-Weather 2014 dataset:")
        logger.info("1. Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/")
        logger.info("2. Download the 'Phoenix2014 Full Frame Videos' (~53GB)")
        logger.info("3. Extract to: data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/")
        return

    logger.info(f"Found {len(video_files)} video files to process")

    # Initialize MediaPipe
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,  # Use most accurate model
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Process each video
    success_count = 0
    failed_videos = []

    for video_path in tqdm(video_files, desc="Extracting features"):
        try:
            # Extract features
            features = process_video(video_path, holistic)

            if features is not None:
                # Save features
                output_path = output_dir / f"{video_path.stem}.npy"
                np.save(output_path, features)
                success_count += 1
            else:
                failed_videos.append(video_path.name)

        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            failed_videos.append(video_path.name)

    holistic.close()

    # Report results
    logger.info(f"\nExtraction complete!")
    logger.info(f"Successfully processed: {success_count}/{len(video_files)} videos")
    logger.info(f"Features saved to: {output_dir}")

    if failed_videos:
        logger.warning(f"Failed videos ({len(failed_videos)}): {failed_videos[:5]}...")

        # Save failed list for debugging
        failed_file = output_dir / "failed_videos.txt"
        with open(failed_file, 'w') as f:
            for video in failed_videos:
                f.write(f"{video}\n")
        logger.info(f"Failed video list saved to: {failed_file}")

    # Save feature statistics
    logger.info("\nComputing feature statistics...")
    all_features = []
    for npy_file in list(output_dir.glob("*.npy"))[:100]:  # Sample first 100 files
        features = np.load(npy_file)
        all_features.append(features)

    if all_features:
        all_features = np.concatenate(all_features, axis=0)
        stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + 1e-6,
            'shape': all_features.shape,
            'num_files': success_count
        }

        # Save statistics
        import pickle
        stats_file = output_dir.parent / "feature_stats.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f"Feature statistics saved to: {stats_file}")
        logger.info(f"Feature dimensions: {all_features.shape}")


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe features from sign language videos')

    parser.add_argument('--video_dir', type=str,
                        default='data/raw_data/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str,
                        default='data/features_enhanced',
                        help='Directory to save extracted features')
    parser.add_argument('--annotation_file', type=str, default=None,
                        help='Optional: Path to annotation CSV file')

    args = parser.parse_args()

    logger.info("Starting MediaPipe feature extraction...")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Check if video directory exists
    if not Path(args.video_dir).exists():
        logger.error(f"Video directory does not exist: {args.video_dir}")
        logger.info("\nPlease ensure the RWTH-PHOENIX dataset is downloaded and extracted.")
        sys.exit(1)

    # Extract features
    extract_features_from_videos(args.video_dir, args.output_dir, args.annotation_file)

    logger.info("\nFeature extraction complete!")
    logger.info("You can now run training with:")
    logger.info("  python src/training/train.py")
    logger.info("  python src/training/train_teacher.py")


if __name__ == "__main__":
    main()