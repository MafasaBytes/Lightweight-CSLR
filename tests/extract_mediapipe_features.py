# =============================================================
# MediaPipe Feature Extraction Script (Sequential, No tqdm)
# For Phoenix 2014 Dataset - Windows Compatible
# =============================================================

import os
import sys

# Must be set BEFORE importing mediapipe
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import logging
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("mediapipe_extractor")

# Import mediapipe
logger.info("Importing MediaPipe...")
import mediapipe as mp
mp_holistic = mp.solutions.holistic
logger.info("MediaPipe imported successfully")

# Dimensions
POSE_DIM = 33 * 3
FACE_DIM = 468 * 3
HAND_DIM = 21 * 3


def extract_landmarks(results):
    pose = np.zeros(POSE_DIM)
    face = np.zeros(FACE_DIM)
    lh = np.zeros(HAND_DIM)
    rh = np.zeros(HAND_DIM)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            face[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*3:(i+1)*3] = [lm.x, lm.y, lm.z]

    return np.concatenate([pose, face, lh, rh], axis=0)


def compute_temporal_features(F):
    T, D = F.shape
    vel = np.zeros_like(F)
    acc = np.zeros_like(F)

    if T > 1:
        vel[1:] = F[1:] - F[:-1]
        vel[0] = vel[1]
    if T > 2:
        acc[2:] = vel[2:] - vel[1:-1]
        acc[:2] = acc[2]

    return np.concatenate([F, vel, acc, F], axis=1)


def process_video(video_dir, holistic, resize_to=(256, 256)):
    """Process a single video directory containing frame images."""
    video_dir = Path(video_dir)
    
    if not video_dir.is_dir():
        return None
    
    frame_files = sorted([
        f for f in video_dir.rglob('*')
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])
    
    if len(frame_files) == 0:
        return None
    
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, resize_to)
        results = holistic.process(img)
        frames.append(extract_landmarks(results))
    
    if len(frames) == 0:
        return None
    
    F = np.array(frames, dtype=np.float32)
    F = compute_temporal_features(F)
    return F


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe features')
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resize_width', type=int, default=256)
    parser.add_argument('--resize_height', type=int, default=256)
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], default=None)
    args = parser.parse_args()

    root = Path(args.video_dir)
    out = Path(args.output_dir)
    resize_to = (args.resize_width, args.resize_height)

    logger.info("=" * 60)
    logger.info("MediaPipe Feature Extraction")
    logger.info(f"Input: {root}")
    logger.info(f"Output: {out}")
    logger.info(f"Resize: {resize_to}")
    logger.info("=" * 60)

    # Create holistic ONCE at the start
    logger.info("Initializing MediaPipe Holistic...")
    holistic = mp_holistic.Holistic(
        static_image_mode=True,  # Changed to True for stability
        model_complexity=2,
        refine_face_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    logger.info("Holistic initialized")

    splits = ['train', 'dev', 'test'] if args.split is None else [args.split]

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            logger.warning(f"Split {split} not found at {split_dir}")
            continue

        out_dir = out / split
        out_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        total = len(video_dirs)
        logger.info(f"\n{split.upper()}: Found {total} videos")

        success = 0
        failed = []

        for i, video_dir in enumerate(video_dirs):
            # Progress logging every 10 videos
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  Processing {i+1}/{total}: {video_dir.name}")

            try:
                features = process_video(video_dir, holistic, resize_to)
                if features is not None:
                    np.save(out_dir / f"{video_dir.name}.npy", features)
                    success += 1
                else:
                    failed.append(video_dir.name)
            except Exception as e:
                logger.warning(f"  Error on {video_dir.name}: {e}")
                failed.append(video_dir.name)

        logger.info(f"{split.upper()}: {success}/{total} processed successfully")
        
        if failed:
            with open(out_dir / "failed.txt", "w") as f:
                for name in failed:
                    f.write(name + "\n")

    holistic.close()
    logger.info("\nFeature extraction complete!")


if __name__ == "__main__":
    main()
