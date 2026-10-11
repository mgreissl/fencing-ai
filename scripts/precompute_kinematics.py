#!/usr/bin/env python3
"""
Precompute and cache kinematic features for fencing clips.
Speeds up training by avoiding real-time YOLO pose inference in the DataLoader.
"""

import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import _get_total_frames, _sample_frame_indices, _read_video
from src.kinematics import extract_kinematic_features, get_pose_model

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute kinematics cache")
    parser.add_argument("--clips_dir", type=str, default="data/clips/foil", help="Clips directory")
    parser.add_argument("--cache_dir", type=str, default="data/kinematics", help="Output directory for cached .npy")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of sampled frames")
    parser.add_argument("--sampling_mode", type=str, default="end_weighted", help="Sampling strategy")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    clip_files = sorted(glob.glob(os.path.join(args.clips_dir, "*.mp4")))
    print(f"Discovered {len(clip_files)} clips in {args.clips_dir}")

    pose_model = get_pose_model()
    cached_count = 0
    computed_count = 0

    for path in tqdm(clip_files, desc="Precomputing kinematics"):
        fname = os.path.splitext(os.path.basename(path))[0]
        cache_path = os.path.join(args.cache_dir, f"{fname}.npy")
        if os.path.exists(cache_path):
            cached_count += 1
            continue

        try:
            total_frames = _get_total_frames(path)
            indices = _sample_frame_indices(
                total_frames,
                args.num_frames,
                sampling_mode=args.sampling_mode,
            )
            frames = _read_video(path, indices)
            feats = extract_kinematic_features(frames, pose_model=pose_model)
            np.save(cache_path, feats)
            computed_count += 1
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Complete! Pre-cached: {cached_count}, Newly computed: {computed_count}")

if __name__ == "__main__":
    main()
