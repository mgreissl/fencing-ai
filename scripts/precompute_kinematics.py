#!/usr/bin/env python3
"""
Precompute and cache kinematic features for fencing clips.
Speeds up training by avoiding real-time YOLO pose inference in the DataLoader.

Features are computed on every frame of each clip and stored at full frame
rate under a directory keyed by the kinematics settings hash, so the dataset
can select the same frame indices as the video stream at load time.
"""

import os
import sys
import glob
import json
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kinematics import (
    get_pose_model,
    kinematics_cache_path,
    load_or_compute_kinematics,
    resolve_kinematics_settings,
)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Precompute kinematics cache")
    parser.add_argument("--clips_dir", type=str, default="data/clips/foil", help="Clips directory")
    parser.add_argument("--cache_dir", type=str, default="data/kinematics", help="Root directory for cached .npy")
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Optional JSON dict overriding DEFAULT_KINEMATICS_SETTINGS, e.g. '{\"max_gap\": 3}'",
    )
    args = parser.parse_args()

    settings = resolve_kinematics_settings(json.loads(args.settings) if args.settings else None)
    clip_files = sorted(glob.glob(os.path.join(args.clips_dir, "*.mp4")))
    print(f"Discovered {len(clip_files)} clips in {args.clips_dir}")
    if not clip_files:
        return
    print(f"Cache directory: {kinematics_cache_path(args.cache_dir, clip_files[0], settings).parent}")

    pose_model = get_pose_model(settings["pose_model"])
    cached_count = 0
    computed_count = 0

    for path in tqdm(clip_files, desc="Precomputing kinematics"):
        fname = os.path.splitext(os.path.basename(path))[0]
        cache_path = kinematics_cache_path(args.cache_dir, path, settings)
        mtime = cache_path.stat().st_mtime_ns if cache_path.exists() else None
        try:
            # Loads valid cache entries, recomputes missing or corrupt ones.
            load_or_compute_kinematics(path, args.cache_dir, settings, pose_model=pose_model)
            if mtime is not None and cache_path.stat().st_mtime_ns == mtime:
                cached_count += 1
            else:
                computed_count += 1
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Complete! Pre-cached: {cached_count}, Newly computed: {computed_count}")

if __name__ == "__main__":
    main()
