#!/usr/bin/env python3
"""
#32: Plot kinematic features over time for signal quality validation.

Usage:
    python scripts/plot_kinematics.py [--clips CLIP1 CLIP2 ...] [--manifest PATH] [--n-clips N] [--output-dir DIR] [--render]

Outputs:
    PNG plots: output_dir/{clip_stem}_kinematics.png
    (optionally) MP4 renders: output_dir/{clip_stem}_kinematics.mp4

Per-clip plot shows:
    - Wrist vx (left & right) over time
    - Extension (left & right)
    - Torso vx (left & right)
    - Validity shading (gray = invalid)
    - Vertical line at light_frame_in_clip

Use --render to additionally write MP4 with render_velocity_trails overlay (slow, requires pose model).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import cv2

from src.manifest import (
    load_manifest,
    load_weapon_records,
)
from src.kinematics import (
    KINEMATIC_FEATURE_NAMES,
    load_or_compute_kinematics,
    render_velocity_trails,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_feature_index(name: str) -> int:
    """Get index of a kinematic feature by name."""
    try:
        return KINEMATIC_FEATURE_NAMES.index(name)
    except ValueError:
        raise ValueError(f"Unknown kinematic feature: {name}. Valid: {KINEMATIC_FEATURE_NAMES}")


def plot_kinematics(
    clip_path: str,
    features: np.ndarray,  # (T, KINEMATIC_FEATURE_DIM)
    light_frame: Optional[float] = None,
    fps: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot kinematic time series with validity shading and light frame marker.

    Features plotted:
    - Wrist velocity (left & right)
    - Arm extension (left & right)
    - Torso velocity (left & right)
    """
    if features.shape[0] == 0:
        log.warning(f"{clip_path}: No features to plot (empty)")
        return

    T = features.shape[0]
    time = np.arange(T)
    if fps:
        time = time / fps

    # Get feature indices
    left_wrist_vx = find_feature_index("left_wrist_vx")
    right_wrist_vx = find_feature_index("right_wrist_vx")
    left_extension = find_feature_index("left_extension")
    right_extension = find_feature_index("right_extension")
    left_torso_vx = find_feature_index("left_torso_vx")
    right_torso_vx = find_feature_index("right_torso_vx")
    left_valid = find_feature_index("left_valid")
    right_valid = find_feature_index("right_valid")
    pair_valid = find_feature_index("pair_valid")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"Kinematic Features: {Path(clip_path).name}", fontsize=14, fontweight='bold')

    # Plot 1: Wrist velocity
    ax = axes[0]
    ax.plot(time, features[:, left_wrist_vx], label="Left Wrist vx", linewidth=1.5, color='C0')
    ax.plot(time, features[:, right_wrist_vx], label="Right Wrist vx", linewidth=1.5, color='C1')
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=features[:, left_valid] == 0, alpha=0.2, color='gray', label='Left Invalid')
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=features[:, right_valid] == 0, alpha=0.15, color='black', label='Right Invalid')
    if light_frame is not None:
        light_time = light_frame / fps if fps else light_frame
        ax.axvline(x=light_time, color='red', linestyle='--', linewidth=2, label='Light Frame', alpha=0.7)
    ax.set_ylabel("Velocity (units/sec)", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Wrist Velocity (X component)", fontsize=12)

    # Plot 2: Arm extension
    ax = axes[1]
    ax.plot(time, features[:, left_extension], label="Left Extension", linewidth=1.5, color='C2')
    ax.plot(time, features[:, right_extension], label="Right Extension", linewidth=1.5, color='C3')
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=features[:, left_valid] == 0, alpha=0.2, color='gray')
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=features[:, right_valid] == 0, alpha=0.15, color='black')
    if light_frame is not None:
        light_time = light_frame / fps if fps else light_frame
        ax.axvline(x=light_time, color='red', linestyle='--', linewidth=2, label='Light Frame', alpha=0.7)
    ax.set_ylabel("Extension Ratio", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Arm Extension (|wrist - shoulder| / torso height)", fontsize=12)

    # Plot 3: Torso velocity
    ax = axes[2]
    ax.plot(time, features[:, left_torso_vx], label="Left Torso vx", linewidth=1.5, color='C4')
    ax.plot(time, features[:, right_torso_vx], label="Right Torso vx", linewidth=1.5, color='C5')
    ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=features[:, pair_valid] == 0, alpha=0.2, color='gray', label='Pair Invalid')
    if light_frame is not None:
        light_time = light_frame / fps if fps else light_frame
        ax.axvline(x=light_time, color='red', linestyle='--', linewidth=2, label='Light Frame', alpha=0.7)
    ax.set_ylabel("Velocity (units/sec)", fontsize=11)
    ax.set_xlabel("Time (seconds)" if fps else "Frame", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Torso Velocity (X component)", fontsize=12)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        log.info(f"Wrote plot: {output_path}")
    plt.close(fig)


def render_and_save_video(
    clip_path: str,
    output_path: Path,
    pose_model=None,
) -> None:
    """
    Render velocity trails over the clip and save as MP4.

    Uses render_velocity_trails to overlay AR foil blade and velocity glow.
    """
    try:
        # Load video frames
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            log.warning(f"Cannot open video: {clip_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            log.warning(f"{clip_path}: No frames read")
            return

        log.info(f"{clip_path}: Rendering {len(frames)} frames...")
        rendered = render_velocity_trails(frames, pose_model=pose_model, mode='foil')

        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for frame in rendered:
            out.write(frame)
        out.release()

        log.info(f"Wrote rendered video: {output_path}")

    except Exception as e:
        log.error(f"Failed to render {clip_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot kinematic features for signal quality validation."
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=None,
        help="Specific clip paths to plot",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest CSV to sample from (if --clips not provided)",
    )
    parser.add_argument(
        "--n-clips",
        type=int,
        default=10,
        help="Number of random clips to sample from manifest (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "kinematics_plots",
        help="Output directory (default: artifacts/kinematics_plots)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "kinematics_cache",
        help="Cache directory for computed kinematics (default: artifacts/kinematics_cache)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Also render MP4 videos with velocity trails (slow)",
    )

    args = parser.parse_args()

    clip_paths = args.clips
    if not clip_paths:
        # Sample from manifest
        if not args.manifest:
            args.manifest = REPO_ROOT / "data" / "clips" / "foil" / "manifest.csv"
        if not args.manifest.exists():
            log.error(f"Manifest not found: {args.manifest}")
            return 1

        records = load_manifest(args.manifest)
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(records), size=min(args.n_clips, len(records)), replace=False)
        clip_paths = [records[i].abs_path(REPO_ROOT) for i in sorted(indices)]
        log.info(f"Sampled {len(clip_paths)} clips from {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    for clip_path in clip_paths:
        try:
            clip_abs = Path(clip_path)
            if not clip_abs.exists():
                log.warning(f"Clip not found: {clip_path}")
                continue

            log.info(f"Processing: {clip_path}")

            # Load or compute kinematics
            features = load_or_compute_kinematics(
                clip_path,
                args.cache_dir,
                settings=None,
                compute_if_missing=True,
            )

            # Try to get FPS and light_frame from manifest (if available)
            fps = 25.0  # Default
            light_frame = None

            # Plot kinematics
            output_png = args.output_dir / f"{clip_abs.stem}_kinematics.png"
            plot_kinematics(
                clip_path,
                features,
                light_frame=light_frame,
                fps=fps,
                output_path=output_png,
            )

            # Render video if requested
            if args.render:
                output_mp4 = args.output_dir / f"{clip_abs.stem}_kinematics.mp4"
                render_and_save_video(clip_path, output_mp4)

        except Exception as e:
            log.error(f"Error processing {clip_path}: {e}")
            continue

    log.info("Done")
    return 0


if __name__ == "__main__":
    exit(main())
