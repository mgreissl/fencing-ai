import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Label mapping: filename prefix → integer class
LABEL_MAP = {"L": 0, "T": 1, "R": 2}
LABEL_NAMES = ["Left", "Tie", "Right"]

# Try to import decord for fast video reading; fall back to torchvision
try:
    from decord import VideoReader, cpu

    DECORD_AVAILABLE = True
    log.info("Using decord for video decoding")
except ImportError:
    DECORD_AVAILABLE = False
    log.info("decord not available; falling back to PyAV")


def _read_video_decord(
    path: str, indices: list[int]
) -> torch.Tensor:
    """
    Read specific frames from a video using decord.

    Returns:
        Tensor of shape (T, C, H, W) with values in [0, 1] float32.
    """
    vr = VideoReader(path, ctx=cpu(0))
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return frames


def _read_video_pyav(
    path: str, indices: list[int]
) -> torch.Tensor:
    """
    Read specific frames from a video using PyAV.

    Returns:
        Tensor of shape (T, C, H, W) with values in [0, 1] float32.
    """
    import av
    import numpy as np

    container = av.open(path)
    stream = container.streams.video[0]

    # Decode all frames (clips are short, so this is fine)
    all_frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
        all_frames.append(arr)
    container.close()

    if not all_frames:
        raise RuntimeError(f"No frames decoded from {path}")

    # Select requested indices (clamped to valid range)
    total = len(all_frames)
    selected = [all_frames[min(i, total - 1)] for i in indices]
    frames = np.stack(selected)  # (T, H, W, C)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return frames


def _read_video(path: str, indices: list[int]) -> torch.Tensor:
    """Read video frames using the best available backend."""
    if DECORD_AVAILABLE:
        return _read_video_decord(path, indices)
    return _read_video_pyav(path, indices)


def _extract_label(filename: str) -> Optional[int]:
    """
    Extract the class label from a clip filename.

    Expected format: {L|R|T}rest_of_filename.mp4
    e.g., "L1234-5.mp4" → 0 (Left)
    """
    basename = os.path.basename(filename)
    if basename and basename[0] in LABEL_MAP:
        return LABEL_MAP[basename[0]]
    return None


def _sample_frame_indices(
    total_frames: int,
    num_frames: int,
    sampling_mode: str = "end_weighted",
    temporal_jitter: bool = False,
    max_shift: int = 1,
) -> list[int]:
    """
    Sample `num_frames` frame indices from a video of `total_frames`.

    Supports two modes:
    - 'end_weighted': Samples fewer frames from the preparation phase (first half)
      and densely from the attack/blade clash phase (second half).
    - 'uniform': Uniformly spaces frames across the entire clip.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Desired number of frames to sample.
        sampling_mode: 'end_weighted' or 'uniform'.
        temporal_jitter: If True, add random offset to the sampling.
        max_shift: Maximum frames to shift for temporal jitter.

    Returns:
        Sorted list of frame indices.
    """
    if total_frames <= 0:
        return list(range(num_frames))

    if total_frames < num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return sorted(indices[:num_frames])

    offset = 0
    if temporal_jitter:
        offset = random.randint(-max_shift, max_shift)

    if sampling_mode == "end_weighted":
        half = total_frames // 2
        n_early = 4
        n_late = num_frames - n_early  # 12 dense frames
        early_tick = half / n_early
        late_tick = (total_frames - half) / n_late

        early = [int(i * early_tick) for i in range(n_early)]
        late = [int(half + i * late_tick + offset) for i in range(n_late)]
        indices = early + late
    else:
        tick = total_frames / num_frames
        indices = [int(tick * i + tick / 2 + offset) for i in range(num_frames)]

    # Clamp to valid range and sort
    indices = sorted([max(0, min(i, total_frames - 1)) for i in indices])
    return indices

    # Clamp to valid range
    indices = [max(0, min(i, total_frames - 1)) for i in indices]
    return indices


def _get_total_frames(path: str) -> int:
    """Get the total frame count of a video."""
    if DECORD_AVAILABLE:
        vr = VideoReader(path, ctx=cpu(0))
        return len(vr)
    else:
        import av

        container = av.open(path)
        stream = container.streams.video[0]
        # Fast frame count without decoding
        total = stream.frames
        if total == 0:
            # Some containers don't report frame count; decode to count
            total = sum(1 for _ in container.decode(stream))
        container.close()
        return total


WEAPON_MAP = {"foil": 0, "sabre": 1, "epee": 2}
WEAPON_NAMES = ["Foil", "Sabre", "Epee"]


def discover_clips(
    clips_dir: Optional[str] = None,
    weapon: str = "foil",
    include_flipped: bool = False,
    flipped_dir: Optional[str] = None,
    max_clips: Optional[int] = None,
) -> list[tuple[str, int, int]]:
    """
    Discover labelled .mp4 clips for specified weapon(s).

    Args:
        clips_dir: Primary directory containing labelled clips (or defaults to data/clips/<weapon>).
        weapon: 'foil' | 'sabre' | 'epee' | 'multi' | 'all'.
        include_flipped: Whether to include legacy flipped directory.
        flipped_dir: Optional legacy directory containing pre-flipped clips.
        max_clips: Maximum number of clips to return.

    Returns:
        List of (filepath, label_int, weapon_id_int) tuples.
    """
    samples = []
    base_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    weapons_to_load = ["foil", "sabre", "epee"] if weapon in ["multi", "all"] else [weapon.lower()]

    for w in weapons_to_load:
        w_id = WEAPON_MAP.get(w, 0)
        target_dirs = []

        if clips_dir and os.path.isdir(clips_dir) and len(weapons_to_load) == 1:
            target_dirs.append(clips_dir)
        else:
            # Modern directory structure
            modern_dir = os.path.join(base_root, "data", "clips", w)
            if os.path.isdir(modern_dir):
                target_dirs.append(modern_dir)
            elif w == "foil":
                # Fallback to final_training_clips if foil
                legacy_dir = os.path.join(base_root, "final_training_clips")
                if os.path.isdir(legacy_dir):
                    target_dirs.append(legacy_dir)

        if include_flipped and flipped_dir and os.path.isdir(flipped_dir) and w == "foil":
            target_dirs.append(flipped_dir)

        w_count = 0
        for directory in target_dirs:
            for fname in sorted(os.listdir(directory)):
                if not fname.lower().endswith(".mp4"):
                    continue
                label = _extract_label(fname)
                if label is not None:
                    samples.append((os.path.join(directory, fname), label, w_id))
                    w_count += 1

        log.info(f"Loaded {w_count} clips for weapon: {w.upper()} (ID={w_id})")

    if not samples:
        log.warning(f"No labelled .mp4 clips found for weapon: {weapon}")

    if max_clips is not None and max_clips > 0:
        samples = samples[:max_clips]

    log.info(
        f"Total discovered clips: {len(samples)} "
        f"(L={sum(1 for _, l, _ in samples if l == 0)}, "
        f"T={sum(1 for _, l, _ in samples if l == 1)}, "
        f"R={sum(1 for _, l, _ in samples if l == 2)})"
    )
    return samples


def _get_base_clip_id(path: str) -> str:
    """Extract a unique clip identifier ignoring -flipped and leading class letter."""
    fname = os.path.basename(path)
    clean = fname.replace("-flipped.mp4", "").replace(".mp4", "")
    return clean[1:] if len(clean) > 1 else clean


def make_splits(
    samples: list[tuple[str, int, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Group-aware stratified train/val/test split.

    Returns:
        Tuple of (train_samples, val_samples, test_samples).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    from collections import defaultdict

    groups = defaultdict(list)
    base_id_labels = {}
    for item in samples:
        path, label = item[0], item[1]
        bid = _get_base_clip_id(path)
        groups[bid].append(item)
        if "-flipped" not in path:
            base_id_labels[bid] = label
        elif bid not in base_id_labels:
            base_id_labels[bid] = label

    unique_bids = sorted(list(groups.keys()))
    bid_labels = [base_id_labels[bid] for bid in unique_bids]

    # First split: train vs (val+test)
    test_val_ratio = val_ratio + test_ratio
    train_bids, rest_bids, train_lbls, rest_lbls = train_test_split(
        unique_bids,
        bid_labels,
        test_size=test_val_ratio,
        stratify=bid_labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / test_val_ratio
    val_bids, test_bids, _, _ = train_test_split(
        rest_bids,
        rest_lbls,
        test_size=relative_test,
        stratify=rest_lbls,
        random_state=seed,
    )

    train_samples = [s for bid in train_bids for s in groups[bid]]
    val_samples = [s for bid in val_bids for s in groups[bid]]
    test_samples = [s for bid in test_bids for s in groups[bid]]

    log.info(
        f"Group-aware split ({len(unique_bids)} base clips): "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    return train_samples, val_samples, test_samples


class FencingVideoDataset(Dataset):
    """
    PyTorch Dataset for fencing video clips.
    """

    def __init__(
        self,
        samples: list,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        sampling_mode: str = "end_weighted",
        temporal_jitter: bool = False,
        max_shift: int = 1,
    ):
        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform
        self.sampling_mode = sampling_mode
        self.temporal_jitter = temporal_jitter
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if len(sample) == 3:
            path, label, weapon_id = sample
        else:
            path, label = sample[0], sample[1]
            weapon_id = 0

        try:
            total_frames = _get_total_frames(path)
            indices = _sample_frame_indices(
                total_frames,
                self.num_frames,
                sampling_mode=self.sampling_mode,
                temporal_jitter=self.temporal_jitter,
                max_shift=self.max_shift,
            )
            frames = _read_video(path, indices)
        except Exception as e:
            log.error(f"Error reading video {path}: {e}")
            frames = torch.zeros(
                self.num_frames, 3, 224, 224, dtype=torch.float32
            )

        # Apply transforms (including dynamic horizontal flip & label swap)
        if self.transform is not None:
            frames, label = self.transform(frames, label)

        return {
            "pixel_values": frames,
            "label": label,
            "weapon_id": weapon_id,
            "path": path,
        }
