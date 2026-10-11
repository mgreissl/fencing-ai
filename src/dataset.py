import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.kinematics import load_or_compute_kinematics, select_kinematic_frames
from src.manifest import ClipRecord, load_weapon_records, merge_records
from src.splits import stratified_subset

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


def read_frames(
    path: str, index_fn: Callable[[int], list[int]]
) -> torch.Tensor:
    """
    Open a video once, choose indices from its frame count, and read them.

    Args:
        path: Video file.
        index_fn: Maps the total frame count to the list of indices to read.

    Returns:
        Tensor of shape (T, C, H, W) with values in [0, 1] float32.
    """
    if DECORD_AVAILABLE:
        vr = VideoReader(path, ctx=cpu(0))
        indices = index_fn(len(vr))
        arr = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    import av

    container = av.open(path)
    try:
        decoded = [
            f.to_ndarray(format="rgb24")
            for f in container.decode(container.streams.video[0])
        ]
    finally:
        container.close()
    if not decoded:
        raise RuntimeError(f"No frames decoded from {path}")
    total = len(decoded)
    selected = [decoded[min(i, total - 1)] for i in index_fn(total)]
    arr = np.stack(selected)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0


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


SAMPLING_MODES = ("onset_window", "end_weighted", "uniform")


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    sampling_mode: str = "onset_window",
    temporal_jitter: bool = False,
    max_shift: int = 1,
    onset_frame: Optional[float] = None,
    clip_fps: Optional[float] = None,
    window_pre_s: float = 1.0,
    window_post_s: float = 0.25,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """
    Sample `num_frames` frame indices from a video of `total_frames`.

    Modes:
    - 'onset_window': uniformly spaced frames over
      [onset - window_pre_s, onset + window_post_s] (seconds, via `clip_fps`)
      around the light onset frame. Falls back to 'uniform' if the onset or
      fps is unknown or the onset lies outside the clip.
    - 'end_weighted': Samples fewer frames from the preparation phase (first half)
      and densely from the attack/blade clash phase (second half).
    - 'uniform': Uniformly spaces frames across the entire clip.

    Args:
        total_frames: Total number of frames in the video.
        num_frames: Desired number of frames to sample.
        sampling_mode: 'onset_window' | 'end_weighted' | 'uniform'.
        temporal_jitter: If True, add random offset to the sampling.
        max_shift: Maximum frames to shift for temporal jitter.
        onset_frame: Frame index of the light onset within the clip.
        clip_fps: Frame rate of the clip (converts seconds to frames).
        window_pre_s: Seconds before the onset included in the window.
        window_post_s: Seconds after the onset included in the window.
        rng: Optional random.Random used for jitter (default: global random).

    Returns:
        Sorted list of `num_frames` frame indices within [0, total_frames).
    """
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(
            f"sampling_mode must be one of {SAMPLING_MODES}, got {sampling_mode!r}"
        )
    if total_frames <= 0:
        return list(range(num_frames))

    if total_frames < num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return sorted(indices[:num_frames])

    offset = 0
    if temporal_jitter:
        offset = (rng or random).randint(-max_shift, max_shift)

    mode = sampling_mode
    if mode == "onset_window":
        if (
            onset_frame is None
            or clip_fps is None
            or clip_fps <= 0
            or not 0 <= onset_frame < total_frames
        ):
            mode = "uniform"

    if mode == "onset_window":
        start = onset_frame - window_pre_s * clip_fps + offset
        end = onset_frame + window_post_s * clip_fps + offset
        indices = [int(round(x)) for x in np.linspace(start, end, num_frames)]
    elif mode == "end_weighted":
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


# Backwards-compatible alias (used by scripts/precompute_kinematics.py)
_sample_frame_indices = sample_frame_indices


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


ALL_WEAPONS = ("foil", "sabre", "epee")


def resolve_weapons(weapon: str, exclude_weapons: Sequence[str] = ("epee",)) -> list[str]:
    """
    Weapons to load for a ``data.weapon`` setting.

    'multi'/'all' expand to foil, sabre, epee minus ``exclude_weapons`` (epee has
    no right of way, so its label follows from the lights and adds noise to a
    shared model). A single weapon name is returned as is, never excluded.
    """
    if weapon.lower() not in ("multi", "all"):
        return [weapon.lower()]
    excluded = {w.lower() for w in exclude_weapons or ()}
    weapons = [w for w in ALL_WEAPONS if w not in excluded]
    if not weapons:
        raise ValueError(f"exclude_weapons={sorted(excluded)} leaves no weapon to load")
    return weapons


def discover_clips(
    clips_dir: Optional[str] = None,
    weapon: str = "foil",
    include_flipped: bool = False,
    flipped_dir: Optional[str] = None,
    max_clips: Optional[int] = None,
    seed: int = 42,
    validate: bool = True,
    exclude_weapons: Sequence[str] = ("epee",),
) -> list[ClipRecord]:
    """
    Discover labelled clips for specified weapon(s).

    Uses ``manifest.csv`` in each clip directory when present; otherwise
    falls back to scanning ``*.mp4`` and parsing filenames.

    Args:
        clips_dir: Primary directory containing labelled clips (or defaults to data/clips/<weapon>).
        weapon: 'foil' | 'sabre' | 'epee' | 'multi' | 'all'.
        include_flipped: Whether to include legacy flipped directory.
        flipped_dir: Optional legacy directory containing pre-flipped clips.
        max_clips: Size of a label-stratified random subset (None/0 = all).
        seed: Seed for the max_clips subset.
        exclude_weapons: Weapons skipped when ``weapon`` is 'multi'/'all'.
        validate: Drop clips that are missing or cannot be decoded (see
            `filter_readable_clips`).

    Returns:
        List of ClipRecord.
    """
    per_dir: list[list[ClipRecord]] = []
    base_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    weapons_to_load = resolve_weapons(weapon, exclude_weapons)

    for w in weapons_to_load:
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
            recs = load_weapon_records(directory, w)
            w_count += len(recs)
            per_dir.append(recs)

        log.info(f"Loaded {w_count} clips for weapon: {w.upper()}")

    samples = merge_records(per_dir)

    if not samples:
        log.warning(f"No labelled clips found for weapon: {weapon}")

    if max_clips is not None and max_clips > 0:
        samples = stratified_subset(samples, max_clips, seed=seed)
    if validate:
        samples = filter_readable_clips(samples)

    log.info(
        f"Total discovered clips: {len(samples)} "
        f"(L={sum(1 for r in samples if r.label == 'L')}, "
        f"T={sum(1 for r in samples if r.label == 'T')}, "
        f"R={sum(1 for r in samples if r.label == 'R')})"
    )
    return samples


def filter_readable_clips(records: list[ClipRecord]) -> list[ClipRecord]:
    """
    Drop clips that are missing or cannot be decoded; warn about each.

    Runs once at dataset build. Clips whose manifest provides `n_frames` are
    only checked for existence; others are opened once to read the frame count.
    """
    valid: list[ClipRecord] = []
    dropped = 0
    for rec in records:
        path = rec.abs_path()
        reason = None
        if not os.path.isfile(path):
            reason = "file not found"
        elif rec.n_frames is None:
            try:
                if _get_total_frames(path) <= 0:
                    reason = "no frames"
            except Exception as exc:  # decoder errors are backend specific
                reason = f"unreadable ({exc})"
        elif rec.n_frames <= 0:
            reason = "manifest n_frames <= 0"
        if reason is None:
            valid.append(rec)
        else:
            dropped += 1
            log.warning(f"Dropping clip {rec.clip_path}: {reason}")
    if dropped:
        log.warning(f"Dropped {dropped} unusable clips; {len(valid)} remain")
    return valid


def kinematics_dataset_kwargs(cfg) -> dict:
    """
    FencingVideoDataset kwargs for the kinematics stream from a Hydra config.

    Reads ``model.use_kinematics`` and ``model.kinematics.{cache_dir, settings,
    compute_if_missing}`` (defaults: data/kinematics, None, False).
    """
    from omegaconf import OmegaConf

    kin = getattr(cfg.model, "kinematics", None)
    settings = getattr(kin, "settings", None) if kin is not None else None
    if settings is not None:
        settings = OmegaConf.to_container(settings, resolve=True)
    return {
        "use_kinematics": bool(getattr(cfg.model, "use_kinematics", False)),
        "kinematics_cache_dir": str(getattr(kin, "cache_dir", None) or "data/kinematics"),
        "kinematics_settings": settings,
        "kinematics_compute_if_missing": bool(getattr(kin, "compute_if_missing", False)),
    }


class FencingVideoDataset(Dataset):
    """
    PyTorch Dataset for fencing video clips.

    `samples` is a list of `ClipRecord`. A clip that cannot be decoded raises
    a RuntimeError (no silent zero-frame substitution).
    """

    def __init__(
        self,
        samples: list[ClipRecord],
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        sampling_mode: str = "onset_window",
        temporal_jitter: bool = False,
        max_shift: int = 1,
        use_kinematics: bool = False,
        kinematics_cache_dir: str = "data/kinematics",
        window_pre_s: float = 1.0,
        window_post_s: float = 0.25,
        kinematics_settings: Optional[dict] = None,
        kinematics_compute_if_missing: bool = False,
    ):
        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform
        self.sampling_mode = sampling_mode
        self.temporal_jitter = temporal_jitter
        self.max_shift = max_shift
        self.use_kinematics = use_kinematics
        self.kinematics_cache_dir = kinematics_cache_dir
        self.kinematics_settings = kinematics_settings
        self.kinematics_compute_if_missing = kinematics_compute_if_missing
        self.window_pre_s = window_pre_s
        self.window_post_s = window_post_s

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        path = sample.abs_path()
        label = sample.label_id
        weapon_id = sample.weapon_id

        sampled: list[int] = []

        def index_fn(total_frames: int) -> list[int]:
            sampled[:] = sample_frame_indices(
                total_frames,
                self.num_frames,
                sampling_mode=self.sampling_mode,
                temporal_jitter=self.temporal_jitter,
                max_shift=self.max_shift,
                onset_frame=sample.light_frame_in_clip,
                clip_fps=sample.clip_fps,
                window_pre_s=self.window_pre_s,
                window_post_s=self.window_post_s,
            )
            return sampled

        try:
            frames = read_frames(path, index_fn)  # opens the video once
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path}: {e}") from e

        # Kinematics: full-rate cached features, rows picked at the video indices
        kinematics = None
        if self.use_kinematics:
            features = load_or_compute_kinematics(
                path,
                self.kinematics_cache_dir,
                settings=self.kinematics_settings,
                compute_if_missing=self.kinematics_compute_if_missing,
            )
            kinematics = torch.from_numpy(
                select_kinematic_frames(features, sampled)
            ).float()

        # Apply transforms (including dynamic horizontal flip & label/kinematics swap)
        if self.transform is not None:
            if kinematics is not None:
                frames, label, kinematics = self.transform(frames, label, kinematics)
            else:
                frames, label = self.transform(frames, label)

        result = {
            "pixel_values": frames,
            "label": label,
            "weapon_id": weapon_id,
            "path": path,
        }
        if kinematics is not None:
            result["kinematics"] = kinematics
        return result
