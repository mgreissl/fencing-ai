"""
src/kinematics.py — Kinematic Sword-Arm & Velocity Tracking for Fencing AI.

Extracts physically grounded kinematic signals from fencer body poses (using YOLO-Pose):
- Lead/weapon wrist trajectory (x, y) and velocity (vx, vy).
- Arm extension ratio: distance(wrist, shoulder) / torso_height.
- Acceleration / jerk spikes (detecting parries and blade contact).
- Torso momentum (distinguishing advancing lunges from retreats/hesitations).
- Mutual closure rate.

Pipeline: pose on every frame of the clip -> person tracking with fixed
left/right identities -> per-clip weapon side -> validity masks and short-gap
interpolation -> derivatives with dt = 1 / fps. The result is a full-rate
(T_clip, KINEMATIC_FEATURE_DIM) array; ``select_kinematic_frames`` picks the
frame indices the video stream samples. See ``KINEMATIC_FEATURE_NAMES``.

Also provides broadcast-style AR velocity glow rendering (Olympic "Fencing Visualized" style).
"""

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

log = logging.getLogger(__name__)

# COCO Keypoint indices
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16

_POSE_MODELS: dict = {}

# Bump whenever the feature computation or layout changes; part of the cache key.
KINEMATIC_FEATURE_VERSION = 2

# ---------------------------------------------------------------------------
# Feature layout
# ---------------------------------------------------------------------------
# Per frame the feature vector is [left fencer | right fencer | pair].
# "left"/"right" are identities fixed per clip (who starts on the left), not
# per-frame positions. Positions are normalized to [0, 1] image coordinates,
# velocities are in units per second (dt = 1 / fps). All entries of a block are
# 0 when its validity channel is 0.
FENCER_FEATURE_NAMES: tuple[str, ...] = (
    "wrist_x",  # weapon wrist x
    "wrist_y",  # weapon wrist y
    "wrist_vx",  # d(wrist_x)/dt
    "wrist_vy",  # d(wrist_y)/dt
    "wrist_acc",  # log(1 + |d^2 wrist/dt^2|)
    "extension",  # |weapon wrist - weapon shoulder| / torso height
    "torso_x",  # hip midpoint x
    "torso_vx",  # d(torso_x)/dt
    "valid",  # 1 if detected (or gap-interpolated), else 0
)
PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "distance",  # right wrist_x - left wrist_x
    "closure",  # -d(distance)/dt, > 0 when the fencers close in
    "pair_valid",  # both fencers valid
)
FENCER_FEATURE_DIM = len(FENCER_FEATURE_NAMES)
LEFT_OFFSET = 0
RIGHT_OFFSET = FENCER_FEATURE_DIM
PAIR_OFFSET = 2 * FENCER_FEATURE_DIM
KINEMATIC_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"left_{n}" for n in FENCER_FEATURE_NAMES)
    + tuple(f"right_{n}" for n in FENCER_FEATURE_NAMES)
    + PAIR_FEATURE_NAMES
)
KINEMATIC_FEATURE_DIM = len(KINEMATIC_FEATURE_NAMES)  # 21

_F_VALID = FENCER_FEATURE_NAMES.index("valid")
_F_X_POSITIONS = (FENCER_FEATURE_NAMES.index("wrist_x"), FENCER_FEATURE_NAMES.index("torso_x"))
_F_X_VELOCITIES = (FENCER_FEATURE_NAMES.index("wrist_vx"), FENCER_FEATURE_NAMES.index("torso_vx"))

# Settings that influence the extracted features. Every key is part of the
# kinematics cache key, so changing any of them invalidates cached features.
DEFAULT_KINEMATICS_SETTINGS: dict = {
    "pose_model": "yolo11n-pose.pt",
    "default_fps": 25.0,
    "keypoint_conf_threshold": 0.3,
    "max_gap": 5,
    "smoothing_window": 5,
    "smoothing_polyorder": 2,
    "track_max_age": 15,
    "track_max_distance": 0.75,
    "y_band": [0.30, 0.78],
    "min_box_height": 0.08,
    "max_box_bottom": 0.95,
    "min_track_coverage": 0.2,
    "side_assignment_frames": 10,
}


def resolve_kinematics_settings(settings: Optional[dict] = None) -> dict:
    """Merge user settings over ``DEFAULT_KINEMATICS_SETTINGS``."""
    unknown = set(settings or {}) - set(DEFAULT_KINEMATICS_SETTINGS)
    if unknown:
        raise KeyError(f"Unknown kinematics settings: {sorted(unknown)}")
    return {**DEFAULT_KINEMATICS_SETTINGS, **(settings or {})}


def kinematics_settings_hash(settings: Optional[dict] = None) -> str:
    """Short, stable hash of the resolved settings and ``KINEMATIC_FEATURE_VERSION``."""
    payload = {**resolve_kinematics_settings(settings), "feature_version": KINEMATIC_FEATURE_VERSION}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def kinematics_cache_path(
    cache_dir: Union[str, os.PathLike],
    clip_path: Union[str, os.PathLike],
    settings: Optional[dict] = None,
) -> Path:
    """
    Cache file for the full-rate features of a clip.

    Layout: ``{cache_dir}/v{version}_{pose_model}_{settings_hash}/{clip_stem}.npy``.
    Any change of the pose model, a feature setting or the feature version
    yields a new directory, so stale features are never reused.
    """
    cfg = resolve_kinematics_settings(settings)
    tag = f"v{KINEMATIC_FEATURE_VERSION}_{Path(cfg['pose_model']).stem}_{kinematics_settings_hash(cfg)}"
    return Path(cache_dir) / tag / f"{Path(clip_path).stem}.npy"


def save_kinematics_atomic(path: Union[str, os.PathLike], array: np.ndarray) -> Path:
    """
    Write ``array`` to ``path`` (.npy) so that readers never see a partial file.

    The data goes to a temporary file in the same directory, is flushed to disk
    and then moved over the target with ``os.replace`` (atomic on POSIX and
    Windows). On failure the temporary file is removed and an existing target
    is left untouched. If several DataLoader workers race, the last complete
    write wins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, np.asarray(array))
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(5):
            try:
                os.replace(tmp_name, path)
                break
            except PermissionError:
                # Windows refuses to replace a file another process has open.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    return path


class KinematicsCacheError(RuntimeError):
    """A cached kinematics array does not belong to the current clip."""


FPS_TOLERANCE = 0.01


def kinematics_meta_path(cache_path: Union[str, os.PathLike]) -> Path:
    """Sidecar metadata file of a cached array: ``{stem}.meta.json``."""
    cache_path = Path(cache_path)
    return cache_path.with_name(f"{cache_path.stem}.meta.json")


def _build_kinematics_meta(clip_path: Union[str, os.PathLike], n_rows: int) -> dict:
    """Facts about the clip the features were computed from."""
    meta = {
        "n_frames": int(n_rows),
        "fps": None,
        "clip_size": None,
        "clip_mtime_ns": None,
    }
    clip = Path(clip_path)
    if clip.is_file():
        stat = clip.stat()
        meta["clip_size"] = int(stat.st_size)
        meta["clip_mtime_ns"] = int(stat.st_mtime_ns)
        fps = _video_fps(clip)
        meta["fps"] = float(fps) if fps and fps > 0 else None
    return meta


def _meta_mismatch(
    meta: Optional[dict], clip_path: Union[str, os.PathLike], n_rows: int
) -> Optional[str]:
    """Why the cached array does not fit the clip, or None if it does.

    The modification time is stored but not compared: copying a data set to
    another machine changes it without changing the content.
    """
    if meta is None:
        return "no metadata file (cache predates clip validation)"
    if meta.get("n_frames") != n_rows:
        return f"metadata says {meta.get('n_frames')} frames but array has {n_rows} rows"
    clip = Path(clip_path)
    if not clip.is_file():
        return None
    size = clip.stat().st_size
    if meta.get("clip_size") is not None and meta["clip_size"] != size:
        return f"clip size changed ({meta['clip_size']} -> {size} bytes)"
    cached_fps = meta.get("fps")
    if cached_fps:
        fps = _video_fps(clip)
        if fps and fps > 0 and abs(fps - cached_fps) > FPS_TOLERANCE:
            return f"clip fps changed ({cached_fps} -> {fps})"
    return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def save_kinematics_cache(
    cache_dir: Union[str, os.PathLike],
    clip_path: Union[str, os.PathLike],
    array: np.ndarray,
    settings: Optional[dict] = None,
) -> Path:
    """
    Store ``array`` as the cache entry of a clip, with its sidecar metadata.

    The array is written first and the metadata second (both atomically), so a
    crash in between leaves an entry without metadata, which readers reject.
    """
    path = save_kinematics_atomic(kinematics_cache_path(cache_dir, clip_path, settings), array)
    _write_json_atomic(kinematics_meta_path(path), _build_kinematics_meta(clip_path, len(array)))
    return path


def load_or_compute_kinematics(
    clip_path: Union[str, os.PathLike],
    cache_dir: Union[str, os.PathLike],
    settings: Optional[dict] = None,
    pose_model=None,
    compute_if_missing: bool = True,
) -> np.ndarray:
    """
    Return the full-rate (T_clip, KINEMATIC_FEATURE_DIM) features of a clip.

    Loads ``kinematics_cache_path(cache_dir, clip_path, settings)`` if it holds a
    readable array with the current layout whose sidecar metadata (frame count,
    fps, clip size; see ``kinematics_meta_path``) matches the clip; otherwise
    computes the features on all frames of the clip and stores them with
    ``save_kinematics_atomic``. Select the sampled frames afterwards with
    ``select_kinematic_frames``, so the cache is independent of sampling mode
    and temporal jitter.

    Args:
        compute_if_missing: if False, raise instead of running pose estimation
            (e.g. inside DataLoader workers): ``FileNotFoundError`` when there
            is no readable cache entry, ``KinematicsCacheError`` when the entry
            does not match the clip (re-cut clip, other fps, no metadata).
            If True, such an entry is recomputed with a warning.
    """
    path = kinematics_cache_path(cache_dir, clip_path, settings)
    mismatch = None
    if path.exists():
        try:
            feats = np.load(path, allow_pickle=False)
            if feats.ndim == 2 and feats.shape[1] == KINEMATIC_FEATURE_DIM:
                meta_path = kinematics_meta_path(path)
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = None
                mismatch = _meta_mismatch(meta, clip_path, len(feats))
                if mismatch is None:
                    return feats.astype(np.float32, copy=False)
        except (OSError, ValueError, EOFError):
            pass  # corrupt or truncated file: recompute below
    if not compute_if_missing:
        if mismatch is not None:
            raise KinematicsCacheError(
                f"Stale kinematics cache {path}: {mismatch}; delete it and re-run "
                "scripts/precompute_kinematics.py."
            )
        raise FileNotFoundError(
            f"No kinematics cache at {path}; run scripts/precompute_kinematics.py first."
        )
    if mismatch is not None:
        log.warning(f"Recomputing stale kinematics cache {path}: {mismatch}")
    feats = extract_kinematic_features(clip_path, pose_model=pose_model, settings=settings)
    save_kinematics_cache(cache_dir, clip_path, feats, settings)
    return feats


def get_pose_model(model_name: str = "yolo11n-pose.pt"):
    """Lazy loader for YOLO pose estimation models, cached per model name."""
    if model_name not in _POSE_MODELS:
        from ultralytics import YOLO

        _POSE_MODELS[model_name] = YOLO(model_name)
    return _POSE_MODELS[model_name]


# ---------------------------------------------------------------------------
# Detections and the YOLO adapter
# ---------------------------------------------------------------------------


@dataclass
class FrameDetections:
    """
    Person detections of a single frame, all coordinates normalized to [0, 1].

    Attributes:
        boxes: (N, 4) boxes as (x1, y1, x2, y2).
        keypoints: (N, 17, 2) COCO keypoints as (x, y).
        keypoint_conf: (N, 17) keypoint confidences.
    """

    boxes: np.ndarray
    keypoints: np.ndarray
    keypoint_conf: np.ndarray

    @classmethod
    def empty(cls) -> "FrameDetections":
        return cls(
            boxes=np.zeros((0, 4)),
            keypoints=np.zeros((0, 17, 2)),
            keypoint_conf=np.zeros((0, 17)),
        )

    def __len__(self) -> int:
        return len(self.boxes)


def pose_results_to_detections(result) -> FrameDetections:
    """Convert one ultralytics pose ``Results`` object into ``FrameDetections``."""
    h, w = result.orig_shape[:2]
    if result.keypoints is None or result.boxes is None or len(result.boxes.xyxy) == 0:
        return FrameDetections.empty()

    scale = np.array([max(w, 1), max(h, 1)], dtype=np.float64)
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4)
    boxes = boxes / np.tile(scale, 2)
    kpts = result.keypoints.xy.cpu().numpy().astype(np.float64).reshape(-1, 17, 2)
    kpts = kpts / scale
    if result.keypoints.conf is None:
        conf = np.ones(kpts.shape[:2])
    else:
        conf = result.keypoints.conf.cpu().numpy().astype(np.float64).reshape(-1, 17)
    return FrameDetections(boxes=boxes, keypoints=kpts, keypoint_conf=conf)


# ---------------------------------------------------------------------------
# Tracking and fencer selection (#30)
# ---------------------------------------------------------------------------


@dataclass
class PersonTrack:
    """A tracked person: per-observation frame index, box and keypoints."""

    frame_idx: np.ndarray  # (K,) int
    boxes: np.ndarray  # (K, 4)
    keypoints: np.ndarray  # (K, 17, 2)
    keypoint_conf: np.ndarray  # (K, 17)

    @property
    def centers(self) -> np.ndarray:
        return np.stack(
            [
                (self.boxes[:, 0] + self.boxes[:, 2]) / 2.0,
                (self.boxes[:, 1] + self.boxes[:, 3]) / 2.0,
            ],
            axis=1,
        )

    @property
    def heights(self) -> np.ndarray:
        return self.boxes[:, 3] - self.boxes[:, 1]


@dataclass
class FencerTracks:
    """
    Full-rate pose of both fencers with fixed identities.

    Index 0 on the fencer axis is the fencer who starts on the left, index 1
    the fencer who starts on the right. Frames where a fencer was not detected
    have ``detected == False`` and zero keypoints.

    Attributes:
        keypoints: (T, 2, 17, 2) normalized keypoints.
        keypoint_conf: (T, 2, 17) keypoint confidences.
        detected: (T, 2) bool, fencer present in frame.
        fps: frame rate of the source clip.
    """

    keypoints: np.ndarray
    keypoint_conf: np.ndarray
    detected: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.detected.shape[0])


def _assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Minimum-cost assignment of rows to columns, dropping pairs above max_cost."""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs = []
        used_r, used_c = set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = np.unravel_index(flat, cost.shape)
            if r not in used_r and c not in used_c:
                pairs.append((int(r), int(c)))
                used_r.add(r)
                used_c.add(c)
    return [(r, c) for r, c in pairs if cost[r, c] <= max_cost]


def track_detections(
    detections: Sequence[FrameDetections],
    max_age: int = 15,
    max_distance: float = 0.75,
) -> list[PersonTrack]:
    """
    Link per-frame detections into person tracks.

    Uses a constant-velocity centroid tracker: each track predicts its box
    centre for the current frame, and detections are assigned by minimum total
    distance (normalized by box height). This keeps identities through crossings
    and short occlusions, which per-frame x-sorting cannot.

    Args:
        detections: per-frame detections for consecutive frames.
        max_age: frames a track may go unobserved before it is closed.
        max_distance: gating threshold on centre distance / box height.

    Returns:
        List of tracks, in order of creation.
    """
    tracks: list[dict] = []
    for t, det in enumerate(detections):
        n_det = len(det)
        centers = np.stack(
            [(det.boxes[:, 0] + det.boxes[:, 2]) / 2.0, (det.boxes[:, 1] + det.boxes[:, 3]) / 2.0],
            axis=1,
        ) if n_det else np.zeros((0, 2))
        heights = det.boxes[:, 3] - det.boxes[:, 1] if n_det else np.zeros(0)

        active = [i for i, tr in enumerate(tracks) if t - tr["last_t"] <= max_age]
        if active and n_det:
            pred = np.stack(
                [tracks[i]["center"] + tracks[i]["vel"] * (t - tracks[i]["last_t"]) for i in active]
            )
            track_h = np.array([tracks[i]["height"] for i in active])
            dist = np.linalg.norm(pred[:, None, :] - centers[None, :, :], axis=-1)
            norm = np.maximum(np.maximum(track_h[:, None], heights[None, :]), 1e-6)
            pairs = _assign(dist / norm, max_distance)
        else:
            pairs = []

        matched = set()
        for r, c in pairs:
            tr = tracks[active[r]]
            dt = t - tr["last_t"]
            vel = (centers[c] - tr["center"]) / dt
            tr["vel"] = vel if tr["n"] == 1 else 0.5 * tr["vel"] + 0.5 * vel
            tr["center"] = centers[c]
            tr["height"] = heights[c]
            tr["last_t"] = t
            tr["n"] += 1
            tr["obs"].append((t, c))
            matched.add(c)

        for c in range(n_det):
            if c not in matched:
                tracks.append(
                    {
                        "center": centers[c],
                        "vel": np.zeros(2),
                        "height": heights[c],
                        "last_t": t,
                        "n": 1,
                        "obs": [(t, c)],
                    }
                )

    out = []
    for tr in tracks:
        idx = np.array([t for t, _ in tr["obs"]], dtype=np.int64)
        out.append(
            PersonTrack(
                frame_idx=idx,
                boxes=np.stack([detections[t].boxes[c] for t, c in tr["obs"]]),
                keypoints=np.stack([detections[t].keypoints[c] for t, c in tr["obs"]]),
                keypoint_conf=np.stack([detections[t].keypoint_conf[c] for t, c in tr["obs"]]),
            )
        )
    return out


def select_fencer_tracks(
    tracks: Sequence[PersonTrack],
    num_frames: int,
    y_band: Sequence[float] = (0.30, 0.78),
    min_box_height: float = 0.08,
    max_box_bottom: float = 0.95,
    min_track_coverage: float = 0.2,
    side_assignment_frames: int = 10,
) -> tuple[Optional[PersonTrack], Optional[PersonTrack]]:
    """
    Pick the two tracks most likely to be the fencers and order them left/right.

    A track is plausible if its median box centre lies in the piste band
    ``y_band``, its median height is at least ``min_box_height``, its median box
    bottom is above ``max_box_bottom`` (rejects a foreground referee) and it is
    present in at least ``min_track_coverage`` of the frames (rejects flicker).
    Plausible tracks are ranked by coverage x median height; the best two win.

    Left/right is decided once per clip from the median x over the first
    ``side_assignment_frames`` observations and then kept fixed. A single
    plausible track is assigned by its side of the frame; the other fencer is
    returned as ``None`` (never duplicated).

    Returns:
        (left_track, right_track), either may be None.
    """
    scored = []
    for tr in tracks:
        centers, heights = tr.centers, tr.heights
        med_y = float(np.median(centers[:, 1]))
        med_h = float(np.median(heights))
        med_bottom = float(np.median(tr.boxes[:, 3]))
        coverage = len(tr.frame_idx) / max(num_frames, 1)
        if not (y_band[0] <= med_y <= y_band[1]):
            continue
        if med_h < min_box_height or med_bottom > max_box_bottom:
            continue
        if coverage < min_track_coverage:
            continue
        side_x = float(np.median(centers[:side_assignment_frames, 0]))
        scored.append((coverage * med_h, side_x, tr))

    scored.sort(key=lambda s: s[0], reverse=True)
    chosen = scored[:2]
    if not chosen:
        return None, None
    if len(chosen) == 1:
        _, side_x, tr = chosen[0]
        return (tr, None) if side_x < 0.5 else (None, tr)
    chosen.sort(key=lambda s: s[1])
    return chosen[0][2], chosen[1][2]


def _tracks_to_arrays(
    left: Optional[PersonTrack], right: Optional[PersonTrack], num_frames: int, fps: float
) -> FencerTracks:
    keypoints = np.zeros((num_frames, 2, 17, 2))
    conf = np.zeros((num_frames, 2, 17))
    detected = np.zeros((num_frames, 2), dtype=bool)
    for slot, tr in enumerate((left, right)):
        if tr is None:
            continue
        keypoints[tr.frame_idx, slot] = tr.keypoints
        conf[tr.frame_idx, slot] = tr.keypoint_conf
        detected[tr.frame_idx, slot] = True
    return FencerTracks(keypoints=keypoints, keypoint_conf=conf, detected=detected, fps=float(fps))


def track_fencers(
    detections: Sequence[FrameDetections],
    fps: float,
    settings: Optional[dict] = None,
) -> FencerTracks:
    """Track people over all frames and return both fencers with fixed identities."""
    cfg = resolve_kinematics_settings(settings)
    tracks = track_detections(
        detections, max_age=cfg["track_max_age"], max_distance=cfg["track_max_distance"]
    )
    left, right = select_fencer_tracks(
        tracks,
        len(detections),
        y_band=cfg["y_band"],
        min_box_height=cfg["min_box_height"],
        max_box_bottom=cfg["max_box_bottom"],
        min_track_coverage=cfg["min_track_coverage"],
        side_assignment_frames=cfg["side_assignment_frames"],
    )
    return _tracks_to_arrays(left, right, len(detections), fps)


# ---------------------------------------------------------------------------
# Weapon side and per-fencer signals (#31)
# ---------------------------------------------------------------------------


@dataclass
class FencerSignals:
    """
    Per-frame raw signals of one fencer; entries are zero where ``valid`` is False.

    Attributes:
        wrist: (T, 2) weapon wrist position.
        torso: (T, 2) hip midpoint.
        extension: (T,) |weapon wrist - weapon shoulder| / torso height.
        valid: (T,) bool.
    """

    wrist: np.ndarray
    torso: np.ndarray
    extension: np.ndarray
    valid: np.ndarray


def determine_weapon_wrist(
    keypoints: np.ndarray,
    keypoint_conf: np.ndarray,
    valid: np.ndarray,
    facing: int,
    conf_threshold: float = 0.3,
) -> int:
    """
    Decide the weapon wrist of a fencer once per clip.

    Each valid frame with both wrists above ``conf_threshold`` votes for the
    wrist that is further towards the opponent. The majority over the clip wins,
    so short phases where the back arm swings forward (fleche, recovery) or the
    weapon arm is withdrawn (parry) do not flip the weapon side.

    Args:
        keypoints: (T, 17, 2) keypoints of one fencer.
        keypoint_conf: (T, 17) keypoint confidences.
        valid: (T,) frames in which the fencer is detected.
        facing: +1 if the fencer faces towards +x (left fencer), -1 otherwise.
        conf_threshold: minimum confidence of both wrists for a vote.

    Returns:
        ``R_WRIST`` or ``L_WRIST``. Defaults to ``R_WRIST`` without votes or on a tie.
    """
    usable = (
        np.asarray(valid, dtype=bool)
        & (keypoint_conf[:, L_WRIST] >= conf_threshold)
        & (keypoint_conf[:, R_WRIST] >= conf_threshold)
    )
    lead = facing * (keypoints[:, R_WRIST, 0] - keypoints[:, L_WRIST, 0])
    right_votes = int(np.sum(usable & (lead > 0)))
    left_votes = int(np.sum(usable & (lead < 0)))
    return L_WRIST if left_votes > right_votes else R_WRIST


def compute_fencer_signals(
    keypoints: np.ndarray,
    keypoint_conf: np.ndarray,
    detected: np.ndarray,
    weapon_wrist: int,
    conf_threshold: float = 0.3,
) -> FencerSignals:
    """
    Compute weapon wrist, torso and arm extension of one fencer for all frames.

    A frame is valid if the fencer is detected and both shoulders, both hips,
    the weapon wrist and its shoulder have confidence >= ``conf_threshold``.
    """
    weapon_shoulder = R_SHOULDER if weapon_wrist == R_WRIST else L_SHOULDER
    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, weapon_wrist, weapon_shoulder]
    valid = np.asarray(detected, dtype=bool) & np.all(
        keypoint_conf[:, required] >= conf_threshold, axis=1
    )

    sh_mid = (keypoints[:, L_SHOULDER] + keypoints[:, R_SHOULDER]) / 2.0
    hip_mid = (keypoints[:, L_HIP] + keypoints[:, R_HIP]) / 2.0
    torso_scale = np.maximum(np.linalg.norm(sh_mid - hip_mid, axis=1), 0.05)
    wrist = keypoints[:, weapon_wrist]
    extension = np.linalg.norm(wrist - keypoints[:, weapon_shoulder], axis=1) / torso_scale

    mask = valid[:, None]
    return FencerSignals(
        wrist=np.where(mask, wrist, 0.0),
        torso=np.where(mask, hip_mid, 0.0),
        extension=np.where(valid, extension, 0.0),
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Missing detections: validity masks and gap filling (#29)
# ---------------------------------------------------------------------------


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, stop), ...]`` half-open index ranges where ``mask`` is True."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def fill_short_gaps(
    values: np.ndarray, valid: np.ndarray, max_gap: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate internal gaps of at most ``max_gap`` invalid frames.

    Gaps at the start or end of the clip are never extrapolated and longer gaps
    stay invalid. Invalid entries of the result are zero, but they are flagged
    by the returned mask and must never be used for derivatives.

    Args:
        values: (T, ...) per-frame values (arbitrary trailing dims).
        valid: (T,) validity mask.
        max_gap: maximum number of consecutive missing frames to fill.

    Returns:
        (filled_values, new_valid)
    """
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    new_valid = valid.copy()
    n = len(valid)
    if n and max_gap > 0 and valid.any():
        valid_idx = np.flatnonzero(valid)
        for start, stop in contiguous_runs(~valid):
            if start > 0 and stop < n and stop - start <= max_gap:
                new_valid[start:stop] = True
        fill_idx = np.flatnonzero(new_valid & ~valid)
        if len(fill_idx):
            flat = values.reshape(n, -1)
            interp = np.stack(
                [np.interp(fill_idx, valid_idx, flat[valid_idx, j]) for j in range(flat.shape[1])],
                axis=1,
            )
            flat = flat.copy()
            flat[fill_idx] = interp
            values = flat.reshape(values.shape)
    mask = new_valid.reshape((n,) + (1,) * (values.ndim - 1))
    return np.where(mask, values, 0.0), new_valid


# ---------------------------------------------------------------------------
# Full-rate derivatives and feature assembly (#28)
# ---------------------------------------------------------------------------


def masked_derivative(
    values: np.ndarray,
    valid: np.ndarray,
    fps: float,
    order: int = 1,
    smoothing_window: int = 0,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Time derivative with the true frame step ``dt = 1 / fps``, per valid run.

    Derivatives are computed separately on every contiguous run of valid frames,
    so invalid (zero) frames never enter a difference. With
    ``smoothing_window >= 3`` and scipy available a Savitzky-Golay derivative
    filter is used (window shrunk to fit short runs); otherwise ``np.gradient``
    is applied ``order`` times. Runs of a single frame and invalid frames get 0.

    Args:
        values: (T, ...) per-frame values.
        valid: (T,) validity mask.
        fps: frame rate of the series.
        order: derivative order (1 = velocity, 2 = acceleration).
        smoothing_window: Savitzky-Golay window in frames (0 or 1 disables).
        polyorder: Savitzky-Golay polynomial order.

    Returns:
        (T, ...) derivative in units per second**order.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    dt = 1.0 / float(fps)
    try:
        from scipy.signal import savgol_filter
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        savgol_filter = None

    for start, stop in contiguous_runs(valid):
        seg = values[start:stop]
        n = stop - start
        window = min(int(smoothing_window), n if n % 2 else n - 1)
        if window % 2 == 0:
            window -= 1
        poly = min(polyorder, window - 1)
        if savgol_filter is not None and window >= 3 and poly >= order:
            out[start:stop] = savgol_filter(
                seg, window, poly, deriv=order, delta=dt, axis=0, mode="interp"
            )
        elif n >= 2:
            deriv = seg
            for _ in range(order):
                deriv = np.gradient(deriv, dt, axis=0)
            out[start:stop] = deriv
    return out


def compute_kinematic_features(
    tracks: FencerTracks, settings: Optional[dict] = None
) -> np.ndarray:
    """
    Compute the full-rate (T, KINEMATIC_FEATURE_DIM) feature array from fencer tracks.

    Pipeline per fencer: weapon wrist decided once per clip (majority vote),
    per-frame signals with a validity mask, short gaps interpolated, then
    derivatives with ``dt = 1 / fps`` on valid runs only. Invalid entries are 0
    and flagged by the ``*_valid`` channels. See ``KINEMATIC_FEATURE_NAMES``.
    """
    cfg = resolve_kinematics_settings(settings)
    fps = float(tracks.fps)
    thr = cfg["keypoint_conf_threshold"]
    deriv_kw = dict(
        smoothing_window=cfg["smoothing_window"], polyorder=cfg["smoothing_polyorder"]
    )

    blocks, wrists_x, valids = [], [], []
    for slot, facing in ((0, 1), (1, -1)):
        kpts = tracks.keypoints[:, slot]
        conf = tracks.keypoint_conf[:, slot]
        detected = tracks.detected[:, slot]
        weapon = determine_weapon_wrist(kpts, conf, detected, facing, thr)
        sig = compute_fencer_signals(kpts, conf, detected, weapon, thr)

        raw = np.concatenate([sig.wrist, sig.torso, sig.extension[:, None]], axis=1)
        raw, valid = fill_short_gaps(raw, sig.valid, cfg["max_gap"])
        wrist, torso_x, extension = raw[:, 0:2], raw[:, 2], raw[:, 4]

        # v = dp/dt, a = d^2p/dt^2 with dt = 1/fps
        wrist_v = masked_derivative(wrist, valid, fps, order=1, **deriv_kw)
        wrist_a = masked_derivative(wrist, valid, fps, order=2, **deriv_kw)
        torso_vx = masked_derivative(torso_x, valid, fps, order=1, **deriv_kw)
        # log(1 + |a|) keeps impact spikes visible without dominating the input scale
        wrist_acc = np.log1p(np.linalg.norm(wrist_a, axis=1))

        blocks.append(
            np.column_stack(
                [wrist, wrist_v, wrist_acc, extension, torso_x, torso_vx, valid]
            )
        )
        wrists_x.append(wrist[:, 0])
        valids.append(valid)

    pair_valid = valids[0] & valids[1]
    distance = np.where(pair_valid, wrists_x[1] - wrists_x[0], 0.0)
    closure = -masked_derivative(distance, pair_valid, fps, order=1, **deriv_kw)
    pair = np.column_stack([distance, closure, pair_valid])

    features = np.concatenate(blocks + [pair], axis=1).astype(np.float32)
    assert features.shape[1] == KINEMATIC_FEATURE_DIM
    return features


def select_kinematic_frames(features: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """
    Pick the rows of a full-rate feature array that the video stream samples.

    Indices are clamped to the valid range, so off-by-one differences between
    video decoders do not crash. An empty feature array yields zeros
    (all channels invalid).
    """
    idx = np.asarray(list(indices), dtype=np.int64)
    if len(features) == 0:
        return np.zeros((len(idx), features.shape[-1]), dtype=np.float32)
    return np.asarray(features)[np.clip(idx, 0, len(features) - 1)]


def _iter_bgr_frames(
    frames_or_video_path: Union[str, os.PathLike, torch.Tensor, np.ndarray, Sequence[np.ndarray]],
):
    """Yield (H, W, 3) uint8 frames in the BGR order YOLO expects."""
    src = frames_or_video_path
    if isinstance(src, (str, os.PathLike)):
        cap = cv2.VideoCapture(str(src))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
    elif isinstance(src, torch.Tensor):
        if src.dim() != 4:
            raise ValueError(f"Unsupported tensor shape: {tuple(src.shape)}")
        if src.shape[1] == 3:  # (T, C, H, W) -> (T, H, W, C)
            src = src.permute(0, 2, 3, 1)
        elif src.shape[-1] != 3:
            raise ValueError(f"Unsupported tensor shape: {tuple(src.shape)}")
        # Dataset tensors are RGB in [0, 1].
        for f in src:
            rgb = (f.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            yield np.ascontiguousarray(rgb[..., ::-1])
    else:
        # numpy frames are passed through unchanged (OpenCV / BGR convention).
        yield from src


def _video_fps(path: Union[str, os.PathLike]) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()


def extract_pose_tracks(
    frames_or_video_path: Union[str, os.PathLike, torch.Tensor, np.ndarray, Sequence[np.ndarray]],
    pose_model=None,
    fps: Optional[float] = None,
    settings: Optional[dict] = None,
    batch_size: int = 32,
) -> FencerTracks:
    """
    Run pose estimation on every frame of a clip and track both fencers.

    Args:
        frames_or_video_path: a video file path (all frames are decoded), a
            tensor (T, C, H, W) / (T, H, W, C) RGB in [0, 1], or uint8 BGR frames.
        pose_model: YOLO pose model; loaded from ``settings["pose_model"]`` if None.
        fps: frame rate; read from the video file if possible, else
            ``settings["default_fps"]``.
        settings: kinematics settings (see ``DEFAULT_KINEMATICS_SETTINGS``).
        batch_size: frames per pose-model call.

    Returns:
        FencerTracks with one entry per frame.
    """
    cfg = resolve_kinematics_settings(settings)
    if pose_model is None:
        pose_model = get_pose_model(cfg["pose_model"])
    if fps is None and isinstance(frames_or_video_path, (str, os.PathLike)):
        fps = _video_fps(frames_or_video_path)
    if not fps or fps <= 0:
        fps = cfg["default_fps"]

    detections: list[FrameDetections] = []
    batch: list[np.ndarray] = []

    def flush():
        if batch:
            results = pose_model(list(batch), verbose=False)
            detections.extend(pose_results_to_detections(r) for r in results)
            batch.clear()

    for frame in _iter_bgr_frames(frames_or_video_path):
        batch.append(frame)
        if len(batch) >= batch_size:
            flush()
    flush()
    return track_fencers(detections, fps=fps, settings=cfg)


def extract_kinematic_features(
    frames: Union[str, os.PathLike, torch.Tensor, np.ndarray, Sequence[np.ndarray]],
    pose_model=None,
    fps: Optional[float] = None,
    settings: Optional[dict] = None,
) -> np.ndarray:
    """
    Extract the full-rate (T, KINEMATIC_FEATURE_DIM) kinematic time-series.

    ``frames`` must be consecutive frames of a clip (or a video path); use
    ``select_kinematic_frames`` afterwards to pick the frames the video stream
    samples. Do not pass already-subsampled frames: derivatives assume
    ``dt = 1 / fps`` between rows.
    """
    tracks = extract_pose_tracks(frames, pose_model=pose_model, fps=fps, settings=settings)
    if tracks.num_frames == 0:
        return np.zeros((0, KINEMATIC_FEATURE_DIM), dtype=np.float32)
    return compute_kinematic_features(tracks, settings=settings)


def render_velocity_trails(
    frames: list[np.ndarray],
    pose_model=None,
    mode: str = "foil",
    trail_length: int = 8,
    blade_length: float = 58.0,
) -> list[np.ndarray]:
    """
    Renders Olympic 'Fencing Visualized' AR broadcast visual effects onto video frames:
    - 'foil': Authentic glowing neon laser foil blades and trailing tip velocity ribbons.
    - 'skeleton': Biomechanical skeletal overlay tracking fencer stances and joint nodes.
    - 'combined': Both neon foil blades and skeletal overlays.
    """
    if pose_model is None:
        pose_model = get_pose_model()

    results = pose_model(frames, verbose=False)
    rendered = [f.copy() for f in frames]
    H, W = frames[0].shape[:2]

    BONES = [
        (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
        (5, 7), (7, 9),                      # Left Arm
        (6, 8), (8, 10),                     # Right Arm
        (11, 13), (13, 15),                  # Left Leg
        (12, 14), (14, 16),                  # Right Leg
    ]

    L_GLOW = (255, 210, 0)      # Electric Cyan (BGR)
    L_CORE = (255, 255, 255)
    R_GLOW = (0, 110, 255)      # Laser Orange / Red (BGR)
    R_CORE = (255, 255, 255)

    left_tip_history = []
    right_tip_history = []

    out_frames = []

    for t, (f, r) in enumerate(zip(rendered, results)):
        target = f.copy()
        glow_layer = np.zeros_like(f, dtype=np.float32)

        candidates = []
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            for b, k in zip(r.boxes, r.keypoints):
                xyxy = b.xyxy[0].cpu().numpy()
                h = xyxy[3] - xyxy[1]
                center_y = (xyxy[1] + xyxy[3]) / (2.0 * max(H, 1))
                bottom_y = xyxy[3] / max(H, 1)
                center_x = (xyxy[0] + xyxy[2]) / (2.0 * max(W, 1))
                if 0.40 <= center_y <= 0.70 and bottom_y < 0.90 and h >= 50:
                    candidates.append((center_x, k.xy[0].cpu().numpy()))

        p_lw, p_lt, p_rw, p_rt = None, None, None, None
        if len(candidates) >= 2:
            candidates.sort(key=lambda c: c[0])
            l_kpts = candidates[0][1]
            r_kpts = candidates[-1][1]

            # Render Skeletons if requested
            if mode in ["skeleton", "combined"]:
                for (i1, i2) in BONES:
                    p1 = (int(l_kpts[i1][0]), int(l_kpts[i1][1]))
                    p2 = (int(l_kpts[i2][0]), int(l_kpts[i2][1]))
                    if p1[0] > 0 and p2[0] > 0:
                        cv2.line(target, p1, p2, (240, 180, 0), 3, cv2.LINE_AA)
                for i in range(5, 17):
                    p = (int(l_kpts[i][0]), int(l_kpts[i][1]))
                    if p[0] > 0:
                        cv2.circle(target, p, 5, (255, 255, 100), -1, cv2.LINE_AA)

                for (i1, i2) in BONES:
                    p1 = (int(r_kpts[i1][0]), int(r_kpts[i1][1]))
                    p2 = (int(r_kpts[i2][0]), int(r_kpts[i2][1]))
                    if p1[0] > 0 and p2[0] > 0:
                        cv2.line(target, p1, p2, (0, 130, 255), 3, cv2.LINE_AA)
                for i in range(5, 17):
                    p = (int(r_kpts[i][0]), int(r_kpts[i][1]))
                    if p[0] > 0:
                        cv2.circle(target, p, 5, (120, 220, 255), -1, cv2.LINE_AA)

            # Lead sword arm & foil projection
            lw = l_kpts[R_WRIST] if l_kpts[R_WRIST][0] > l_kpts[L_WRIST][0] else l_kpts[L_WRIST]
            le = l_kpts[R_ELBOW] if l_kpts[R_WRIST][0] > l_kpts[L_WRIST][0] else l_kpts[L_ELBOW]
            u_l = (lw - le) / (np.linalg.norm(lw - le) + 1e-6)
            l_tip = lw + u_l * blade_length
            left_tip_history.append((int(l_tip[0]), int(l_tip[1])))

            rw = r_kpts[L_WRIST] if r_kpts[L_WRIST][0] < r_kpts[R_WRIST][0] else r_kpts[R_WRIST]
            re = r_kpts[L_ELBOW] if r_kpts[L_WRIST][0] < r_kpts[R_WRIST][0] else r_kpts[R_ELBOW]
            u_r = (rw - re) / (np.linalg.norm(rw - re) + 1e-6)
            r_tip = rw + u_r * blade_length
            right_tip_history.append((int(r_tip[0]), int(r_tip[1])))

            p_lw, p_lt = (int(lw[0]), int(lw[1])), (int(l_tip[0]), int(l_tip[1]))
            p_rw, p_rt = (int(rw[0]), int(rw[1])), (int(r_tip[0]), int(r_tip[1]))

            if mode in ["foil", "combined"]:
                cv2.line(glow_layer, p_lw, p_lt, L_GLOW, 6, cv2.LINE_AA)
                cv2.circle(glow_layer, p_lt, 8, L_GLOW, -1, cv2.LINE_AA)
                cv2.line(glow_layer, p_rw, p_rt, R_GLOW, 6, cv2.LINE_AA)
                cv2.circle(glow_layer, p_rt, 8, R_GLOW, -1, cv2.LINE_AA)

        if mode in ["foil", "combined"]:
            for hist, color in [(left_tip_history, L_GLOW), (right_tip_history, R_GLOW)]:
                if len(hist) > 1:
                    recent = hist[-trail_length:]
                    for idx in range(len(recent) - 1):
                        fade = (idx + 1) / float(len(recent))
                        c = tuple(ch * fade for ch in color)
                        w = int(2 + 4 * fade)
                        cv2.line(glow_layer, recent[idx], recent[idx + 1], c, w, cv2.LINE_AA)

            bloom_wide = cv2.GaussianBlur(glow_layer, (25, 25), 0)
            bloom_narrow = cv2.GaussianBlur(glow_layer, (9, 9), 0)
            total_bloom = np.clip(bloom_wide * 0.8 + bloom_narrow * 1.2 + glow_layer * 0.5, 0, 255).astype(np.uint8)
            cv2.add(target, total_bloom, dst=target)

            if p_lw is not None and p_lt is not None:
                cv2.line(target, p_lw, p_lt, L_CORE, 2, cv2.LINE_AA)
                cv2.circle(target, p_lt, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(target, p_rw, p_rt, R_CORE, 2, cv2.LINE_AA)
                cv2.circle(target, p_rt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        out_frames.append(target)

    return out_frames


def flip_kinematic_features(
    kinematics: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Horizontal reflection of kinematic features of shape (..., KINEMATIC_FEATURE_DIM).

    - Swaps the left and right fencer blocks.
    - Reflects positions (x -> 1 - x) of valid frames; invalid frames stay 0.
    - Negates horizontal velocities (vx -> -vx).
    - y, acceleration magnitude, extension, validity and the pair block
      (distance, closure) are invariant.

    Applying it twice returns the input. Works on torch tensors and numpy arrays.
    """
    is_torch = isinstance(kinematics, torch.Tensor)
    flipped = kinematics.clone() if is_torch else np.array(kinematics, copy=True)
    for src, dst in ((LEFT_OFFSET, RIGHT_OFFSET), (RIGHT_OFFSET, LEFT_OFFSET)):
        block = kinematics[..., src : src + FENCER_FEATURE_DIM]
        flipped[..., dst : dst + FENCER_FEATURE_DIM] = block
        valid = block[..., _F_VALID] > 0.5
        for i in _F_X_POSITIONS:
            reflected = 1.0 - block[..., i]
            zero = torch.zeros_like(reflected) if is_torch else np.zeros_like(reflected)
            where = torch.where if is_torch else np.where
            flipped[..., dst + i] = where(valid, reflected, zero)
        for i in _F_X_VELOCITIES:
            flipped[..., dst + i] = -block[..., i]
    return flipped
