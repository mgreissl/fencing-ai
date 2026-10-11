"""
src/kinematics.py — Kinematic Sword-Arm & Velocity Tracking for Fencing AI.

Extracts physically grounded kinematic signals from fencer body poses (using YOLO-Pose):
- Lead/weapon wrist trajectory (x, y) and velocity (vx, vy).
- Arm extension ratio: distance(wrist, shoulder) / torso_height.
- Acceleration / jerk spikes (detecting parries and blade contact).
- Torso momentum (distinguishing advancing lunges from retreats/hesitations).
- Mutual closure rate.

Also provides broadcast-style AR velocity glow rendering (Olympic "Fencing Visualized" style).
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Sequence, Union

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

_YOLO_MODEL = None

# Settings that influence the extracted features. Every key is part of the
# kinematics cache key, so changing any of them invalidates cached features.
DEFAULT_KINEMATICS_SETTINGS: dict = {
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


def get_pose_model(model_name: str = "yolo11n-pose.pt"):
    """Lazy loader for YOLO pose estimation model."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(model_name)
    return _YOLO_MODEL


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


def _get_fencer_metrics(kpts: np.ndarray, is_left_fencer: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract lead wrist position, torso position, and torso scale for a fencer.

    Args:
        kpts: (17, 2) array of (x, y) coordinates normalized to [0, 1].
        is_left_fencer: True if fencer is on the left facing right.

    Returns:
        (lead_wrist_xy, torso_xy, torso_scale)
    """
    # Shoulder midpoint and hip midpoint
    l_sh, r_sh = kpts[L_SHOULDER], kpts[R_SHOULDER]
    l_hp, r_hp = kpts[L_HIP], kpts[R_HIP]

    sh_mid = (l_sh + r_sh) / 2.0
    hp_mid = (l_hp + r_hp) / 2.0
    torso_scale = max(float(np.linalg.norm(sh_mid - hp_mid)), 0.05)

    l_wr, r_wr = kpts[L_WRIST], kpts[R_WRIST]

    # For Left fencer facing Right, lead wrist is the one with greater X (closest to opponent)
    # For Right fencer facing Left, lead wrist is the one with smaller X
    if is_left_fencer:
        lead_wrist = r_wr if r_wr[0] > l_wr[0] else l_wr
        shoulder = r_sh if r_wr[0] > l_wr[0] else l_sh
    else:
        lead_wrist = l_wr if l_wr[0] < r_wr[0] else r_wr
        shoulder = l_sh if l_wr[0] < r_wr[0] else r_sh

    # Extension: distance from shoulder to lead wrist normalized by torso height
    extension = float(np.linalg.norm(lead_wrist - shoulder)) / torso_scale

    return lead_wrist, hp_mid, extension


def extract_kinematic_features(
    frames: Union[torch.Tensor, np.ndarray],
    pose_model=None,
) -> np.ndarray:
    """
    Extract a (T, 16) kinematic time-series from a sequence of T video frames.

    Args:
        frames: Tensor of shape (T, C, H, W) in [0, 1] or numpy (T, H, W, C) in [0, 255].
        pose_model: Optional pre-loaded YOLO model.

    Returns:
        np.ndarray of shape (T, 16) with normalized kinematic features.
    """
    if pose_model is None:
        pose_model = get_pose_model()

    # Convert frames to list of (H, W, 3) uint8 numpy arrays
    if isinstance(frames, torch.Tensor):
        if frames.dim() == 4 and frames.shape[1] == 3:  # (T, C, H, W)
            np_frames = [
                (f.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                for f in frames
            ]
        elif frames.dim() == 4 and frames.shape[-1] == 3:  # (T, H, W, C)
            np_frames = [(f.cpu().numpy() * 255.0).astype(np.uint8) for f in frames]
        else:
            raise ValueError(f"Unsupported tensor shape: {frames.shape}")
    else:
        np_frames = list(frames)

    T = len(np_frames)
    if T == 0:
        return np.zeros((16, 16), dtype=np.float32)

    H, W = np_frames[0].shape[:2]

    # Run YOLO pose inference
    results = pose_model(np_frames, verbose=False)

    raw_left_wrist = np.zeros((T, 2), dtype=np.float32)
    raw_right_wrist = np.zeros((T, 2), dtype=np.float32)
    raw_left_torso = np.zeros((T, 2), dtype=np.float32)
    raw_right_torso = np.zeros((T, 2), dtype=np.float32)
    raw_left_ext = np.zeros(T, dtype=np.float32)
    raw_right_ext = np.zeros(T, dtype=np.float32)

    for t, r in enumerate(results):
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        # Filter detected persons to on-piste competitors
        # (rejects foreground referee at y > 0.78 and background spectators at y < 0.35)
        candidates = []
        for b, k in zip(r.boxes, r.keypoints):
            xyxy = b.xyxy[0].cpu().numpy()
            h = xyxy[3] - xyxy[1]
            center_x = (xyxy[0] + xyxy[2]) / (2.0 * max(W, 1))
            center_y = (xyxy[1] + xyxy[3]) / (2.0 * max(H, 1))
            if 0.30 <= center_y <= 0.78 and h >= 50:
                kpts_norm = k.xy[0].cpu().numpy().copy()
                kpts_norm[:, 0] /= max(W, 1)
                kpts_norm[:, 1] /= max(H, 1)
                candidates.append((center_x, h, kpts_norm))

        if len(candidates) == 0:
            continue

        # Sort by horizontal position on the piste
        candidates.sort(key=lambda c: c[0])
        k_left = candidates[0][2]
        k_right = candidates[-1][2] if len(candidates) > 1 else candidates[0][2]

        l_wrist, l_torso, l_ext = _get_fencer_metrics(k_left, is_left_fencer=True)
        r_wrist, r_torso, r_ext = _get_fencer_metrics(k_right, is_left_fencer=False)

        raw_left_wrist[t] = l_wrist
        raw_right_wrist[t] = r_wrist
        raw_left_torso[t] = l_torso
        raw_right_torso[t] = r_torso
        raw_left_ext[t] = l_ext
        raw_right_ext[t] = r_ext

    # Compute first-order velocities (differences across time)
    v_left_wrist = np.zeros_like(raw_left_wrist)
    v_right_wrist = np.zeros_like(raw_right_wrist)
    v_left_wrist[1:] = (raw_left_wrist[1:] - raw_left_wrist[:-1]) * 10.0  # scaled
    v_right_wrist[1:] = (raw_right_wrist[1:] - raw_right_wrist[:-1]) * 10.0

    # Compute accelerations (jerk/impulse spikes)
    a_left_wrist = np.zeros(T, dtype=np.float32)
    a_right_wrist = np.zeros(T, dtype=np.float32)
    a_left_wrist[1:] = np.abs(v_left_wrist[1:, 0] - v_left_wrist[:-1, 0]) * 10.0
    a_right_wrist[1:] = np.abs(v_right_wrist[1:, 0] - v_right_wrist[:-1, 0]) * 10.0

    # Torso advance velocity (X component)
    v_left_torso = np.zeros(T, dtype=np.float32)
    v_right_torso = np.zeros(T, dtype=np.float32)
    v_left_torso[1:] = (raw_left_torso[1:, 0] - raw_left_torso[:-1, 0]) * 10.0
    v_right_torso[1:] = (raw_right_torso[1:, 0] - raw_right_torso[:-1, 0]) * 10.0

    # Interaction: separation and closure rate
    distance = raw_right_wrist[:, 0] - raw_left_wrist[:, 0]
    closure = np.zeros(T, dtype=np.float32)
    closure[1:] = (distance[:-1] - distance[1:]) * 10.0

    # Assemble 16-dimensional feature vector per frame
    # Left: [wrist_x, wrist_y, vx, vy, ax, ext, torso_vx] = 7
    # Right: [wrist_x, wrist_y, vx, vy, ax, ext, torso_vx] = 7
    # Interaction: [distance, closure] = 2
    # Total = 16
    features = np.column_stack(
        [
            raw_left_wrist,         # 2 (0, 1)
            v_left_wrist,           # 2 (2, 3)
            a_left_wrist,           # 1 (4)
            raw_left_ext,           # 1 (5)
            v_left_torso,           # 1 (6)
            raw_right_wrist,        # 2 (7, 8)
            v_right_wrist,          # 2 (9, 10)
            a_right_wrist,          # 1 (11)
            raw_right_ext,          # 1 (12)
            v_right_torso,          # 1 (13)
            distance,               # 1 (14)
            closure,                # 1 (15)
        ]
    ).astype(np.float32)

    return features


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


def flip_kinematic_features(kinematics: torch.Tensor) -> torch.Tensor:
    """
    Apply horizontal reflection to kinematic features tensor of shape (..., 16):
    - Swaps Left fencer (indices 0..6) and Right fencer (indices 7..13).
    - Reflects horizontal coordinates (x -> 1.0 - x) and velocities (vx -> -vx).
    """
    flipped = kinematics.clone()
    left_feat = kinematics[..., :7].clone()
    right_feat = kinematics[..., 7:14].clone()

    # Reflect X position (0), X velocity (2), and torso X velocity (6)
    left_feat[..., 0] = 1.0 - left_feat[..., 0]
    left_feat[..., 2] = -left_feat[..., 2]
    left_feat[..., 6] = -left_feat[..., 6]

    right_feat[..., 0] = 1.0 - right_feat[..., 0]
    right_feat[..., 2] = -right_feat[..., 2]
    right_feat[..., 6] = -right_feat[..., 6]

    flipped[..., :7] = right_feat
    flipped[..., 7:14] = left_feat
    return flipped

