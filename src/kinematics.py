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
from typing import Optional, Union

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


def get_pose_model(model_name: str = "yolo11n-pose.pt"):
    """Lazy loader for YOLO pose estimation model."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO

        _YOLO_MODEL = YOLO(model_name)
    return _YOLO_MODEL


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

