"""Tests for full-rate kinematic features, frame selection and flipping (#28)."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings as hyp_settings, strategies as st

from src.kinematics import (
    FENCER_FEATURE_NAMES,
    FrameDetections,
    FencerTracks,
    KINEMATIC_FEATURE_DIM,
    KINEMATIC_FEATURE_NAMES,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    compute_kinematic_features,
    extract_pose_tracks,
    flip_kinematic_features,
    masked_derivative,
    select_kinematic_frames,
    track_fencers,
)

F = len(FENCER_FEATURE_NAMES)
IDX = {name: i for i, name in enumerate(KINEMATIC_FEATURE_NAMES)}

# COCO left/right keypoint pairs, swapped by a horizontal mirror.
_MIRROR_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def skeleton(cx: float, cy: float = 0.5, h: float = 0.3, facing: int = 1) -> np.ndarray:
    """Right-handed fencer: right wrist extended towards the opponent."""
    kpts = np.tile([cx, cy], (17, 1)).astype(np.float64)
    kpts[[L_SHOULDER, R_SHOULDER], 1] = cy - 0.25 * h
    kpts[L_SHOULDER, 0] = cx - 0.02 * facing
    kpts[R_SHOULDER, 0] = cx + 0.02 * facing
    kpts[[L_HIP, R_HIP], 1] = cy + 0.05 * h
    kpts[R_WRIST] = [cx + 0.1 * facing, cy - 0.2 * h]
    kpts[L_WRIST] = [cx - 0.05 * facing, cy - 0.3 * h]
    return kpts


def make_tracks(left_x, right_x, fps: float) -> FencerTracks:
    """FencerTracks from per-frame hip x positions; NaN marks a missing detection."""
    n = len(left_x)
    kpts = np.zeros((n, 2, 17, 2))
    detected = np.zeros((n, 2), dtype=bool)
    for slot, (xs, facing) in enumerate(((left_x, 1), (right_x, -1))):
        for t, x in enumerate(xs):
            if not np.isnan(x):
                kpts[t, slot] = skeleton(x, facing=facing)
                detected[t, slot] = True
    conf = detected[..., None].astype(np.float64) * np.ones(17)
    return FencerTracks(keypoints=kpts, keypoint_conf=conf, detected=detected, fps=fps)


def test_layout_constants_match_output():
    assert KINEMATIC_FEATURE_DIM == len(KINEMATIC_FEATURE_NAMES) == 2 * F + 3
    tracks = make_tracks(np.linspace(0.2, 0.3, 10), np.linspace(0.8, 0.7, 10), 25.0)
    feats = compute_kinematic_features(tracks)
    assert feats.shape == (10, KINEMATIC_FEATURE_DIM)
    assert feats.dtype == np.float32


@pytest.mark.parametrize("smoothing_window", [0, 5])
@pytest.mark.parametrize("order", [1, 2])
def test_masked_derivative_uses_true_dt(order, smoothing_window):
    fps = 30.0
    t = np.arange(40) / fps
    values = 0.1 + 1.5 * t + 2.0 * t**2
    valid = np.ones(40, dtype=bool)
    deriv = masked_derivative(values, valid, fps, order=order, smoothing_window=smoothing_window)
    expected = 1.5 + 4.0 * t if order == 1 else np.full_like(t, 4.0)
    # np.gradient is exact for quadratics in the interior, one-sided at the edges.
    np.testing.assert_allclose(deriv[2:-2], expected[2:-2], rtol=1e-6)


def test_masked_derivative_never_crosses_invalid_frames():
    values = np.concatenate([np.full(10, 0.2), np.zeros(5), np.full(10, 0.8)])
    valid = values > 0
    deriv = masked_derivative(values, valid, 25.0, order=1, smoothing_window=5)
    # Both runs are static: no spike at the jump, invalid frames exactly 0.
    np.testing.assert_allclose(deriv, 0.0, atol=1e-9)
    assert np.all(deriv[~valid] == 0)


@pytest.mark.parametrize("fps", [15.0, 30.0, 50.0])
def test_constant_velocity_is_independent_of_fps_and_sampling(fps):
    v = 0.6  # normalized width per second
    n = 60
    left = 0.1 + v * np.arange(n) / fps
    right = 0.9 - 0.5 * v * np.arange(n) / fps
    feats = compute_kinematic_features(make_tracks(left, right, fps))
    np.testing.assert_allclose(feats[:, IDX["left_wrist_vx"]], v, rtol=1e-5)
    np.testing.assert_allclose(feats[:, IDX["right_wrist_vx"]], -0.5 * v, rtol=1e-5)
    np.testing.assert_allclose(feats[:, IDX["left_torso_vx"]], v, rtol=1e-5)
    np.testing.assert_allclose(feats[:, IDX["left_wrist_acc"]], 0.0, atol=1e-5)
    np.testing.assert_allclose(feats[:, IDX["closure"]], 1.5 * v, rtol=1e-5)

    # End-weighted indices: sparse early, dense late. Velocity must not depend on spacing.
    indices = [0, 7, 15, 22, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57]
    sel = select_kinematic_frames(feats, indices)
    np.testing.assert_allclose(sel[:, IDX["left_wrist_vx"]], v, rtol=1e-5)


def test_short_gap_is_filled_without_velocity_spike():
    fps = 25.0
    left = 0.2 + 0.5 * np.arange(30) / fps
    left[10:13] = np.nan
    feats = compute_kinematic_features(make_tracks(left, np.full(30, 0.8), fps))
    assert np.all(feats[:, IDX["left_valid"]] == 1)
    np.testing.assert_allclose(feats[:, IDX["left_wrist_vx"]], 0.5, rtol=1e-5)
    np.testing.assert_allclose(feats[11, IDX["left_wrist_x"]], 0.2 + 0.5 * 11 / fps + 0.1)


def test_long_gap_is_invalid_and_zero():
    fps = 25.0
    left = 0.2 + 0.5 * np.arange(40) / fps
    left[10:25] = np.nan
    feats = compute_kinematic_features(make_tracks(left, np.full(40, 0.8), fps))
    gap = slice(10, 25)
    assert np.all(feats[gap, IDX["left_valid"]] == 0)
    assert np.all(feats[gap, IDX["pair_valid"]] == 0)
    left_block = feats[gap, IDX["left_wrist_x"] : IDX["left_wrist_x"] + F]
    assert np.all(left_block == 0)
    assert np.all(feats[gap, IDX["distance"]] == 0)
    ok = np.r_[0:10, 25:40]
    np.testing.assert_allclose(feats[ok, IDX["left_wrist_vx"]], 0.5, rtol=1e-5)


def test_single_fencer_leaves_other_block_invalid():
    tracks = make_tracks(np.linspace(0.2, 0.4, 20), np.full(20, np.nan), 25.0)
    feats = compute_kinematic_features(tracks)
    assert np.all(feats[:, IDX["left_valid"]] == 1)
    right = feats[:, F : 2 * F]
    assert np.all(right == 0)
    assert np.all(feats[:, 2 * F :] == 0)


def test_select_kinematic_frames_clamps_and_handles_empty():
    feats = np.arange(5 * KINEMATIC_FEATURE_DIM, dtype=np.float32).reshape(5, -1)
    sel = select_kinematic_frames(feats, [0, 2, 7, -1])
    np.testing.assert_array_equal(sel, feats[[0, 2, 4, 0]])
    empty = select_kinematic_frames(np.zeros((0, KINEMATIC_FEATURE_DIM)), [0, 1, 2])
    assert empty.shape == (3, KINEMATIC_FEATURE_DIM) and np.all(empty == 0)


# --- flip ------------------------------------------------------------------


@st.composite
def feature_arrays(draw):
    t = draw(st.integers(1, 12))
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    feats = rng.uniform(-2, 2, size=(t, KINEMATIC_FEATURE_DIM)).astype(np.float32)
    for prefix in ("left_", "right_"):
        start = IDX[prefix + "wrist_x"]
        valid = rng.integers(0, 2, size=t).astype(bool)
        feats[:, IDX[prefix + "valid"]] = valid
        feats[:, start : start + F] *= valid[:, None]
    return feats


@hyp_settings(max_examples=50, deadline=None)
@given(feature_arrays())
def test_flip_is_an_involution(feats):
    # 1 - (1 - x) is exact up to float32 rounding.
    np.testing.assert_allclose(
        flip_kinematic_features(flip_kinematic_features(feats)), feats, atol=1e-6
    )
    tensor = torch.from_numpy(feats)
    twice = flip_kinematic_features(flip_kinematic_features(tensor))
    assert isinstance(twice, torch.Tensor)
    torch.testing.assert_close(twice, tensor)


def test_flip_swap_semantics():
    feats = np.zeros((1, KINEMATIC_FEATURE_DIM), dtype=np.float32)
    feats[0, IDX["left_wrist_x"]] = 0.3
    feats[0, IDX["left_wrist_vx"]] = 1.0
    feats[0, IDX["left_extension"]] = 1.2
    feats[0, IDX["left_valid"]] = 1.0
    feats[0, IDX["distance"]] = 0.4
    out = flip_kinematic_features(feats)
    assert out[0, IDX["right_wrist_x"]] == pytest.approx(0.7)
    assert out[0, IDX["right_wrist_vx"]] == pytest.approx(-1.0)
    assert out[0, IDX["right_extension"]] == pytest.approx(1.2)
    assert out[0, IDX["right_valid"]] == 1.0
    # Invalid fencer stays all-zero instead of x -> 1.
    assert np.all(out[0, :F] == 0)
    assert out[0, IDX["distance"]] == pytest.approx(0.4)


def _detections_from(people_per_frame, mirror: bool) -> list[FrameDetections]:
    frames = []
    for people in people_per_frame:
        boxes, kpts = [], []
        for cx, facing in people:
            k = skeleton(cx, facing=facing)
            box = np.array([cx - 0.06, 0.35, cx + 0.06, 0.65])
            if mirror:
                k = k[_MIRROR_PERM].copy()
                k[:, 0] = 1.0 - k[:, 0]
                box = np.array([1.0 - box[2], box[1], 1.0 - box[0], box[3]])
            boxes.append(box)
            kpts.append(k)
        frames.append(
            FrameDetections(
                boxes=np.array(boxes).reshape(-1, 4),
                keypoints=np.array(kpts).reshape(-1, 17, 2),
                keypoint_conf=np.ones((len(kpts), 17)),
            )
        )
    return frames


def test_flip_matches_features_of_mirrored_video():
    fps = 25.0
    n = 30
    rng = np.random.default_rng(0)
    left = 0.25 + np.cumsum(rng.normal(0.004, 0.003, n))
    right = 0.75 - np.cumsum(rng.normal(0.003, 0.003, n))
    people = [[(left[t], 1), (right[t], -1)] for t in range(n)]
    people[12] = [(right[12], -1)]  # left fencer missing for one frame
    people[20] = []  # nobody detected
    feats = compute_kinematic_features(track_fencers(_detections_from(people, False), fps))
    mirrored = compute_kinematic_features(track_fencers(_detections_from(people, True), fps))
    np.testing.assert_allclose(flip_kinematic_features(feats), mirrored, atol=1e-5)


# --- pose adapter end to end -------------------------------------------------


class _FakePoseModel:
    """Returns empty-but-valid results and records batch sizes."""

    def __init__(self):
        self.batches = []

    def __call__(self, frames, verbose=False):
        self.batches.append(len(frames))
        out = []
        for f in frames:
            r = type("R", (), {})()
            r.orig_shape = f.shape[:2]
            r.boxes = None
            r.keypoints = None
            out.append(r)
        return out


def test_extract_pose_tracks_runs_on_all_frames_of_tensor():
    frames = torch.rand(37, 3, 8, 12)
    model = _FakePoseModel()
    tracks = extract_pose_tracks(frames, pose_model=model, fps=30.0, batch_size=16)
    assert sum(model.batches) == 37 and max(model.batches) <= 16
    assert tracks.num_frames == 37 and tracks.fps == 30.0
    assert not tracks.detected.any()


def test_extract_pose_tracks_reads_video_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    for i in range(23):
        writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
    writer.release()
    model = _FakePoseModel()
    tracks = extract_pose_tracks(str(path), pose_model=model, batch_size=8)
    assert sum(model.batches) == 23
    assert tracks.num_frames == 23
    assert tracks.fps == pytest.approx(20.0)
