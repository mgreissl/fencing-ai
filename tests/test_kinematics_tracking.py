"""Tests for the fencer tracker and left/right identity assignment (#30)."""

import numpy as np
import pytest

from src.kinematics import (
    FrameDetections,
    L_SHOULDER,
    L_WRIST,
    R_SHOULDER,
    R_WRIST,
    pose_results_to_detections,
    track_detections,
    track_fencers,
)


def make_skeleton(cx: float, cy: float, h: float) -> np.ndarray:
    """Crude normalized (17, 2) stick figure centred at (cx, cy) with height h."""
    kpts = np.tile([cx, cy], (17, 1)).astype(np.float64)
    kpts[[L_SHOULDER, R_SHOULDER], 1] = cy - 0.25 * h
    kpts[[L_WRIST, R_WRIST], 1] = cy - 0.2 * h
    kpts[[11, 12], 1] = cy + 0.05 * h
    return kpts


def make_frame(people: list[tuple[float, float, float]]) -> FrameDetections:
    """Build detections for people given as (cx, cy, h) in normalized coords."""
    if not people:
        return FrameDetections.empty()
    boxes = np.array(
        [[cx - 0.2 * h, cy - 0.5 * h, cx + 0.2 * h, cy + 0.5 * h] for cx, cy, h in people]
    )
    kpts = np.stack([make_skeleton(cx, cy, h) for cx, cy, h in people])
    return FrameDetections(
        boxes=boxes,
        keypoints=kpts,
        keypoint_conf=np.ones(kpts.shape[:2]),
    )


def test_tracker_keeps_identities_when_fencers_cross():
    n = 40
    frames = []
    for t in range(n):
        a = (0.2 + 0.02 * t, 0.50, 0.3)  # starts left, moves right
        b = (0.8 - 0.02 * t, 0.52, 0.3)  # starts right, moves left
        people = [b, a] if t % 2 else [a, b]  # detection order must not matter
        if t == 15:
            people = [a]  # occlusion at the crossing point
        frames.append(make_frame(people))

    tracks = track_detections(frames)
    long_tracks = [tr for tr in tracks if len(tr.frame_idx) > 30]
    assert len(long_tracks) == 2

    fencers = track_fencers(frames, fps=25.0)
    left_x = fencers.keypoints[:, 0, 11, 0]
    right_x = fencers.keypoints[:, 1, 11, 0]
    valid = fencers.detected
    # Left identity started on the left and ends on the right after crossing.
    assert left_x[0] < 0.3 and left_x[-1] > 0.9
    assert right_x[0] > 0.7 and right_x[-1] < 0.1
    # Motion of each identity is monotone: no swap back and forth.
    assert np.all(np.diff(left_x[valid[:, 0]]) > 0)
    assert np.all(np.diff(right_x[valid[:, 1]]) < 0)


def test_single_detection_is_not_duplicated():
    frames = [make_frame([(0.3, 0.5, 0.3)]) for _ in range(20)]
    fencers = track_fencers(frames, fps=25.0)
    assert fencers.detected[:, 0].all()
    assert not fencers.detected[:, 1].any()
    assert np.all(fencers.keypoints[:, 1] == 0)


def test_single_detection_on_right_half_goes_to_right_fencer():
    frames = [make_frame([(0.7, 0.5, 0.3)]) for _ in range(20)]
    fencers = track_fencers(frames, fps=25.0)
    assert not fencers.detected[:, 0].any()
    assert fencers.detected[:, 1].all()


def test_referee_coach_and_spectators_are_rejected():
    frames = []
    for t in range(30):
        people = [
            (0.5, 0.88, 0.35),  # foreground referee, below the piste band
            (0.3 + 0.002 * t, 0.5, 0.3),  # left fencer
            (0.7 - 0.002 * t, 0.5, 0.3),  # right fencer
            (0.95, 0.5, 0.04),  # tiny coach far away
            (0.5, 0.15, 0.2),  # spectator in the stands
        ]
        if t % 7 == 0:
            people.append((0.1, 0.5, 0.3))  # short-lived flicker detection
        frames.append(make_frame(people))
    fencers = track_fencers(frames, fps=25.0)
    assert fencers.detected.all()
    np.testing.assert_allclose(fencers.keypoints[0, 0, 11, 0], 0.3)
    np.testing.assert_allclose(fencers.keypoints[0, 1, 11, 0], 0.7)
    assert np.all(np.abs(fencers.keypoints[:, :, 11, 1] - 0.5 - 0.015) < 1e-9)


def test_no_detections_gives_all_invalid():
    frames = [FrameDetections.empty() for _ in range(5)]
    fencers = track_fencers(frames, fps=30.0)
    assert fencers.keypoints.shape == (5, 2, 17, 2)
    assert not fencers.detected.any()
    assert fencers.fps == 30.0


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)


class _FakeResult:
    """Mimics the parts of an ultralytics Results object the adapter touches."""

    def __init__(self, boxes_px, kpts_px, conf, hw):
        self.orig_shape = hw
        self.boxes = type("B", (), {"xyxy": _FakeTensor(boxes_px)})()
        self.keypoints = type(
            "K",
            (),
            {
                "xy": _FakeTensor(kpts_px),
                "conf": None if conf is None else _FakeTensor(conf),
            },
        )()


def test_pose_results_adapter_normalizes_coordinates():
    h, w = 200, 400
    kpts = np.full((1, 17, 2), [100.0, 50.0])
    res = _FakeResult([[40.0, 20.0, 120.0, 180.0]], kpts, np.full((1, 17), 0.9), (h, w))
    det = pose_results_to_detections(res)
    np.testing.assert_allclose(det.boxes, [[0.1, 0.1, 0.3, 0.9]])
    np.testing.assert_allclose(det.keypoints[0, 0], [0.25, 0.25])
    np.testing.assert_allclose(det.keypoint_conf, 0.9, rtol=1e-6)


def test_pose_results_adapter_handles_missing_conf_and_empty():
    res = _FakeResult(np.zeros((0, 4)), np.zeros((0, 17, 2)), None, (10, 10))
    det = pose_results_to_detections(res)
    assert len(det) == 0
    res = _FakeResult([[0, 0, 5, 5]], np.ones((1, 17, 2)), None, (10, 10))
    det = pose_results_to_detections(res)
    assert det.keypoint_conf.shape == (1, 17)
    assert np.all(det.keypoint_conf == 1.0)


@pytest.mark.parametrize("n_frames", [1, 3])
def test_short_clips_do_not_crash(n_frames):
    frames = [make_frame([(0.3, 0.5, 0.3), (0.7, 0.5, 0.3)]) for _ in range(n_frames)]
    fencers = track_fencers(frames, fps=25.0)
    assert fencers.detected.all()
