"""Tests for per-clip weapon-side selection (#31)."""

import numpy as np

from src.kinematics import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    compute_fencer_signals,
    determine_weapon_wrist,
)


def make_pose(n_frames: int, cx: float, lead_dx: float, back_dx: float, lead: int):
    """(T, 17, 2) poses with the `lead` wrist at cx + lead_dx, the other at cx + back_dx."""
    kpts = np.tile([cx, 0.5], (n_frames, 17, 1)).astype(np.float64)
    kpts[:, [L_SHOULDER, R_SHOULDER], 1] = 0.4
    kpts[:, [L_HIP, R_HIP], 1] = 0.55
    back = L_WRIST if lead == R_WRIST else R_WRIST
    kpts[:, lead, 0] = cx + lead_dx
    kpts[:, back, 0] = cx + back_dx
    kpts[:, [L_WRIST, R_WRIST], 1] = 0.42
    return kpts


def test_majority_vote_ignores_back_arm_swinging_forward():
    n = 50
    kpts = make_pose(n, 0.3, lead_dx=0.08, back_dx=-0.05, lead=R_WRIST)
    # Fleche: back arm swings ahead of the weapon arm in 15 frames.
    kpts[30:45, L_WRIST, 0] = 0.3 + 0.12
    conf = np.ones((n, 17))
    valid = np.ones(n, dtype=bool)
    assert determine_weapon_wrist(kpts, conf, valid, facing=1) == R_WRIST

    sig = compute_fencer_signals(kpts, conf, valid, weapon_wrist=R_WRIST)
    # The weapon wrist stays the right wrist in every frame, also during the fleche.
    np.testing.assert_allclose(sig.wrist[:, 0], kpts[:, R_WRIST, 0])


def test_right_fencer_facing_left_left_handed():
    n = 20
    kpts = make_pose(n, 0.7, lead_dx=-0.08, back_dx=0.05, lead=L_WRIST)
    conf = np.ones((n, 17))
    assert determine_weapon_wrist(kpts, conf, np.ones(n, bool), facing=-1) == L_WRIST


def test_low_confidence_and_invalid_frames_do_not_vote():
    n = 20
    kpts = make_pose(n, 0.3, lead_dx=0.08, back_dx=-0.05, lead=L_WRIST)
    kpts[:12, R_WRIST, 0] = 0.5  # noisy right wrist far ahead ...
    conf = np.ones((n, 17))
    conf[:8, R_WRIST] = 0.05  # ... but low confidence in 8 frames
    valid = np.ones(n, dtype=bool)
    valid[8:12] = False  # and the fencer is not detected in 4 more
    assert determine_weapon_wrist(kpts, conf, valid, facing=1) == L_WRIST


def test_default_is_right_wrist_without_votes():
    n = 5
    kpts = np.zeros((n, 17, 2))
    conf = np.zeros((n, 17))
    assert determine_weapon_wrist(kpts, conf, np.zeros(n, bool), facing=1) == R_WRIST


def test_signals_validity_and_extension():
    n = 4
    kpts = make_pose(n, 0.3, lead_dx=0.15, back_dx=-0.05, lead=R_WRIST)
    conf = np.ones((n, 17))
    conf[2, R_SHOULDER] = 0.1  # weapon shoulder unreliable -> frame invalid
    detected = np.array([True, True, True, False])
    sig = compute_fencer_signals(kpts, conf, detected, weapon_wrist=R_WRIST)
    np.testing.assert_array_equal(sig.valid, [True, True, False, False])
    # Torso = hip midpoint; scale = |shoulder_mid - hip_mid| = 0.15.
    np.testing.assert_allclose(sig.torso[0], [0.3, 0.55])
    expected_ext = np.hypot(0.15, 0.02) / 0.15
    np.testing.assert_allclose(sig.extension[:2], expected_ext)
    assert np.all(sig.wrist[~sig.valid] == 0)
    assert np.all(sig.extension[~sig.valid] == 0)
