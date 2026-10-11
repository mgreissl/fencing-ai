"""Tests for missing-detection handling: validity masks and gap filling (#29)."""

import numpy as np
import pytest

from src.kinematics import contiguous_runs, fill_short_gaps


def test_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert contiguous_runs(mask) == [(1, 3), (5, 6), (7, 10)]
    assert contiguous_runs(np.zeros(3, bool)) == []
    assert contiguous_runs(np.ones(3, bool)) == [(0, 3)]


def test_short_gap_is_linearly_interpolated():
    t = np.arange(10, dtype=np.float64)
    values = np.stack([0.1 * t, 0.5 + 0.0 * t], axis=1)
    valid = np.ones(10, dtype=bool)
    valid[3:6] = False
    values[~valid] = 0.0  # missing detections arrive as zeros
    filled, new_valid = fill_short_gaps(values, valid, max_gap=3)
    assert new_valid.all()
    np.testing.assert_allclose(filled[:, 0], 0.1 * t)
    np.testing.assert_allclose(filled[:, 1], 0.5)


def test_long_gap_stays_invalid_and_zero():
    values = np.arange(12, dtype=np.float64)
    valid = np.ones(12, dtype=bool)
    valid[2:7] = False  # 5 missing frames
    filled, new_valid = fill_short_gaps(values, valid, max_gap=4)
    np.testing.assert_array_equal(new_valid, valid)
    assert np.all(filled[~new_valid] == 0)
    np.testing.assert_allclose(filled[new_valid], values[new_valid])


def test_edge_gaps_are_not_extrapolated():
    values = np.arange(8, dtype=np.float64)
    valid = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    filled, new_valid = fill_short_gaps(values, valid, max_gap=5)
    np.testing.assert_array_equal(new_valid, valid)
    assert np.all(filled[~valid] == 0)


@pytest.mark.parametrize("max_gap", [0, -1])
def test_no_filling_when_disabled(max_gap):
    values = np.arange(5, dtype=np.float64)
    valid = np.array([1, 0, 1, 1, 1], dtype=bool)
    filled, new_valid = fill_short_gaps(values, valid, max_gap=max_gap)
    np.testing.assert_array_equal(new_valid, valid)
    assert filled[1] == 0


def test_all_invalid_and_empty():
    filled, new_valid = fill_short_gaps(np.ones((4, 2)), np.zeros(4, bool), max_gap=3)
    assert not new_valid.any() and np.all(filled == 0)
    filled, new_valid = fill_short_gaps(np.zeros((0, 2)), np.zeros(0, bool), max_gap=3)
    assert filled.shape == (0, 2) and new_valid.shape == (0,)
