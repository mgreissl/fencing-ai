"""Tests for source-fps handling in scripts/preprocess_pipeline.py (#5)."""

import math

import pytest

from scripts import preprocess_pipeline as pp


@pytest.mark.parametrize("raw, expected", [(25.0, 25.0), (30.0, 30.0), (29.97, 29.97)])
def test_valid_source_fps(raw, expected):
    assert pp.validate_source_fps(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [0.0, -1.0, float("nan"), float("inf"), 1.0, 500.0])
def test_invalid_source_fps(raw):
    assert pp.validate_source_fps(raw) is None


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0, 50.0, 59.94])
def test_frame_time_roundtrip(fps):
    assert pp.seconds_to_frames(pp.frame_to_seconds(1000, fps), fps) == 1000


def test_seek_time_does_not_assume_25_fps():
    # Frame 30000 of a 30 fps video is t = 1000 s, not 1200 s (30000 / 25).
    assert pp.frame_to_seconds(30000, 30.0) == pytest.approx(1000.0)
    assert not math.isclose(pp.frame_to_seconds(30000, 30.0), 30000 / 25.0)


def test_durations_scale_with_fps():
    assert pp.seconds_to_frames(pp.JUMP_S, 25.0) == 260
    assert pp.seconds_to_frames(pp.HIDE_S, 25.0) == 200
    assert pp.seconds_to_frames(pp.HIDE_S, 30.0) == 240
