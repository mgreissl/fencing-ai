"""Tests for native-fps clips and optional clip_fps resampling (#7)."""

import numpy as np
import pytest

from preprocess_helpers import make_timeline
from scripts import preprocess_pipeline as pp


def test_default_clip_fps_is_native_source_fps():
    for fps in (25.0, 29.97, 30.0, 50.0):
        window = pp.compute_window(2000, fps)
        assert window.clip_fps == fps
        assert (np.diff(window.src_indices) == 1).all()


def test_config_default_keeps_native_fps():
    assert pp.PipelineConfig().clip_fps is None


def test_resampled_window_keeps_onset_exactly():
    window = pp.compute_window(1000, 30.0, clip_fps=15.0)
    assert window.clip_fps == 15.0
    assert window.n_frames == round(2.0 * 15) + round(0.3 * 15)
    assert window.src_indices[window.light_frame_in_clip] == 1000
    assert (np.diff(window.src_indices) == 2).all()


def test_resampled_window_spans_same_duration():
    native = pp.compute_window(1000, 30.0)
    half = pp.compute_window(1000, 30.0, clip_fps=15.0)
    assert half.src_indices[0] == native.src_indices[0]


def test_clip_fps_above_source_is_rejected():
    with pytest.raises(ValueError):
        pp.compute_window(1000, 25.0, clip_fps=30.0)


def test_resampled_window_outside_video_is_rejected():
    assert pp.compute_window(30, 30.0, clip_fps=15.0) is None


def test_analyse_uses_configured_clip_fps():
    onset = 1000
    timeline = make_timeline(
        fps=30.0,
        runs=[(onset, onset + 90)],
        scores=lambda frame: (3, 2) if frame < onset + 10 else (4, 2),
    )
    (cand,) = pp.analyse_timeline(timeline, "sabre", pp.PipelineConfig(clip_fps=15.0))
    assert cand.window.clip_fps == 15.0
    (native,) = pp.analyse_timeline(timeline, "sabre")
    assert native.window.clip_fps == 30.0


def test_cli_clip_fps_option():
    import argparse

    parser = argparse.ArgumentParser()
    pp.add_cli_args(parser)
    cfg = pp.config_from_args(parser.parse_args(["--clip-fps", "15"]))
    assert cfg.clip_fps == 15.0
    assert pp.config_from_args(parser.parse_args([])).clip_fps is None
