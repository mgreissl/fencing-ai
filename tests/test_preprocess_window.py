"""Tests for light-onset aligned windows and onset detection (#6)."""

import numpy as np
import pytest

from scripts import preprocess_pipeline as pp


@pytest.mark.parametrize("fps", [25.0, 30.0, 50.0])
def test_window_is_aligned_to_onset(fps):
    window = pp.compute_window(onset_frame=1000, source_fps=fps, pre_s=2.0, post_s=0.3)
    n_pre = round(2.0 * fps)
    n_post = round(0.3 * fps)
    assert window.n_frames == n_pre + n_post
    assert window.light_frame_in_clip == n_pre
    assert window.src_indices[window.light_frame_in_clip] == 1000
    assert window.src_indices[0] == 1000 - n_pre
    assert window.src_indices[-1] == 1000 + n_post - 1
    assert window.clip_fps == fps


def test_window_frames_are_contiguous_at_native_fps():
    window = pp.compute_window(500, 25.0)
    diffs = np.diff(window.src_indices)
    assert (diffs == 1).all()


def test_window_outside_video_is_rejected():
    assert pp.compute_window(onset_frame=20, source_fps=25.0) is None
    assert pp.compute_window(onset_frame=995, source_fps=25.0, total_frames=1000) is None
    assert pp.compute_window(onset_frame=900, source_fps=25.0, total_frames=1000)


def lit_from_runs(n, runs):
    lit = np.zeros(n, dtype=bool)
    for start, stop in runs:
        lit[start:stop] = True
    return lit


def test_onsets_one_per_light_run():
    lit = lit_from_runs(1000, [(200, 275), (500, 580)])
    assert pp.find_light_onsets(lit, min_dark_frames=12) == [200, 500]


def test_flicker_inside_a_run_is_not_a_new_onset():
    lit = lit_from_runs(1000, [(200, 220), (225, 275)])
    assert pp.find_light_onsets(lit, min_dark_frames=12) == [200]


def test_light_already_on_at_scan_start_is_ignored():
    lit = lit_from_runs(1000, [(100, 160), (400, 450)])
    assert pp.find_light_onsets(lit, min_dark_frames=12, first_valid=100) == [400]


def test_onset_needs_dark_frames_after_scan_start():
    lit = lit_from_runs(1000, [(105, 150)])
    assert pp.find_light_onsets(lit, min_dark_frames=12, first_valid=100) == []


def test_aggregate_hit_type_merges_late_second_light():
    left = [0, 1, 1, 1]
    right = [0, 0, 0, 1]
    assert pp.aggregate_hit_type(left, right) == "On-On"


@pytest.mark.parametrize(
    "left, right, expected",
    [
        ([0, 0], [0, 0], "No-No"),
        ([1, 1], [0, 0], "On-No"),
        ([0, 2], [1, 1], "Off-On"),
        ([2, 1], [0, 0], "On-No"),
        ([2, 2], [2, 2], "Off-Off"),
    ],
)
def test_aggregate_hit_type(left, right, expected):
    assert pp.aggregate_hit_type(left, right) == expected


def make_timeline(fps=25.0, n_frames=2000, runs=(), left_state=pp.STATE_ON,
                  right_state=pp.STATE_NO, scores=None, stride=6):
    left = np.zeros(n_frames, dtype=np.int8)
    right = np.zeros(n_frames, dtype=np.int8)
    for start, stop in runs:
        left[start:stop] = left_state
        right[start:stop] = right_state
    score_frames = np.arange(0, n_frames, stride)
    score_list = [scores(i) if callable(scores) else scores for i in score_frames]
    return pp.Timeline(
        fps=fps, n_frames=n_frames, first_valid=0,
        left_state=left, right_state=right,
        score_frames=score_frames, scores=score_list,
    )


def test_analyse_timeline_emits_window_aligned_candidate():
    onset = 1000
    timeline = make_timeline(
        runs=[(onset, onset + 75)],
        scores=lambda frame: (3, 2) if frame < onset + 10 else (4, 2),
    )
    candidates = pp.analyse_timeline(timeline, "sabre")
    assert len(candidates) == 1
    cand = candidates[0]
    assert (cand.onset_frame, cand.hit_type, cand.label) == (onset, "On-No", "L")
    assert cand.window.src_indices[cand.window.light_frame_in_clip] == onset
    assert cand.score_before == (3, 2) and cand.score_after == (4, 2)


def test_analyse_timeline_skips_unreadable_scores():
    timeline = make_timeline(runs=[(1000, 1075)], scores=None)
    assert pp.analyse_timeline(timeline, "sabre") == []


def test_analyse_timeline_skips_window_before_video_start():
    timeline = make_timeline(
        runs=[(20, 95)], scores=lambda frame: (3, 2) if frame < 30 else (4, 2)
    )
    assert pp.analyse_timeline(timeline, "sabre") == []
