"""Score changes after the timeout window (video review) are caught (#53)."""

import pytest

from preprocess_helpers import make_timeline
from scripts import preprocess_pipeline as pp


def timeline(score_fn, runs, n_frames=2000):
    return make_timeline(runs=runs, n_frames=n_frames, left_state=pp.STATE_ON,
                         right_state=pp.STATE_NO, scores=score_fn)


def schedule(*steps):
    """Score by frame: steps are (first_frame, score), sorted."""
    def fn(frame):
        current = steps[0][1]
        for first, score in steps:
            if frame >= first:
                current = score
        return current
    return fn


def analyse(tl):
    return pp.analyse_timeline(tl, "sabre")


def test_overturned_touch_after_timeout_is_rejected():
    # +1 left at 620, overturned (back to 5-5) at 1000, after the 12 s timeout
    fn = schedule((0, (5, 5)), (620, (6, 5)), (1000, (5, 5)))
    first, second = analyse(timeline(fn, [(600, 700), (1500, 1600)]))
    assert first.label is None and first.reason == "score_changed_later"
    assert second.reason != "score_changed_later"


def test_further_change_before_video_end_is_rejected():
    fn = schedule((0, (5, 5)), (620, (6, 5)), (1000, (7, 5)))
    (only,) = analyse(timeline(fn, [(600, 700)]))
    assert only.reason == "score_changed_later"


def test_stable_score_stays_accepted():
    fn = schedule((0, (5, 5)), (620, (6, 5)))
    first, second = analyse(timeline(fn, [(600, 700), (1500, 1600)]))
    assert first.label == "L" and first.score_after == (6, 5)


def test_unchanged_score_stays_accepted_for_tie():
    fn = schedule((0, (5, 5)))
    tl = make_timeline(runs=[(600, 700), (1500, 1600)], scores=fn,
                       left_state=pp.STATE_ON, right_state=pp.STATE_ON)
    assert analyse(tl)[0].label == "T"


def test_unreadable_late_score_does_not_reject():
    fn = lambda f: (5, 5) if f < 620 else ((6, 5) if f < 1100 else None)
    first, _ = analyse(timeline(fn, [(600, 700), (1500, 1600)]))
    assert first.label == "L"


def test_changes_beyond_lookahead_limit_are_ignored():
    # bout reset to 0-0 long after the touch must not reject it
    reset = int(pp.PipelineConfig().late_score_max_s * 25) + 800
    fn = schedule((0, (5, 5)), (620, (6, 5)), (reset, (0, 0)))
    tl = timeline(fn, [(600, 700), (reset + 500, reset + 600)],
                  n_frames=reset + 1000)
    assert analyse(tl)[0].label == "L"


def test_rejection_is_counted_like_other_reasons():
    fn = schedule((0, (5, 5)), (620, (6, 5)), (1000, (5, 5)))
    exchanges = analyse(timeline(fn, [(600, 700), (1500, 1600)]))
    assert pp.count_reasons(exchanges)["score_changed_later"] == 1
