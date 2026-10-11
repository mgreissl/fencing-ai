"""Tests for score stabilisation over a sequence of reads (#9)."""

import numpy as np
import pytest

from preprocess_helpers import make_timeline
from scripts import preprocess_pipeline as pp

FPS = 25.0
STRIDE = 6  # frames between score samples (~0.25 s)
ONSET = 1000


def samples_from(fn, start=0, stop=2000, stride=STRIDE):
    frames = np.arange(start, stop, stride)
    return frames, [fn(int(f)) for f in frames]


def step(before, after, at):
    return lambda f: before if f < at else after


# --- read_score_before -----------------------------------------------------

def test_before_stable():
    frames, scores = samples_from(lambda f: (3, 2))
    assert pp.read_score_before(frames, scores, ONSET, n_stable=3) == ((3, 2), None)


def test_before_uses_only_reads_before_onset():
    frames, scores = samples_from(step((3, 2), (4, 2), ONSET))
    assert pp.read_score_before(frames, scores, ONSET, n_stable=3)[0] == (3, 2)


def test_before_unstable_when_reads_disagree():
    frames, scores = samples_from(step((3, 2), (4, 2), ONSET - 8))
    score, reason = pp.read_score_before(frames, scores, ONSET, n_stable=3)
    assert score is None and reason == "score_unstable_before"


def test_before_unreadable_when_all_reads_missing():
    frames, scores = samples_from(lambda f: None)
    score, reason = pp.read_score_before(frames, scores, ONSET, n_stable=3)
    assert score is None and reason == "score_unreadable_before"


def test_before_partly_missing_is_unstable():
    frames, scores = samples_from(lambda f: None if f > ONSET - 12 else (3, 2))
    score, reason = pp.read_score_before(frames, scores, ONSET, n_stable=3)
    assert score is None and reason == "score_unstable_before"


def test_before_too_few_reads():
    frames = np.array([ONSET - 6])
    assert pp.read_score_before(frames, [(3, 2)], ONSET, 3)[1] == "score_unreadable_before"


# --- score_window_end ------------------------------------------------------

def test_window_end_closed_by_next_onset():
    assert pp.score_window_end(1000, 1300, 5000, FPS, 12.0) == (1299, "next_onset")


def test_window_end_closed_by_timeout():
    assert pp.score_window_end(1000, 5000, 9000, FPS, 12.0) == (1300, "timeout")
    assert pp.score_window_end(1000, None, 9000, FPS, 12.0) == (1300, "timeout")


def test_window_end_closed_by_video_end():
    assert pp.score_window_end(1000, None, 1100, FPS, 12.0) == (1099, "video_end")


# --- resolve_score_after ---------------------------------------------------

def resolve(fn, end=1299, closed_by="next_onset", before=(3, 2), n_stable=4,
            min_unchanged_s=3.0, onset=ONSET):
    frames, scores = samples_from(fn)
    return pp.resolve_score_after(
        frames, scores, before, onset, end, closed_by, FPS, n_stable, min_unchanged_s
    )


def test_after_change_after_one_second():
    result = resolve(step((3, 2), (4, 2), ONSET + 25))
    assert (result.kind, result.score) == ("changed", (4, 2))


def test_after_delayed_decision_is_not_missed():
    # Referee decision 7 s after the light; the old fixed read at ~1.4 s
    # would have seen the unchanged score and called it a tie.
    result = resolve(step((3, 2), (4, 2), ONSET + 175), end=ONSET + 300,
                     closed_by="timeout")
    assert (result.kind, result.score) == ("changed", (4, 2))


def test_after_single_misread_is_ignored():
    result = resolve(lambda f: (9, 9) if f == ONSET + 62 else
                     ((4, 2) if f > ONSET + 25 else (3, 2)))
    assert (result.kind, result.score) == ("changed", (4, 2))


def test_after_unchanged_confirmed_by_next_onset():
    result = resolve(lambda f: (3, 2), end=ONSET + 200, closed_by="next_onset")
    assert (result.kind, result.score) == ("unchanged", (3, 2))


def test_after_unchanged_confirmed_by_timeout():
    result = resolve(lambda f: (3, 2), end=ONSET + 300, closed_by="timeout")
    assert result.kind == "unchanged"


def test_after_unchanged_but_next_exchange_starts_too_soon_is_undecided():
    result = resolve(lambda f: (3, 2), end=ONSET + 50, closed_by="next_onset")
    assert result.kind == "undecided"


def test_after_unchanged_at_video_end_is_undecided():
    result = resolve(lambda f: (3, 2), end=ONSET + 200, closed_by="video_end")
    assert result.kind == "undecided"


def test_after_change_still_in_progress_at_window_end_is_undecided():
    # Only two reads of the new value before the window closes.
    result = resolve(step((3, 2), (4, 2), ONSET + 190), end=ONSET + 200,
                     closed_by="timeout")
    assert result.kind == "undecided"


def test_after_two_changes_are_unstable():
    def fn(f):
        if f < ONSET + 25:
            return (3, 2)
        return (4, 2) if f < ONSET + 100 else (4, 3)

    assert resolve(fn).kind == "unstable"


def test_after_revert_is_unstable():
    def fn(f):
        return (4, 2) if ONSET + 25 <= f < ONSET + 100 else (3, 2)

    assert resolve(fn).kind == "unstable"


def test_after_all_missing_is_unreadable():
    assert resolve(lambda f: None).kind == "unreadable"


# --- analyse_timeline integration ------------------------------------------

def test_delayed_decision_is_labelled_by_the_eventual_score_not_tie():
    timeline = make_timeline(
        n_frames=2500,
        runs=[(ONSET, ONSET + 75)],
        left_state=pp.STATE_ON, right_state=pp.STATE_ON,
        scores=step((3, 2), (3, 3), ONSET + 175),
    )
    (exchange,) = pp.accepted_exchanges(pp.analyse_timeline(timeline, "sabre"))
    assert exchange.label == "R"


def test_unchanged_score_until_next_light_is_a_tie():
    second = ONSET + 300
    timeline = make_timeline(
        n_frames=2500,
        runs=[(ONSET, ONSET + 75), (second, second + 75)],
        left_state=pp.STATE_ON, right_state=pp.STATE_ON,
        scores=step((3, 2), (4, 2), second + 100),
    )
    exchanges = pp.accepted_exchanges(pp.analyse_timeline(timeline, "sabre"))
    assert [(e.onset_frame, e.label) for e in exchanges] == [(ONSET, "T"), (second, "L")]


def test_undecided_exchange_taints_the_next_one():
    second = ONSET + 60  # next light 2.4 s later, score still unchanged
    timeline = make_timeline(
        n_frames=2500,
        runs=[(ONSET, ONSET + 30), (second, second + 30)],
        left_state=pp.STATE_ON, right_state=pp.STATE_ON,
        scores=step((3, 2), (4, 2), second + 150),
    )
    exchanges = pp.analyse_timeline(timeline, "sabre")
    reasons = {e.onset_frame: e.reason for e in exchanges}
    assert reasons[ONSET] == "score_unchanged_undecided"
    assert reasons[second] == "previous_exchange_undecided"
    assert pp.accepted_exchanges(exchanges) == []
