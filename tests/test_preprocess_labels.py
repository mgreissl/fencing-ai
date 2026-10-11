"""Tests for per-weapon label semantics and rejected cases (#10)."""

import pytest

from preprocess_helpers import make_timeline
from scripts import preprocess_pipeline as pp

BEFORE = (5, 5)


def label(hit_type, delta, weapon):
    after = (BEFORE[0] + delta[0], BEFORE[1] + delta[1])
    return pp.label_exchange(hit_type, BEFORE, after, weapon)


# (hit_type, (d_left, d_right), expected label) for the priority weapons.
PRIORITY_LABELLED = [
    ("On-On", (1, 0), "L"),
    ("On-On", (0, 1), "R"),
    ("On-On", (0, 0), "T"),
    ("On-No", (1, 0), "L"),
    ("No-On", (0, 1), "R"),
]
FOIL_ONLY_LABELLED = [
    ("On-Off", (1, 0), "L"),
    ("On-Off", (0, 0), "R"),
    ("Off-On", (0, 1), "R"),
    ("Off-On", (0, 0), "L"),
]


@pytest.mark.parametrize("weapon", ["foil", "sabre"])
@pytest.mark.parametrize("hit_type, delta, expected", PRIORITY_LABELLED)
def test_priority_weapons_labelled(weapon, hit_type, delta, expected):
    decision = label(hit_type, delta, weapon)
    assert decision.label == expected
    assert decision.reason is None


@pytest.mark.parametrize("hit_type, delta, expected", FOIL_ONLY_LABELLED)
def test_foil_off_target_priority_labels(hit_type, delta, expected):
    assert label(hit_type, delta, "foil").label == expected


@pytest.mark.parametrize("hit_type", ["On-Off", "Off-On"])
def test_sabre_has_no_off_target_labels(hit_type):
    decision = label(hit_type, (1, 0), "sabre")
    assert decision.label is None and decision.reason == "hit_type_not_labellable"


@pytest.mark.parametrize("weapon", ["foil", "sabre"])
def test_both_plus_one_is_rejected_for_priority_weapons(weapon):
    hit_types = pp.LABEL_HIT_TYPES[weapon]
    assert hit_types
    for hit_type in hit_types:
        decision = label(hit_type, (1, 1), weapon)
        assert decision.label is None
        assert decision.reason == "both_scores_increased"


@pytest.mark.parametrize("hit_type, delta", [
    ("On-No", (0, 1)),
    ("No-On", (1, 0)),
    ("On-Off", (0, 1)),
    ("Off-On", (1, 0)),
])
def test_score_for_the_unlit_side_is_inconsistent(hit_type, delta):
    decision = label(hit_type, delta, "foil")
    assert decision.label is None
    assert decision.reason == "score_inconsistent_with_lights"


@pytest.mark.parametrize("hit_type", ["On-No", "No-On"])
def test_single_light_without_score_change_is_rejected(hit_type):
    decision = label(hit_type, (0, 0), "sabre")
    assert decision.label is None
    assert decision.reason == "single_light_no_score_change"


@pytest.mark.parametrize("delta", [(-1, 0), (0, -1), (-2, 1)])
def test_negative_change_is_rejected(delta):
    decision = label("On-On", delta, "foil")
    assert decision.label is None and decision.reason == "negative_score_change"


@pytest.mark.parametrize("delta", [(2, 0), (0, 3), (5, 5)])
def test_card_like_jumps_are_rejected(delta):
    decision = label("On-On", delta, "sabre")
    assert decision.label is None and decision.reason == "score_jump_gt_one"


@pytest.mark.parametrize("hit_type", ["Off-No", "No-Off", "Off-Off", "No-No"])
def test_unlabellable_hit_types(hit_type):
    decision = label(hit_type, (0, 0), "foil")
    assert decision.label is None and decision.reason == "hit_type_not_labellable"


# --- epee: T is a double touch, not "no priority" --------------------------

def test_epee_double_touch_is_t():
    assert label("On-On", (1, 1), "epee").label == "T"


@pytest.mark.parametrize("hit_type, delta, expected", [
    ("On-No", (1, 0), "L"),
    ("No-On", (0, 1), "R"),
])
def test_epee_single_touch(hit_type, delta, expected):
    assert label(hit_type, delta, "epee").label == expected


def test_epee_both_lit_without_score_change_is_not_a_tie():
    decision = label("On-On", (0, 0), "epee")
    assert decision.label is None
    assert decision.reason == "no_score_change_both_lit"


@pytest.mark.parametrize("delta", [(1, 0), (0, 1)])
def test_epee_both_lit_single_point_is_rejected(delta):
    decision = label("On-On", delta, "epee")
    assert decision.label is None
    assert decision.reason == "one_sided_score_both_lit"


def test_epee_single_light_with_double_score_is_rejected():
    decision = label("On-No", (1, 1), "epee")
    assert decision.label is None
    assert decision.reason == "both_scores_increased"


def test_epee_off_target_does_not_exist():
    decision = label("On-Off", (1, 0), "epee")
    assert decision.reason == "hit_type_not_labellable"


# --- compatibility wrapper and pipeline wiring -----------------------------

def test_determine_label_wrapper_returns_plain_label():
    assert pp.determine_label("On-On", 3, 2, 4, 2, weapon="sabre") == "L"
    assert pp.determine_label("On-No", 3, 2, 3, 2, weapon="sabre") is None


def test_rejection_reason_reaches_the_exchange():
    onset = 1000
    timeline = make_timeline(
        n_frames=2500,
        runs=[(onset, onset + 75)],
        left_state=pp.STATE_ON,
        right_state=pp.STATE_NO,
        scores=lambda frame: (3, 2) if frame < onset + 10 else (3, 3),
    )
    (exchange,) = pp.analyse_timeline(timeline, "sabre")
    assert exchange.label is None
    assert exchange.reason == "score_inconsistent_with_lights"
