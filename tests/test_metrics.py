import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import f1_score

from src.metrics import (
    accuracy,
    bootstrap_ci,
    classification_summary,
    confusion_matrix,
    expected_calibration_error,
    group_metrics,
    light_only_predict,
    macro_f1,
    majority_baseline_predict,
    majority_class,
    mcnemar_test,
    on_on_majority,
    per_class_recall,
)


def test_confusion_and_accuracy():
    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 2, 0]
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 6)


def test_recall_nan_for_absent_class():
    recall = per_class_recall([0, 0, 2], [0, 1, 2])
    assert recall[0] == pytest.approx(0.5)
    assert np.isnan(recall[1])
    assert recall[2] == 1.0


@given(
    st.lists(st.tuples(st.integers(0, 2), st.integers(0, 2)), min_size=1, max_size=60)
)
@settings(max_examples=50, deadline=None)
def test_macro_f1_matches_sklearn(pairs):
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    present = sorted(set(y_true) | set(y_pred))
    expected = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_macro_f1_punishes_majority_predictor():
    y_true = [0] * 8 + [1] + [2]
    y_pred = [0] * 10
    assert accuracy(y_true, y_pred) == 0.8
    assert macro_f1(y_true, y_pred) < 0.4


def test_ece_perfectly_calibrated_and_overconfident():
    # confidence 1.0, always right -> 0
    probs = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    assert expected_calibration_error(probs, [0, 1]) == pytest.approx(0.0)
    # confidence 1.0 but always wrong -> 1
    assert expected_calibration_error(probs, [1, 0]) == pytest.approx(1.0)
    # 0.8 confidence, 50% accuracy in one bin -> 0.3
    probs = np.array([[0.8, 0.1, 0.1]] * 4)
    assert expected_calibration_error(probs, [0, 0, 1, 1]) == pytest.approx(0.3)


def test_summary_keys():
    probs = np.eye(3)[[0, 1, 2]]
    s = classification_summary([0, 1, 2], [0, 1, 2], probs)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0 and s["ece"] == 0.0
    assert s["recall"] == [1.0, 1.0, 1.0]


def test_group_metrics():
    out = group_metrics([0, 0, 2, 2], [0, 1, 2, 2], ["On-On", "On-On", "No-On", "No-On"])
    assert out["On-On"]["accuracy"] == 0.5 and out["On-On"]["n"] == 2
    assert out["No-On"]["accuracy"] == 1.0


def test_majority():
    assert majority_class([0, 1, 1, 2]) == 1
    assert majority_class([]) == 1
    assert majority_baseline_predict([2, 2, 0], 4).tolist() == [2, 2, 2, 2]


def test_light_only_rule():
    hits = ["On-No", "No-On", "On-Off", "Off-On", "On-On", ""]
    preds = light_only_predict(hits, on_on_label=1, fallback_label=0)
    assert preds.tolist() == [0, 2, 0, 2, 1, 0]


def test_on_on_majority():
    labels = [0, 0, 0, 1, 1, 2]
    hits = ["On-No", "On-No", "On-No", "On-On", "On-On", "On-On"]
    assert on_on_majority(labels, hits) == 1
    assert on_on_majority(labels, ["On-No"] * 6) == 0  # no On-On -> overall majority


def test_bootstrap_ci_brackets_point_and_is_seeded():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4).astype(str)
    y_true = rng.integers(0, 3, size=120)
    y_pred = np.where(rng.random(120) < 0.7, y_true, rng.integers(0, 3, size=120))
    point, lo, hi = bootstrap_ci(accuracy, y_true, y_pred, groups, n_boot=300, seed=1)
    assert lo <= point <= hi
    assert (point, lo, hi) == bootstrap_ci(
        accuracy, y_true, y_pred, groups, n_boot=300, seed=1
    )


def test_bootstrap_clusters_widen_interval():
    # perfectly correlated within match: each match is all right or all wrong
    groups = np.repeat(np.arange(20), 10).astype(str)
    y_true = np.zeros(200, dtype=int)
    y_pred = np.where(np.repeat(np.arange(20) % 2, 10) == 0, 0, 1)
    _, lo_c, hi_c = bootstrap_ci(accuracy, y_true, y_pred, groups, n_boot=500, seed=0)
    _, lo_i, hi_i = bootstrap_ci(
        accuracy, y_true, y_pred, np.arange(200).astype(str), n_boot=500, seed=0
    )
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_mcnemar_known_values():
    a = np.array([1] * 10 + [0] * 0 + [1] * 5 + [0] * 5, dtype=bool)
    b = np.array([0] * 10 + [0] * 0 + [1] * 5 + [1] * 5, dtype=bool)
    res = mcnemar_test(a, b)
    assert res["b"] == 10 and res["c"] == 5
    assert res["p_value"] == pytest.approx(0.3018, abs=1e-3)


def test_mcnemar_identical_and_mismatch():
    same = [True, False, True]
    assert mcnemar_test(same, same)["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_test([True], [True, False])


def test_evaluation_report_baselines_and_on_on():
    from src.metrics import evaluation_report

    # train: On-On clips are mostly Tie; overall majority is Left
    train_labels = [0, 0, 0, 0, 1, 1, 1, 2]
    train_hits = ["On-No", "On-No", "On-Off", "On-Off", "On-On", "On-On", "On-On", "No-On"]
    y_true = [0, 2, 1, 0, 1]
    hits = ["On-No", "No-On", "On-On", "Off-On", "On-On"]
    y_pred = [0, 2, 1, 2, 0]
    probs = np.eye(3)[y_pred]
    rep = evaluation_report(
        y_true, y_pred, probs, hits, ["foil"] * 5, train_labels, train_hits
    )
    assert rep["overall"]["accuracy"] == pytest.approx(3 / 5)
    assert rep["on_on"]["n"] == 2 and rep["on_on"]["accuracy"] == pytest.approx(0.5)
    assert rep["per_hit_type"]["No-On"]["accuracy"] == 1.0
    # majority = Left(0): correct for items 0 and 3
    assert rep["baselines"]["majority"]["accuracy"] == pytest.approx(2 / 5)
    assert rep["baselines"]["majority"]["predicts"] == "L"
    # light rule predicts L, R, T, R, T; truth 0, 2, 1, 0, 1 -> 4 of 5 right
    assert rep["baselines"]["light_only"]["accuracy"] == pytest.approx(4 / 5)
    assert rep["baselines"]["light_only"]["on_on_accuracy"] == pytest.approx(1.0)
    assert rep["per_weapon"]["foil"]["n"] == 5


def test_evaluation_report_without_on_on_and_unknown_hits():
    from src.metrics import evaluation_report

    rep = evaluation_report(
        [0, 2], [0, 2], np.eye(3)[[0, 2]], ["", ""], ["foil", "foil"], [0, 0, 1], ["", "", ""]
    )
    assert rep["on_on"] is None
    assert "unknown" in rep["per_hit_type"]
    assert rep["baselines"]["light_only"]["on_on_accuracy"] is None
    assert rep["baselines"]["light_only"]["accuracy"] == pytest.approx(0.5)  # all -> L


def test_prediction_roundtrip(tmp_path):
    from src.metrics import read_predictions, write_predictions

    rows = [
        dict(clip_path="a.mp4", match_id="m1", hit_type="On-On", weapon="foil",
             label="L", pred="T", p_L=0.2, p_T=0.7, p_R=0.1),
        dict(clip_path="b.mp4", match_id="m2", hit_type="", weapon="epee",
             label="R", pred="R", p_L=0.1, p_T=0.1, p_R=0.8),
    ]
    write_predictions(rows, tmp_path / "p.csv")
    got = read_predictions(tmp_path / "p.csv")
    assert got["label"].tolist() == [0, 2] and got["pred"].tolist() == [1, 2]
    assert got["probs"].shape == (2, 3) and got["probs"][0, 1] == 0.7
    assert got["match_id"].tolist() == ["m1", "m2"]
