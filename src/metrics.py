"""Evaluation metrics, baselines and paired statistics (pure numpy/scipy).

Labels are integer class ids: 0=Left, 1=Tie, 2=Right.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

N_CLASSES = 3
LEFT, TIE, RIGHT = 0, 1, 2

# Light-only rule: which side's light alone decides the call.
LIGHT_RULE = {"On-No": LEFT, "No-On": RIGHT, "On-Off": LEFT, "Off-On": RIGHT}


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = N_CLASSES
) -> NDArray[np.int64]:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == np.asarray(y_pred)))


def per_class_recall(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = N_CLASSES
) -> NDArray[np.float64]:
    """Recall per class; NaN for classes absent from y_true."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / support
    recall[support == 0] = np.nan
    return recall


def per_class_f1(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = N_CLASSES
) -> NDArray[np.float64]:
    """F1 per class (0 when precision and recall are both 0); NaN if absent
    from both y_true and y_pred."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / denom  # = 2PR/(P+R)
    f1[denom == 0] = np.nan
    return f1


def macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = N_CLASSES
) -> float:
    """Unweighted mean of per-class F1 over classes present in y_true or y_pred."""
    f1 = per_class_f1(y_true, y_pred, n_classes)
    if np.all(np.isnan(f1)):
        return float("nan")
    return float(np.nanmean(f1))


def expected_calibration_error(
    probs: NDArray[np.float64], y_true: Sequence[int], n_bins: int = 10
) -> float:
    """ECE with equal-width confidence bins: sum_b |B_b|/n |acc_b - conf_b|."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    if len(y_true) == 0:
        return float("nan")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index in [0, n_bins-1]; confidence 1.0 goes to the last bin
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1], right=True), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def classification_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Optional[NDArray[np.float64]] = None,
    n_classes: int = N_CLASSES,
    ece_bins: int = 10,
) -> dict:
    """Accuracy, macro-F1, per-class recall (and ECE if probs given)."""
    summary = {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "recall": per_class_recall(y_true, y_pred, n_classes).tolist(),
    }
    if probs is not None:
        summary["ece"] = expected_calibration_error(probs, y_true, ece_bins)
    return summary


def group_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str]
) -> dict[str, dict]:
    """Accuracy / macro-F1 per group value (e.g. hit_type or weapon)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups, dtype=object)
    out: dict[str, dict] = {}
    for g in sorted({str(x) for x in groups}):
        mask = groups == g
        out[g] = {
            "n": int(mask.sum()),
            "accuracy": accuracy(y_true[mask], y_pred[mask]),
            "macro_f1": macro_f1(y_true[mask], y_pred[mask]),
        }
    return out


# --- Baselines ---------------------------------------------------------------


def majority_class(labels: Sequence[int], default: int = TIE) -> int:
    """Most frequent label (lowest id wins ties); `default` if empty."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return default
    return int(np.bincount(labels, minlength=N_CLASSES).argmax())


def majority_baseline_predict(train_labels: Sequence[int], n: int) -> NDArray[np.int64]:
    """Predict the train-set majority class for all `n` items."""
    return np.full(n, majority_class(train_labels), dtype=np.int64)


def on_on_majority(
    train_labels: Sequence[int], train_hit_types: Sequence[str]
) -> int:
    """Majority label among train clips with hit_type 'On-On'.

    Falls back to the overall majority if there are no On-On training clips.
    """
    labels = np.asarray(train_labels, dtype=np.int64)
    mask = np.asarray(train_hit_types, dtype=object) == "On-On"
    if not mask.any():
        return majority_class(labels)
    return majority_class(labels[mask])


def light_only_predict(
    hit_types: Sequence[str], on_on_label: int, fallback_label: int
) -> NDArray[np.int64]:
    """Rule using only the lights: On-No/On-Off -> L, No-On/Off-On -> R,
    On-On -> `on_on_label`, unknown hit_type -> `fallback_label`."""
    preds = []
    for hit in hit_types:
        if hit == "On-On":
            preds.append(on_on_label)
        else:
            preds.append(LIGHT_RULE.get(hit, fallback_label))
    return np.asarray(preds, dtype=np.int64)


# --- Uncertainty / paired tests ---------------------------------------------


def bootstrap_ci(
    stat_fn: Callable[[NDArray, NDArray], float],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI, resampling whole groups (matches) with replacement.

    Returns (point_estimate, lower, upper) with percentile interval at
    1-alpha coverage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    members: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[str(g)].append(i)
    idx_by_group = [np.asarray(v, dtype=np.int64) for v in members.values()]
    point = float(stat_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    n_groups = len(idx_by_group)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        stats[b] = stat_fn(y_true[idx], y_pred[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar_test(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> dict[str, float]:
    """Exact (binomial) McNemar test on paired per-clip correctness.

    b = A right & B wrong, c = A wrong & B right. Two-sided p-value of
    Binomial(b + c, 0.5). p = 1 when there are no discordant pairs.
    """
    from scipy.stats import binomtest

    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired predictions must have the same length")
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n_disc = b + c
    p = 1.0 if n_disc == 0 else float(binomtest(b, n_disc, 0.5).pvalue)
    return {"b": b, "c": c, "p_value": p}


# --- Reporting / prediction files -------------------------------------------

LABEL_LETTERS = ("L", "T", "R")
PREDICTION_COLUMNS = (
    "clip_path",
    "match_id",
    "hit_type",
    "weapon",
    "label",
    "pred",
    "p_L",
    "p_T",
    "p_R",
)


def evaluation_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: NDArray[np.float64],
    hit_types: Sequence[str],
    weapons: Sequence[str],
    train_labels: Sequence[int],
    train_hit_types: Sequence[str],
    ece_bins: int = 10,
) -> dict:
    """Overall, per-hit-type and per-weapon metrics plus rule baselines.

    Baselines (scored on the same clips):
    - ``majority``: always the train-set majority class.
    - ``light_only``: the light rule (On-On -> train On-On majority; clips
      with unknown hit_type -> train majority).
    ``on_on`` reports the model's accuracy on On-On clips separately (None
    if the evaluation set has none).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    hits = np.asarray(hit_types, dtype=object)
    hits = np.where(hits == "", "unknown", hits)
    train_majority = majority_class(train_labels)

    majority_pred = np.full(len(y_true), train_majority, dtype=np.int64)
    light_pred = light_only_predict(
        np.where(hits == "unknown", "", hits),
        on_on_label=on_on_majority(train_labels, train_hit_types),
        fallback_label=train_majority,
    )

    per_hit = group_metrics(y_true, y_pred, hits)
    on_on_mask = hits == "On-On"
    report = {
        "overall": classification_summary(y_true, y_pred, probs, ece_bins=ece_bins),
        "per_hit_type": per_hit,
        "per_weapon": group_metrics(y_true, y_pred, weapons),
        "on_on": per_hit.get("On-On"),
        "baselines": {
            "majority": {
                **classification_summary(y_true, majority_pred),
                "predicts": LABEL_LETTERS[train_majority],
            },
            "light_only": {
                **classification_summary(y_true, light_pred),
                "on_on_accuracy": (
                    accuracy(y_true[on_on_mask], light_pred[on_on_mask])
                    if on_on_mask.any()
                    else None
                ),
                "per_hit_type": group_metrics(y_true, light_pred, hits),
            },
        },
    }
    return report


def write_predictions(rows: Sequence[dict], path) -> None:
    """Write per-clip predictions. Each row needs the PREDICTION_COLUMNS keys
    (label/pred as L/T/R letters)."""
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in PREDICTION_COLUMNS})


def read_predictions(path) -> dict[str, NDArray]:
    """Read a predictions CSV into arrays: label/pred as int ids, probs (n, 3)."""
    import csv
    from pathlib import Path

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    letter_id = {c: i for i, c in enumerate(LABEL_LETTERS)}
    return {
        "clip_path": np.asarray([r["clip_path"] for r in rows], dtype=object),
        "match_id": np.asarray([r["match_id"] for r in rows], dtype=object),
        "hit_type": np.asarray([r["hit_type"] for r in rows], dtype=object),
        "weapon": np.asarray([r["weapon"] for r in rows], dtype=object),
        "label": np.asarray([letter_id[r["label"]] for r in rows], dtype=np.int64),
        "pred": np.asarray([letter_id[r["pred"]] for r in rows], dtype=np.int64),
        "probs": np.asarray(
            [[float(r["p_L"]), float(r["p_T"]), float(r["p_R"])] for r in rows],
            dtype=np.float64,
        ).reshape(len(rows), 3),
    }
