"""Evaluation metrics, baselines and paired statistics (pure numpy/scipy).

Labels are integer class ids: 0=Left, 1=Tie, 2=Right.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Optional, Sequence

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


def on_on_majority_by_weapon(
    train_labels: Sequence[int],
    train_hit_types: Sequence[str],
    train_weapons: Sequence[str],
) -> dict[str, int]:
    """On-On majority label per weapon of the training clips.

    Epee On-On is always a double hit (Tie) by rule. A weapon without On-On
    training clips falls back to the pooled On-On majority.
    """
    labels = np.asarray(train_labels, dtype=np.int64)
    hits = np.asarray(train_hit_types, dtype=object)
    weapons = np.asarray(train_weapons, dtype=object)
    pooled = on_on_majority(labels, hits)
    out = {}
    for weapon in sorted({str(w) for w in weapons}):
        mask = (weapons == weapon) & (hits == "On-On")
        out[weapon] = majority_class(labels[mask]) if mask.any() else pooled
    out["epee"] = TIE
    return out


def light_only_predict(
    hit_types: Sequence[str],
    on_on_label: int | Mapping[str, int],
    fallback_label: int,
    weapons: Optional[Sequence[str]] = None,
) -> NDArray[np.int64]:
    """Rule using only the lights: On-No/On-Off -> L, No-On/Off-On -> R,
    On-On -> `on_on_label`, unknown hit_type -> `fallback_label`.

    With ``weapons``, epee On-On is Tie (a double hit, no right of way) and
    ``on_on_label`` may be a ``{weapon: label}`` mapping (weapons missing from
    it use ``fallback_label``).
    """
    preds = []
    for i, hit in enumerate(hit_types):
        weapon = None if weapons is None else str(weapons[i])
        if hit == "On-On":
            if weapon == "epee":
                preds.append(TIE)
            elif isinstance(on_on_label, Mapping):
                preds.append(on_on_label.get(weapon, fallback_label))
            else:
                preds.append(on_on_label)
        else:
            preds.append(LIGHT_RULE.get(hit, fallback_label))
    return np.asarray(preds, dtype=np.int64)


# --- Hybrid prediction: light rules plus model -------------------------------

# Single-light clips are decided by the lights (same rules as the labelling in
# scripts/preprocess_pipeline.py): On-No -> L, No-On -> R for every weapon; epee
# On-On is a double hit (Tie).
SINGLE_LIGHT_RULE = {"On-No": LEFT, "No-On": RIGHT}
# Hit types whose label the lights do not determine (non-epee).
AMBIGUOUS_HIT_TYPES = ("On-On", "On-Off", "Off-On")


def rule_prediction(hit_type: str, weapon: str) -> Optional[int]:
    """Label fixed by the lights alone, or None if the model must decide.

    Clips without hit_type (or On-On with unknown weapon) return None.
    """
    if hit_type in SINGLE_LIGHT_RULE:
        return SINGLE_LIGHT_RULE[hit_type]
    if hit_type == "On-On" and weapon == "epee":
        return TIE
    return None


def ambiguous_mask(
    hit_types: Sequence[str], weapons: Sequence[str]
) -> NDArray[np.bool_]:
    """Clips the lights do not decide: On-On (not epee) and foil On-Off/Off-On."""
    return np.asarray(
        [
            hit in AMBIGUOUS_HIT_TYPES and str(weapon) != "epee"
            for hit, weapon in zip(hit_types, weapons)
        ],
        dtype=bool,
    )


def ambiguous_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    hit_types: Sequence[str],
    weapons: Sequence[str],
) -> float:
    """Macro-F1 on the ambiguous clips only; NaN if there are none."""
    keep = ambiguous_mask(hit_types, weapons)
    if not keep.any():
        return float("nan")
    return macro_f1(np.asarray(y_true)[keep], np.asarray(y_pred)[keep])


def hybrid_predict(
    model_pred: Sequence[int], hit_types: Sequence[str], weapons: Sequence[str]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Rule prediction where the lights decide, else the model's.

    Returns (predictions, from_rule mask).
    """
    preds = np.array(model_pred, dtype=np.int64)
    from_rule = np.zeros(len(preds), dtype=bool)
    for i, (hit, weapon) in enumerate(zip(hit_types, weapons)):
        rule = rule_prediction(hit, str(weapon))
        if rule is not None:
            preds[i] = rule
            from_rule[i] = True
    return preds, from_rule


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
# pred = final system prediction (rule where the lights decide, else model);
# model_pred = raw model prediction; source = "rule" | "model". p_* are model
# probabilities.
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
    "model_pred",
    "source",
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
    train_weapons: Optional[Sequence[str]] = None,
    model_pred: Optional[Sequence[int]] = None,
) -> dict:
    """Overall, per-hit-type and per-weapon metrics plus rule baselines.

    Baselines (scored on the same clips):
    - ``majority``: always the train-set majority class.
    - ``light_only``: the light rule per weapon (On-On -> train On-On
      majority of that weapon, epee On-On -> Tie; clips with unknown hit_type
      -> train majority). Without ``train_weapons`` the On-On majority is
      pooled across weapons.
    ``on_on`` reports the system's accuracy on On-On clips separately (None
    if the evaluation set has none).

    ``y_pred`` is the final system prediction. ``ambiguous`` scores it on the
    clips the lights do not decide (On-On, foil On-Off/Off-On); the rules never
    touch those, so it is the model's own accuracy there. With ``model_pred``
    (the raw model output, differing from ``y_pred`` under hybrid rules) a
    ``model_only`` block gives the model's metrics on all and on ambiguous
    clips.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    hits = np.asarray(hit_types, dtype=object)
    hits = np.where(hits == "", "unknown", hits)
    train_majority = majority_class(train_labels)

    majority_pred = np.full(len(y_true), train_majority, dtype=np.int64)
    if train_weapons is None:
        on_on_label = on_on_majority(train_labels, train_hit_types)
    else:
        on_on_label = on_on_majority_by_weapon(
            train_labels, train_hit_types, train_weapons
        )
    light_pred = light_only_predict(
        np.where(hits == "unknown", "", hits),
        on_on_label=on_on_label,
        fallback_label=train_majority,
        weapons=weapons,
    )

    per_hit = group_metrics(y_true, y_pred, hits)
    on_on_mask = hits == "On-On"
    amb_mask = ambiguous_mask(hits, weapons)
    report = {
        "overall": classification_summary(y_true, y_pred, probs, ece_bins=ece_bins),
        "per_hit_type": per_hit,
        "per_weapon": group_metrics(y_true, y_pred, weapons),
        "on_on": per_hit.get("On-On"),
        "ambiguous": classification_summary(y_true[amb_mask], y_pred[amb_mask]),
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
                "ambiguous_accuracy": (
                    accuracy(y_true[amb_mask], light_pred[amb_mask])
                    if amb_mask.any()
                    else None
                ),
                "per_hit_type": group_metrics(y_true, light_pred, hits),
            },
        },
    }
    if model_pred is not None:
        model_pred = np.asarray(model_pred, dtype=np.int64)
        report["model_only"] = {
            "overall": classification_summary(
                y_true, model_pred, probs, ece_bins=ece_bins
            ),
            "ambiguous": classification_summary(y_true[amb_mask], model_pred[amb_mask]),
            "per_hit_type": group_metrics(y_true, model_pred, hits),
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
            full = {"model_pred": row["pred"], "source": "model", **row}
            writer.writerow({k: full[k] for k in PREDICTION_COLUMNS})


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
        # older files lack these columns: model prediction = pred, source = model
        "model_pred": np.asarray(
            [letter_id[r.get("model_pred") or r["pred"]] for r in rows],
            dtype=np.int64,
        ),
        "source": np.asarray([r.get("source") or "model" for r in rows], dtype=object),
        "probs": np.asarray(
            [[float(r["p_L"]), float(r["p_T"]), float(r["p_R"])] for r in rows],
            dtype=np.float64,
        ).reshape(len(rows), 3),
    }
