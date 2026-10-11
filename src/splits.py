"""Group-aware, class-stratified train/val/test splits and grouped k-fold.

Clips from the same match (or tournament) never end up in different splits.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import replace
from typing import Optional, Sequence

from src.manifest import VALID_SPLITS, ClipRecord

log = logging.getLogger(__name__)

N_CLASSES = 3


def group_key(rec: ClipRecord, group_by: str = "match") -> str:
    """Grouping key of a record. Falls back to match_id (then the clip itself)."""
    if group_by == "tournament":
        if rec.tournament:
            return f"tournament:{rec.tournament}"
        return f"match:{rec.match_id}" if rec.match_id else f"clip:{rec.clip_path}"
    if group_by != "match":
        raise ValueError(f"group_by must be 'match' or 'tournament', got {group_by!r}")
    return f"match:{rec.match_id}" if rec.match_id else f"clip:{rec.clip_path}"


def _group_counts(
    records: Sequence[ClipRecord], group_by: str
) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * N_CLASSES)
    for rec in records:
        counts[group_key(rec, group_by)][rec.label_id] += 1
    return counts


def assign_groups(
    records: Sequence[ClipRecord],
    fractions: Sequence[float],
    seed: int = 42,
    group_by: str = "match",
) -> dict[str, int]:
    """Greedily assign groups to bins so per-class counts follow ``fractions``.

    Larger groups are placed first (random tie-break); each group goes to the
    bin whose relative class-count deviation from its target grows least.
    Deterministic given ``seed``.
    """
    counts = _group_counts(records, group_by)
    total = [sum(c[k] for c in counts.values()) for k in range(N_CLASSES)]
    n_bins = len(fractions)
    target = [[f * total[k] for k in range(N_CLASSES)] for f in fractions]
    current = [[0.0] * N_CLASSES for _ in range(n_bins)]

    rng = random.Random(seed)
    keys = sorted(counts)
    rng.shuffle(keys)
    keys.sort(key=lambda key: -sum(counts[key]))  # stable: keeps shuffle for ties

    assignment: dict[str, int] = {}
    for key in keys:
        best_bins: list[int] = []
        best_cost = float("inf")
        for b in range(n_bins):
            if fractions[b] <= 0:
                continue
            cost = 0.0
            for k in range(N_CLASSES):
                scale = max(total[k], 1)
                after = current[b][k] + counts[key][k]
                before = current[b][k]
                cost += ((after - target[b][k]) ** 2 - (before - target[b][k]) ** 2) / scale
            if cost < best_cost - 1e-12:
                best_cost, best_bins = cost, [b]
            elif abs(cost - best_cost) <= 1e-12:
                best_bins.append(b)
        chosen = rng.choice(best_bins)
        assignment[key] = chosen
        for k in range(N_CLASSES):
            current[chosen][k] += counts[key][k]
    return assignment


def make_group_splits(
    records: Sequence[ClipRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_by: str = "match",
) -> list[ClipRecord]:
    """Return records with ``split`` filled in (train/val/test), grouped."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")
    assignment = assign_groups(
        records, [train_ratio, val_ratio, test_ratio], seed=seed, group_by=group_by
    )
    out = [
        replace(rec, split=VALID_SPLITS[assignment[group_key(rec, group_by)]])
        for rec in records
    ]
    _log_split_summary(out, group_by)
    return out


def make_group_kfold_splits(
    records: Sequence[ClipRecord],
    n_folds: int,
    fold: int,
    seed: int = 42,
    group_by: str = "match",
) -> list[ClipRecord]:
    """Grouped, stratified k-fold: ``fold`` is test, ``fold+1`` is val, rest train."""
    if n_folds < 3:
        raise ValueError("cv_folds must be >= 3 (train, val and test folds)")
    if not 0 <= fold < n_folds:
        raise ValueError(f"cv_fold must be in [0, {n_folds}), got {fold}")
    assignment = assign_groups(
        records, [1.0 / n_folds] * n_folds, seed=seed, group_by=group_by
    )
    test_fold, val_fold = fold, (fold + 1) % n_folds
    out = []
    for rec in records:
        f = assignment[group_key(rec, group_by)]
        name = "test" if f == test_fold else "val" if f == val_fold else "train"
        out.append(replace(rec, split=name))
    _log_split_summary(out, group_by)
    return out


def stratified_subset(
    records: Sequence[ClipRecord], max_clips: int, seed: int = 42
) -> list[ClipRecord]:
    """Random subset of ``max_clips`` records, proportional per label.

    Original order is preserved. Every non-empty class keeps at least one
    clip when ``max_clips`` allows it. Deterministic given ``seed``.
    """
    if max_clips <= 0 or max_clips >= len(records):
        return list(records)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_label[rec.label].append(i)
    labels = sorted(by_label)
    total = len(records)
    quota = {k: max_clips * len(by_label[k]) / total for k in labels}
    take = {k: int(quota[k]) for k in labels}
    if max_clips >= len(labels):
        for k in labels:
            take[k] = max(take[k], 1)
    # distribute / trim to hit max_clips exactly (largest remainder first)
    order = sorted(labels, key=lambda k: quota[k] - int(quota[k]), reverse=True)
    while sum(take.values()) < max_clips:
        for k in order:
            if take[k] < len(by_label[k]) and sum(take.values()) < max_clips:
                take[k] += 1
    while sum(take.values()) > max_clips:
        k = max(labels, key=lambda lab: take[lab])
        take[k] -= 1
    rng = random.Random(seed)
    keep: set[int] = set()
    for k in labels:
        keep.update(rng.sample(by_label[k], min(take[k], len(by_label[k]))))
    return [rec for i, rec in enumerate(records) if i in keep]


def select_split(records: Sequence[ClipRecord], split: str) -> list[ClipRecord]:
    return [r for r in records if r.split == split]


def groups_by_split(
    records: Sequence[ClipRecord], group_by: str = "match"
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        out[rec.split].add(group_key(rec, group_by))
    return dict(out)


def assert_no_group_leak(
    records: Sequence[ClipRecord], group_by: str = "match"
) -> None:
    """Raise if any group appears in more than one split."""
    seen: dict[str, str] = {}
    for rec in records:
        key = group_key(rec, group_by)
        if seen.setdefault(key, rec.split) != rec.split:
            raise AssertionError(
                f"group {key} appears in splits {seen[key]!r} and {rec.split!r}"
            )


def has_complete_split(records: Sequence[ClipRecord]) -> bool:
    """True if every record already carries a valid split value."""
    return bool(records) and all(r.split in VALID_SPLITS for r in records)


def _log_split_summary(records: Sequence[ClipRecord], group_by: str) -> None:
    for name in VALID_SPLITS:
        part = select_split(records, name)
        labels = [r.label for r in part]
        log.info(
            "split %-5s: %4d clips, %3d groups (L=%d T=%d R=%d)",
            name,
            len(part),
            len({group_key(r, group_by) for r in part}),
            labels.count("L"),
            labels.count("T"),
            labels.count("R"),
        )
        if not part:
            log.warning("split '%s' is empty (too few %s groups?)", name, group_by)


SPLIT_FILENAME = "split_manifest.csv"


def build_splits(
    records: Sequence[ClipRecord],
    group_by: str = "match",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    cv_folds: int = 0,
    cv_fold: int = 0,
    use_manifest_split: bool = False,
) -> list[ClipRecord]:
    """Assign train/val/test to every record according to the data config.

    Priority: grouped k-fold (``cv_folds > 0``), then the manifest's own
    ``split`` column (``use_manifest_split`` and complete), else a grouped
    stratified ratio split. Group leakage is asserted for computed splits.
    """
    if cv_folds and cv_folds > 0:
        out = make_group_kfold_splits(records, cv_folds, cv_fold, seed, group_by)
    elif use_manifest_split and has_complete_split(records):
        log.info("Using split assignment from the manifest")
        return list(records)
    else:
        if use_manifest_split:
            log.warning("Manifest split incomplete; computing a grouped split")
        out = make_group_splits(
            records, train_ratio, val_ratio, test_ratio, seed, group_by
        )
    assert_no_group_leak(out, group_by)
    return out


def save_split_manifest(records: Sequence[ClipRecord], directory) -> "Path":
    """Write the split assignment (manifest copy with `split` filled)."""
    from pathlib import Path

    from src.manifest import write_manifest

    path = Path(directory) / SPLIT_FILENAME
    write_manifest(records, path)
    log.info("Saved split assignment to %s", path)
    return path


def load_split_manifest(directory) -> list[ClipRecord]:
    """Load the split assignment saved by training. Fails loudly if absent."""
    from pathlib import Path

    from src.manifest import load_manifest

    path = Path(directory) / SPLIT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Splits are never recomputed at evaluation time; "
            "re-run training so it writes the split file next to the checkpoint."
        )
    records = load_manifest(path)
    if not has_complete_split(records):
        raise ValueError(f"{path} has rows without a valid split value")
    return records
