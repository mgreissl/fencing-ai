#!/usr/bin/env python
"""Match-level bootstrap CIs and paired McNemar test on saved predictions.

Consumes ``predictions_<split>.csv`` written by ``python -m src.evaluate``.

Examples:
    python scripts/bootstrap_eval.py checkpoints/predictions_test.csv
    python scripts/bootstrap_eval.py a/predictions_test.csv --compare b/predictions_test.csv
    python scripts/bootstrap_eval.py fold0/predictions_test.csv fold1/predictions_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import (  # noqa: E402
    accuracy,
    bootstrap_ci,
    macro_f1,
    mcnemar_test,
    read_predictions,
)


def load_merged(paths: list[str]) -> dict:
    """Concatenate prediction files (e.g. CV folds); clip paths must be unique."""
    parts = [read_predictions(p) for p in paths]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if len(set(merged["clip_path"])) != len(merged["clip_path"]):
        raise ValueError("duplicate clip_path across prediction files")
    return merged


def report_ci(name: str, data: dict, n_boot: int, seed: int) -> None:
    groups = data["match_id"]
    if (groups == "").any():
        # clips without a match id become singleton groups
        groups = np.where(groups == "", data["clip_path"], groups)
        print("warning: clips without match_id treated as independent groups")
    print(f"{name}: n={len(data['label'])} clips, {len(set(groups))} matches")
    for metric_name, fn in (("accuracy", accuracy), ("macro_f1", macro_f1)):
        point, lo, hi = bootstrap_ci(
            fn, data["label"], data["pred"], groups, n_boot=n_boot, seed=seed
        )
        print(f"  {metric_name:<9} {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("predictions", nargs="+", help="predictions CSV file(s)")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="second system's predictions CSV(s) for a paired McNemar test",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a = load_merged(args.predictions)
    report_ci("A", a, args.n_boot, args.seed)

    if args.compare:
        b = load_merged(args.compare)
        report_ci("B", b, args.n_boot, args.seed)
        order_b = {c: i for i, c in enumerate(b["clip_path"])}
        shared = [i for i, c in enumerate(a["clip_path"]) if c in order_b]
        if len(shared) != len(a["clip_path"]) or len(shared) != len(order_b):
            print(f"note: pairing on {len(shared)} shared clips only")
        idx_a = np.asarray(shared)
        idx_b = np.asarray([order_b[a["clip_path"][i]] for i in shared])
        correct_a = a["pred"][idx_a] == a["label"][idx_a]
        correct_b = b["pred"][idx_b] == b["label"][idx_b]
        res = mcnemar_test(correct_a, correct_b)
        print(
            f"McNemar (exact): A-only correct={res['b']}, B-only correct={res['c']}, "
            f"p={res['p_value']:.4f}"
        )


if __name__ == "__main__":
    main()
