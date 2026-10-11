#!/usr/bin/env python
"""Run an ablation (experiments x seeds x folds) and aggregate the test metrics.

Each run trains with ``python -m src.train +experiment=<name>`` and evaluates
the produced checkpoint with ``python -m src.evaluate``. Runs land in
``<root>/<experiment>/seed<S>`` (with ``/fold<k>`` for cross-validation) and
the aggregate is written to ``<root>/runs.csv``, ``summary.csv``, ``summary.md``.

Examples:
    python scripts/run_ablation.py --seeds 42 43 44 -o wandb.enabled=false
    python scripts/run_ablation.py --experiments video_only kinematics_only \\
        --cv-folds 5 --skip-existing
    python scripts/run_ablation.py --aggregate-only --root checkpoints/ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXPERIMENTS = ("video_only", "kinematics_only", "video_kinematics")
DEFAULT_SEEDS = (42, 43, 44)
METRICS_FILE = "metrics_test.json"
# column name -> key inside metrics_test.json["overall"] (loss is top-level)
METRIC_KEYS = {
    "accuracy": ("overall", "accuracy"),
    "macro_f1": ("overall", "macro_f1"),
    "ece": ("overall", "ece"),
    "loss": ("loss",),
}


def run_dir(root: Path, experiment: str, seed: int) -> Path:
    return Path(root) / experiment / f"seed{seed}"


def _quote(value: str) -> str:
    """Hydra override value safe for paths with ':' (Windows drives)."""
    return "'" + value.replace("\\", "/").replace("'", "\\'") + "'"


def _path_arg(path: Path) -> str:
    try:
        path = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        pass
    return _quote(path.as_posix())


def build_commands(
    experiment: str,
    seed: int,
    root: Path,
    cv_folds: int = 0,
    extra_overrides: Sequence[str] = (),
) -> list[list[str]]:
    """Train command followed by one evaluate command per fold (or just one)."""
    out = run_dir(root, experiment, seed)
    train = [
        sys.executable,
        "-m",
        "src.train",
        f"+experiment={experiment}",
        f"training.seed={seed}",
        f"training.checkpoint_dir={_path_arg(out)}",
        f"hydra.run.dir={_path_arg(out / 'hydra_train')}",
    ]
    if cv_folds > 0:
        train.append(f"data.cv_folds={cv_folds}")
    train += list(extra_overrides)
    return [train]


def build_fold_commands(
    experiment: str,
    seed: int,
    root: Path,
    cv_folds: int = 0,
    extra_overrides: Sequence[str] = (),
) -> list[tuple[list[str], list[str], Path]]:
    """(train, evaluate, metrics path) per run; one run per fold with CV."""
    out = run_dir(root, experiment, seed)
    folds = range(cv_folds) if cv_folds > 0 else [None]
    runs = []
    for fold in folds:
        train = build_commands(experiment, seed, root, cv_folds, extra_overrides)[0]
        ckpt = out if fold is None else out / f"fold{fold}"
        if fold is not None:
            train.append(f"data.cv_fold={fold}")
        evaluate = [
            sys.executable,
            "-m",
            "src.evaluate",
            f"training.checkpoint_dir={_path_arg(ckpt)}",
            f"hydra.run.dir={_path_arg(ckpt / 'hydra_eval')}",
        ]
        runs.append((train, evaluate, ckpt / METRICS_FILE))
    return runs


def _get(metrics: dict, keys: Sequence[str]) -> Optional[float]:
    node = metrics
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return None if node is None else float(node)


def collect_runs(root: Path) -> list[dict]:
    """One row per ``metrics_test.json`` under ``root/<experiment>/seed<S>[/fold<k>]``."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob(METRICS_FILE)):
        parts = path.relative_to(root).parts[:-1]
        if len(parts) < 2 or not parts[1].startswith("seed"):
            continue
        fold = parts[2][len("fold"):] if len(parts) > 2 and parts[2].startswith("fold") else ""
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "experiment": parts[0],
            "seed": int(parts[1][len("seed"):]),
            "fold": fold,
        }
        for name, keys in METRIC_KEYS.items():
            row[name] = _get(metrics, keys)
        rows.append(row)
    return rows


def aggregate(rows: Sequence[dict]) -> list[dict]:
    """Mean and sample std (ddof=1, 0 for a single run) per experiment."""
    summary = []
    for experiment in dict.fromkeys(r["experiment"] for r in rows):
        group = [r for r in rows if r["experiment"] == experiment]
        entry = {
            "experiment": experiment,
            "n_runs": len(group),
            "n_seeds": len({r["seed"] for r in group}),
        }
        for name in METRIC_KEYS:
            values = [r[name] for r in group if r[name] is not None]
            entry[f"{name}_mean"] = statistics.fmean(values) if values else None
            entry[f"{name}_std"] = (
                statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
            )
        summary.append(entry)
    return summary


def to_markdown(summary: Sequence[dict], commit: str = "") -> str:
    header = ["experiment", "runs", "seeds"] + [f"{m} (mean ± std)" for m in METRIC_KEYS]
    lines = []
    if commit:
        lines += [f"Commit: `{commit}`", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for e in summary:
        cells = [e["experiment"], str(e["n_runs"]), str(e["n_seeds"])]
        for name in METRIC_KEYS:
            mean, std = e[f"{name}_mean"], e[f"{name}_std"]
            cells.append("n/a" if mean is None else f"{mean:.4f} ± {std:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate(root: Path, commit: str = "") -> list[dict]:
    """Write runs.csv, summary.csv, summary.md into ``root``; return the summary."""
    root = Path(root)
    rows = collect_runs(root)
    summary = aggregate(rows)
    _write_csv(root / "runs.csv", rows)
    _write_csv(root / "summary.csv", summary)
    (root / "summary.md").write_text(to_markdown(summary, commit), encoding="utf-8")
    return summary


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--cv-folds", type=int, default=0, help="run all folds 0..N-1")
    parser.add_argument("--root", type=Path, default=Path("checkpoints/ablation"))
    parser.add_argument(
        "-o", "--override", action="append", default=[],
        help="extra Hydra override for training (repeatable), e.g. -o wandb.enabled=false",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.aggregate_only:
        for experiment in args.experiments:
            for seed in args.seeds:
                runs = build_fold_commands(
                    experiment, seed, args.root, args.cv_folds, args.override
                )
                for train, evaluate, metrics_path in runs:
                    if args.skip_existing and metrics_path.exists():
                        print(f"skip {metrics_path}")
                        continue
                    for cmd in (train, evaluate):
                        print(" ".join(cmd), flush=True)
                        if not args.dry_run:
                            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if args.dry_run:
        return 0

    summary = write_aggregate(args.root, _git_commit())
    print((Path(args.root) / "summary.md").read_text(encoding="utf-8"))
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
