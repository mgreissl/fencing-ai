#!/usr/bin/env python3
"""
#11: Read filled audit sheet and report label noise rates with CIs.

Usage:
    python scripts/audit_report.py [--sheet PATH] [--output-json PATH]

Input:
    Audit sheet CSV with columns: clip_path, weapon, hit_type, auto_label, human_label, notes
    Rows with empty human_label are skipped.
    A human label "X" means "not identifiable" and is reported separately.

Output:
    - Markdown summary to stdout
    - JSON with per-class noise rates and confusion matrix (--output-json)

Metrics:
    - Per-class noise rate: P(auto_label != human_label) with Wilson 95% CI
    - Per-weapon noise rate
    - Confusion matrix: rows = human_label, cols = auto_label
    - "Unidentifiable" count (human_label = "X")
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(n_errors: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportion.

    Args:
        n_errors: number of errors (numerator)
        n_total: total count
        confidence: confidence level (default 0.95 = 95%)

    Returns:
        (lower, upper) bounds of the CI for the error rate
    """
    if n_total == 0:
        return 0.0, 0.0

    p_hat = n_errors / n_total
    z = 1.96 if confidence == 0.95 else 1.645  # For different CIs, adjust as needed

    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def load_audit_sheet(path: Path) -> list[dict]:
    """Load filled audit sheet, skip rows with empty human_label."""
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("human_label") or "").strip():
                rows.append(row)
    return rows


def compute_noise_rates(rows: list[dict]) -> dict:
    """
    Compute label noise rates by class and by weapon.

    Returns:
        {
            "total": {
                "n_clips": int,
                "n_errors": int,
                "error_rate": float,
                "ci_lower": float,
                "ci_upper": float,
            },
            "by_class": {
                "L": {...},
                "T": {...},
                "R": {...},
            },
            "by_weapon": {
                "foil": {...},
                ...
            },
            "confusion_matrix": {
                human_label: { auto_label: count, ... },
                ...
            },
            "unidentifiable": int,
        }
    """
    results = {
        "by_class": {},
        "by_weapon": {},
        "confusion_matrix": {},
        "unidentifiable": 0,
    }

    class_stats = {}
    weapon_stats = {}
    confusion = {}

    for row in rows:
        auto = (row.get("auto_label") or "").strip().upper()
        human = (row.get("human_label") or "").strip().upper()
        weapon = (row.get("weapon") or "").strip().lower()

        if not human:
            continue

        if human == "X":
            results["unidentifiable"] += 1
            continue

        # Track by class
        if human not in class_stats:
            class_stats[human] = {"total": 0, "errors": 0}
        class_stats[human]["total"] += 1
        if auto != human:
            class_stats[human]["errors"] += 1

        # Track by weapon
        if weapon not in weapon_stats:
            weapon_stats[weapon] = {"total": 0, "errors": 0}
        weapon_stats[weapon]["total"] += 1
        if auto != human:
            weapon_stats[weapon]["errors"] += 1

        # Confusion matrix
        if human not in confusion:
            confusion[human] = {}
        if auto not in confusion[human]:
            confusion[human][auto] = 0
        confusion[human][auto] += 1

    # Compute CIs
    for label, stats in class_stats.items():
        n_e, n_t = stats["errors"], stats["total"]
        lower, upper = wilson_ci(n_e, n_t)
        results["by_class"][label] = {
            "n_clips": n_t,
            "n_errors": n_e,
            "error_rate": n_e / n_t if n_t > 0 else 0.0,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    for weapon, stats in weapon_stats.items():
        n_e, n_t = stats["errors"], stats["total"]
        lower, upper = wilson_ci(n_e, n_t)
        results["by_weapon"][weapon] = {
            "n_clips": n_t,
            "n_errors": n_e,
            "error_rate": n_e / n_t if n_t > 0 else 0.0,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    # Total
    total_errors = sum(s["errors"] for s in class_stats.values())
    total_clips = sum(s["total"] for s in class_stats.values())
    if total_clips > 0:
        lower, upper = wilson_ci(total_errors, total_clips)
    else:
        lower, upper = 0.0, 0.0

    results["total"] = {
        "n_clips": total_clips,
        "n_errors": total_errors,
        "error_rate": total_errors / total_clips if total_clips > 0 else 0.0,
        "ci_lower": lower,
        "ci_upper": upper,
    }

    results["confusion_matrix"] = {h: dict(c) for h, c in confusion.items()}

    return results


def format_markdown(results: dict) -> str:
    """Format results as markdown."""
    md = "# Label Audit Report\n\n"

    total = results["total"]
    md += "## Summary\n\n"
    md += f"- **Total clips audited:** {total['n_clips']}\n"
    md += f"- **Total errors:** {total['n_errors']}\n"
    md += f"- **Overall noise rate:** {total['error_rate']:.2%} "
    md += f"[{total['ci_lower']:.2%}, {total['ci_upper']:.2%}]\n"
    md += f"- **Unidentifiable:** {results['unidentifiable']}\n\n"

    md += "## Per-Class Noise Rate\n\n"
    md += "| Label | N Clips | N Errors | Error Rate | 95% CI |\n"
    md += "|-------|---------|----------|------------|--------|\n"
    for label in sorted(results["by_class"].keys()):
        stats = results["by_class"][label]
        md += (
            f"| {label} | {stats['n_clips']} | {stats['n_errors']} | "
            f"{stats['error_rate']:.2%} | "
            f"[{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}] |\n"
        )

    md += "\n## Per-Weapon Noise Rate\n\n"
    md += "| Weapon | N Clips | N Errors | Error Rate | 95% CI |\n"
    md += "|--------|---------|----------|------------|--------|\n"
    for weapon in sorted(results["by_weapon"].keys()):
        stats = results["by_weapon"][weapon]
        md += (
            f"| {weapon} | {stats['n_clips']} | {stats['n_errors']} | "
            f"{stats['error_rate']:.2%} | "
            f"[{stats['ci_lower']:.2%}, {stats['ci_upper']:.2%}] |\n"
        )

    if results["confusion_matrix"]:
        md += "\n## Confusion Matrix (Human Label vs Auto Label)\n\n"
        md += "| Human \\ Auto | "
        auto_labels = sorted(set(a for row in results["confusion_matrix"].values() for a in row))
        md += " | ".join(auto_labels) + " |\n"
        md += "|" + "|".join(["---"] * (len(auto_labels) + 1)) + "|\n"
        for human_label in sorted(results["confusion_matrix"].keys()):
            row_data = results["confusion_matrix"][human_label]
            md += f"| {human_label} |"
            for auto_label in auto_labels:
                md += f" {row_data.get(auto_label, 0)} |"
            md += "\n"

    return md


def main():
    parser = argparse.ArgumentParser(
        description="Report label noise rates from filled audit sheet."
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=REPO_ROOT / "audit" / "audit_sheet.csv",
        help="Path to filled audit sheet (default: audit/audit_sheet.csv)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write JSON results (optional)",
    )

    args = parser.parse_args()

    if not args.sheet.exists():
        log.error(f"Audit sheet not found: {args.sheet}")
        return 1

    rows = load_audit_sheet(args.sheet)
    if not rows:
        log.error("No rows with human_label found in audit sheet")
        return 1

    log.info(f"Loaded {len(rows)} audited clips (+ {len(load_audit_sheet(args.sheet)) - len(rows)} unidentifiable)")

    results = compute_noise_rates(rows)

    # Print markdown
    md = format_markdown(results)
    print(md)

    # Write JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log.info(f"Wrote JSON results to {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
