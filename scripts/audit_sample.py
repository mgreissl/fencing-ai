#!/usr/bin/env python3
"""
#11: Draw a stratified random sample of clips for manual label audit.

Usage:
    python scripts/audit_sample.py [--manifest PATH] [--output-dir DIR] [--n-clips N] [--seed S] [--stratify-by-hittype]

Outputs:
    audit/{audit_sheet.csv, index.html}

The audit sheet has columns: clip_path, weapon, hit_type, auto_label, human_label (empty), notes (empty)
The HTML embeds each clip as <video controls> so a human can review and fill the CSV.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.manifest import (
    LABEL_MAP,
    WEAPON_MAP,
    load_manifest,
    load_weapon_records,
    write_manifest,
)


def stratified_sample(
    records,
    n: int = 200,
    seed: Optional[int] = None,
    stratify_by_hittype: bool = False,
) -> list:
    """
    Draw stratified random sample from records, stratified by (weapon, label) or (weapon, label, hit_type).

    Returns:
        List of sampled ClipRecord objects.
    """
    rng = np.random.RandomState(seed)

    # Group by strata
    if stratify_by_hittype:
        groups = {}
        for rec in records:
            key = (rec.weapon, rec.label, rec.hit_type or "unknown")
            groups.setdefault(key, []).append(rec)
    else:
        groups = {}
        for rec in records:
            key = (rec.weapon, rec.label)
            groups.setdefault(key, []).append(rec)

    # Proportional allocation: stratum size = n * (stratum_count / total_count)
    total = len(records)
    sampled = []
    for key, group in groups.items():
        stratum_size = max(1, int(np.round(n * len(group) / total)))
        indices = rng.choice(len(group), size=min(stratum_size, len(group)), replace=False)
        sampled.extend([group[i] for i in indices])

    # Shuffle and truncate to exactly n
    rng.shuffle(sampled)
    return sampled[:n]


def write_audit_sheet(records, output_path: Path) -> None:
    """
    Write audit sheet CSV with columns:
    clip_path, weapon, hit_type, auto_label, human_label (empty), notes (empty)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_path", "weapon", "hit_type", "auto_label", "human_label", "notes"],
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "clip_path": rec.clip_path,
                "weapon": rec.weapon,
                "hit_type": rec.hit_type or "",
                "auto_label": rec.label,
                "human_label": "",
                "notes": "",
            })
    log.info(f"Wrote audit sheet to {output_path} ({len(records)} clips)")


def write_audit_index_html(records, output_path: Path, repo_root: Path = REPO_ROOT) -> None:
    """
    Write HTML index with embedded <video controls> for each clip.

    The human reviewer can fill the CSV offline after previewing clips here.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Fencing Clip Audit</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background: #f5f5f5; }
        .clip-container {
            background: white;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .clip-header {
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .clip-meta {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }
        video {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .clip-num { color: #0066cc; }
    </style>
</head>
<body>
    <h1>Fencing Clip Audit</h1>
    <p>Use the CSV file to record human labels. Columns: clip_path, weapon, hit_type, auto_label, human_label, notes</p>
"""

    for i, rec in enumerate(records, 1):
        clip_abs = Path(rec.abs_path(repo_root))
        clip_rel = clip_abs.relative_to(repo_root)
        video_url = clip_rel.as_posix()

        label_rev = {v: k for k, v in LABEL_MAP.items()}
        auto_label = label_rev.get(rec.label_id, rec.label)

        html += f"""    <div class="clip-container">
        <div class="clip-header"><span class="clip-num">#{i}</span> {rec.clip_path}</div>
        <div class="clip-meta">
            Weapon: <strong>{rec.weapon}</strong> |
            Hit Type: <strong>{rec.hit_type or "unknown"}</strong> |
            Auto Label: <strong>{auto_label}</strong> |
            Match: <strong>{rec.match_id or "?"}</strong>
        </div>
        <video controls width="640">
            <source src="{video_url}" type="video/mp4">
            Video not supported
        </video>
    </div>
"""

    html += """</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    log.info(f"Wrote audit index to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Draw stratified random sample of clips for manual label audit."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest CSV. If not provided, scans data/clips/{weapon}/manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "audit",
        help="Output directory for audit sheet and index (default: audit/)",
    )
    parser.add_argument(
        "--n-clips",
        type=int,
        default=200,
        help="Number of clips to sample (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--stratify-by-hittype",
        action="store_true",
        help="Stratify by (weapon, label, hit_type) instead of (weapon, label)",
    )

    args = parser.parse_args()

    # Load records
    records = []
    if args.manifest:
        records = load_manifest(args.manifest)
    else:
        data_dir = REPO_ROOT / "data" / "clips"
        for weapon in WEAPON_MAP:
            weapon_dir = data_dir / weapon
            if weapon_dir.is_dir():
                try:
                    recs = load_weapon_records(weapon_dir, weapon, root=REPO_ROOT)
                    records.extend(recs)
                except Exception as e:
                    log.warning(f"Could not load {weapon_dir}: {e}")

    if not records:
        log.error("No records loaded. Check --manifest or data/clips layout.")
        return 1

    log.info(f"Loaded {len(records)} clips")

    # Sample
    sampled = stratified_sample(
        records,
        n=args.n_clips,
        seed=args.seed,
        stratify_by_hittype=args.stratify_by_hittype,
    )
    log.info(f"Sampled {len(sampled)} clips (stratified by {'weapon, label, hit_type' if args.stratify_by_hittype else 'weapon, label'})")

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_audit_sheet(sampled, args.output_dir / "audit_sheet.csv")
    write_audit_index_html(sampled, args.output_dir / "index.html")

    return 0


if __name__ == "__main__":
    exit(main())
