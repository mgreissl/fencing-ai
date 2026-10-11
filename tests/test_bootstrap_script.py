import subprocess
import sys
from pathlib import Path

from src.metrics import write_predictions

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bootstrap_eval.py"


def _rows(correct_flags, prefix):
    rows = []
    for i, ok in enumerate(correct_flags):
        rows.append(
            dict(
                clip_path=f"{i}.mp4",
                match_id=f"m{i // 3}",
                hit_type="On-On",
                weapon="foil",
                label="L",
                pred="L" if ok else "R",
                p_L=0.9,
                p_T=0.05,
                p_R=0.05,
            )
        )
    return rows


def test_script_runs_with_compare(tmp_path):
    write_predictions(_rows([True] * 20 + [False] * 4, "a"), tmp_path / "a.csv")
    write_predictions(_rows([True] * 12 + [False] * 12, "b"), tmp_path / "b.csv")
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(tmp_path / "a.csv"),
            "--compare",
            str(tmp_path / "b.csv"),
            "--n-boot",
            "50",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "95% CI" in out
    assert "McNemar" in out and "A-only correct=8" in out
