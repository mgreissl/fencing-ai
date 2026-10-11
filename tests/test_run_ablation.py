import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_ablation.py"
spec = importlib.util.spec_from_file_location("run_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def _write_metrics(directory: Path, accuracy, macro_f1, ece=0.1, loss=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics_test.json").write_text(
        json.dumps(
            {
                "loss": loss,
                "overall": {"accuracy": accuracy, "macro_f1": macro_f1, "ece": ece},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def root(tmp_path):
    _write_metrics(tmp_path / "video_only" / "seed42", 0.60, 0.50)
    _write_metrics(tmp_path / "video_only" / "seed43", 0.70, 0.60)
    _write_metrics(tmp_path / "video_only" / "seed44", 0.80, 0.70)
    _write_metrics(tmp_path / "kinematics_only" / "seed42", 0.50, 0.40)
    (tmp_path / "video_only" / "seed42" / "unrelated.json").write_text("{}")
    return tmp_path


def test_collect_runs(root):
    rows = ablation.collect_runs(root)
    assert len(rows) == 4
    first = [r for r in rows if r["experiment"] == "video_only" and r["seed"] == 42][0]
    assert first["accuracy"] == 0.60 and first["fold"] == ""


def test_collect_runs_with_folds(tmp_path):
    _write_metrics(tmp_path / "video_only" / "seed1" / "fold0", 0.5, 0.5)
    _write_metrics(tmp_path / "video_only" / "seed1" / "fold1", 0.7, 0.5)
    rows = ablation.collect_runs(tmp_path)
    assert sorted(r["fold"] for r in rows) == ["0", "1"]


def test_aggregate_mean_and_sample_std(root):
    summary = {e["experiment"]: e for e in ablation.aggregate(ablation.collect_runs(root))}
    video = summary["video_only"]
    assert video["n_runs"] == 3 and video["n_seeds"] == 3
    assert video["accuracy_mean"] == pytest.approx(0.70)
    assert video["accuracy_std"] == pytest.approx(0.10)  # ddof=1
    assert video["macro_f1_mean"] == pytest.approx(0.60)
    single = summary["kinematics_only"]
    assert single["accuracy_mean"] == pytest.approx(0.50)
    assert single["accuracy_std"] == 0.0


def test_missing_metric_is_none(tmp_path):
    directory = tmp_path / "exp" / "seed1"
    directory.mkdir(parents=True)
    (directory / "metrics_test.json").write_text(json.dumps({"loss": 2.0}))
    (summary,) = ablation.aggregate(ablation.collect_runs(tmp_path))
    assert summary["accuracy_mean"] is None and summary["loss_mean"] == 2.0


def test_write_aggregate_outputs(root):
    ablation.write_aggregate(root, commit="abc123")
    md = (root / "summary.md").read_text(encoding="utf-8")
    assert "abc123" in md
    assert "| video_only | 3 | 3 | 0.7000 ± 0.1000 |" in md
    header = (root / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("experiment,n_runs,n_seeds,accuracy_mean,accuracy_std")
    assert len((root / "runs.csv").read_text(encoding="utf-8").splitlines()) == 5


def test_build_fold_commands(tmp_path):
    runs = ablation.build_fold_commands(
        "kinematics_only", 7, tmp_path, cv_folds=2, extra_overrides=["wandb.enabled=false"]
    )
    assert len(runs) == 2
    train, evaluate, metrics_path = runs[1]
    assert "+experiment=kinematics_only" in train
    assert "training.seed=7" in train
    assert "data.cv_folds=2" in train and "data.cv_fold=1" in train
    assert "wandb.enabled=false" in train
    assert metrics_path == tmp_path / "kinematics_only" / "seed7" / "fold1" / "metrics_test.json"
    assert evaluate[2] == "src.evaluate"
    assert any("fold1" in arg and arg.startswith("training.checkpoint_dir=") for arg in evaluate)
    assert not any(a.startswith("+experiment") for a in evaluate)


def test_dry_run_prints_without_running(tmp_path, capsys):
    code = ablation.main(
        ["--root", str(tmp_path), "--experiments", "video_only", "--seeds", "1", "2", "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("src.train") == 2 and out.count("src.evaluate") == 2
