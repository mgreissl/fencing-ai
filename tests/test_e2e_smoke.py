"""End-to-end smoke test: tiny random VideoMAE, synthetic clips, 1 epoch on CPU.

Runs the real ``python -m src.train`` and ``python -m src.evaluate`` entry
points (Hydra, checkpoint, saved split, merged checkpoint config) without
network access or W&B.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.kinematics import (
    KINEMATIC_FEATURE_DIM,
    kinematics_cache_path,
    save_kinematics_atomic,
)
from src.manifest import MANIFEST_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent

N_FRAMES_CLIP = 10
CLIP_FPS = 10.0
SIZE = 32
# (match_id, split) -> 2 clips each; 6 matches, every split non-empty
MATCHES = [
    ("m1", "train"), ("m2", "train"), ("m3", "train"), ("m4", "train"),
    ("m5", "val"), ("m6", "test"),
]
LABELS = ["L", "T", "R"]
HIT_TYPES = ["On-On", "On-No", "No-On"]

BACKBONE = "{hidden_size:32,num_hidden_layers:2,num_attention_heads:2," \
    "intermediate_size:64,image_size:32,patch_size:16,tubelet_size:2,num_frames:4}"


def _write_clip(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (64, 48)
    )
    assert writer.isOpened()
    for _ in range(N_FRAMES_CLIP):
        writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    writer.release()


def _make_dataset(root: Path, cache_dir: Path) -> Path:
    clips_dir = root / "clips"
    clips_dir.mkdir()
    rows = []
    for i, (match_id, split) in enumerate(MATCHES):
        for j in range(2):
            n = 2 * i + j
            path = clips_dir / f"{LABELS[n % 3]}_foil_{match_id}_{j}.mp4"
            _write_clip(path, seed=n)
            save_kinematics_atomic(
                kinematics_cache_path(cache_dir, path),
                np.random.default_rng(n).normal(
                    size=(N_FRAMES_CLIP, KINEMATIC_FEATURE_DIM)
                ).astype(np.float32),
            )
            rows.append(
                {
                    "clip_path": path.as_posix(),
                    "label": LABELS[n % 3],
                    "weapon": "foil",
                    "match_id": match_id,
                    "source_url": "",
                    "tournament": f"t{i % 2}",
                    "hit_type": HIT_TYPES[n % 3],
                    "score_left_before": 0,
                    "score_right_before": 0,
                    "score_left_after": 1,
                    "score_right_after": 0,
                    "light_time_s": 0.5,
                    "source_fps": CLIP_FPS,
                    "clip_fps": CLIP_FPS,
                    "n_frames": N_FRAMES_CLIP,
                    "light_frame_in_clip": 6,
                    "split": split,
                }
            )
    with open(clips_dir / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return clips_dir


def _run(module: str, overrides: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, *overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@pytest.mark.parametrize(
    "experiment",
    [
        "video_kinematics",
        pytest.param("video_only", marks=pytest.mark.slow),
        pytest.param("kinematics_only", marks=pytest.mark.slow),
    ],
)
def test_train_then_evaluate(tmp_path, experiment):
    cache_dir = tmp_path / "kin"
    clips_dir = _make_dataset(tmp_path, cache_dir)
    ckpt = tmp_path / "ckpt"

    train_overrides = [
        f"+experiment={experiment}",
        "data.weapon=foil",
        f"data.clips_dir='{clips_dir.as_posix()}'",
        "data.use_manifest_split=true",
        "data.num_frames=4",
        "data.frame_size=32",
        "data.num_workers=0",
        f"model.backbone_config={BACKBONE}",
        "model.use_lora=false",
        "model.freeze_backbone=false",
        "model.head.hidden_dim=16",
        "model.kinematics.hidden_dim=16",
        f"model.kinematics.cache_dir='{cache_dir.as_posix()}'",
        "training.epochs=1",
        "training.batch_size=2",
        "training.mixed_precision=false",
        "training.device=cpu",
        "training.scheduler.warmup_epochs=0",
        f"training.checkpoint_dir='{ckpt.as_posix()}'",
        f"hydra.run.dir='{(tmp_path / 'hydra_train').as_posix()}'",
        "wandb.enabled=false",
    ]
    _run("src.train", train_overrides)
    assert (ckpt / "best_model.pt").is_file()
    assert (ckpt / "split_manifest.csv").is_file()

    # Evaluate: only the checkpoint dir is given; model, data and kinematics
    # settings must come from the checkpoint config, the split from the file.
    _run(
        "src.evaluate",
        [
            f"training.checkpoint_dir='{ckpt.as_posix()}'",
            f"hydra.run.dir='{(tmp_path / 'hydra_eval').as_posix()}'",
        ],
    )
    predictions = ckpt / "predictions_test.csv"
    metrics = json.loads((ckpt / "metrics_test.json").read_text(encoding="utf-8"))
    assert predictions.is_file()
    assert 0.0 <= metrics["overall"]["accuracy"] <= 1.0
    assert metrics["ambiguous"]["n"] >= 0 and "model_only" in metrics

    with open(predictions, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {r["source"] for r in rows} <= {"rule", "model"}
    assert {r["match_id"] for r in rows} == {"m6"}  # saved split, not recomputed
