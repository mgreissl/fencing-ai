import logging

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.manifest import ClipRecord
from src.metrics import ambiguous_macro_f1
from src.train import _selection_key, _selection_score, _validate


def test_ambiguous_macro_f1_uses_only_ambiguous_clips():
    y_true = [0, 2, 1, 0, 2]
    y_pred = [2, 0, 1, 0, 2]
    hits = ["On-No", "No-On", "On-On", "On-Off", "On-On"]
    weapons = ["foil"] * 5
    # ambiguous = idx 2, 3, 4 -> all correct
    assert ambiguous_macro_f1(y_true, y_pred, hits, weapons) == pytest.approx(1.0)


def test_ambiguous_macro_f1_nan_when_empty():
    value = ambiguous_macro_f1([0], [0], ["On-No"], ["foil"])
    assert np.isnan(value)


def test_selection_key_on_on():
    assert _selection_key("on_on_macro_f1") == "val/on_on_macro_f1"


def test_selection_score_uses_requested_metric():
    metrics = {"val/macro_f1": 0.5, "val/on_on_macro_f1": 0.8}
    assert _selection_score(metrics, "on_on_macro_f1", set()) == (0.8, "on_on_macro_f1")


def test_selection_score_falls_back_once_with_warning(caplog):
    metrics = {"val/macro_f1": 0.5}  # no ambiguous val clips
    warned: set = set()
    with caplog.at_level(logging.WARNING, logger="src.train"):
        first = _selection_score(metrics, "on_on_macro_f1", warned)
        second = _selection_score(metrics, "on_on_macro_f1", warned)
    assert first == second == (0.5, "macro_f1")
    assert len([r for r in caplog.records if "falling back" in r.message]) == 1


class _FakeClips(Dataset):
    """Yields batches shaped like FencingVideoDataset; `samples` for metadata."""

    def __init__(self, labels, hit_types, weapons):
        self.samples = [
            ClipRecord(f"/clips/{i}.mp4", "LTR"[lab], weapon, match_id=f"m{i}", hit_type=hit)
            for i, (lab, hit, weapon) in enumerate(zip(labels, hit_types, weapons))
        ]
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        return {
            "pixel_values": torch.zeros(2, 3, 4, 4),
            "label": self.labels[i],
            "weapon_id": rec.weapon_id,
            "path": rec.abs_path(),
        }


class _Constant(nn.Module):
    """Always predicts class `cls` (logits independent of input)."""

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values, weapon_id=None, kinematics=None):
        out = torch.zeros(pixel_values.shape[0], 3) + self.dummy
        out[:, self.cls] = 5.0
        return out


CFG = OmegaConf.create(
    {
        "model": {"class_names": ["Left", "Tie", "Right"]},
        "eval": {"flip_tta": False},
        "training": {"epochs": 1, "mixed_precision": False},
        "wandb": {"log_media_every_n_epochs": 1, "max_misclassified_samples": 1},
    }
)


def test_validate_logs_on_on_macro_f1():
    labels = [0, 2, 1, 1]  # single-light L, R then two On-On ties
    hits = ["On-No", "No-On", "On-On", "On-On"]
    ds = _FakeClips(labels, hits, ["foil"] * 4)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    metrics = _validate(
        _Constant(1), loader, nn.CrossEntropyLoss(), torch.device("cpu"), 0, False, CFG
    )
    # always Tie: accuracy 2/4 overall, but perfect on the ambiguous On-On clips
    assert metrics["val/accuracy"] == pytest.approx(0.5)
    assert metrics["val/on_on_macro_f1"] == pytest.approx(1.0)
    assert metrics["val/macro_f1"] < 1.0


def test_validate_omits_on_on_metric_without_ambiguous_clips():
    ds = _FakeClips([0, 2], ["On-No", "No-On"], ["foil"] * 2)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    metrics = _validate(
        _Constant(1), loader, nn.CrossEntropyLoss(), torch.device("cpu"), 0, False, CFG
    )
    assert "val/on_on_macro_f1" not in metrics
