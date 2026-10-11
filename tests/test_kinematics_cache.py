"""Tests for the kinematics cache key (#33) and atomic cache writes (#35)."""

from pathlib import Path

import numpy as np
import pytest

import src.kinematics as kin
from src.kinematics import (
    DEFAULT_KINEMATICS_SETTINGS,
    kinematics_cache_path,
    kinematics_settings_hash,
)


def test_cache_path_is_stable_and_uses_defaults(tmp_path):
    p1 = kinematics_cache_path(tmp_path, "data/clips/foil/L1234-5.mp4", None)
    p2 = kinematics_cache_path(tmp_path, Path("other/dir/L1234-5.mp4"), {})
    p3 = kinematics_cache_path(tmp_path, "L1234-5.mp4", dict(DEFAULT_KINEMATICS_SETTINGS))
    assert p1 == p2 == p3
    assert p1.name == "L1234-5.npy"
    assert p1.parent.parent == tmp_path
    assert "yolo11n-pose" in p1.parent.name


@pytest.mark.parametrize(
    "override",
    [
        {"pose_model": "yolo11m-pose.pt"},
        {"max_gap": 2},
        {"smoothing_window": 0},
        {"keypoint_conf_threshold": 0.5},
        {"y_band": [0.25, 0.8]},
        {"default_fps": 30.0},
    ],
)
def test_cache_path_changes_with_settings(tmp_path, override):
    base = kinematics_cache_path(tmp_path, "clip.mp4", None)
    changed = kinematics_cache_path(tmp_path, "clip.mp4", override)
    assert base != changed
    assert kinematics_settings_hash(None) != kinematics_settings_hash(override)


def test_cache_path_changes_with_feature_version(tmp_path, monkeypatch):
    base = kinematics_cache_path(tmp_path, "clip.mp4", None)
    monkeypatch.setattr(kin, "KINEMATIC_FEATURE_VERSION", kin.KINEMATIC_FEATURE_VERSION + 1)
    assert kinematics_cache_path(tmp_path, "clip.mp4", None) != base


def test_unknown_setting_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        kinematics_cache_path(tmp_path, "clip.mp4", {"num_frames": 16})


def test_get_pose_model_caches_per_name(monkeypatch):
    import sys
    import types

    created = []

    class FakeYOLO:
        def __init__(self, name):
            created.append(name)
            self.name = name

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(kin, "_POSE_MODELS", {})
    a = kin.get_pose_model("a-pose.pt")
    b = kin.get_pose_model("b-pose.pt")
    assert a.name == "a-pose.pt" and b.name == "b-pose.pt"
    assert kin.get_pose_model("a-pose.pt") is a
    assert created == ["a-pose.pt", "b-pose.pt"]
