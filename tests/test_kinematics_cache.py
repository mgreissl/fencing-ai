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


# --- atomic writes and load-or-compute (#35) -------------------------------


def _tmp_leftovers(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.suffix == ".tmp"]


def test_save_atomic_roundtrip(tmp_path):
    target = tmp_path / "sub" / "clip.npy"
    arr = np.arange(42, dtype=np.float32).reshape(2, 21)
    assert kin.save_kinematics_atomic(target, arr) == target
    np.testing.assert_array_equal(np.load(target), arr)
    assert _tmp_leftovers(target.parent) == []


def test_failed_save_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "clip.npy"

    def broken_save(fh, arr, *args, **kwargs):
        fh.write(b"\x93NUMPY partial")
        raise OSError("disk full")

    monkeypatch.setattr(kin.np, "save", broken_save)
    with pytest.raises(OSError):
        kin.save_kinematics_atomic(target, np.zeros((3, 21)))
    assert not target.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_failed_save_keeps_previous_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "clip.npy"
    old = np.ones((4, 21), dtype=np.float32)
    kin.save_kinematics_atomic(target, old)

    def broken_save(fh, arr, *args, **kwargs):
        fh.write(b"garbage")
        raise RuntimeError("killed")

    monkeypatch.setattr(kin.np, "save", broken_save)
    with pytest.raises(RuntimeError):
        kin.save_kinematics_atomic(target, np.zeros((4, 21)))
    monkeypatch.undo()
    np.testing.assert_array_equal(np.load(target), old)


@pytest.fixture
def fake_extract(monkeypatch):
    calls = []

    def extract(clip_path, pose_model=None, fps=None, settings=None):
        calls.append(str(clip_path))
        return np.full((7, kin.KINEMATIC_FEATURE_DIM), len(calls), dtype=np.float32)

    monkeypatch.setattr(kin, "extract_kinematic_features", extract)
    return calls


def test_load_or_compute_computes_once_then_loads(tmp_path, fake_extract):
    first = kin.load_or_compute_kinematics("clips/L1.mp4", tmp_path)
    second = kin.load_or_compute_kinematics("clips/L1.mp4", tmp_path)
    assert fake_extract == ["clips/L1.mp4"]
    np.testing.assert_array_equal(first, second)
    assert second.shape == (7, kin.KINEMATIC_FEATURE_DIM)
    assert kinematics_cache_path(tmp_path, "clips/L1.mp4").exists()


def test_load_or_compute_recomputes_for_new_settings(tmp_path, fake_extract):
    kin.load_or_compute_kinematics("L1.mp4", tmp_path)
    kin.load_or_compute_kinematics("L1.mp4", tmp_path, settings={"max_gap": 1})
    assert len(fake_extract) == 2


def test_load_or_compute_can_require_precomputed_cache(tmp_path, fake_extract):
    with pytest.raises(FileNotFoundError):
        kin.load_or_compute_kinematics("L1.mp4", tmp_path, compute_if_missing=False)
    assert fake_extract == []


def test_load_or_compute_replaces_corrupt_or_stale_file(tmp_path, fake_extract):
    path = kinematics_cache_path(tmp_path, "L1.mp4")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x93NUMPY truncated")
    feats = kin.load_or_compute_kinematics("L1.mp4", tmp_path)
    assert feats.shape == (7, kin.KINEMATIC_FEATURE_DIM)
    np.save(path, np.zeros((5, 16), dtype=np.float32))  # old 16-dim layout
    feats = kin.load_or_compute_kinematics("L1.mp4", tmp_path)
    assert feats.shape[1] == kin.KINEMATIC_FEATURE_DIM
    assert len(fake_extract) == 2


# --- cache metadata validation (#52) ----------------------------------------


_CLIP_FPS = {"value": 30.0}


@pytest.fixture
def clip_file(tmp_path, monkeypatch):
    clip = tmp_path / "clips" / "L1.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"x" * 100)
    _CLIP_FPS["value"] = 30.0
    monkeypatch.setattr(kin, "_video_fps", lambda path: _CLIP_FPS["value"])
    return clip


def test_cache_stores_metadata_next_to_array(tmp_path, fake_extract, clip_file):
    import json

    kin.load_or_compute_kinematics(clip_file, tmp_path / "cache")
    path = kinematics_cache_path(tmp_path / "cache", clip_file)
    meta = json.loads(kin.kinematics_meta_path(path).read_text())
    assert meta["n_frames"] == 7
    assert meta["fps"] == 30.0
    assert meta["clip_size"] == 100
    assert "clip_mtime_ns" in meta
    assert _tmp_leftovers(path.parent) == []


def test_valid_cache_is_loaded_without_recompute(tmp_path, fake_extract, clip_file):
    cache = tmp_path / "cache"
    kin.load_or_compute_kinematics(clip_file, cache)
    kin.load_or_compute_kinematics(clip_file, cache, compute_if_missing=False)
    assert len(fake_extract) == 1


def test_cache_without_metadata_is_rejected(tmp_path, fake_extract, clip_file):
    cache = tmp_path / "cache"
    path = kinematics_cache_path(cache, clip_file)
    kin.save_kinematics_atomic(path, np.zeros((7, kin.KINEMATIC_FEATURE_DIM), "f4"))
    with pytest.raises(kin.KinematicsCacheError, match="metadata"):
        kin.load_or_compute_kinematics(clip_file, cache, compute_if_missing=False)
    kin.load_or_compute_kinematics(clip_file, cache)  # recompute allowed
    assert len(fake_extract) == 1


@pytest.mark.parametrize("change", ["size", "fps", "rows"])
def test_stale_cache_is_rejected_or_recomputed(tmp_path, fake_extract, clip_file, change):
    cache = tmp_path / "cache"
    kin.load_or_compute_kinematics(clip_file, cache)
    if change == "size":
        clip_file.write_bytes(b"x" * 250)
    elif change == "fps":
        _CLIP_FPS["value"] = 25.0
    else:
        path = kinematics_cache_path(cache, clip_file)
        kin.save_kinematics_atomic(path, np.zeros((9, kin.KINEMATIC_FEATURE_DIM), "f4"))
    with pytest.raises(kin.KinematicsCacheError, match=change if change != "rows" else "rows"):
        kin.load_or_compute_kinematics(clip_file, cache, compute_if_missing=False)
    feats = kin.load_or_compute_kinematics(clip_file, cache)
    assert len(fake_extract) == 2 and feats.shape[0] == 7
    kin.load_or_compute_kinematics(clip_file, cache, compute_if_missing=False)  # fresh now
