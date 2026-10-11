import numpy as np
import pytest
import torch

import src.dataset as dataset_module
from src.dataset import FencingVideoDataset
from src.kinematics import KINEMATIC_FEATURE_DIM
from src.manifest import ClipRecord

N_CLIP = 40


def _fake_read_frames(path, index_fn):
    indices = index_fn(N_CLIP)
    frames = torch.zeros(len(indices), 3, 8, 8)
    frames[:, 0, 0, 0] = torch.tensor(indices, dtype=torch.float32)
    return frames


@pytest.fixture
def patched(monkeypatch, tmp_path):
    full = np.tile(
        np.arange(N_CLIP, dtype=np.float32)[:, None], (1, KINEMATIC_FEATURE_DIM)
    )
    calls = []

    def fake_load(clip_path, cache_dir, settings=None, pose_model=None, compute_if_missing=True):
        calls.append((clip_path, cache_dir, settings, compute_if_missing))
        return full

    monkeypatch.setattr(dataset_module, "read_frames", _fake_read_frames)
    monkeypatch.setattr(dataset_module, "load_or_compute_kinematics", fake_load)
    return calls


def _record(tmp_path):
    return ClipRecord(str(tmp_path / "a.mp4"), "L", "foil")


def test_kinematics_rows_match_video_indices(patched, tmp_path):
    ds = FencingVideoDataset(
        [_record(tmp_path)],
        num_frames=8,
        sampling_mode="uniform",
        temporal_jitter=True,
        max_shift=3,
        use_kinematics=True,
    )
    for _ in range(5):  # jitter differs per call; both streams must agree
        item = ds[0]
        video_idx = item["pixel_values"][:, 0, 0, 0]
        assert item["kinematics"].shape == (8, KINEMATIC_FEATURE_DIM)
        assert torch.equal(item["kinematics"][:, 0], video_idx)


def test_config_passed_and_missing_cache_not_computed(patched, tmp_path):
    settings = {"max_gap": 3}
    ds = FencingVideoDataset(
        [_record(tmp_path)],
        num_frames=4,
        use_kinematics=True,
        kinematics_cache_dir="cache_here",
        kinematics_settings=settings,
    )
    ds[0]
    _, cache_dir, got_settings, compute = patched[0]
    assert cache_dir == "cache_here"
    assert got_settings == settings
    assert compute is False


def test_dataset_does_not_write_cache(patched, tmp_path):
    cache = tmp_path / "kin"
    ds = FencingVideoDataset(
        [_record(tmp_path)], num_frames=4, use_kinematics=True,
        kinematics_cache_dir=str(cache),
    )
    ds[0]
    assert not cache.exists()


@pytest.mark.parametrize("n_rows, ok", [(N_CLIP, True), (N_CLIP + 1, True), (N_CLIP - 1, True), (N_CLIP + 2, False), (N_CLIP - 5, False)])
def test_feature_length_must_match_decoded_frames(monkeypatch, tmp_path, n_rows, ok):
    full = np.zeros((n_rows, KINEMATIC_FEATURE_DIM), dtype=np.float32)
    monkeypatch.setattr(dataset_module, "read_frames", _fake_read_frames)
    monkeypatch.setattr(
        dataset_module, "load_or_compute_kinematics", lambda *a, **k: full
    )
    ds = FencingVideoDataset([_record(tmp_path)], num_frames=4, use_kinematics=True)
    if ok:
        assert ds[0]["kinematics"].shape == (4, KINEMATIC_FEATURE_DIM)
    else:
        with pytest.raises(RuntimeError, match="a.mp4.*frames"):
            ds[0]
