import pytest

from src.dataset import filter_readable_clips
from src.manifest import ClipRecord
from tests.test_sampling import _make_video


def _corrupt_with_manifest_frames(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    return ClipRecord(str(path), "L", "foil", n_frames=60)


def test_exists_mode_trusts_manifest_n_frames(tmp_path):
    rec = _corrupt_with_manifest_frames(tmp_path)
    assert filter_readable_clips([rec], mode="exists") == [rec]
    assert filter_readable_clips([rec]) == [rec]  # default is "exists"


def test_full_mode_drops_corrupt_clip_with_manifest_n_frames(tmp_path):
    rec = _corrupt_with_manifest_frames(tmp_path)
    assert filter_readable_clips([rec], mode="full") == []


def test_full_mode_keeps_readable_clip(tmp_path):
    video = tmp_path / "ok.mp4"
    _make_video(video)
    rec = ClipRecord(str(video), "L", "foil", n_frames=20)
    assert filter_readable_clips([rec], mode="full") == [rec]


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="validate_clips"):
        filter_readable_clips([], mode="bogus")


def test_discover_clips_passes_mode(tmp_path):
    from src.dataset import discover_clips

    (tmp_path / "L1.mp4").write_bytes(b"x")
    with pytest.raises(ValueError, match="validate_clips"):
        discover_clips(str(tmp_path), "foil", validate_mode="bogus")


def _records(tmp_path):
    with_onset = ClipRecord(
        str(tmp_path / "a.mp4"), "L", "foil", light_frame_in_clip=30, clip_fps=30.0
    )
    no_onset = ClipRecord(str(tmp_path / "b.mp4"), "R", "foil")
    no_fps = ClipRecord(
        str(tmp_path / "c.mp4"), "T", "foil", light_frame_in_clip=30
    )
    return with_onset, no_onset, no_fps


def test_missing_onset_fallback_is_logged_once(tmp_path, caplog):
    import logging

    from src.dataset import FencingVideoDataset

    records = list(_records(tmp_path))
    with caplog.at_level(logging.WARNING, logger="src.dataset"):
        ds = FencingVideoDataset(records, sampling_mode="onset_window")
    msgs = [r.getMessage() for r in caplog.records if "onset" in r.getMessage()]
    assert len(msgs) == 1 and "2 of 3" in msgs[0]
    assert ds.n_onset_fallback == 2 and len(ds) == 3


def test_no_warning_for_other_sampling_modes(tmp_path, caplog):
    import logging

    from src.dataset import FencingVideoDataset

    with caplog.at_level(logging.WARNING, logger="src.dataset"):
        ds = FencingVideoDataset(list(_records(tmp_path)), sampling_mode="uniform")
    assert not [r for r in caplog.records if "onset" in r.getMessage()]
    assert ds.n_onset_fallback == 0


def test_require_onset_drops_clips_without_onset(tmp_path):
    from src.dataset import FencingVideoDataset

    records = list(_records(tmp_path))
    ds = FencingVideoDataset(
        records, sampling_mode="onset_window", require_onset=True
    )
    assert ds.samples == [records[0]]
    assert ds.n_onset_fallback == 0
    # not applicable when the mode does not use the onset
    ds = FencingVideoDataset(records, sampling_mode="uniform", require_onset=True)
    assert len(ds) == 3


def test_require_onset_read_from_config():
    from omegaconf import OmegaConf

    from src.dataset import kinematics_dataset_kwargs

    cfg = OmegaConf.create(
        {"model": {"use_kinematics": False}, "data": {"require_onset": True}}
    )
    assert kinematics_dataset_kwargs(cfg)["require_onset"] is True
    cfg = OmegaConf.create({"model": {"use_kinematics": False}})
    assert kinematics_dataset_kwargs(cfg)["require_onset"] is False
