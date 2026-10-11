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
