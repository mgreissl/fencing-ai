import random

import numpy as np
import pytest

from src.dataset import (
    FencingVideoDataset,
    filter_readable_clips,
    read_frames,
    sample_frame_indices,
)
from src.manifest import ClipRecord

MODES = ["onset_window", "end_weighted", "uniform"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("total", [16, 17, 45, 120, 400])
def test_indices_length_sorted_in_range(mode, total):
    idx = sample_frame_indices(
        total, 16, sampling_mode=mode, onset_frame=total * 0.8, clip_fps=30.0
    )
    assert len(idx) == 16
    assert idx == sorted(idx)
    assert 0 <= idx[0] and idx[-1] < total


@pytest.mark.parametrize("mode", MODES)
def test_short_clip_padded(mode):
    idx = sample_frame_indices(5, 16, sampling_mode=mode, onset_frame=3, clip_fps=30)
    assert len(idx) == 16 and max(idx) == 4


def test_onset_window_covers_window():
    # onset at frame 90, fps 30: window [60, 97.5] -> frames 60..98
    idx = sample_frame_indices(
        150,
        16,
        sampling_mode="onset_window",
        onset_frame=90,
        clip_fps=30.0,
        window_pre_s=1.0,
        window_post_s=0.25,
    )
    assert idx[0] == 60
    assert idx[-1] == 98  # round(97.5) -> 98 (banker's: 98)
    assert all(b >= a for a, b in zip(idx, idx[1:]))
    # evenly spaced up to rounding
    steps = np.diff(idx)
    assert steps.max() - steps.min() <= 1


def test_onset_window_clamps_at_clip_start():
    idx = sample_frame_indices(
        100, 16, "onset_window", onset_frame=10, clip_fps=30, window_pre_s=1.0
    )
    assert idx[0] == 0 and len(idx) == 16


@pytest.mark.parametrize(
    "onset,fps", [(None, 30.0), (50, None), (50, 0.0), (500, 30.0), (-1, 30.0)]
)
def test_onset_window_falls_back_to_uniform(onset, fps):
    expected = sample_frame_indices(120, 16, "uniform")
    got = sample_frame_indices(
        120, 16, "onset_window", onset_frame=onset, clip_fps=fps
    )
    assert got == expected


def test_jitter_shifts_window_and_is_reproducible():
    base = sample_frame_indices(
        200, 16, "onset_window", onset_frame=100, clip_fps=30
    )
    a = sample_frame_indices(
        200, 16, "onset_window", True, 5, 100, 30, rng=random.Random(1)
    )
    b = sample_frame_indices(
        200, 16, "onset_window", True, 5, 100, 30, rng=random.Random(1)
    )
    assert a == b
    assert abs(a[0] - base[0]) <= 5 + 1


def test_invalid_mode():
    with pytest.raises(ValueError):
        sample_frame_indices(100, 16, "bogus")


def test_filter_drops_missing_and_trusts_manifest_frames(tmp_path):
    present = tmp_path / "ok.mp4"
    present.write_bytes(b"not a video")
    good_meta = ClipRecord(str(present), "L", "foil", n_frames=60)
    missing = ClipRecord(str(tmp_path / "gone.mp4"), "L", "foil", n_frames=60)
    corrupt = ClipRecord(str(present), "T", "foil")  # n_frames unknown -> opened
    zero = ClipRecord(str(present), "R", "foil", n_frames=0)
    kept = filter_readable_clips([good_meta, missing, corrupt, zero])
    assert kept == [good_meta]


def _make_video(path, n_frames=20, size=32):
    av = pytest.importorskip("av")
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((size, size, 3), i * 10 % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as exc:  # encoder unavailable
        pytest.skip(f"cannot encode test video: {exc}")


def test_read_frames_opens_once_and_dataset_raises(tmp_path):
    video = tmp_path / "v.mp4"
    _make_video(video)
    calls = []

    def index_fn(total):
        calls.append(total)
        return [0, total // 2, total - 1]

    frames = read_frames(str(video), index_fn)
    assert frames.shape[0] == 3 and frames.shape[1] == 3
    assert len(calls) == 1 and calls[0] >= 19

    bad = ClipRecord(str(tmp_path / "bad.mp4"), "L", "foil")
    (tmp_path / "bad.mp4").write_bytes(b"garbage")
    ds = FencingVideoDataset([bad], num_frames=4)
    with pytest.raises(RuntimeError, match="bad.mp4"):
        ds[0]
