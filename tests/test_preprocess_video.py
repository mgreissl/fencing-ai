"""Integration test: synthetic video through scan, analysis and clip writing."""

import cv2
import numpy as np
import pytest

from scripts import preprocess_pipeline as pp

FPS = 30.0
N_FRAMES = 500
ONSET = 300


def write_synthetic_video(path, fps=FPS, n_frames=N_FRAMES, onset=ONSET, length=90):
    """Black video with the left (red) light on from ``onset`` for ``length``."""
    box = pp.OVERLAY.left_on
    y0, y1, x0, x1 = box.region
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, pp.OVERLAY.frame_size
    )
    if not writer.isOpened():
        pytest.skip("FFV1 writer unavailable")
    for i in range(n_frames):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = i % 256  # unique-ish marker per frame
        if onset <= i < onset + length:
            frame[y0:y1, x0:x1] = box.template
        writer.write(frame)
    writer.release()


def scripted_scores(frame):
    # Reader is called with frames, not indices; use the marker channel.
    return (3, 2)


@pytest.fixture()
def synthetic_video(tmp_path):
    path = tmp_path / "match.avi"
    write_synthetic_video(path)
    return path


def test_scan_finds_onset_and_reads_fps(synthetic_video):
    cfg = pp.PipelineConfig(intro_s=2.0)
    timeline = pp.scan_video(str(synthetic_video), "sabre", cfg, scripted_scores)
    assert timeline.fps == pytest.approx(FPS)
    assert timeline.n_frames == N_FRAMES
    onsets = pp.find_light_onsets(timeline.lit, 15, first_valid=timeline.first_valid)
    assert onsets == [ONSET]


def test_clip_is_aligned_to_light_onset(synthetic_video, tmp_path):
    cfg = pp.PipelineConfig(intro_s=2.0)
    timeline = pp.scan_video(str(synthetic_video), "sabre", cfg, scripted_scores)
    window = pp.compute_window(ONSET, timeline.fps, total_frames=timeline.n_frames)
    out = tmp_path / "clip.mp4"

    saved = pp.write_clips(str(synthetic_video), [(window, str(out))])
    assert saved == [str(out)]

    cap = cv2.VideoCapture(str(out))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    assert len(frames) == window.n_frames == 69
    y0, y1, x0, x1 = pp.OVERLAY.left_on.region
    mean_red = [float(f[y0:y1, x0:x1, 2].mean()) for f in frames]
    k = window.light_frame_in_clip
    assert max(mean_red[:k]) < 60
    assert mean_red[k] > 150


def test_window_start_matches_source_time(synthetic_video, tmp_path):
    # The green marker channel encodes (source index % 256); the first clip
    # frame must be the frame 2.0 s before the onset at the real 30 fps.
    cfg = pp.PipelineConfig(intro_s=2.0)
    timeline = pp.scan_video(str(synthetic_video), "sabre", cfg, scripted_scores)
    window = pp.compute_window(ONSET, timeline.fps, total_frames=timeline.n_frames)
    out = tmp_path / "clip.mp4"
    pp.write_clips(str(synthetic_video), [(window, str(out))])
    cap = cv2.VideoCapture(str(out))
    ok, first = cap.read()
    cap.release()
    assert ok
    expected = (ONSET - 60) % 256
    assert abs(int(first[100, 300, 1]) - expected) <= 6  # lossy mp4v


def test_invalid_fps_video_is_skipped(tmp_path):
    missing = tmp_path / "missing.avi"
    assert pp.scan_video(str(missing), "sabre") is None
