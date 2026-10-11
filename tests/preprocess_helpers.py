"""Synthetic timelines and videos shared by the preprocessing tests."""

import cv2
import numpy as np
import pytest

from scripts import preprocess_pipeline as pp


def make_timeline(fps=25.0, n_frames=2000, runs=(), left_state=pp.STATE_ON,
                  right_state=pp.STATE_NO, scores=None, stride=6):
    left = np.zeros(n_frames, dtype=np.int8)
    right = np.zeros(n_frames, dtype=np.int8)
    for start, stop in runs:
        left[start:stop] = left_state
        right[start:stop] = right_state
    score_frames = np.arange(0, n_frames, stride)
    score_list = [scores(i) if callable(scores) else scores for i in score_frames]
    return pp.Timeline(
        fps=fps, n_frames=n_frames, first_valid=0,
        left_state=left, right_state=right,
        score_frames=score_frames, scores=score_list,
    )


def encode_frame_index(frame, index):
    """Store a frame index in two pixels (survives lossless codecs)."""
    frame[50, 50, 0] = index % 256
    frame[50, 50, 2] = index // 256


def decode_frame_index(frame):
    return int(frame[50, 50, 0]) + 256 * int(frame[50, 50, 2])


def write_synthetic_match(path, fps=30.0, n_frames=1500, onset=600, length=90,
                          left_light=True, right_light=False):
    """Black lossless video with scoring lights on from ``onset``."""
    cfg = pp.OVERLAY
    boxes = []
    if left_light:
        boxes.append(cfg.left_on)
    if right_light:
        boxes.append(cfg.right_on)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, cfg.frame_size
    )
    if not writer.isOpened():
        pytest.skip("FFV1 writer unavailable")
    for i in range(n_frames):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        encode_frame_index(frame, i)
        if onset <= i < onset + length:
            for box in boxes:
                y0, y1, x0, x1 = box.region
                frame[y0:y1, x0:x1] = box.template
        writer.write(frame)
    writer.release()
