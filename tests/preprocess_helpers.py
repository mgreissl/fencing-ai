"""Synthetic timelines shared by the preprocessing tests."""

import numpy as np

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
