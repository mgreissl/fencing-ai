import random

import pytest
import torch

from src.transforms import TrainVideoTransform


def _clip(n_frames=6, size=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    frame = torch.rand(3, size, size, generator=gen)
    return frame.expand(n_frames, -1, -1, -1).clone()


def _jitter_only(**kwargs):
    return TrainVideoTransform(
        frame_size=16,
        flip_prob=0.0,
        color_jitter=True,
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1,
        **kwargs,
    )


def test_color_jitter_identical_params_across_frames():
    out, _ = _jitter_only()(_clip(), label=0)
    for t in range(1, out.shape[0]):
        assert torch.allclose(out[0], out[t], atol=1e-6)


def test_color_jitter_changes_between_calls():
    transform = _jitter_only()
    first, _ = transform(_clip(), label=0)
    second, _ = transform(_clip(), label=0)
    assert not torch.allclose(first, second)


def test_color_jitter_does_not_reseed_global_rng(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("global RNG must not be reseeded")

    transform = _jitter_only()
    monkeypatch.setattr(torch, "manual_seed", fail)
    monkeypatch.setattr(random, "seed", fail)
    transform(_clip(), label=0)
