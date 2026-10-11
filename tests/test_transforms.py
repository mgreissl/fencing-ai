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


def test_build_from_default_config_has_no_dead_crop_settings():
    from omegaconf import OmegaConf

    from src.transforms import build_train_transforms

    cfg = OmegaConf.load("configs/default.yaml")
    assert "random_resized_crop" not in cfg.augmentation.train
    transform = build_train_transforms(cfg)
    assert not hasattr(transform, "crop_scale")


from src.transforms import (  # noqa: E402
    IMAGENET_MEAN,
    EvalVideoTransform,
    build_eval_transforms,
    build_train_transforms,
    mask_overlay,
)


def test_mask_overlay_fills_bottom_rows_only():
    frames = torch.ones(2, 3, 360, 640)
    out = mask_overlay(frames, 0.82)
    assert torch.equal(out[:, :, :295], frames[:, :, :295])
    expected = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).expand(2, 3, 65, 640)
    assert torch.allclose(out[:, :, 295:], expected)
    assert frames.min() == 1.0  # input not mutated


def test_mask_overlay_custom_fill_and_noop():
    frames = torch.ones(1, 3, 10, 4)
    assert torch.equal(mask_overlay(frames, 1.0), frames)
    out = mask_overlay(frames, 0.5, fill=0.0)
    assert out[:, :, 5:].sum() == 0 and out[:, :, :5].min() == 1.0


def test_eval_transform_masks_overlay():
    frames = torch.ones(2, 3, 360, 640)
    out, _ = EvalVideoTransform(frame_size=36, overlay_mask_y_min_frac=0.82)(frames, 0)
    assert torch.allclose(out[:, :, -3:], torch.zeros_like(out[:, :, -3:]), atol=1e-5)
    assert out[:, :, :20].abs().min() > 0.5  # white area stays bright after norm


def _overlay_cfg(enabled=True):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("configs/default.yaml")
    cfg.data.overlay_mask.enabled = enabled
    return cfg


def test_builders_read_overlay_config():
    cfg = _overlay_cfg()
    assert cfg.data.overlay_mask.y_min_frac == 0.82
    assert build_train_transforms(cfg).overlay_mask_y_min_frac == 0.82
    assert build_eval_transforms(cfg).overlay_mask_y_min_frac == 0.82
    off = _overlay_cfg(enabled=False)
    assert build_train_transforms(off).overlay_mask_y_min_frac is None
    assert build_eval_transforms(off).overlay_mask_y_min_frac is None


def test_flipped_masked_clip_has_no_overlay_pixels():
    # overlay marker: bright red band in the bottom rows; scene is dark green
    frames = torch.zeros(4, 3, 360, 640)
    frames[:, 1] = 0.3
    frames[:, 0, 300:330, 100:300] = 1.0
    frames[:, 1:, 300:330, 100:300] = 0.0
    transform = TrainVideoTransform(
        frame_size=72, overlay_mask_y_min_frac=0.82, flip_prob=1.0, color_jitter=False
    )
    out, label = transform(frames, label=0)
    assert label == 2  # flipped
    band = out[:, :, int(72 * 0.82) + 1 :, :]
    assert torch.allclose(band, torch.zeros_like(band), atol=1e-4)
    # the scene itself is still present above the band
    assert out[:, 1, :10, :].abs().min() > 0
