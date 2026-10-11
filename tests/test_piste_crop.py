import pytest
import torch
from omegaconf import OmegaConf

from src.overlay import overlay_mask_y_min_frac
from src.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    EvalVideoTransform,
    TrainVideoTransform,
    build_eval_transforms,
    build_train_transforms,
    crop_piste,
)


def _banded_clip(height=100, width=160, n_frames=3):
    """Rows 25..79 are 0.5, everything else is 1.0."""
    frames = torch.ones(n_frames, 3, height, width)
    frames[:, :, 25:80, :] = 0.5
    return frames


def test_crop_piste_selects_rows():
    out = crop_piste(_banded_clip(), 0.25, 0.8)
    assert out.shape == (3, 3, 55, 160)
    assert torch.all(out == 0.5)


def test_crop_piste_open_end_keeps_bottom():
    assert crop_piste(_banded_clip(), 0.25, None).shape[-2] == 75


def test_crop_piste_rejects_empty_band():
    with pytest.raises(ValueError, match="piste_crop"):
        crop_piste(_banded_clip(), 0.8, 0.25)


def _expected(value, channel):
    return (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]


def test_eval_transform_crops_before_resize():
    tf = EvalVideoTransform(frame_size=16, piste_crop=(0.25, 0.8))
    out, _ = tf(_banded_clip(), 0)
    assert out.shape == (3, 3, 16, 16)
    for c in range(3):
        assert torch.allclose(out[:, c], torch.tensor(_expected(0.5, c)), atol=1e-5)


def test_train_transform_crops_before_resize_and_flip():
    tf = TrainVideoTransform(
        frame_size=16, flip_prob=1.0, color_jitter=False, piste_crop=(0.25, 0.8)
    )
    out, label = tf(_banded_clip(), 0)
    assert label == 2
    for c in range(3):
        assert torch.allclose(out[:, c], torch.tensor(_expected(0.5, c)), atol=1e-5)


def test_crop_off_by_default_matches_old_behaviour():
    tf = EvalVideoTransform(frame_size=16)
    out, _ = tf(_banded_clip(), 0)
    assert out[:, 0].max() > _expected(0.5, 0) + 0.1  # bright rows survive


def _cfg(**crop):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.data.piste_crop = {"enabled": True, **crop}
    return cfg


def test_default_config_has_crop_disabled():
    cfg = OmegaConf.load("configs/default.yaml")
    assert cfg.data.piste_crop.enabled is False
    assert build_eval_transforms(cfg).piste_crop is None
    assert build_train_transforms(cfg).piste_crop is None


def test_config_null_y_max_uses_overlay_top():
    cfg = _cfg(y_min_frac=0.25, y_max_frac=None)
    expected = (0.25, pytest.approx(overlay_mask_y_min_frac()))
    assert build_eval_transforms(cfg).piste_crop == expected
    assert build_train_transforms(cfg).piste_crop == expected


def test_config_explicit_band_and_mask_override():
    cfg = _cfg(y_min_frac=0.3, y_max_frac=0.7)
    assert build_eval_transforms(cfg).piste_crop == (0.3, 0.7)
    cfg = _cfg(y_min_frac=0.3, y_max_frac=None)
    cfg.data.overlay_mask.y_min_frac = 0.6
    assert build_eval_transforms(cfg).piste_crop == (0.3, 0.6)


def test_mask_skipped_when_crop_already_excludes_band():
    tf = EvalVideoTransform(
        frame_size=16, overlay_mask_y_min_frac=0.82, piste_crop=(0.25, 0.8)
    )
    out, _ = tf(_banded_clip(), 0)
    for c in range(3):
        assert torch.allclose(out[:, c], torch.tensor(_expected(0.5, c)), atol=1e-5)
