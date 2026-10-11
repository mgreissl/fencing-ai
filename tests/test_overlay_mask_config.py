import pytest
from omegaconf import OmegaConf

from src.overlay import load_overlay, overlay_mask_y_min_frac
from src.transforms import build_eval_transforms, build_train_transforms


def _cfg(**mask):
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.data.overlay_mask = {"enabled": True, **mask}
    return cfg


def test_frac_derived_from_score_region():
    overlay = load_overlay()
    frac = overlay_mask_y_min_frac(overlay)
    assert frac == pytest.approx(295 / 360)
    assert frac * overlay.frame_size[1] <= overlay.score_left[0]
    assert frac * overlay.frame_size[1] <= overlay.mask_band_y


def test_margin_moves_frac():
    assert overlay_mask_y_min_frac(margin_px=0) > overlay_mask_y_min_frac()


def test_transform_default_uses_overlay_geometry():
    cfg = _cfg()
    expected = overlay_mask_y_min_frac()
    assert build_train_transforms(cfg).overlay_mask_y_min_frac == pytest.approx(expected)
    assert build_eval_transforms(cfg).overlay_mask_y_min_frac == pytest.approx(expected)


def test_config_override_wins():
    cfg = _cfg(y_min_frac=0.9)
    assert build_eval_transforms(cfg).overlay_mask_y_min_frac == 0.9
    cfg = _cfg(y_min_frac=None)
    assert build_eval_transforms(cfg).overlay_mask_y_min_frac == pytest.approx(
        overlay_mask_y_min_frac()
    )
