"""Tests for src/overlay.py: config loading, light detection, masking."""

import numpy as np
import pytest

from src import overlay


@pytest.fixture(scope="module")
def cfg():
    return overlay.load_overlay("fie_default")


def blank(cfg):
    width, height = cfg.frame_size
    return np.zeros((height, width, 3), dtype=np.uint8)


def paint(frame, box):
    y0, y1, x0, x1 = box.region
    frame[y0:y1, x0:x1] = box.template
    return frame


def test_config_geometry_loaded(cfg):
    assert cfg.name == "fie_default"
    assert cfg.frame_size == (640, 360)
    assert cfg.left_on.region == (330, 334, 140, 260)
    assert cfg.right_on.region == (330, 334, 380, 500)
    assert cfg.left_off.max_sad == 7000
    assert cfg.score_left == (309, 325, 265, 285)
    assert cfg.mask_band_y == 330


def test_template_shape_matches_region(cfg):
    for box in (cfg.left_on, cfg.right_on, cfg.left_off, cfg.right_off):
        y0, y1, x0, x1 = box.region
        assert box.template.shape == (y1 - y0, x1 - x0, 3)


def test_unknown_overlay_raises():
    with pytest.raises(FileNotFoundError):
        overlay.load_overlay("does_not_exist")


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (False, False, "No-No"),
        (True, False, "On-No"),
        (False, True, "No-On"),
        (True, True, "On-On"),
    ],
)
def test_on_target_lights(cfg, left, right, expected):
    frame = blank(cfg)
    if left:
        paint(frame, cfg.left_on)
    if right:
        paint(frame, cfg.right_on)
    assert overlay.check_lights(frame, cfg, "sabre") == expected
    assert overlay.any_light(frame, cfg, "sabre") == (left or right)


def test_off_target_only_counts_for_foil(cfg):
    frame = paint(blank(cfg), cfg.left_off)
    assert overlay.check_lights(frame, cfg, "foil") == "Off-No"
    assert overlay.check_lights(frame, cfg, "epee") == "No-No"
    assert overlay.any_light(frame, cfg, "foil")
    assert not overlay.any_light(frame, cfg, "epee")


def test_on_target_wins_over_off_target(cfg):
    frame = paint(paint(blank(cfg), cfg.left_on), cfg.left_off)
    assert overlay.check_lights(frame, cfg, "foil") == "On-No"


def test_off_on_combination(cfg):
    frame = paint(paint(blank(cfg), cfg.left_off), cfg.right_on)
    assert overlay.check_lights(frame, cfg, "foil") == "Off-On"


def test_other_resolution_is_resized(cfg):
    frame = paint(blank(cfg), cfg.left_on)
    import cv2

    big = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_NEAREST)
    assert overlay.check_lights(big, cfg, "sabre") == "On-No"


def test_mask_light_region_blanks_band(cfg):
    frames = np.full((2, 360, 640, 3), 200, dtype=np.uint8)
    masked = overlay.mask_light_region(frames, cfg)
    assert masked is not frames
    assert (masked[:, 330:] == 0).all()
    assert (masked[:, :330] == 200).all()
    assert (frames == 200).all()


def test_mask_band_scales_with_resolution(cfg):
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    masked = overlay.mask_light_region(frame, cfg)
    assert (masked[660:] == 0).all()
    assert (masked[:660] == 200).all()


def test_score_crops_shape(cfg):
    frame = blank(cfg)
    left, right = overlay.score_crops(frame, cfg)
    assert left.shape == (1, 16 * 20 * 3)
    assert right.shape == (1, 16 * 20 * 3)


def test_probe_rejects_bright_lights(cfg):
    frame = blank(cfg)
    frame[300:305, 200:440] = 30
    ok, reason = overlay.probe_frame(frame, cfg)
    assert ok
    frame[330:334, 140:260] = 250
    ok, reason = overlay.probe_frame(frame, cfg)
    assert not ok and "Light regions too bright" in reason


def test_probe_rejects_missing_header(cfg):
    frame = blank(cfg)
    frame[300:305, 200:440] = 200
    ok, reason = overlay.probe_frame(frame, cfg)
    assert not ok and "No dark header" in reason
