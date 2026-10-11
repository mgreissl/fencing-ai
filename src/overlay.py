"""Broadcast overlay geometry and pixel-level checks (lights, score crops).

The geometry and thresholds live in ``configs/overlay/<name>.yaml`` (one file
per broadcast template). This module is the only place that turns them into
pixel checks; preprocessing, harvesting and the data pipeline (light masking)
all go through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLAY_CONFIG_DIR = REPO_ROOT / "configs" / "overlay"
DEFAULT_OVERLAY = "fie_default"

Region = tuple[int, int, int, int]  # (y0, y1, x0, x1), half-open


@dataclass(frozen=True, eq=False)
class LightBox:
    """One light box: pixel region, lit template and match threshold."""

    region: Region
    template: NDArray[np.uint8]
    max_sad: int


@dataclass(frozen=True, eq=False)
class OverlayConfig:
    """Geometry and thresholds of one broadcast overlay template."""

    name: str
    frame_size: tuple[int, int]  # (width, height)
    left_on: LightBox
    right_on: LightBox
    left_off: LightBox
    right_off: LightBox
    mask_band_y: int
    score_left: Region
    score_right: Region
    header_region: Region
    header_max_mean: float
    unlit_max_mean: float
    raw: dict


def _region(values: Sequence[int]) -> Region:
    y0, y1, x0, x1 = (int(v) for v in values)
    return (y0, y1, x0, x1)


def _light_box(spec: dict) -> LightBox:
    template = cv2.imread(str(REPO_ROOT / spec["template"]))
    if template is None:
        raise FileNotFoundError(f"Light template not found: {spec['template']}")
    return LightBox(
        region=_region(spec["region"]),
        template=template,
        max_sad=int(spec["max_sad"]),
    )


def resolve_overlay_path(name_or_path: str | Path) -> Path:
    """Return the YAML path for an overlay name (``fie_default``) or a path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    return OVERLAY_CONFIG_DIR / f"{name_or_path}.yaml"


@lru_cache(maxsize=None)
def load_overlay(name_or_path: str | Path = DEFAULT_OVERLAY) -> OverlayConfig:
    """Load an overlay config by name or YAML path (cached)."""
    path = resolve_overlay_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Overlay config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    lights = raw["lights"]
    width, height = (int(v) for v in raw["frame_size"])
    probe = raw["probe"]
    return OverlayConfig(
        name=str(raw["name"]),
        frame_size=(width, height),
        left_on=_light_box(lights["left_on"]),
        right_on=_light_box(lights["right_on"]),
        left_off=_light_box(lights["left_off"]),
        right_off=_light_box(lights["right_off"]),
        mask_band_y=int(lights["mask_band_y"]),
        score_left=_region(raw["score"]["left"]),
        score_right=_region(raw["score"]["right"]),
        header_region=_region(probe["header_region"]),
        header_max_mean=float(probe["header_max_mean"]),
        unlit_max_mean=float(probe["unlit_max_mean"]),
        raw=raw,
    )


def normalize_frame(frame: NDArray[np.uint8], cfg: OverlayConfig) -> NDArray[np.uint8]:
    """Resize a BGR frame to the overlay's reference size if needed."""
    width, height = cfg.frame_size
    if frame.shape[:2] != (height, width):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def _crop(frame: NDArray[np.uint8], region: Region) -> NDArray[np.uint8]:
    y0, y1, x0, x1 = region
    return frame[y0:y1, x0:x1]


def _is_lit(frame: NDArray[np.uint8], box: LightBox) -> bool:
    diff = np.abs(_crop(frame, box.region).astype(int) - box.template.astype(int))
    return bool(diff.sum() <= box.max_sad)


def light_states(
    frame: NDArray[np.uint8], cfg: OverlayConfig, weapon: str = "foil"
) -> tuple[str, str]:
    """Return (left, right) light state, each ``On``, ``Off`` or ``No``.

    ``On`` is the coloured on-target light, ``Off`` the white off-target light
    (foil only). On-target takes precedence over off-target.
    """
    frame = normalize_frame(frame, cfg)

    def side(on_box: LightBox, off_box: LightBox) -> str:
        if _is_lit(frame, on_box):
            return "On"
        if weapon == "foil" and _is_lit(frame, off_box):
            return "Off"
        return "No"

    return (
        side(cfg.left_on, cfg.left_off),
        side(cfg.right_on, cfg.right_off),
    )


def check_lights(
    frame: NDArray[np.uint8], cfg: OverlayConfig, weapon: str = "foil"
) -> str:
    """Return the light pattern of a frame, e.g. ``On-On`` or ``Off-No``."""
    left, right = light_states(frame, cfg, weapon)
    return f"{left}-{right}"


def any_light(
    frame: NDArray[np.uint8], cfg: OverlayConfig, weapon: str = "foil"
) -> bool:
    """True if any scoring light (on- or off-target) is lit."""
    return check_lights(frame, cfg, weapon) != "No-No"


def score_crops(
    frame: NDArray[np.uint8], cfg: OverlayConfig
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Flattened (1, n) left/right scoreboard digit crops for the classifier."""
    frame = normalize_frame(frame, cfg)
    left = _crop(frame, cfg.score_left).reshape(1, -1)
    right = _crop(frame, cfg.score_right).reshape(1, -1)
    return left, right


def light_mask_band(cfg: OverlayConfig, frame_height: int | None = None) -> int:
    """First pixel row (inclusive) of the light-box band for a frame height."""
    reference_height = cfg.frame_size[1]
    if frame_height is None or frame_height == reference_height:
        return cfg.mask_band_y
    return int(round(cfg.mask_band_y * frame_height / reference_height))


def mask_light_region(
    frames: NDArray[np.uint8], cfg: OverlayConfig, fill: int = 0
) -> NDArray[np.uint8]:
    """Return a copy with all rows ``y >= mask_band_y`` set to ``fill``.

    Accepts (H, W, C) or (..., H, W, C) arrays of any frame size; the band is
    scaled from the reference frame height.
    """
    masked = np.array(frames, copy=True)
    band = light_mask_band(cfg, masked.shape[-3])
    masked[..., band:, :, :] = fill
    return masked


def probe_frame(frame: NDArray[np.uint8], cfg: OverlayConfig) -> tuple[bool, str]:
    """Check that a frame shows the expected overlay (dark header, unlit lights)."""
    frame = normalize_frame(frame, cfg)
    header_mean = float(np.mean(_crop(frame, cfg.header_region)))
    if header_mean > cfg.header_max_mean:
        return False, f"No dark header (mean={header_mean:.1f} > {cfg.header_max_mean:g})"

    light_l = float(np.mean(_crop(frame, cfg.left_on.region)))
    light_r = float(np.mean(_crop(frame, cfg.right_on.region)))
    if light_l > cfg.unlit_max_mean or light_r > cfg.unlit_max_mean:
        return False, f"Light regions too bright ({light_l:.1f}, {light_r:.1f})"

    return True, f"header={header_mean:.1f}"
