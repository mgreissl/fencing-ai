"""Broadcast overlay geometry and pixel-level checks (lights, score crops).

The geometry and thresholds live in ``configs/overlay/<name>.yaml`` (one file
per broadcast template). This module is the only place that turns them into
pixel checks; preprocessing, harvesting and the data pipeline (light masking)
all go through it.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Sequence

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLAY_CONFIG_DIR = REPO_ROOT / "configs" / "overlay"
DEFAULT_OVERLAY = "fie_default"

logger = logging.getLogger(__name__)

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
    min_score_confidence: float
    classifier_npz: Path
    classifier_pickle: Path
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
        min_score_confidence=float(raw["score"]["min_confidence"]),
        classifier_npz=REPO_ROOT / raw["classifier"]["npz"],
        classifier_pickle=REPO_ROOT / raw["classifier"]["pickle"],
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


class DigitClassifier:
    """Multinomial logistic regression over flattened scoreboard digit crops.

    Holds plain numpy coefficients, so it does not depend on the scikit-learn
    version that trained the model.
    """

    def __init__(
        self,
        coef: NDArray[np.float64],
        intercept: NDArray[np.float64],
        classes: NDArray[np.int64],
    ) -> None:
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.classes = np.asarray(classes)

    def predict_proba(self, x: NDArray) -> NDArray[np.float64]:
        """Class probabilities, shape (n_samples, n_classes)."""
        logits = np.asarray(x, dtype=np.float64) @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)  # stable softmax
        expo = np.exp(logits)
        return expo / expo.sum(axis=1, keepdims=True)

    def predict(self, x: NDArray) -> NDArray:
        """Most likely class per sample."""
        return self.classes[self.predict_proba(x).argmax(axis=1)]

    def save_npz(self, path: str | Path) -> None:
        np.savez(path, coef=self.coef, intercept=self.intercept, classes=self.classes)

    @classmethod
    def load_npz(cls, path: str | Path) -> "DigitClassifier":
        with np.load(path) as data:
            return cls(data["coef"], data["intercept"], data["classes"])


def load_pickled_classifier(path: str | Path) -> DigitClassifier:
    """Load the legacy scikit-learn pickle and copy out its coefficients.

    Isolated here because pickles are tied to the scikit-learn version. The
    pickled model lacks ``classes_``; the labels are the digits 0-15.
    """
    with open(path, "rb") as fh:
        model = pickle.load(fh)  # noqa: S301 - trusted repo resource
    n_classes = model.coef_.shape[0]
    return DigitClassifier(model.coef_, model.intercept_, np.arange(n_classes))


def export_classifier_npz(pickle_path: str | Path, npz_path: str | Path) -> None:
    """Convert the legacy pickle into the version-independent npz format."""
    load_pickled_classifier(pickle_path).save_npz(npz_path)


def load_digit_classifier(
    npz_path: str | Path | None = None,
    pickle_path: str | Path | None = None,
    cfg: OverlayConfig | None = None,
) -> DigitClassifier:
    """Load the digit classifier from npz, falling back to the pickle."""
    cfg = cfg or load_overlay(DEFAULT_OVERLAY)
    npz_path = Path(npz_path or cfg.classifier_npz)
    pickle_path = Path(pickle_path or cfg.classifier_pickle)
    if npz_path.exists():
        return DigitClassifier.load_npz(npz_path)
    if pickle_path.exists():
        logger.warning("%s missing, falling back to pickle %s", npz_path, pickle_path)
        return load_pickled_classifier(pickle_path)
    raise FileNotFoundError(f"No digit classifier found ({npz_path}, {pickle_path})")


class ScoreRead(NamedTuple):
    """A confident reading of the scoreboard."""

    left: int
    right: int
    confidence: float  # lower of the two digit probabilities


def read_score(
    frame: NDArray[np.uint8],
    cfg: OverlayConfig,
    classifier: DigitClassifier,
    min_confidence: float | None = None,
) -> ScoreRead | None:
    """Read both scores; None if either digit is below the confidence threshold."""
    threshold = cfg.min_score_confidence if min_confidence is None else min_confidence
    left_crop, right_crop = score_crops(frame, cfg)
    proba = classifier.predict_proba(np.vstack([left_crop, right_crop]))
    best = proba.argmax(axis=1)
    confidence = float(proba[np.arange(2), best].min())
    if confidence < threshold:
        return None
    classes = getattr(classifier, "classes", np.arange(proba.shape[1]))
    return ScoreRead(int(classes[best[0]]), int(classes[best[1]]), confidence)


SCORE_MASK_MARGIN_PX = 14


def overlay_mask_y_min_frac(
    cfg: OverlayConfig | str | Path | None = None,
    margin_px: int = SCORE_MASK_MARGIN_PX,
) -> float:
    """First row to blank as a fraction of frame height, from overlay geometry.

    The top of the score digit region minus ``margin_px``, divided by the
    reference frame height (309 - 14 = 295 of 360 rows = 0.8194 for
    ``fie_default``). Everything below it (scoreboard and light boxes) is masked.
    """
    if cfg is None or isinstance(cfg, (str, Path)):
        cfg = load_overlay(DEFAULT_OVERLAY if cfg is None else cfg)
    score_top = min(cfg.score_left[0], cfg.score_right[0])
    y_min = min(score_top - margin_px, cfg.mask_band_y)
    return y_min / cfg.frame_size[1]
