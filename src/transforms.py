import random
from typing import Optional, Sequence

import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2 as T


# ImageNet normalization (matches VideoMAE pre-training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label indices (must match config.model.class_names order)
LABEL_LEFT = 0
LABEL_TIE = 1
LABEL_RIGHT = 2


def mask_overlay(
    frames: torch.Tensor,
    y_min_frac: float,
    fill: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Blank the scoreboard/overlay band at the bottom of every frame.

    Rows from ``int(H * y_min_frac)`` to the bottom are set to ``fill``
    (default: ImageNet mean, i.e. zero after normalisation). Apply it to raw
    [0, 1] frames before resize and flip so no overlay pixels survive either.

    The fraction defaults to the overlay geometry
    (``src.overlay.overlay_mask_y_min_frac``, 295 / 360 for ``fie_default``);
    ``data.overlay_mask.y_min_frac`` overrides it when set.

    Args:
        frames: Tensor (..., C, H, W).
        y_min_frac: First masked row as a fraction of the frame height.
        fill: Per-channel fill values, or a scalar.

    Returns:
        A new tensor; the input is not modified.
    """
    height = frames.shape[-2]
    y_min = min(int(height * y_min_frac), height)
    if fill is None:
        fill = IMAGENET_MEAN
    fill_tensor = torch.as_tensor(fill, dtype=frames.dtype, device=frames.device)
    out = frames.clone()
    if y_min < height:
        out[..., y_min:, :] = fill_tensor.reshape(-1, 1, 1) if fill_tensor.ndim else fill_tensor
    return out


def _overlay_mask_frac(cfg) -> Optional[float]:
    """Mask start fraction if the mask is enabled, else None.

    ``data.overlay_mask.y_min_frac`` wins when set; otherwise it is derived
    from the overlay geometry (``data.overlay_mask.overlay``, default
    ``fie_default``).
    """
    mask = OmegaConf.select(cfg, "data.overlay_mask")
    if mask is None or not mask.get("enabled", False):
        return None
    frac = mask.get("y_min_frac")
    if frac is not None:
        return float(frac)
    from src.overlay import overlay_mask_y_min_frac

    return overlay_mask_y_min_frac(mask.get("overlay"))


def build_train_transforms(cfg) -> "TrainVideoTransform":
    """Build training-time augmentation pipeline from Hydra config."""
    return TrainVideoTransform(
        frame_size=cfg.data.frame_size,
        overlay_mask_y_min_frac=_overlay_mask_frac(cfg),
        random_affine=cfg.augmentation.train.random_affine.enabled,
        affine_translate=tuple(cfg.augmentation.train.random_affine.translate),
        affine_scale=tuple(cfg.augmentation.train.random_affine.scale),
        gaussian_blur=cfg.augmentation.train.gaussian_blur.enabled,
        blur_kernel_size=cfg.augmentation.train.gaussian_blur.kernel_size,
        blur_sigma=tuple(cfg.augmentation.train.gaussian_blur.sigma),
        flip_prob=cfg.augmentation.train.horizontal_flip.prob
        if cfg.augmentation.train.horizontal_flip.enabled
        else 0.0,
        color_jitter=cfg.augmentation.train.color_jitter.enabled,
        brightness=cfg.augmentation.train.color_jitter.brightness,
        contrast=cfg.augmentation.train.color_jitter.contrast,
        saturation=cfg.augmentation.train.color_jitter.saturation,
        hue=cfg.augmentation.train.color_jitter.hue,
        grayscale_prob=cfg.augmentation.train.random_grayscale.prob
        if cfg.augmentation.train.random_grayscale.enabled
        else 0.0,
    )


def build_eval_transforms(cfg) -> "EvalVideoTransform":
    """Build evaluation-time (deterministic) transform pipeline."""
    return EvalVideoTransform(
        frame_size=cfg.data.frame_size,
        overlay_mask_y_min_frac=_overlay_mask_frac(cfg),
    )


class TrainVideoTransform:
    """
    Training augmentations applied consistently across all frames of a clip.

    The same random parameters (flip decision, color jitter params)
    are applied to every frame so that temporal coherence is preserved.

    Horizontal flipping also swaps Left↔Right labels (fencing is symmetric).
    """

    def __init__(
        self,
        frame_size: int = 224,
        overlay_mask_y_min_frac: Optional[float] = None,
        flip_prob: float = 0.5,
        random_affine: bool = False,
        affine_translate: tuple = (0.05, 0.05),
        affine_scale: tuple = (0.95, 1.05),
        gaussian_blur: bool = False,
        blur_kernel_size: int = 3,
        blur_sigma: tuple = (0.1, 1.0),
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        grayscale_prob: float = 0.0,
    ):
        self.frame_size = frame_size
        self.overlay_mask_y_min_frac = overlay_mask_y_min_frac
        self.flip_prob = flip_prob
        self.grayscale_prob = grayscale_prob

        # Direct resize preserves full widescreen piste and both fencers
        self.resize = T.Resize(
            size=(frame_size, frame_size),
            antialias=True,
        )
        # Mild geometric jitter; v2 draws one parameter set per call, so every
        # frame of the (T, C, H, W) clip is transformed identically. Uncovered
        # borders are filled with the ImageNet mean (zero after normalisation).
        self.affine_transform = (
            T.RandomAffine(
                degrees=0,
                translate=tuple(affine_translate),
                scale=tuple(affine_scale),
                fill=IMAGENET_MEAN,
            )
            if random_affine
            else None
        )
        self.blur_transform = (
            T.GaussianBlur(kernel_size=blur_kernel_size, sigma=tuple(blur_sigma))
            if gaussian_blur
            else None
        )
        self.color_jitter_transform = (
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
            if color_jitter
            else None
        )
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self,
        frames: torch.Tensor,
        label: int,
        kinematics: Optional[torch.Tensor] = None,
    ):
        """
        Apply augmentations to a clip.

        Args:
            frames: Tensor of shape (T, C, H, W) with values in [0, 1].
            label: Integer class label.
            kinematics: Optional kinematic tensor (T, 21).

        Returns:
            Tuple of (augmented_frames, possibly_swapped_label) or
            (augmented_frames, possibly_swapped_label, possibly_flipped_kinematics).
        """
        # --- Mask scoreboard overlay (before resize and flip) ---
        if self.overlay_mask_y_min_frac is not None:
            frames = mask_overlay(frames, self.overlay_mask_y_min_frac)

        # --- Direct Resize (preserves both fencers and entire piste) ---
        frames = torch.stack([self.resize(f) for f in frames])

        # --- Mild affine (translate/scale) and blur, same for all frames ---
        if self.affine_transform is not None:
            frames = self.affine_transform(frames)
        if self.blur_transform is not None:
            frames = self.blur_transform(frames)

        # --- Random Horizontal Flip (with label & kinematics swap) ---
        if random.random() < self.flip_prob:
            frames = torch.flip(frames, dims=[-1])  # Flip width dimension
            if label == LABEL_LEFT:
                label = LABEL_RIGHT
            elif label == LABEL_RIGHT:
                label = LABEL_LEFT
            # LABEL_TIE stays the same

            if kinematics is not None:
                from src.kinematics import flip_kinematic_features

                kinematics = flip_kinematic_features(kinematics)

        # --- Color Jitter (v2 samples one parameter set per call, so the whole
        # (T, C, H, W) clip gets identical jitter without touching global RNG) ---
        if self.color_jitter_transform is not None:
            frames = self.color_jitter_transform(frames)

        # --- Random Grayscale ---
        if self.grayscale_prob > 0 and random.random() < self.grayscale_prob:
            gray = T.Grayscale(num_output_channels=3)
            frames = torch.stack([gray(f) for f in frames])

        # --- Normalize ---
        frames = torch.stack([self.normalize(f) for f in frames])

        if kinematics is not None:
            return frames, label, kinematics
        return frames, label


class EvalVideoTransform:
    """
    Deterministic evaluation transforms: direct resize + normalize.
    Preserves full widescreen piste geometry without cropping out fencers.
    """

    def __init__(
        self,
        frame_size: int = 224,
        overlay_mask_y_min_frac: Optional[float] = None,
    ):
        self.frame_size = frame_size
        self.overlay_mask_y_min_frac = overlay_mask_y_min_frac
        self.transform = T.Compose(
            [
                T.Resize(
                    size=(frame_size, frame_size),
                    antialias=True,
                ),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(
        self,
        frames: torch.Tensor,
        label: int,
        kinematics: Optional[torch.Tensor] = None,
    ):
        """
        Apply deterministic transforms to a clip.

        Args:
            frames: Tensor of shape (T, C, H, W) with values in [0, 1].
            label: Integer class label (unchanged).
            kinematics: Optional kinematic tensor (T, 21).

        Returns:
            Tuple of (transformed_frames, label) or (transformed_frames, label, kinematics).
        """
        if self.overlay_mask_y_min_frac is not None:
            frames = mask_overlay(frames, self.overlay_mask_y_min_frac)
        frames = torch.stack([self.transform(f) for f in frames])
        if kinematics is not None:
            return frames, label, kinematics
        return frames, label
