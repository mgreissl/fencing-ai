import random
from typing import Optional

import torch
from torchvision.transforms import v2 as T


# ImageNet normalization (matches VideoMAE pre-training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label indices (must match config.model.class_names order)
LABEL_LEFT = 0
LABEL_TIE = 1
LABEL_RIGHT = 2


def build_train_transforms(cfg) -> "TrainVideoTransform":
    """Build training-time augmentation pipeline from Hydra config."""
    return TrainVideoTransform(
        frame_size=cfg.data.frame_size,
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
    return EvalVideoTransform(frame_size=cfg.data.frame_size)


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
        flip_prob: float = 0.5,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        grayscale_prob: float = 0.0,
    ):
        self.frame_size = frame_size
        self.flip_prob = flip_prob
        self.grayscale_prob = grayscale_prob

        # Direct resize preserves full widescreen piste and both fencers
        self.resize = T.Resize(
            size=(frame_size, frame_size),
            antialias=True,
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
        # --- Direct Resize (preserves both fencers and entire piste) ---
        frames = torch.stack([self.resize(f) for f in frames])

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

    def __init__(self, frame_size: int = 224):
        self.frame_size = frame_size
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
        frames = torch.stack([self.transform(f) for f in frames])
        if kinematics is not None:
            return frames, label, kinematics
        return frames, label
