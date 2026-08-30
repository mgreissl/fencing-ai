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
        crop_scale=tuple(cfg.augmentation.train.random_resized_crop.scale),
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

    The same random parameters (crop region, flip decision, color jitter params)
    are applied to every frame so that temporal coherence is preserved.

    Horizontal flipping also swaps Left↔Right labels (fencing is symmetric).
    """

    def __init__(
        self,
        frame_size: int = 224,
        crop_scale: tuple = (0.8, 1.0),
        flip_prob: float = 0.5,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        grayscale_prob: float = 0.0,
    ):
        self.frame_size = frame_size
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.grayscale_prob = grayscale_prob

        # Build per-frame spatial transforms (applied with same random state)
        self.resize_crop = T.RandomResizedCrop(
            size=(frame_size, frame_size),
            scale=crop_scale,
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
        self, frames: torch.Tensor, label: int
    ) -> tuple[torch.Tensor, int]:
        """
        Apply augmentations to a clip.

        Args:
            frames: Tensor of shape (T, C, H, W) with values in [0, 1].
            label: Integer class label.

        Returns:
            Tuple of (augmented_frames, possibly_swapped_label).
        """
        T_len = frames.shape[0]

        # --- Random Resized Crop (same crop for all frames) ---
        # Get crop parameters once, apply to every frame
        i, j, h, w = self.resize_crop.get_params(
            frames[0], scale=self.crop_scale, ratio=(0.75, 1.333)
        )
        frames = torch.stack(
            [
                T.functional.resized_crop(
                    f, i, j, h, w, (self.frame_size, self.frame_size), antialias=True
                )
                for f in frames
            ]
        )

        # --- Random Horizontal Flip (with label swap) ---
        if random.random() < self.flip_prob:
            frames = torch.flip(frames, dims=[-1])  # Flip width dimension
            if label == LABEL_LEFT:
                label = LABEL_RIGHT
            elif label == LABEL_RIGHT:
                label = LABEL_LEFT
            # LABEL_TIE stays the same

        # --- Color Jitter (same params for all frames) ---
        if self.color_jitter_transform is not None:
            # Use a fixed seed to ensure the same random jitter across all frames
            seed = random.randint(0, 2**32 - 1)
            jittered = []
            for f in frames:
                torch.manual_seed(seed)
                random.seed(seed)
                jittered.append(self.color_jitter_transform(f))
            frames = torch.stack(jittered)

        # --- Random Grayscale ---
        if self.grayscale_prob > 0 and random.random() < self.grayscale_prob:
            gray = T.Grayscale(num_output_channels=3)
            frames = torch.stack([gray(f) for f in frames])

        # --- Normalize ---
        frames = torch.stack([self.normalize(f) for f in frames])

        return frames, label


class EvalVideoTransform:
    """
    Deterministic evaluation transforms: resize + center crop + normalize.
    No random augmentation — ensures reproducible evaluation metrics.
    """

    def __init__(self, frame_size: int = 224):
        self.frame_size = frame_size
        self.transform = T.Compose(
            [
                T.Resize(
                    size=int(frame_size * 1.143),  # ~256 for 224
                    antialias=True,
                ),
                T.CenterCrop(size=(frame_size, frame_size)),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(
        self, frames: torch.Tensor, label: int
    ) -> tuple[torch.Tensor, int]:
        """
        Apply deterministic transforms to a clip.

        Args:
            frames: Tensor of shape (T, C, H, W) with values in [0, 1].
            label: Integer class label (unchanged).

        Returns:
            Tuple of (transformed_frames, label).
        """
        frames = torch.stack([self.transform(f) for f in frames])
        return frames, label
