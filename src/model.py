import logging
from typing import Optional

import torch
import torch.nn as nn
from src.kinematics import KINEMATIC_FEATURE_DIM, KINEMATIC_FEATURE_NAMES
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

log = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Select the best available device.

    Args:
        preference: "auto" | "mps" | "cuda" | "cpu"

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    return torch.device(preference)


class FencingClassificationHead(nn.Module):
    """
    Two-layer MLP classification head with LayerNorm, GELU, and Dropout.

    Architecture:
        LayerNorm → Dropout → Linear(in, hidden) → GELU → Dropout → Linear(hidden, num_classes)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# Fixed input scales that bring the kinematic features to roughly unit range.
# Positions, extension and the validity channels are already O(1) and stay
# unscaled; velocities are in image units per second and log-accelerations are
# O(1-5). The values are order-of-magnitude constants, not fitted statistics.
_VELOCITY_SCALE = 0.5  # 1 / (typical |v| of ~2 image units per second)
_ACCELERATION_SCALE = 0.2  # 1 / (typical log(1 + |a|) of ~5)


def _kinematic_input_scale(in_features: int) -> torch.Tensor:
    """Per-feature scale vector; ones when the layout is not the standard one."""
    scale = torch.ones(in_features)
    if in_features != KINEMATIC_FEATURE_DIM:
        return scale
    for i, name in enumerate(KINEMATIC_FEATURE_NAMES):
        if name.endswith(("_vx", "_vy")) or name == "closure":
            scale[i] = _VELOCITY_SCALE
        elif name.endswith("_acc"):
            scale[i] = _ACCELERATION_SCALE
    return scale


class KinematicEncoder(nn.Module):
    """
    Temporal 1D-CNN encoder for fencer kinematics of shape (B, T, D_kin).

    Features are scaled by fixed per-feature constants (validity channels stay
    unscaled), then passed through two Conv1d blocks with GroupNorm(1, C),
    which normalises per sample and so behaves identically for any batch size
    and in train/eval mode.
    """

    def __init__(
        self,
        in_features: int = KINEMATIC_FEATURE_DIM,
        hidden_dim: int = 64,
        out_dim: int = 128,
    ):
        super().__init__()
        self.register_buffer("input_scale", _kinematic_input_scale(in_features))
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_kin) -> scale, then transpose to (B, D_kin, T) for Conv1D
        x = (x * self.input_scale).transpose(1, 2)
        feat = self.conv(x).squeeze(-1)  # (B, out_dim)
        return self.proj(feat)


def bilateral_pool(
    sequence_output: torch.Tensor, config, allow_fallback: bool = False
) -> torch.Tensor:
    """
    Pool VideoMAE tokens into [global mean, left-minus-right mean].

    The token grid (time, height, width) follows from the backbone config:
    height = width = image_size // patch_size; time = N // (height * width),
    which equals num_frames // tubelet_size for the configured clip length.
    The left half is the first width // 2 columns, the right half the rest.
    If N does not factor into that grid, a ValueError is raised; with
    allow_fallback=True only the global mean is used (with a warning) and the
    difference half is zero.

    Args:
        sequence_output: (B, N, C) token features.
        config: VideoMAEConfig (image_size, patch_size).
        allow_fallback: degrade to global mean + zeros instead of raising.

    Returns:
        (B, 2 * C) pooled features.
    """
    B, N, C = sequence_output.shape
    side = int(config.image_size) // int(config.patch_size)
    grid_t, rest = divmod(N, side * side)
    z_global = sequence_output.mean(dim=1)
    if rest != 0 or grid_t == 0 or side < 2:
        msg = f"Token count {N} does not fit a {side}x{side} grid"
        if not allow_fallback:
            raise ValueError(
                f"{msg}; set model.allow_pool_fallback=true to fall back to "
                "global pooling without the left-right difference"
            )
        log.warning(f"{msg}; no bilateral pooling")
        return torch.cat([z_global, torch.zeros_like(z_global)], dim=-1)
    grid = sequence_output.reshape(B, grid_t, side, side, C)
    half = side // 2
    z_left = grid[:, :, :, :half, :].mean(dim=(1, 2, 3))
    z_right = grid[:, :, :, half:, :].mean(dim=(1, 2, 3))
    return torch.cat([z_global, z_left - z_right], dim=-1)


class FencingVideoMAE(nn.Module):
    """
    VideoMAE backbone + custom classification head for fencing referee AI.

    The backbone is loaded from a HuggingFace checkpoint (pre-trained on
    Kinetics-400). Only the classification head is trainable by default;
    the backbone can optionally be unfrozen for full fine-tuning.
    """

    def __init__(
        self,
        backbone_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        freeze_backbone: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_layers: list = None,
        lora_dropout: float = 0.1,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
        use_kinematics: bool = False,
        kinematics_in_features: int = KINEMATIC_FEATURE_DIM,
        kinematics_hidden_dim: int = 64,
        kinematics_out_dim: int = 128,
        use_video: bool = True,
        backbone_config: Optional[dict] = None,
        allow_pool_fallback: bool = False,
    ):
        super().__init__()
        self.allow_pool_fallback = allow_pool_fallback

        if lora_layers is None:
            lora_layers = [8, 9, 10, 11]

        self.use_video = use_video
        self.use_kinematics = use_kinematics
        if not use_video and not use_kinematics:
            raise ValueError("use_video=False requires use_kinematics=True")

        if use_video:
            self._build_video_stream(
                backbone_name,
                backbone_config,
                freeze_backbone,
                use_lora,
                lora_r,
                lora_alpha,
                lora_layers,
                lora_dropout,
            )
        else:
            log.info("Video stream disabled: kinematics-only classifier")
            self.video_feature_dim = 0

        # Kinematic stream setup
        if self.use_kinematics:
            self.kinematics_dim = kinematics_out_dim
            self.kinematic_encoder = KinematicEncoder(
                in_features=kinematics_in_features,
                hidden_dim=kinematics_hidden_dim,
                out_dim=kinematics_out_dim,
            )
            self.classifier_in_features = self.video_feature_dim + self.kinematics_dim
            log.info(f"Kinematic encoder enabled (in={kinematics_in_features}, out={kinematics_out_dim})")
        else:
            self.classifier_in_features = self.video_feature_dim

        # Learned weapon conditioning embedding (0=Foil, 1=Sabre, 2=Epee),
        # added to the video features, or to the kinematic features without video
        weapon_dim = self.video_feature_dim if use_video else self.kinematics_dim
        self.weapon_embed = nn.Embedding(3, weapon_dim)

        # Custom classification head
        self.classifier = FencingClassificationHead(
            in_features=self.classifier_in_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"Parameters: {total:,} total, {trainable:,} trainable "
            f"({100 * trainable / total:.2f}%)"
        )

    def _build_video_stream(
        self,
        backbone_name,
        backbone_config,
        freeze_backbone,
        use_lora,
        lora_r,
        lora_alpha,
        lora_layers,
        lora_dropout,
    ):
        if backbone_config:
            # Randomly initialised backbone from an explicit config (no download)
            log.info(f"Building random VideoMAE from backbone_config: {dict(backbone_config)}")
            pretrained = VideoMAEForVideoClassification(
                VideoMAEConfig(**dict(backbone_config))
            )
        else:
            log.info(f"Loading VideoMAE backbone: {backbone_name}")
            pretrained = VideoMAEForVideoClassification.from_pretrained(backbone_name)

        # Extract base VideoMAE encoder and fc_norm
        base_videomae = pretrained.videomae
        self.fc_norm = (
            pretrained.fc_norm
            if getattr(pretrained, "fc_norm", None) is not None
            else nn.LayerNorm(base_videomae.config.hidden_size)
        )

        # Apply LoRA if requested
        if use_lora:
            from peft import get_peft_model, LoraConfig

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                layers_to_transform=lora_layers,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone = get_peft_model(base_videomae, lora_config)
            log.info(
                f"LoRA enabled on layers {lora_layers} (r={lora_r}, alpha={lora_alpha})"
            )
        else:
            self.backbone = base_videomae
            if freeze_backbone:
                self._freeze_backbone()
                log.info("Backbone frozen — only classification head will be trained")
            else:
                log.info("Full model is trainable (backbone NOT frozen)")

        # Get the hidden dimension from the backbone config
        backbone_hidden_size = base_videomae.config.hidden_size
        self.video_feature_dim = 2 * backbone_hidden_size
        self.backbone_config = base_videomae.config
        log.info(f"Backbone hidden size: {backbone_hidden_size}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters (encoder + fc_norm)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fc_norm.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        weapon_id: Optional[torch.Tensor] = None,
        kinematics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values: Tensor of shape (B, T, C, H, W) — batch of video clips.
            weapon_id: Optional tensor of shape (B,) with weapon integers (0=Foil, 1=Sabre, 2=Epee).
            kinematics: Tensor of shape (B, T, D_kin) with kinematic time-series;
                required when the model was built with use_kinematics=True.

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        if self.use_kinematics and kinematics is None:
            raise ValueError(
                "Model was built with use_kinematics=True but no kinematics were "
                "passed; check that the dataset yields a 'kinematics' tensor."
            )

        pooled = None
        if self.use_video:
            # VideoMAE expects (B, T, C, H, W); sequence output (B, N, hidden)
            outputs = self.backbone(pixel_values=pixel_values)
            sequence_output = self.fc_norm(outputs.last_hidden_state)
            pooled = bilateral_pool(
                sequence_output, self.backbone_config, self.allow_pool_fallback
            )

        # Add learned weapon conditioning if provided
        if pooled is not None and weapon_id is not None:
            pooled = pooled + self.weapon_embed(weapon_id)

        # Fuse kinematic features if enabled
        if self.use_kinematics:
            kin_feat = self.kinematic_encoder(kinematics)
            if pooled is None:  # kinematics-only
                pooled = kin_feat
                if weapon_id is not None:
                    pooled = pooled + self.weapon_embed(weapon_id)
            else:
                pooled = torch.cat([pooled, kin_feat], dim=-1)

        # Classification logits
        logits = self.classifier(pooled)
        return logits


def build_model(cfg) -> FencingVideoMAE:
    """
    Build the fencing model from Hydra config.

    Args:
        cfg: OmegaConf config object.

    Returns:
        FencingVideoMAE model instance.
    """
    use_lora = getattr(cfg.model, "use_lora", False)
    lora_r = getattr(cfg.model.lora, "r", 8) if hasattr(cfg.model, "lora") else 8
    lora_alpha = getattr(cfg.model.lora, "alpha", 16) if hasattr(cfg.model, "lora") else 16
    lora_layers = list(getattr(cfg.model.lora, "layers", [8, 9, 10, 11])) if hasattr(cfg.model, "lora") else [8, 9, 10, 11]
    lora_dropout = getattr(cfg.model.lora, "dropout", 0.1) if hasattr(cfg.model, "lora") else 0.1

    use_kinematics = getattr(cfg.model, "use_kinematics", False)
    kin_in = getattr(cfg.model.kinematics, "feature_dim", KINEMATIC_FEATURE_DIM) if hasattr(cfg.model, "kinematics") else KINEMATIC_FEATURE_DIM
    kin_hidden = getattr(cfg.model.kinematics, "hidden_dim", 64) if hasattr(cfg.model, "kinematics") else 64
    kin_out = getattr(cfg.model.kinematics, "out_dim", 128) if hasattr(cfg.model, "kinematics") else 128

    backbone_config = getattr(cfg.model, "backbone_config", None)
    if backbone_config is not None:
        from omegaconf import OmegaConf

        backbone_config = OmegaConf.to_container(backbone_config, resolve=True)

    model = FencingVideoMAE(
        backbone_name=cfg.model.backbone,
        freeze_backbone=cfg.model.freeze_backbone,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
        lora_dropout=lora_dropout,
        hidden_dim=cfg.model.head.hidden_dim,
        num_classes=cfg.model.head.num_classes,
        dropout=cfg.model.head.dropout,
        use_kinematics=use_kinematics,
        kinematics_in_features=kin_in,
        kinematics_hidden_dim=kin_hidden,
        kinematics_out_dim=kin_out,
        use_video=bool(getattr(cfg.model, "use_video", True)),
        backbone_config=backbone_config,
        allow_pool_fallback=bool(getattr(cfg.model, "allow_pool_fallback", False)),
    )
    return model

