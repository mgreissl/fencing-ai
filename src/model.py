import logging
from typing import Optional

import torch
import torch.nn as nn
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
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        if lora_layers is None:
            lora_layers = [8, 9, 10, 11]

        log.info(f"Loading VideoMAE backbone: {backbone_name}")

        # Load the pre-trained model
        pretrained = VideoMAEForVideoClassification.from_pretrained(backbone_name)

        # Extract base VideoMAE encoder
        base_videomae = pretrained.videomae

        # Apply LoRA if requested
        if use_lora:
            from peft import get_peft_model, LoraConfig

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                layers_to_transform=lora_layers,
                lora_dropout=dropout,
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

        self.fc_norm = (
            pretrained.fc_norm
            if hasattr(pretrained, "fc_norm")
            else nn.LayerNorm(base_videomae.config.hidden_size)
        )

        # Get the hidden dimension from the backbone config
        backbone_hidden_size = base_videomae.config.hidden_size
        # Bilateral pooling feature dimension: global pool (768) + spatial difference (768) = 1536
        self.classifier_in_features = 2 * backbone_hidden_size
        log.info(f"Backbone hidden size: {backbone_hidden_size} (Bilateral pooled features: {self.classifier_in_features})")

        # Learned weapon conditioning embedding (0=Foil, 1=Sabre, 2=Epee)
        self.weapon_embed = nn.Embedding(3, self.classifier_in_features)

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

    def _freeze_backbone(self):
        """Freeze all backbone parameters (encoder + fc_norm)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.fc_norm.parameters():
            param.requires_grad = False

    def forward(
        self, pixel_values: torch.Tensor, weapon_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values: Tensor of shape (B, T, C, H, W) — batch of video clips.
            weapon_id: Optional tensor of shape (B,) with weapon integers (0=Foil, 1=Sabre, 2=Epee).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        # VideoMAE expects (B, T, C, H, W) — this is our input format
        outputs = self.backbone(pixel_values=pixel_values)

        # Sequence output: (B, num_patches, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Normalize patch tokens with fc_norm
        sequence_output = self.fc_norm(sequence_output)

        B, N, C = sequence_output.shape
        if N == 1568:
            # 16 frames / 2 = 8 temporal slices; 224/16 = 14 height; 224/16 = 14 width
            grid = sequence_output.view(B, 8, 14, 14, C)
            # Global pool: average across all space and time
            z_global = grid.mean(dim=(1, 2, 3))
            # Left half of screen (W: 0..6) vs Right half of screen (W: 7..13)
            z_left = grid[:, :, :, :7, :].mean(dim=(1, 2, 3))
            z_right = grid[:, :, :, 7:, :].mean(dim=(1, 2, 3))
            z_diff = z_left - z_right  # Anti-symmetric under horizontal flip
            pooled = torch.cat([z_global, z_diff], dim=-1)
        else:
            z_global = sequence_output.mean(dim=1)
            pooled = torch.cat([z_global, torch.zeros_like(z_global)], dim=-1)

        # Add learned weapon conditioning if provided
        if weapon_id is not None:
            pooled = pooled + self.weapon_embed(weapon_id)

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

    model = FencingVideoMAE(
        backbone_name=cfg.model.backbone,
        freeze_backbone=cfg.model.freeze_backbone,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_layers=lora_layers,
        hidden_dim=cfg.model.head.hidden_dim,
        num_classes=cfg.model.head.num_classes,
        dropout=cfg.model.head.dropout,
    )
    return model
