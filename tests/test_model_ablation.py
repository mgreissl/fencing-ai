import pytest
import torch
from omegaconf import OmegaConf
from transformers import VideoMAEConfig

from src.kinematics import KINEMATIC_FEATURE_DIM
from src.model import FencingVideoMAE, bilateral_pool, build_model

TINY_BACKBONE = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "image_size": 32,
    "patch_size": 16,
    "tubelet_size": 2,
    "num_frames": 4,
}
N_FRAMES = 4
SIZE = 32


def _tiny(**kwargs) -> FencingVideoMAE:
    defaults = dict(
        backbone_config=TINY_BACKBONE,
        use_lora=False,
        freeze_backbone=False,
        hidden_dim=16,
        use_kinematics=False,
    )
    defaults.update(kwargs)
    return FencingVideoMAE(**defaults)


def _inputs(batch=2):
    return (
        torch.rand(batch, N_FRAMES, 3, SIZE, SIZE),
        torch.zeros(batch, dtype=torch.long),
        torch.rand(batch, N_FRAMES, KINEMATIC_FEATURE_DIM),
    )


@pytest.mark.parametrize(
    "image_size, patch_size, num_frames, tubelet",
    [(224, 16, 16, 2), (64, 16, 8, 2), (32, 16, 4, 2), (48, 8, 6, 3)],
)
def test_bilateral_pool_derives_grid_from_config(
    image_size, patch_size, num_frames, tubelet
):
    side = image_size // patch_size
    grid_t = num_frames // tubelet
    config = VideoMAEConfig(image_size=image_size, patch_size=patch_size)
    tokens = torch.zeros(2, grid_t * side * side, 3)
    grid = tokens.view(2, grid_t, side, side, 3)
    grid[:, :, :, : side // 2, 0] = 2.0  # left half
    grid[:, :, :, side // 2 :, 0] = 1.0  # right half
    pooled = bilateral_pool(tokens, config)
    assert pooled.shape == (2, 6)
    left, right = 2.0, 1.0
    n_left = side // 2
    expected_global = (left * n_left + right * (side - n_left)) / side
    assert torch.allclose(pooled[:, 0], torch.tensor(expected_global))
    assert torch.allclose(pooled[:, 3], torch.tensor(left - right))


def test_tiny_random_backbone_forward_video_only():
    model = _tiny()
    pixel_values, weapon_id, _ = _inputs()
    assert model.video_feature_dim == 64
    logits = model(pixel_values, weapon_id=weapon_id)
    assert logits.shape == (2, 3)


def test_kinematics_only_has_no_backbone():
    model = _tiny(use_video=False, use_kinematics=True, backbone_config=None)
    assert not hasattr(model, "backbone")
    assert not hasattr(model, "fc_norm")
    pixel_values, weapon_id, kin = _inputs()
    logits = model(pixel_values, weapon_id=weapon_id, kinematics=kin)
    assert logits.shape == (2, 3)
    assert model.classifier_in_features == model.kinematics_dim


def test_kinematics_only_ignores_pixels():
    model = _tiny(use_video=False, use_kinematics=True, backbone_config=None).eval()
    pixel_values, weapon_id, kin = _inputs()
    a = model(pixel_values, weapon_id=weapon_id, kinematics=kin)
    b = model(torch.zeros_like(pixel_values), weapon_id=weapon_id, kinematics=kin)
    assert torch.equal(a, b)


def test_video_plus_kinematics_concatenates():
    model = _tiny(use_kinematics=True, kinematics_out_dim=16)
    assert model.classifier_in_features == 64 + 16
    pixel_values, weapon_id, kin = _inputs()
    assert model(pixel_values, weapon_id=weapon_id, kinematics=kin).shape == (2, 3)


def test_no_stream_is_rejected():
    with pytest.raises(ValueError, match="use_video"):
        _tiny(use_video=False, use_kinematics=False)


def test_build_model_reads_use_video_and_backbone_config():
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "unused",
                "freeze_backbone": False,
                "use_lora": False,
                "use_video": True,
                "use_kinematics": False,
                "backbone_config": TINY_BACKBONE,
                "head": {"hidden_dim": 8, "num_classes": 3, "dropout": 0.1},
            }
        }
    )
    model = build_model(cfg)
    assert model.use_video and model.backbone_config.hidden_size == 32
    cfg.model.use_video = False
    cfg.model.use_kinematics = True
    assert not build_model(cfg).use_video


@pytest.mark.parametrize("name", ["video_only", "kinematics_only", "video_kinematics"])
def test_experiment_configs(name):
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "configs" / "experiment" / f"{name}.yaml"
    exp = OmegaConf.load(path)
    assert exp.model.use_video == (name != "kinematics_only")
    assert exp.model.use_kinematics == (name != "video_only")


def test_lora_dropout_is_separate_from_head_dropout():
    from peft.tuners.lora import LoraLayer

    model = _tiny(use_lora=True, lora_layers=[0, 1], dropout=0.3, lora_dropout=0.1)
    probs = {
        layer.lora_dropout["default"].p
        for layer in model.backbone.modules()
        if isinstance(layer, LoraLayer)
    }
    assert probs == {0.1}


def test_build_model_reads_lora_dropout_default():
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "unused",
                "freeze_backbone": False,
                "use_lora": True,
                "use_kinematics": False,
                "backbone_config": TINY_BACKBONE,
                "lora": {"r": 2, "alpha": 4, "layers": [0, 1]},
                "head": {"hidden_dim": 8, "num_classes": 3, "dropout": 0.3},
            }
        }
    )
    from peft.tuners.lora import LoraLayer

    model = build_model(cfg)
    probs = {
        layer.lora_dropout["default"].p
        for layer in model.backbone.modules()
        if isinstance(layer, LoraLayer)
    }
    assert probs == {0.1}
    cfg.model.lora.dropout = 0.05
    model = build_model(cfg)
    assert {
        layer.lora_dropout["default"].p
        for layer in model.backbone.modules()
        if isinstance(layer, LoraLayer)
    } == {0.05}
def test_bilateral_pool_raises_on_mismatched_token_count_by_default():
    config = VideoMAEConfig(image_size=32, patch_size=16)
    with pytest.raises(ValueError, match="grid"):
        bilateral_pool(torch.rand(1, 7, 4), config)


def test_model_allow_pool_fallback_flag():
    model = _tiny(allow_pool_fallback=True)
    assert model.allow_pool_fallback is True
    assert _tiny().allow_pool_fallback is False
    # one frame less than the config expects: 1 temporal slot -> still fits;
    # use a token count that cannot factor into the grid instead
    config = VideoMAEConfig(image_size=32, patch_size=16)
    pooled = bilateral_pool(torch.rand(1, 7, 4), config, allow_fallback=True)
    assert torch.equal(pooled[:, 4:], torch.zeros(1, 4))
