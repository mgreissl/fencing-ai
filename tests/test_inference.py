import torch
import torch.nn as nn

from src.inference import flip_logits_lr, predict_with_flip_tta
from src.kinematics import KINEMATIC_FEATURE_DIM, flip_kinematic_features


class _AsymmetricModel(nn.Module):
    """Tiny classifier that is deliberately NOT flip-equivariant."""

    def __init__(self, use_kinematics: bool = True):
        super().__init__()
        torch.manual_seed(0)
        self.use_kinematics = use_kinematics
        self.video = nn.Linear(3 * 4 * 4, 3)
        self.kin = nn.Linear(KINEMATIC_FEATURE_DIM, 3)
        self.weapon = nn.Embedding(3, 3)

    def forward(self, pixel_values, weapon_id=None, kinematics=None):
        b = pixel_values.shape[0]
        out = self.video(pixel_values.mean(dim=1).reshape(b, -1))
        if kinematics is not None:
            out = out + self.kin(kinematics.mean(dim=1))
        if weapon_id is not None:
            out = out + self.weapon(weapon_id)
        return out


def _batch(b=3):
    g = torch.Generator().manual_seed(1)
    pixels = torch.randn(b, 5, 3, 4, 4, generator=g)
    kin = torch.rand(b, 5, KINEMATIC_FEATURE_DIM, generator=g)
    kin[..., [8, 17, 20]] = 1.0  # validity channels (flip is an involution on valid data)
    weapon = torch.tensor([0, 1, 0])[:b]
    return pixels, weapon, kin


def test_flip_logits_swaps_left_and_right_only():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(flip_logits_lr(x), torch.tensor([[3.0, 2.0, 1.0]]))


def test_tta_output_is_exactly_lr_symmetric():
    model = _AsymmetricModel().eval()
    pixels, weapon, kin = _batch()
    with torch.no_grad():
        plain = model(pixels, weapon, kin)
        plain_flipped = model(torch.flip(pixels, dims=[-1]), weapon, flip_kinematic_features(kin))
        assert not torch.allclose(plain, flip_logits_lr(plain_flipped))  # model is asymmetric
        p = torch.softmax(predict_with_flip_tta(model, pixels, weapon, kin), dim=-1)
        p_flip = torch.softmax(
            predict_with_flip_tta(
                model, torch.flip(pixels, dims=[-1]), weapon, flip_kinematic_features(kin)
            ),
            dim=-1,
        )
    assert torch.allclose(p[:, 0], p_flip[:, 2], atol=1e-6)
    assert torch.allclose(p[:, 2], p_flip[:, 0], atol=1e-6)
    assert torch.allclose(p[:, 1], p_flip[:, 1], atol=1e-6)


def test_tta_without_kinematics():
    model = _AsymmetricModel(use_kinematics=False).eval()
    pixels, weapon, _ = _batch()
    with torch.no_grad():
        out = predict_with_flip_tta(model, pixels, weapon, None)
        out_flip = predict_with_flip_tta(model, torch.flip(pixels, dims=[-1]), weapon, None)
    assert torch.allclose(out, flip_logits_lr(out_flip), atol=1e-6)
