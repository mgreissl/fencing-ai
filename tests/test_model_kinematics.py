import pytest
import torch
import torch.nn as nn

from src.model import FencingVideoMAE


class _Boom(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("backbone must not run before the kinematics check")


def _stub_model(use_kinematics: bool) -> FencingVideoMAE:
    model = FencingVideoMAE.__new__(FencingVideoMAE)
    nn.Module.__init__(model)
    model.backbone = _Boom()
    model.use_kinematics = use_kinematics
    return model


def test_missing_kinematics_raises():
    model = _stub_model(use_kinematics=True)
    with pytest.raises(ValueError, match="kinematics"):
        model(torch.zeros(1, 2, 3, 4, 4), weapon_id=None, kinematics=None)
