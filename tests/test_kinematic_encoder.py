import torch

from src.kinematics import KINEMATIC_FEATURE_DIM, KINEMATIC_FEATURE_NAMES
from src.model import KinematicEncoder


def test_encoder_default_width_matches_feature_dim():
    enc = KinematicEncoder()
    out = enc(torch.zeros(2, 16, KINEMATIC_FEATURE_DIM))
    assert out.shape == (2, 128)


def test_encoder_is_batch_independent_in_train_mode():
    torch.manual_seed(0)
    enc = KinematicEncoder().train()
    x = torch.randn(4, 16, KINEMATIC_FEATURE_DIM)
    alone = enc(x[:1])
    in_batch = enc(x)[:1]
    assert torch.allclose(alone, in_batch, atol=1e-5)
    enc.eval()
    assert torch.allclose(enc(x)[:1], alone, atol=1e-5)


def test_batch_size_one_train_mode_works():
    KinematicEncoder().train()(torch.randn(1, 16, KINEMATIC_FEATURE_DIM))


def test_validity_channels_unscaled_and_velocities_scaled():
    enc = KinematicEncoder()
    scale = enc.input_scale
    for i, name in enumerate(KINEMATIC_FEATURE_NAMES):
        if name.endswith("valid"):
            assert scale[i] == 1.0
    assert scale[KINEMATIC_FEATURE_NAMES.index("left_wrist_vx")] < 1.0
