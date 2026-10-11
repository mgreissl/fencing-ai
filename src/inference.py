from typing import Optional

import torch
import torch.nn as nn

from src.kinematics import flip_kinematic_features


def flip_logits_lr(logits: torch.Tensor) -> torch.Tensor:
    """Swap the Left and Right class entries (order L, T, R) of (..., 3) logits."""
    return logits[..., [2, 1, 0]]


def predict_with_flip_tta(
    model: nn.Module,
    pixel_values: torch.Tensor,
    weapon_id: Optional[torch.Tensor] = None,
    kinematics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flip test-time averaging: mean of f(x) and the L/R-swapped f(flip(x)).

    Mirroring a clip swaps which fencer is on which side, so a correct
    prediction for flip(x) is the prediction for x with Left and Right
    exchanged. Averaging both makes the output exactly L/R-symmetric:
    avg(flip(x)) == swap_LR(avg(x)). Logits are averaged, so CrossEntropy and
    softmax work on the result as usual.

    Args:
        model: Callable ``model(pixel_values, weapon_id=..., kinematics=...)``
            returning (B, 3) logits in the order Left, Tie, Right.
        pixel_values: (B, T, C, H, W) clips.
        weapon_id: Optional (B,) weapon ids.
        kinematics: Optional (B, T, D_kin) features; flipped consistently.

    Returns:
        Averaged logits of shape (B, 3).
    """
    logits = model(pixel_values, weapon_id=weapon_id, kinematics=kinematics)
    flipped_kin = None if kinematics is None else flip_kinematic_features(kinematics)
    flipped_logits = model(
        torch.flip(pixel_values, dims=[-1]), weapon_id=weapon_id, kinematics=flipped_kin
    )
    return 0.5 * (logits + flip_logits_lr(flipped_logits))
