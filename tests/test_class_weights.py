import pytest
import torch

from src.train import build_class_weights


def test_inverse_freq_balances_classes():
    labels = [0] * 6 + [1] * 2 + [2] * 4
    w = build_class_weights(labels, "inverse_freq")
    # N / (K * n_k) = 12 / (3 * n_k)
    assert w.tolist() == pytest.approx([2 / 3, 2.0, 1.0])
    # every class contributes equally to the weighted loss mass
    counts = torch.tensor([6.0, 2.0, 4.0])
    assert (w * counts).tolist() == pytest.approx([4.0, 4.0, 4.0])


def test_none_returns_no_weights():
    assert build_class_weights([0, 1, 2], "none") is None


def test_absent_class_gets_zero_weight_and_rest_stay_normalised():
    w = build_class_weights([0, 0, 2], "inverse_freq")
    assert w.tolist() == pytest.approx([0.75, 0.0, 1.5])


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="class_weights"):
        build_class_weights([0, 1], "sqrt")


def test_weights_computed_from_given_labels_only():
    train = build_class_weights([0, 0, 0, 1], "inverse_freq")
    other = build_class_weights([2, 2, 2, 1], "inverse_freq")
    assert train.tolist() != other.tolist()

