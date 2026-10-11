from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import src.evaluate as evaluate_module
import src.inference as inference_module
import src.train as train_module
from src.inference import amp_enabled, autocast_context
from tests.test_train_selection import CFG, _Constant, _FakeClips


def test_amp_enabled_requires_flag_and_accelerator():
    cuda, mps, cpu = (SimpleNamespace(type=t) for t in ("cuda", "mps", "cpu"))
    assert amp_enabled(True, cuda) and amp_enabled(True, mps)
    assert not amp_enabled(False, cuda) and not amp_enabled(False, mps)
    assert not amp_enabled(True, cpu)


def test_autocast_context_passes_enabled_flag(monkeypatch):
    calls = []

    def spy(device_type, enabled):
        calls.append((device_type, enabled))
        return torch.autocast("cpu", enabled=False)

    monkeypatch.setattr(inference_module.torch.amp, "autocast", spy)
    autocast_context(SimpleNamespace(type="cuda"), use_amp=False)
    autocast_context(SimpleNamespace(type="cuda"), use_amp=True)
    assert calls == [("cuda", False), ("cuda", True)]


def _spy_on(monkeypatch, module, record):
    original = module.autocast_context

    def spy(device, use_amp):
        record.append(use_amp)
        return original(device, use_amp)

    monkeypatch.setattr(module, "autocast_context", spy)


def _loader():
    ds = _FakeClips([0, 1], ["On-On", "On-On"], ["foil", "foil"])
    return DataLoader(ds, batch_size=2, shuffle=False)


def test_validate_respects_mixed_precision_flag(monkeypatch):
    record = []
    _spy_on(monkeypatch, train_module, record)
    for flag in (False, True):
        cfg = OmegaConf.merge(CFG, {"training": {"mixed_precision": flag}})
        train_module._validate(
            _Constant(0), _loader(), nn.CrossEntropyLoss(), torch.device("cpu"),
            0, False, cfg,
        )
    assert record == [False, True]


def test_evaluate_respects_use_amp(monkeypatch):
    record = []
    _spy_on(monkeypatch, evaluate_module, record)
    for flag in (False, True):
        evaluate_module.evaluate(
            _Constant(0), _loader(), torch.device("cpu"), ["Left", "Tie", "Right"],
            use_amp=flag,
        )
    assert record == [False, True]
