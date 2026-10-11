import subprocess
import sys

import pytest

from src import cli


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli, "module_exists", lambda module: True)
    return recorded


@pytest.mark.parametrize(
    "command, module",
    [(name, module) for name, (module, _) in cli.COMMANDS.items()],
)
def test_dispatch_module(calls, command, module):
    assert cli.main([command, "--foo", "bar", "a.b=1"]) == 0
    cmd, kwargs = calls[0]
    assert cmd == [sys.executable, "-m", module, "--foo", "bar", "a.b=1"]
    assert kwargs["cwd"] == cli.REPO_ROOT


def test_hydra_overrides_pass_through(calls):
    cli.main(["train", "data.weapon=foil", "training.epochs=3"])
    assert calls[0][0][-2:] == ["data.weapon=foil", "training.epochs=3"]
    assert calls[0][0][2] == "src.train"


def test_help_flag_reaches_module(calls):
    cli.main(["preprocess", "--help"])
    assert calls[0][0][-1] == "--help"


def test_exit_code_propagates(monkeypatch):
    monkeypatch.setattr(cli, "module_exists", lambda module: True)
    monkeypatch.setattr(
        cli.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3)
    )
    assert cli.main(["bootstrap", "preds.csv"]) == 3


def test_missing_module_reports_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(cli, "module_exists", lambda module: False)
    called = []
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: called.append(a))
    assert cli.main(["ablation"]) == 2
    assert not called
    assert "not available" in capsys.readouterr().err


def test_unknown_command_exits(calls):
    with pytest.raises(SystemExit):
        cli.main(["nope"])

