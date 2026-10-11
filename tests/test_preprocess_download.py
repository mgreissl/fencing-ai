"""yt-dlp invocation: environment interpreter, cookies, TLS (#72, #73, #74)."""

import subprocess
import sys

import pytest

from scripts import preprocess_pipeline as pp

URL = "https://www.youtube.com/watch?v=abcdefghijk"


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), **kwargs})
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: boom")

    monkeypatch.setattr(pp.sp, "run", fake_run)
    return calls


def test_uses_interpreter_of_active_environment(tmp_path, captured):
    pp.download_match(URL, "abcdefghijk", str(tmp_path))
    cmd = captured[0]["cmd"]
    assert cmd[0] == sys.executable
    assert "yt-dlp" not in cmd
