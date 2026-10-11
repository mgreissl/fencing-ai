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


def command(tmp_path, captured, **options):
    pp.download_match(URL, "abcdefghijk", str(tmp_path),
                      pp.DownloadOptions(**options))
    return captured[0]["cmd"]


def test_default_uses_no_cookies_and_no_remote_components(tmp_path, captured):
    cmd = command(tmp_path, captured)
    assert "--cookies-from-browser" not in cmd
    assert "--cookies" not in cmd
    assert "--remote-components" not in cmd


@pytest.mark.parametrize("browser", ["firefox", "edge", "chrome"])
def test_cookies_browser_is_configurable(tmp_path, captured, browser):
    cmd = command(tmp_path, captured, cookies_browser=browser)
    assert cmd[cmd.index("--cookies-from-browser") + 1] == browser


def test_cookies_file(tmp_path, captured):
    cmd = command(tmp_path, captured, cookies_file="c.txt")
    assert cmd[cmd.index("--cookies") + 1] == "c.txt"
    assert "--cookies-from-browser" not in cmd


def test_cookies_file_wins_over_browser(tmp_path, captured):
    cmd = command(tmp_path, captured, cookies_file="c.txt",
                  cookies_browser="edge")
    assert "--cookies-from-browser" not in cmd


def test_remote_components_opt_in(tmp_path, captured):
    cmd = command(tmp_path, captured, remote_components=True)
    assert cmd[cmd.index("--remote-components") + 1] == "ejs:github"


def parse(*argv):
    import argparse
    parser = argparse.ArgumentParser()
    pp.add_cli_args(parser)
    return parser.parse_args(argv)


def test_cli_defaults_to_no_cookies():
    opts = pp.download_options_from_args(parse())
    assert opts == pp.DownloadOptions()
    assert opts.cookies_browser == "none"


def test_cli_options():
    opts = pp.download_options_from_args(parse(
        "--cookies-browser", "chrome", "--cookies-file", "c.txt",
        "--remote-components"))
    assert (opts.cookies_browser, opts.cookies_file, opts.remote_components) == (
        "chrome", "c.txt", True)


def test_cli_rejects_unknown_browser():
    with pytest.raises(SystemExit):
        parse("--cookies-browser", "safari")
