"""yt-dlp invocation: environment interpreter, cookies, TLS (#72, #73, #74)."""

import contextlib
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
    with contextlib.suppress(pp.DownloadError):
        pp.download_match(URL, "abcdefghijk", str(tmp_path))
    cmd = captured[0]["cmd"]
    assert cmd[0] == sys.executable
    assert "yt-dlp" not in cmd


def command(tmp_path, captured, **options):
    with contextlib.suppress(pp.DownloadError):
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


def test_cli_tls_options():
    assert pp.download_options_from_args(parse()).no_check_certificates is False
    opts = pp.download_options_from_args(parse(
        "--ca-file", "ca.pem", "--no-check-certificates"))
    assert (opts.ca_file, opts.no_check_certificates) == ("ca.pem", True)


def test_cli_rejects_unknown_browser():
    with pytest.raises(SystemExit):
        parse("--cookies-browser", "safari")


# --- TLS: OS trust store, error visibility (#73) ----------------------------

def test_verification_stays_on_by_default(tmp_path, captured):
    assert "--no-check-certificates" not in command(tmp_path, captured)


def test_no_check_certificates_is_explicit_opt_in(tmp_path, captured):
    cmd = command(tmp_path, captured, no_check_certificates=True)
    assert "--no-check-certificates" in cmd


def test_ca_file_is_passed_via_environment(tmp_path, captured):
    command(tmp_path, captured, ca_file="proxy-ca.pem")
    env = captured[0]["env"]
    assert env["FENCING_CA_FILE"] == "proxy-ca.pem"
    assert env["SSL_CERT_FILE"] == "proxy-ca.pem"


def test_environment_untouched_without_ca_file(tmp_path, captured):
    command(tmp_path, captured)
    assert "FENCING_CA_FILE" not in captured[0]["env"]


def test_wrapper_runs_yt_dlp_with_truststore():
    """The wrapper starts for real (no network): imports truststore, yt_dlp."""
    import yt_dlp.version

    cmd = pp.build_download_command(URL, "unused")
    version_cmd = cmd[:cmd.index("--sleep-requests")] + ["--version"]
    result = subprocess.run(version_cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == yt_dlp.version.__version__


def test_failure_raises_with_error_tail(tmp_path, monkeypatch):
    stderr = "\n".join(["noise"] * 20 + [
        "ERROR: [youtube] x: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get "
        "local issuer certificate"])
    monkeypatch.setattr(
        pp.sp, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", stderr))
    with pytest.raises(pp.DownloadError) as excinfo:
        pp.download_match(URL, "abcdefghijk", str(tmp_path))
    message = str(excinfo.value)
    assert "CERTIFICATE_VERIFY_FAILED" in message
    assert "\n" not in message
    assert message.count("noise") <= 4


def test_error_tail_is_bounded():
    tail = pp.error_tail("x" * 5000)
    assert len(tail) <= pp.ERROR_TAIL_CHARS


def test_pipeline_logs_download_error_into_failed_csv(tmp_path, monkeypatch,
                                                       capsys):
    import csv

    (tmp_path / "foil_videos.txt").write_text(URL + "\n")

    def failing_download(*args):
        raise pp.DownloadError("ERROR: unable to get local issuer certificate")

    monkeypatch.setattr(pp, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(pp, "PIPELINE_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(pp, "download_match", failing_download)
    monkeypatch.setattr(pp, "load_processed_urls", lambda weapon, **kwargs: set())
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)
    pp.run_pipeline(weapon="foil")

    assert "local issuer certificate" in capsys.readouterr().out
    with open(tmp_path / "state" / "failed_foil.csv", newline="",
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert "download failed" in rows[0]["reason"]
    assert "local issuer certificate" in rows[0]["reason"]
