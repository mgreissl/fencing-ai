"""Failed matches are not marked processed and are logged for retry (#49)."""

import csv

from scripts import preprocess_pipeline as pp

OK_URL = "https://www.youtube.com/watch?v=okokokokok1"
BAD_DL_URL = "https://www.youtube.com/watch?v=baddownload"
BAD_PROC_URL = "https://www.youtube.com/watch?v=badprocess1"


def run(tmp_path, monkeypatch):
    (tmp_path / "foil_videos.txt").write_text(
        "\n".join([OK_URL, BAD_DL_URL, BAD_PROC_URL]) + "\n"
    )
    state = tmp_path / "state"
    marked = []

    def fake_download(url, match_id, precut_dir):
        if url == BAD_DL_URL:
            return None, ""
        path = tmp_path / f"{match_id}.mp4"
        path.write_bytes(b"")
        return str(path), "title"

    def fake_process(fpath, clips_dir, **kwargs):
        if kwargs["source_url"] == BAD_PROC_URL:
            raise RuntimeError("decode exploded")
        return 2

    monkeypatch.setattr(pp, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(pp, "PIPELINE_STATE_DIR", str(state))
    monkeypatch.setattr(pp, "download_match", fake_download)
    monkeypatch.setattr(pp, "process_video_file", fake_process)
    monkeypatch.setattr(pp, "load_processed_urls", lambda weapon: set())
    monkeypatch.setattr(pp, "mark_processed", lambda weapon, url: marked.append(url))
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)
    pp.run_pipeline(weapon="foil")
    return marked, state


def test_only_successful_urls_marked_processed(tmp_path, monkeypatch):
    marked, _ = run(tmp_path, monkeypatch)
    assert marked == [OK_URL]


def test_failures_logged_with_reason(tmp_path, monkeypatch):
    _, state = run(tmp_path, monkeypatch)
    with open(state / "failed_foil.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["url"] for r in rows] == [BAD_DL_URL, BAD_PROC_URL]
    assert all(r["weapon"] == "foil" and r["timestamp"] for r in rows)
    assert "download" in rows[0]["reason"]
    assert "RuntimeError" in rows[1]["reason"] and "decode exploded" in rows[1]["reason"]
