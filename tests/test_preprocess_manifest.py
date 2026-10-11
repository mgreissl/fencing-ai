"""Tests for the clip manifest, match ids and pipeline state files (#13)."""

import csv
import os

import pytest

from preprocess_helpers import decode_frame_index, write_synthetic_match
from scripts import preprocess_pipeline as pp

EXPECTED_COLUMNS = [
    "clip_path", "label", "weapon", "match_id", "source_url", "tournament",
    "hit_type", "score_left_before", "score_right_before", "score_left_after",
    "score_right_after", "light_time_s", "source_fps", "clip_fps", "n_frames",
    "light_frame_in_clip", "split",
]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_manifest_columns_are_the_shared_contract():
    assert pp.MANIFEST_COLUMNS == EXPECTED_COLUMNS


@pytest.mark.parametrize("url, expected", [
    ("https://youtube.com/watch?v=-6RjMy8ljN4", "-6RjMy8ljN4"),
    ("https://www.youtube.com/watch?v=wm_gzZNyZxs&t=120s", "wm_gzZNyZxs"),
    ("https://www.youtube.com/watch?list=PLxx&v=vfuYagEG1II", "vfuYagEG1II"),
    ("https://youtu.be/wNERxVECnH4?t=5", "wNERxVECnH4"),
    ("https://www.youtube.com/shorts/8wkZFc4HQpc", "8wkZFc4HQpc"),
])
def test_parse_match_id(url, expected):
    assert pp.parse_match_id(url) == expected


@pytest.mark.parametrize("url", ["", "https://example.com/video", "not a url"])
def test_parse_match_id_rejects_non_youtube(url):
    with pytest.raises(ValueError):
        pp.parse_match_id(url)


def test_parse_tournament():
    assert pp.parse_tournament("  FE  M S  Grand Prix Seoul 2024  ") == "M S Grand Prix Seoul 2024"
    assert pp.parse_tournament(None) == ""
    assert pp.parse_tournament("") == ""


def test_repo_relative_posix_path():
    path = os.path.join(pp.BASE_DIR, "data", "clips", "foil", "L_foil_x_0.mp4")
    assert pp.repo_relative_posix(path) == "data/clips/foil/L_foil_x_0.mp4"


def test_repo_relative_posix_outside_repo_is_absolute(tmp_path):
    result = pp.repo_relative_posix(str(tmp_path / "a.mp4"))
    assert "\\" not in result and not result.startswith("..")


def row(path, **overrides):
    base = dict.fromkeys(pp.MANIFEST_COLUMNS, "")
    base.update(clip_path=path, label="L")
    base.update(overrides)
    return base


def test_manifest_append_writes_header_once(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.append_manifest_rows(manifest, [row("a.mp4")])
    pp.append_manifest_rows(manifest, [row("b.mp4")])
    rows = read_rows(manifest)
    assert [r["clip_path"] for r in rows] == ["a.mp4", "b.mp4"]
    with open(manifest, encoding="utf-8") as fh:
        assert fh.readline().strip().split(",") == EXPECTED_COLUMNS
        assert sum(1 for line in fh if line.startswith("clip_path,")) == 0


def test_manifest_rerun_replaces_rows_of_same_clip(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.append_manifest_rows(manifest, [row("a.mp4", label="L"), row("b.mp4")])
    pp.append_manifest_rows(manifest, [row("a.mp4", label="R")])
    rows = {r["clip_path"]: r["label"] for r in read_rows(manifest)}
    assert rows == {"a.mp4": "R", "b.mp4": "L"}


def test_manifest_with_no_rows_creates_nothing(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.append_manifest_rows(manifest, [])
    assert not manifest.exists()


def test_processed_urls_union_of_state_and_legacy(tmp_path):
    state = tmp_path / "state"
    legacy = tmp_path / "scripts"
    state.mkdir()
    legacy.mkdir()
    (state / "processed_foil_urls.txt").write_text("u1\nu2\n")
    (legacy / "processed_foil_urls.txt").write_text("u2\nu3\n")
    urls = pp.load_processed_urls("foil", state_dir=str(state), legacy_dir=str(legacy))
    assert urls == {"u1", "u2", "u3"}


def test_processed_urls_legacy_only(tmp_path):
    legacy = tmp_path / "scripts"
    legacy.mkdir()
    (legacy / "processed_epee_urls.txt").write_text("u3\n")
    urls = pp.load_processed_urls(
        "epee", state_dir=str(tmp_path / "state"), legacy_dir=str(legacy)
    )
    assert urls == {"u3"}


def test_mark_processed_writes_to_state_dir(tmp_path):
    state = tmp_path / "state"
    pp.mark_processed("foil", "u9", state_dir=str(state))
    pp.mark_processed("foil", "u10", state_dir=str(state))
    text = (state / "processed_foil_urls.txt").read_text().split()
    assert text == ["u9", "u10"]


def test_process_video_file_writes_manifest_and_clip(tmp_path, monkeypatch):
    onset, change_at = 600, 600 + 20

    def scripted_score(frame):
        return (3, 2) if decode_frame_index(frame) < change_at else (4, 2)

    monkeypatch.setattr(pp, "check_score", scripted_score)
    video = tmp_path / "abcDEF12345.avi"
    write_synthetic_match(video, fps=30.0, n_frames=1500, onset=onset)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()

    saved = pp.process_video_file(
        str(video), str(out_dir), "sabre",
        cfg=pp.PipelineConfig(intro_s=2.0), min_video_s=1.0,
        source_url="https://youtube.com/watch?v=abcDEF12345",
        tournament="Test Open 2024",
    )
    assert saved == 1

    (clip,) = [p for p in out_dir.iterdir() if p.suffix == ".mp4"]
    assert clip.name == "L_sabre_abcDEF12345_0.mp4"

    (entry,) = read_rows(out_dir / "manifest.csv")
    assert list(entry) == EXPECTED_COLUMNS
    assert entry["clip_path"].endswith("L_sabre_abcDEF12345_0.mp4")
    assert "\\" not in entry["clip_path"]
    assert entry["label"] == "L" and entry["weapon"] == "sabre"
    assert entry["match_id"] == "abcDEF12345"
    assert entry["source_url"] == "https://youtube.com/watch?v=abcDEF12345"
    assert entry["tournament"] == "Test Open 2024"
    assert entry["hit_type"] == "On-No"
    assert (entry["score_left_before"], entry["score_right_before"]) == ("3", "2")
    assert (entry["score_left_after"], entry["score_right_after"]) == ("4", "2")
    assert float(entry["light_time_s"]) == pytest.approx(20.0)
    assert float(entry["source_fps"]) == pytest.approx(30.0)
    assert float(entry["clip_fps"]) == pytest.approx(30.0)
    assert entry["n_frames"] == "69"
    assert entry["light_frame_in_clip"] == "60"
    assert entry["split"] == ""


def test_no_manifest_without_labelled_clips(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "check_score", lambda frame: (3, 2))
    video = tmp_path / "abcDEF12345.avi"
    write_synthetic_match(video, n_frames=400, onset=200)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    pp.process_video_file(
        str(video), str(out_dir), "sabre",
        cfg=pp.PipelineConfig(intro_s=2.0), min_video_s=1.0,
    )
    assert not (out_dir / "manifest.csv").exists()  # nothing labelled
