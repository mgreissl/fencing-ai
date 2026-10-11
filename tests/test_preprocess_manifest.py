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


def legacy_layout(tmp_path):
    """New versioned log plus the two legacy locations (state/, scripts/)."""
    new = tmp_path / "state" / "v2"
    old_state = tmp_path / "state"
    old_scripts = tmp_path / "scripts"
    for directory in (new, old_scripts):
        directory.mkdir(parents=True)
    (new / "processed_foil_urls.txt").write_text("u1\nu2\n")
    (old_state / "processed_foil_urls.txt").write_text("u2\nu3\n")
    (old_scripts / "processed_foil_urls.txt").write_text("u4\n")
    return str(new), (str(old_state), str(old_scripts))


def test_legacy_logs_ignored_by_default(tmp_path):
    new, legacy = legacy_layout(tmp_path)
    urls = pp.load_processed_urls("foil", state_dir=new, legacy_dirs=legacy)
    assert urls == {"u1", "u2"}


def test_resume_legacy_unions_all_logs(tmp_path):
    new, legacy = legacy_layout(tmp_path)
    urls = pp.load_processed_urls("foil", state_dir=new, legacy_dirs=legacy,
                                  resume_legacy=True)
    assert urls == {"u1", "u2", "u3", "u4"}


def test_missing_logs_give_empty_set(tmp_path):
    urls = pp.load_processed_urls("epee", state_dir=str(tmp_path / "x"),
                                  legacy_dirs=(str(tmp_path / "y"),))
    assert urls == set()


def test_state_dir_is_versioned():
    assert os.path.normpath(pp.PIPELINE_STATE_DIR) == os.path.normpath(
        os.path.join(pp.BASE_DIR, "state", pp.STATE_VERSION))
    assert pp.STATE_VERSION == "v2"
    assert os.path.dirname(pp.rejections_path("foil")) == pp.PIPELINE_STATE_DIR
    assert os.path.dirname(pp.failed_path("foil")) == pp.PIPELINE_STATE_DIR


def test_default_state_dir_follows_module_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PIPELINE_STATE_DIR", str(tmp_path / "v9"))
    pp.mark_processed("foil", "u5")
    assert pp.load_processed_urls("foil", legacy_dirs=()) == {"u5"}


def test_resume_legacy_cli_flag():
    import argparse

    parser = argparse.ArgumentParser()
    pp.add_cli_args(parser)
    assert parser.parse_args([]).resume_legacy is False
    assert parser.parse_args(["--resume-legacy"]).resume_legacy is True


def test_pipeline_ignores_legacy_unless_asked(tmp_path, monkeypatch):
    seen = []
    (tmp_path / "foil_videos.txt").write_text(
        "https://www.youtube.com/watch?v=aaaaaaaaaaa\n"
        "https://www.youtube.com/watch?v=bbbbbbbbbbb\n")
    legacy = tmp_path / "state"
    legacy.mkdir()
    (legacy / "processed_foil_urls.txt").write_text(
        "https://www.youtube.com/watch?v=aaaaaaaaaaa\n")
    monkeypatch.setattr(pp, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(pp, "PIPELINE_STATE_DIR", str(legacy / "v2"))
    monkeypatch.setattr(pp, "LEGACY_STATE_DIRS", (str(legacy),))
    monkeypatch.setattr(pp, "download_match",
                        lambda url, *a: seen.append(url) or (None, ""))
    monkeypatch.setattr(pp.time, "sleep", lambda s: None)

    pp.run_pipeline(weapon="foil")
    assert len(seen) == 2
    seen.clear()
    pp.run_pipeline(weapon="foil", resume_legacy=True)
    assert seen == ["https://www.youtube.com/watch?v=bbbbbbbbbbb"]


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
    assert clip.name == "L_sabre_abcDEF12345_20000.mp4"  # onset 600 / 30 fps

    (entry,) = read_rows(out_dir / "manifest.csv")
    assert list(entry) == EXPECTED_COLUMNS
    assert entry["clip_path"].endswith("L_sabre_abcDEF12345_20000.mp4")
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


def test_clip_name_uses_onset_time_not_running_index():
    exchange = pp.Exchange(onset_frame=751, hit_type="On-No", label="R")
    assert pp.clip_id(exchange, "foil", "m1", fps=25.0) == "R_foil_m1_30040"
    other = pp.Exchange(onset_frame=900, hit_type="On-No", label="L")
    assert pp.clip_id(other, "foil", "m1", fps=25.0) == "L_foil_m1_36000"


def test_clip_name_parses_with_legacy_filename_parser():
    from src.manifest import parse_legacy_filename

    parsed = parse_legacy_filename("T_foil_abc-DEF_12345_30040.mp4")
    assert parsed is not None and parsed["match_id"] == "abc-DEF_12345"


def test_replace_manifest_rows_swaps_all_rows_of_a_match(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.append_manifest_rows(manifest, [
        row("a_0.mp4", match_id="A"), row("a_1.mp4", match_id="A"),
        row("b_0.mp4", match_id="B"),
    ])
    pp.replace_manifest_rows(manifest, "A", [row("a_500.mp4", match_id="A")])
    rows = read_rows(manifest)
    assert sorted(r["clip_path"] for r in rows) == ["a_500.mp4", "b_0.mp4"]
    assert list(rows[0]) == pp.MANIFEST_COLUMNS


def test_replace_manifest_rows_with_no_rows_drops_the_match(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.append_manifest_rows(manifest, [row("a_0.mp4", match_id="A"),
                                       row("b_0.mp4", match_id="B")])
    pp.replace_manifest_rows(manifest, "A", [])
    assert [r["clip_path"] for r in read_rows(manifest)] == ["b_0.mp4"]


def test_replace_manifest_rows_does_not_create_empty_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pp.replace_manifest_rows(manifest, "A", [])
    assert not manifest.exists()


def test_reprocessing_replaces_rows_instead_of_duplicating(tmp_path, monkeypatch):
    onset, change_at = 600, 620
    monkeypatch.setattr(
        pp, "check_score",
        lambda frame: (3, 2) if decode_frame_index(frame) < change_at else (4, 2))
    video = tmp_path / "abcDEF12345.avi"
    write_synthetic_match(video, fps=30.0, n_frames=1500, onset=onset)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    manifest = out_dir / "manifest.csv"
    pp.append_manifest_rows(manifest, [
        row("data/clips/sabre/R_sabre_abcDEF12345_0.mp4", match_id="abcDEF12345",
            label="R"),
        row("data/clips/sabre/L_sabre_other_0.mp4", match_id="other"),
    ])

    for _ in range(2):
        pp.process_video_file(
            str(video), str(out_dir), "sabre",
            cfg=pp.PipelineConfig(intro_s=2.0), min_video_s=1.0)

    rows = read_rows(manifest)
    mine = [r for r in rows if r["match_id"] == "abcDEF12345"]
    assert len(mine) == 1 and mine[0]["clip_path"].endswith("_20000.mp4")
    assert len(rows) == 2  # the other match is untouched


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
