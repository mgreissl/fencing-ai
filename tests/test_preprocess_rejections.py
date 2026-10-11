"""Tests for the per-reason rejection log (#15)."""

import csv
from collections import Counter

import cv2
import numpy as np
import pytest

from preprocess_helpers import make_timeline
from scripts import preprocess_pipeline as pp

ONSET = 1000


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def exchanges_for_demo():
    # 1: both lights, score unchanged until the next light -> accepted T.
    # 2: single light, score never changes -> rejected.
    second = ONSET + 300
    timeline = make_timeline(
        n_frames=3000,
        runs=[(ONSET, ONSET + 75)],
        left_state=pp.STATE_ON,
        right_state=pp.STATE_ON,
        scores=lambda frame: (3, 2),
    )
    timeline.left_state[second:second + 75] = pp.STATE_ON
    return timeline


def test_count_reasons_includes_accepted():
    exchanges = pp.analyse_timeline(exchanges_for_demo(), "sabre")
    counts = pp.count_reasons(exchanges)
    assert counts == Counter({"accepted": 1, "single_light_no_score_change": 1})


def test_rejection_rows_have_reason_and_time(tmp_path):
    exchanges = pp.analyse_timeline(exchanges_for_demo(), "sabre")
    path = tmp_path / "state" / "rejections_sabre.csv"
    pp.append_rejections(path, "vid123", "sabre", exchanges, fps=25.0)
    rows = read_rows(path)
    rejected = [e for e in exchanges if not e.accepted]
    assert len(rows) == len(rejected) >= 1
    row = rows[0]
    assert row["match_id"] == "vid123" and row["weapon"] == "sabre"
    assert row["reason"] == rejected[0].reason
    assert float(row["light_time_s"]) == pytest.approx(rejected[0].onset_frame / 25.0)
    assert row["hit_type"] == rejected[0].hit_type


def test_header_written_once_and_file_is_append_safe(tmp_path):
    exchanges = pp.analyse_timeline(exchanges_for_demo(), "sabre")
    path = tmp_path / "rejections_sabre.csv"
    pp.append_rejections(path, "a", "sabre", exchanges, fps=25.0)
    first = len(read_rows(path))
    pp.append_rejections(path, "b", "sabre", exchanges, fps=25.0)
    rows = read_rows(path)
    assert len(rows) == 2 * first
    assert {r["match_id"] for r in rows} == {"a", "b"}
    with open(path, encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.startswith("match_id,")) == 1


def test_video_level_rejection_row(tmp_path):
    path = tmp_path / "rejections_foil.csv"
    pp.append_video_rejection(path, "vid", "foil", "video_too_short")
    (row,) = read_rows(path)
    assert row["reason"] == "video_too_short"
    assert row["light_time_s"] == "" and row["hit_type"] == ""


def test_summary_lists_counts_by_reason():
    counts = Counter({"accepted": 2, "score_unstable_before": 5})
    text = pp.format_reason_counts(counts)
    assert "score_unstable_before" in text and "5" in text
    assert text.index("score_unstable_before") < text.index("accepted")


def test_read_reason_counts_from_file(tmp_path):
    path = tmp_path / "rejections_foil.csv"
    pp.append_video_rejection(path, "a", "foil", "video_too_short")
    pp.append_video_rejection(path, "b", "foil", "video_too_short")
    pp.append_video_rejection(path, "c", "foil", "invalid_source_fps")
    assert pp.read_reason_counts(path) == Counter(
        {"video_too_short": 2, "invalid_source_fps": 1}
    )
    assert pp.read_reason_counts(tmp_path / "missing.csv") == Counter()


def test_process_video_file_logs_short_video(tmp_path):
    video = tmp_path / "shortvid.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (640, 360)
    )
    for _ in range(30):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()

    log = tmp_path / "rejections_sabre.csv"
    counts = Counter()
    saved = pp.process_video_file(
        str(video), str(tmp_path), "sabre", rejections_path=log, reason_counts=counts
    )
    assert saved == 0
    assert counts == Counter({"video_too_short": 1})
    assert read_rows(log)[0]["reason"] == "video_too_short"


def test_process_video_file_without_lights_logs_no_light(tmp_path):
    video = tmp_path / "dark.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (640, 360)
    )
    for _ in range(120):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()

    counts = Counter()
    cfg = pp.PipelineConfig(intro_s=1.0)
    saved = pp.process_video_file(
        str(video), str(tmp_path), "sabre", cfg=cfg, reason_counts=counts,
        min_video_s=1.0,
    )
    assert saved == 0
    assert counts == Counter({"no_light_detected": 1})
