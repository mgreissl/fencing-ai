#!/usr/bin/env python3
"""
scripts/preprocess_pipeline.py — Multi-Weapon Preprocessing Pipeline.

Processes match video URLs from {weapon}_videos.txt:
1. Downloads match video in 360p via yt-dlp (with android client fallback).
2. Scans the video once: light state per frame, scoreboard digits at a stride.
3. Auto-labels referee decision (L, R, T) from scoreboard digit OCR changes.
4. Saves clips aligned to the light onset (PRE_S before to POST_S after, at
   the source fps) into data/clips/{weapon}/ and describes every clip in
   data/clips/{weapon}/manifest.csv. Rejections go to
   state/v2/rejections_{weapon}.csv, progress to
   state/v2/processed_{weapon}_urls.txt.
5. Removes match video immediately to conserve disk space.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess as sp
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import overlay  # noqa: E402

# Overlay geometry/thresholds: single source of truth in configs/overlay/.
OVERLAY = overlay.load_overlay(overlay.DEFAULT_OVERLAY)

# Digit classifier (npz, pickle fallback) for scoreboard numbers 0-15.
SCORE_CLASSIFIER = overlay.load_digit_classifier(cfg=OVERLAY)

INTRO_S = 60.0  # skipped at the start of every match video
MIN_VIDEO_S = 100.0


def validate_source_fps(fps):
    """Return fps as float if plausible, else None (never assume 25)."""
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fps) or not 5.0 <= fps <= 120.0:
        return None
    return fps


def frame_to_seconds(frame_index, fps):
    """Source frame index to seconds."""
    return frame_index / fps


def seconds_to_frames(seconds, fps):
    """Seconds to the nearest source frame count."""
    return int(round(seconds * fps))


def check_lights(frame, weapon="foil"):
    """Check whether left/right on-target or off-target lights are lit."""
    return overlay.check_lights(frame, OVERLAY, weapon)


def check_score(frame):
    """Return (left, right) scores, or None if a digit read is not confident."""
    read = overlay.read_score(frame, OVERLAY, SCORE_CLASSIFIER)
    return None if read is None else (read.left, read.right)


# Label semantics (the class is the referee's decision, read from the score):
#
# Foil and sabre award the point by priority (right of way). The label is the
# fencer who was awarded the touch:
#   L / R  the left / right score went up by exactly one.
#   T      both on-target lights, score unchanged after the decision window:
#          simultaneous action, no touch awarded. This also contains annulled
#          touches (for example after a video review), which cannot be told
#          apart from the video.
#   Foil only, one on-target and one off-target light: the off-target hit does
#   not score, so the score only goes up for the on-target fencer if he had
#   priority (On-Off +1 left -> L). If the score stays unchanged the referee
#   gave priority to the off-target fencer (On-Off -> R, Off-On -> L). That
#   outcome also covers simultaneous actions and annulments, so T and R/L mean
#   different things (T: "no point awarded", R/L: "priority to the off-target
#   fencer") for the same unchanged score. ``on_off_unchanged`` selects:
#   "priority" (default) keeps the R/L label, "reject" drops these exchanges.
#
# Epee has no priority; both touches count. The label is the fencer who scored
# (the one whose score went up), not the one who was hit:
#   L / R  single light and that fencer's score went up by one.
#   T      both lights and both scores went up by one (double touch).
#
# Everything else is rejected, not labelled: negative changes, jumps of more
# than one point (cards, penalties, corrections), both scores up in foil or
# sabre, a score change for a fencer whose light was not lit, a single light
# without score change (annulled or card), epee with both lights but no or
# only one point, and hit types that cannot be mapped (off-target only).

LABEL_HIT_TYPES = {
    "foil": ("On-On", "On-No", "No-On", "On-Off", "Off-On"),
    "sabre": ("On-On", "On-No", "No-On"),
    "epee": ("On-On", "On-No", "No-On"),
}


@dataclass(frozen=True)
class LabelDecision:
    """A label (L/T/R) or the reason why the exchange is not labellable."""

    label: str = None
    reason: str = None


def _reject(reason):
    return LabelDecision(None, reason)


ON_OFF_UNCHANGED_MODES = ("priority", "reject")


def label_exchange(hit_type, before, after, weapon="foil",
                   on_off_unchanged="priority"):
    """Label one exchange from the light pattern and the score change.

    ``before`` and ``after`` are (left, right) scores; ``after`` must be the
    stabilised, confirmed score (see ``resolve_score_after``).
    ``on_off_unchanged`` decides what a foil On-Off / Off-On with unchanged
    score becomes: ``priority`` (label for the off-target fencer) or ``reject``.
    """
    if on_off_unchanged not in ON_OFF_UNCHANGED_MODES:
        raise ValueError(f"on_off_unchanged must be one of {ON_OFF_UNCHANGED_MODES}")
    delta_left, delta_right = after[0] - before[0], after[1] - before[1]
    if delta_left < 0 or delta_right < 0:
        return _reject("negative_score_change")
    if delta_left > 1 or delta_right > 1:
        return _reject("score_jump_gt_one")
    if hit_type not in LABEL_HIT_TYPES.get(weapon, ()):
        return _reject("hit_type_not_labellable")
    delta = (delta_left, delta_right)

    if weapon == "epee":
        return _label_epee(hit_type, delta)
    return _label_priority(hit_type, delta, on_off_unchanged)


def _label_epee(hit_type, delta):
    if hit_type == "On-On":
        if delta == (1, 1):
            return LabelDecision("T")
        if delta == (0, 0):
            return _reject("no_score_change_both_lit")
        return _reject("one_sided_score_both_lit")
    if delta == (1, 1):
        return _reject("both_scores_increased")
    if hit_type == "On-No":
        if delta == (1, 0):
            return LabelDecision("L")
    elif delta == (0, 1):  # No-On
        return LabelDecision("R")
    if delta == (0, 0):
        return _reject("single_light_no_score_change")
    return _reject("score_inconsistent_with_lights")


def _label_priority(hit_type, delta, on_off_unchanged="priority"):
    if delta == (1, 1):
        return _reject("both_scores_increased")
    if (on_off_unchanged == "reject" and delta == (0, 0)
            and hit_type in ("On-Off", "Off-On")):
        return _reject("off_target_score_unchanged")
    table = {
        "On-On": {(1, 0): "L", (0, 1): "R", (0, 0): "T"},
        "On-No": {(1, 0): "L"},
        "No-On": {(0, 1): "R"},
        "On-Off": {(1, 0): "L", (0, 0): "R"},
        "Off-On": {(0, 1): "R", (0, 0): "L"},
    }
    label = table[hit_type].get(delta)
    if label is not None:
        return LabelDecision(label)
    if delta == (0, 0):
        return _reject("single_light_no_score_change")
    return _reject("score_inconsistent_with_lights")


def determine_label(hit_type, left, right, next_left, next_right, weapon="foil"):
    """Label (L/T/R) or None; see ``label_exchange`` for the reason."""
    return label_exchange(
        hit_type, (left, right), (next_left, next_right), weapon
    ).label


# ---------------------------------------------------------------------------
# Pure helpers: windows, light onsets, hit type
# ---------------------------------------------------------------------------

# Clip window around the light onset (seconds).
PRE_S = 2.0
POST_S = 0.3

# Light state codes stored per frame in a Timeline.
STATE_NO, STATE_ON, STATE_OFF = 0, 1, 2
STATE_CODES = {"No": STATE_NO, "On": STATE_ON, "Off": STATE_OFF}


@dataclass(frozen=True)
class Window:
    """Source frames of one saved clip, aligned to the light onset."""

    src_indices: tuple
    light_frame_in_clip: int  # 0-based index of the onset frame in the clip
    clip_fps: float
    source_fps: float

    @property
    def n_frames(self):
        return len(self.src_indices)


def compute_window(onset_frame, source_fps, pre_s=PRE_S, post_s=POST_S,
                   total_frames=None, clip_fps=None):
    """Window of ``pre_s`` before to ``post_s`` after the light onset.

    By default frames are taken at the native source fps. With ``clip_fps``
    below the source fps, source frames are subsampled; the onset frame is
    always kept exactly. The onset is clip frame ``round(pre_s * clip_fps)``.
    Returns None if the window leaves the video.
    """
    out_fps = source_fps if clip_fps is None else float(clip_fps)
    if out_fps > source_fps * 1.001:
        raise ValueError(f"clip_fps {out_fps} exceeds source fps {source_fps}")
    n_pre = seconds_to_frames(pre_s, out_fps)
    n_post = seconds_to_frames(post_s, out_fps)
    offsets = np.arange(-n_pre, n_post)  # clip frames relative to the onset
    src = onset_frame + np.rint(offsets * (source_fps / out_fps)).astype(int)
    if src[0] < 0 or (total_frames is not None and src[-1] >= total_frames):
        return None
    return Window(
        src_indices=tuple(int(i) for i in src),
        light_frame_in_clip=n_pre,
        clip_fps=out_fps,
        source_fps=source_fps,
    )


def find_light_onsets(lit, min_dark_frames, first_valid=0):
    """Frame indices where a light starts after ``min_dark_frames`` of darkness.

    ``lit`` is a per-frame boolean sequence. Frames before ``first_valid`` were
    not evaluated, so an onset needs ``min_dark_frames`` evaluated dark frames
    in front of it.
    """
    lit = np.asarray(lit, dtype=bool)
    onsets = []
    last_lit = first_valid - 1
    for i in np.flatnonzero(lit):
        i = int(i)
        if i < first_valid:
            continue
        if i - last_lit > min_dark_frames and i - first_valid >= min_dark_frames:
            onsets.append(i)
        last_lit = i
    return onsets


def aggregate_hit_type(left_states, right_states):
    """Combine per-frame light states of a short window into one hit type.

    Per side the strongest state wins (On > Off > No), so a second light that
    comes on a few frames after the first one is still part of the exchange.
    """
    def strongest(states):
        states = set(int(s) for s in states)
        if STATE_ON in states:
            return "On"
        if STATE_OFF in states:
            return "Off"
        return "No"

    return f"{strongest(left_states)}-{strongest(right_states)}"


# ---------------------------------------------------------------------------
# Video scan: lights per frame, scoreboard sampled at a fixed stride
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Tunable parameters of the preprocessing pipeline."""

    pre_s: float = PRE_S
    post_s: float = POST_S
    clip_fps: float = None        # None = native source fps
    intro_s: float = INTRO_S
    min_dark_s: float = 0.5       # lights must be off this long before an onset
    hit_window_s: float = 0.5     # lights are merged over this long after onset
    score_stride_s: float = 0.25  # scoreboard sampling interval
    n_stable_before: int = 3      # identical reads needed before the onset
    n_stable_after: int = 4       # identical reads needed after the onset
    score_timeout_s: float = 12.0  # wait this long for the referee decision
    min_unchanged_s: float = 3.0  # unchanged score counts only after this long
    on_off_unchanged: str = "priority"  # foil On-Off/Off-On, score unchanged


@dataclass
class Timeline:
    """Per-frame light states and strided scoreboard reads of one video."""

    fps: float
    n_frames: int
    first_valid: int
    left_state: np.ndarray
    right_state: np.ndarray
    score_frames: np.ndarray
    scores: list = field(default_factory=list)  # (left, right) or None

    @property
    def lit(self):
        return (self.left_state != STATE_NO) | (self.right_state != STATE_NO)


def scan_video(vid_path, weapon, cfg=None, score_reader=None):
    """Read a video once: light state of every frame, scores at a stride.

    Returns None if the video has no valid fps.
    """
    cfg = cfg or PipelineConfig()
    score_reader = score_reader or check_score
    cap = cv2.VideoCapture(vid_path)
    try:
        fps = validate_source_fps(cap.get(cv2.CAP_PROP_FPS))
        if fps is None:
            return None
        stride = max(1, seconds_to_frames(cfg.score_stride_s, fps))
        first_valid = seconds_to_frames(cfg.intro_s, fps)

        left, right, score_frames, scores = [], [], [], []
        idx = 0
        while cap.grab():
            frame = None
            if idx >= first_valid:
                ok, frame = cap.retrieve()
                if not ok:
                    frame = None
            if frame is None:
                left.append(STATE_NO)
                right.append(STATE_NO)
            else:
                frame = overlay.normalize_frame(frame, OVERLAY)
                state_left, state_right = overlay.light_states(frame, OVERLAY, weapon)
                left.append(STATE_CODES[state_left])
                right.append(STATE_CODES[state_right])
                if idx % stride == 0:
                    score_frames.append(idx)
                    scores.append(score_reader(frame))
            idx += 1
    finally:
        cap.release()

    return Timeline(
        fps=fps,
        n_frames=idx,
        first_valid=first_valid,
        left_state=np.asarray(left, dtype=np.int8),
        right_state=np.asarray(right, dtype=np.int8),
        score_frames=np.asarray(score_frames, dtype=int),
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Score stabilisation
# ---------------------------------------------------------------------------

@dataclass
class AfterScore:
    """Outcome of reading the scoreboard after a light onset.

    kind: ``changed`` (new stable score), ``unchanged`` (stable and confirmed
    unchanged), ``undecided`` (window closed too early to tell),
    ``unstable`` (several changes or a revert) or ``unreadable``.
    """

    kind: str
    score: tuple = None
    decision_frame: int = None
    closed_by: str = ""


def _stable_runs(frames, scores, n_stable):
    """Runs of at least ``n_stable`` consecutive identical reads.

    Returns [(value, first_frame), ...]; adjacent runs with the same value
    (for example split by a misread) are merged.
    """
    runs = []
    current, count, first = None, 0, None
    for frame, score in zip(frames, scores):
        if score is not None and score == current:
            count += 1
        else:
            current, count, first = score, (0 if score is None else 1), frame
        if score is not None and count == n_stable:
            if not runs or runs[-1][0] != current:
                runs.append((current, int(first)))
    return runs


def read_score_before(frames, scores, onset, n_stable):
    """Stable score just before the onset: (score, None) or (None, reason)."""
    end = int(np.searchsorted(frames, onset, side="left"))
    tail = scores[max(0, end - n_stable):end]
    if len(tail) < n_stable or all(t is None for t in tail):
        return None, "score_unreadable_before"
    if any(t is None for t in tail) or any(t != tail[0] for t in tail):
        return None, "score_unstable_before"
    return tail[0], None


def score_window_end(onset, next_onset, n_frames, fps, timeout_s):
    """Last frame to read the post-touch score and why the window closed."""
    timeout_end = onset + seconds_to_frames(timeout_s, fps)
    candidates = [(timeout_end, "timeout"), (n_frames - 1, "video_end")]
    if next_onset is not None:
        candidates.append((next_onset - 1, "next_onset"))
    return min(candidates, key=lambda c: c[0])


def resolve_score_after(frames, scores, before, onset, end_frame, closed_by, fps,
                        n_stable, min_unchanged_s):
    """Read the scoreboard after a touch until it is stable or the window ends.

    A change must hold for ``n_stable`` consecutive reads (single misreads are
    ignored). An unchanged score is only reported once the window closed by
    the next light or the timeout and at least ``min_unchanged_s`` were
    observed; otherwise the decision may still be pending.
    """
    lo = int(np.searchsorted(frames, onset, side="right"))
    hi = int(np.searchsorted(frames, end_frame, side="right"))
    window_frames, window_scores = frames[lo:hi], scores[lo:hi]
    runs = _stable_runs(window_frames, window_scores, n_stable)
    if not runs:
        return AfterScore("unreadable", closed_by=closed_by)

    values = [value for value, _ in runs]
    last_read = [x for x in window_scores if x is not None][-1]
    if last_read != values[-1]:  # the scoreboard was still changing
        return AfterScore("undecided", closed_by=closed_by)

    if len(values) == 1 and values[0] == before:
        observed_s = (end_frame - onset) / fps
        if closed_by in ("next_onset", "timeout") and observed_s >= min_unchanged_s:
            return AfterScore("unchanged", before, None, closed_by)
        return AfterScore("undecided", closed_by=closed_by)
    if len(values) == 1:
        return AfterScore("changed", values[0], runs[0][1], closed_by)
    if len(values) == 2 and values[0] == before and values[1] != before:
        return AfterScore("changed", values[1], runs[1][1], closed_by)
    return AfterScore("unstable", closed_by=closed_by)


@dataclass
class Exchange:
    """One detected light event with its label or rejection reason."""

    onset_frame: int
    hit_type: str
    label: str = None          # L/T/R, None if rejected
    reason: str = None         # rejection reason, None if accepted
    score_before: tuple = None
    score_after: tuple = None
    window: Window = None

    @property
    def accepted(self):
        return self.label is not None


def accepted_exchanges(exchanges):
    """Exchanges that received a label."""
    return [e for e in exchanges if e.accepted]


AFTER_REJECTION = {
    "unreadable": "score_unreadable_after",
    "unstable": "score_unstable_after",
    "undecided": "score_unchanged_undecided",
}


def analyse_timeline(timeline, weapon, cfg=None):
    """Turn a scanned timeline into exchanges with labels or reasons (pure)."""
    cfg = cfg or PipelineConfig()
    fps = timeline.fps
    frames, scores = timeline.score_frames, timeline.scores
    onsets = find_light_onsets(
        timeline.lit,
        seconds_to_frames(cfg.min_dark_s, fps),
        first_valid=timeline.first_valid,
    )
    exchanges = []
    previous_undecided = False
    for i, onset in enumerate(onsets):
        next_onset = onsets[i + 1] if i + 1 < len(onsets) else None
        hit_end = onset + max(1, seconds_to_frames(cfg.hit_window_s, fps))
        hit_type = aggregate_hit_type(
            timeline.left_state[onset:hit_end], timeline.right_state[onset:hit_end]
        )
        exchange = Exchange(onset, hit_type)
        exchanges.append(exchange)

        before, before_reason = read_score_before(frames, scores, onset,
                                                  cfg.n_stable_before)
        after = None
        if before is not None:
            end_frame, closed_by = score_window_end(
                onset, next_onset, timeline.n_frames, fps, cfg.score_timeout_s
            )
            after = resolve_score_after(
                frames, scores, before, onset, end_frame, closed_by, fps,
                cfg.n_stable_after, cfg.min_unchanged_s,
            )
        was_undecided = previous_undecided
        previous_undecided = bool(
            after is not None and after.kind == "undecided"
            and after.closed_by == "next_onset"
        )

        window = compute_window(onset, fps, cfg.pre_s, cfg.post_s,
                                total_frames=timeline.n_frames,
                                clip_fps=cfg.clip_fps)
        if hit_type not in LABEL_HIT_TYPES.get(weapon, ()):
            exchange.reason = (
                "off_target_only"
                if "Off" in hit_type and "On" not in hit_type
                else "hit_type_not_labellable"
            )
        elif window is None:
            exchange.reason = "window_outside_video"
        elif was_undecided:
            exchange.reason = "previous_exchange_undecided"
        elif before is None:
            exchange.reason = before_reason
        elif after.kind not in ("changed", "unchanged"):
            exchange.reason = AFTER_REJECTION[after.kind]
        else:
            decision = label_exchange(hit_type, before, after.score, weapon,
                                      cfg.on_off_unchanged)
            if decision.label is None:
                exchange.reason = decision.reason
            else:
                exchange.label = decision.label
                exchange.score_before, exchange.score_after = before, after.score
                exchange.window = window
    return exchanges


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------

def write_clips(vid_path, jobs):
    """Write clips for (Window, output_path) jobs with one sequential read.

    Returns the output paths that received all their frames.
    """
    if not jobs:
        return []
    wanted = {}  # source frame -> job indices (once per occurrence)
    for job_id, (window, _) in enumerate(jobs):
        for src in window.src_indices:
            wanted.setdefault(src, []).append(job_id)
    last_needed = max(wanted)

    width, height = OVERLAY.frame_size
    writers, written = {}, [0] * len(jobs)
    cap = cv2.VideoCapture(vid_path)
    try:
        idx = 0
        while idx <= last_needed and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                frame = overlay.normalize_frame(frame, OVERLAY)
                for job_id in wanted[idx]:
                    if job_id not in writers:
                        window, path = jobs[job_id]
                        writers[job_id] = cv2.VideoWriter(
                            path, cv2.VideoWriter_fourcc(*"mp4v"),
                            window.clip_fps, (width, height),
                        )
                    writers[job_id].write(frame)
                    written[job_id] += 1
            idx += 1
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    return [path for (window, path), n in zip(jobs, written)
            if n == window.n_frames]


# ---------------------------------------------------------------------------
# Rejection log: why was a candidate exchange not turned into a clip?
# ---------------------------------------------------------------------------

# Machine-local progress/failure/rejection logs. Versioned: a change of the
# labelling or clip window bumps STATE_VERSION so that logs of an older
# pipeline (state/*.txt, scripts/*.txt) never mark URLs as processed.
STATE_VERSION = "v2"
PIPELINE_STATE_DIR = os.path.join(BASE_DIR, "state", STATE_VERSION)
REJECTION_COLUMNS = ["match_id", "weapon", "onset_frame", "light_time_s",
                     "hit_type", "reason"]


def rejections_path(weapon):
    """Per-weapon rejection log: state/rejections_{weapon}.csv."""
    return os.path.join(PIPELINE_STATE_DIR, f"rejections_{weapon}.csv")


def count_reasons(exchanges):
    """Counter of rejection reasons; accepted exchanges count as ``accepted``."""
    return Counter("accepted" if e.accepted else e.reason for e in exchanges)


def _append_rows(path, rows):
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REJECTION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_rejections(path, match_id, weapon, exchanges, fps):
    """Append one row per rejected exchange (header written once)."""
    rows = [
        {
            "match_id": match_id,
            "weapon": weapon,
            "onset_frame": e.onset_frame,
            "light_time_s": f"{frame_to_seconds(e.onset_frame, fps):.3f}",
            "hit_type": e.hit_type,
            "reason": e.reason,
        }
        for e in exchanges
        if not e.accepted
    ]
    if rows:
        _append_rows(path, rows)


def append_video_rejection(path, match_id, weapon, reason):
    """Append a row for a whole video that produced no candidates."""
    _append_rows(path, [{"match_id": match_id, "weapon": weapon, "onset_frame": "",
                         "light_time_s": "", "hit_type": "", "reason": reason}])


def read_reason_counts(path):
    """Counter of reasons over a rejection log (empty if it does not exist)."""
    counts = Counter()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            counts.update(row["reason"] for row in csv.DictReader(fh))
    return counts


def format_reason_counts(counts):
    """Readable lines, rejection reasons first (most frequent on top)."""
    total = sum(counts.values()) or 1
    ordered = sorted(
        counts.items(), key=lambda kv: (kv[0] == "accepted", -kv[1], kv[0])
    )
    return "\n".join(
        f"    {reason:<32s}{n:>6d}  ({100.0 * n / total:4.1f}%)"
        for reason, n in ordered
    )


# ---------------------------------------------------------------------------
# Manifest (one row per saved clip) and pipeline state files
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "clip_path", "label", "weapon", "match_id", "source_url", "tournament",
    "hit_type", "score_left_before", "score_right_before", "score_left_after",
    "score_right_after", "light_time_s", "source_fps", "clip_fps", "n_frames",
    "light_frame_in_clip", "split",
]
MANIFEST_NAME = "manifest.csv"

LEGACY_STATE_DIRS = (os.path.join(BASE_DIR, "state"),
                     os.path.join(BASE_DIR, "scripts"))
_YOUTUBE_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")


def parse_match_id(url):
    """YouTube video id of a match URL (stable, filesystem-safe)."""
    found = _YOUTUBE_ID.search(url or "")
    if not found:
        raise ValueError(f"Cannot parse a YouTube video id from {url!r}")
    return found.group(1)


def parse_tournament(title):
    """Tournament name from the video title (whitespace-normalised, may be empty)."""
    return re.sub(r"\s+", " ", re.sub(r"^\s*FE\s+", "", title or "")).strip()


def repo_relative_posix(path):
    """Path relative to the repo root with forward slashes (absolute if outside)."""
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, BASE_DIR)
    except ValueError:  # different drive
        rel = None
    if rel is None or rel.startswith(".."):
        return path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def manifest_row(exchange, weapon, clip_path, match_id, source_url, tournament,
                 source_fps):
    """Manifest row for one saved clip."""
    window = exchange.window
    return {
        "clip_path": repo_relative_posix(clip_path),
        "label": exchange.label,
        "weapon": weapon,
        "match_id": match_id,
        "source_url": source_url,
        "tournament": tournament,
        "hit_type": exchange.hit_type,
        "score_left_before": exchange.score_before[0],
        "score_right_before": exchange.score_before[1],
        "score_left_after": exchange.score_after[0],
        "score_right_after": exchange.score_after[1],
        "light_time_s": f"{frame_to_seconds(exchange.onset_frame, source_fps):.3f}",
        "source_fps": f"{source_fps:g}",
        "clip_fps": f"{window.clip_fps:g}",
        "n_frames": window.n_frames,
        "light_frame_in_clip": window.light_frame_in_clip,
        "split": "",
    }


def append_manifest_rows(path, rows):
    """Append rows to the manifest (header once). Re-processed clips replace
    their earlier rows instead of duplicating them."""
    rows = list(rows)
    if not rows:
        return
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))

    new_paths = {row["clip_path"] for row in rows}
    if any(old["clip_path"] in new_paths for old in existing):
        kept = [old for old in existing if old["clip_path"] not in new_paths]
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(kept + rows)
        os.replace(tmp_path, path)
        return

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def _processed_log(weapon, directory):
    return os.path.join(directory, f"processed_{weapon}_urls.txt")


def load_processed_urls(weapon, state_dir=None, legacy_dirs=None,
                        resume_legacy=False):
    """URLs already processed by the current pipeline version.

    Logs of the old pipeline (``LEGACY_STATE_DIRS``) are only read with
    ``resume_legacy``; they were written with other labels and clip windows.
    """
    state_dir = state_dir or PIPELINE_STATE_DIR
    directories = [state_dir]
    if resume_legacy:
        directories += LEGACY_STATE_DIRS if legacy_dirs is None else legacy_dirs
    urls = set()
    for directory in directories:
        log = _processed_log(weapon, directory)
        if os.path.exists(log):
            with open(log, encoding="utf-8") as fh:
                urls.update(line.strip() for line in fh if line.strip())
    return urls


def mark_processed(weapon, url, state_dir=None):
    """Record a URL as processed in state/v2/processed_{weapon}_urls.txt."""
    state_dir = state_dir or PIPELINE_STATE_DIR
    os.makedirs(state_dir, exist_ok=True)
    with open(_processed_log(weapon, state_dir), "a", encoding="utf-8") as fh:
        fh.write(url + "\n")


FAILED_COLUMNS = ["url", "weapon", "reason", "timestamp"]


def failed_path(weapon):
    """Per-weapon failure log: state/failed_{weapon}.csv."""
    return os.path.join(PIPELINE_STATE_DIR, f"failed_{weapon}.csv")


def record_failure(weapon, url, reason):
    """Append a failed match to state/failed_{weapon}.csv so it can be retried."""
    path = failed_path(weapon)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(FAILED_COLUMNS)
        writer.writerow([url, weapon, reason, time.strftime("%Y-%m-%dT%H:%M:%S")])


def process_video_file(vid_path, output_dir, weapon="foil", cfg=None,
                       rejections_path=None, reason_counts=None,
                       min_video_s=MIN_VIDEO_S, match_id=None, source_url="",
                       tournament="", manifest_path=None):
    """Cut and label exchanges from a single match video.

    Saved clips are named ``{label}_{weapon}_{match_id}_{n}.mp4`` and described
    in the manifest (default ``{output_dir}/manifest.csv``). Rejection reasons
    are added to ``reason_counts`` (a Counter) and appended to the CSV at
    ``rejections_path`` when given.
    """
    cfg = cfg or PipelineConfig()
    vid_name = match_id or os.path.splitext(os.path.basename(vid_path))[0]
    manifest_path = manifest_path or os.path.join(output_dir, MANIFEST_NAME)
    counts = Counter()

    def finish(saved):
        if reason_counts is not None:
            reason_counts.update(counts)
        if counts:
            print("  Outcome per candidate exchange:")
            print(format_reason_counts(counts))
        return saved

    def reject_video(reason):
        counts[reason] += 1
        if rejections_path:
            append_video_rejection(rejections_path, vid_name, weapon, reason)
        return finish(0)

    timeline = scan_video(vid_path, weapon, cfg)
    if timeline is None:
        print(f"Video {vid_name} has no valid fps, skipping.")
        return reject_video("invalid_source_fps")
    if timeline.n_frames < seconds_to_frames(min_video_s, timeline.fps):
        print(f"Video {vid_name} too short ({timeline.n_frames} frames), skipping.")
        return reject_video("video_too_short")

    exchanges = analyse_timeline(timeline, weapon, cfg)
    if not exchanges:
        return reject_video("no_light_detected")

    jobs, job_exchanges = [], []
    for exchange in accepted_exchanges(exchanges):
        clip_id = f"{exchange.label}_{weapon}_{vid_name}_{len(jobs)}"
        jobs.append((exchange.window, os.path.join(output_dir, f"{clip_id}.mp4")))
        job_exchanges.append(exchange)
    saved = write_clips(vid_path, jobs)
    for (_, path), exchange in zip(jobs, job_exchanges):
        if path not in saved:
            exchange.label, exchange.reason = None, "clip_write_incomplete"

    append_manifest_rows(manifest_path, [
        manifest_row(exchange, weapon, path, vid_name, source_url, tournament,
                     timeline.fps)
        for (_, path), exchange in zip(jobs, job_exchanges) if path in saved
    ])

    counts.update(count_reasons(exchanges))
    if rejections_path:
        append_rejections(rejections_path, vid_name, weapon, exchanges, timeline.fps)

    print(f"  Successfully labelled {len(saved)} {weapon.upper()} clips.")
    return finish(len(saved))


COOKIE_BROWSERS = ("none", "firefox", "edge", "chrome")
ERROR_TAIL_LINES = 4
ERROR_TAIL_CHARS = 500

# yt-dlp runs in a child process of this interpreter. truststore makes it use
# the OS certificate store (which holds the CA of a TLS-intercepting proxy);
# certificate verification stays on. With an explicit CA file the OS store is
# not used and yt-dlp's bundled certifi path points to that file instead.
YTDLP_WRAPPER = """
import os, sys
ca_file = os.environ.get("FENCING_CA_FILE")
if ca_file:
    import certifi
    certifi.where = lambda: ca_file
else:
    import truststore
    truststore.inject_into_ssl()
import yt_dlp
yt_dlp.main(sys.argv[1:])
"""


class DownloadError(RuntimeError):
    """yt-dlp failed; the message holds the tail of its error output."""


def error_tail(stderr, max_lines=ERROR_TAIL_LINES, max_chars=ERROR_TAIL_CHARS):
    """Last lines of yt-dlp's stderr as one bounded single-line string."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])[-max_chars:]


@dataclass
class DownloadOptions:
    """How yt-dlp is called. The defaults need no browser and no cookies."""

    cookies_browser: str = "none"   # none, firefox, edge or chrome
    cookies_file: str = None        # Netscape cookie file, wins over the browser
    remote_components: bool = False  # let yt-dlp fetch its JS challenge solver
    ca_file: str = None             # extra/alternative CA bundle (PEM)
    no_check_certificates: bool = False  # INSECURE: disables TLS verification


def build_download_command(url, precut_dir, options=None):
    """yt-dlp command line for one match (360p, info json next to the video)."""
    options = options or DownloadOptions()
    cmd = [sys.executable, "-c", YTDLP_WRAPPER]
    if options.no_check_certificates:
        cmd += ["--no-check-certificates"]
    if options.remote_components:
        cmd += ["--remote-components", "ejs:github"]
    if options.cookies_file:
        cmd += ["--cookies", options.cookies_file]
    elif options.cookies_browser != "none":
        cmd += ["--cookies-from-browser", options.cookies_browser]
    cmd += [
        "--sleep-requests", "1.5",
        "-f", "18/134/best[height<=360]",
        "--no-warnings",
        "--write-info-json",
        "-o", os.path.join(precut_dir, "%(id)s.%(ext)s"),
        url,
    ]
    return cmd


def download_match(url, match_id, precut_dir, options=None):
    """Download a match in 360p; return (video_path, title) or (None, "")."""
    options = options or DownloadOptions()
    cmd = build_download_command(url, precut_dir, options)
    env = dict(os.environ)
    if options.ca_file:
        env["FENCING_CA_FILE"] = env["SSL_CERT_FILE"] = options.ca_file
    result = sp.run(cmd, env=env, stderr=sp.PIPE, text=True, errors="replace")
    if result.returncode != 0:
        raise DownloadError(
            error_tail(result.stderr) or f"yt-dlp exited with {result.returncode}"
        )

    title = ""
    info_path = os.path.join(precut_dir, f"{match_id}.info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as fh:
            title = json.load(fh).get("title", "")
    for fname in sorted(os.listdir(precut_dir)):
        stem, ext = os.path.splitext(fname)
        if stem == match_id and ext in (".mp4", ".mkv", ".webm"):
            return os.path.join(precut_dir, fname), title
    return None, title


def run_pipeline(weapon="sabre", max_matches=None, cfg=None,
                 download_options=None, resume_legacy=False):
    """Main pipeline execution loop."""
    cfg = cfg or PipelineConfig()
    download_options = download_options or DownloadOptions()
    urls_file = os.path.join(BASE_DIR, f"{weapon}_videos.txt")
    if not os.path.exists(urls_file):
        print(f"Error: {urls_file} not found.")
        return

    with open(urls_file) as f:
        all_urls = [line.strip() for line in f if line.strip().startswith("http")]

    processed_urls = load_processed_urls(weapon, resume_legacy=resume_legacy)
    pending_urls = [u for u in all_urls if u not in processed_urls]
    if max_matches:
        pending_urls = pending_urls[:max_matches]

    print(f"=== Running Preprocessing Pipeline for {weapon.upper()} ===")
    print(f"Total verified URLs: {len(all_urls)}")
    print(f"Already processed: {len(processed_urls)}")
    print(f"Pending to process: {len(pending_urls)}")

    precut_dir = os.path.join(BASE_DIR, f"precut_{weapon}")
    clips_dir = os.path.join(BASE_DIR, "data", "clips", weapon)
    os.makedirs(precut_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    total_new_clips = 0
    reason_counts = Counter()
    rejection_log = rejections_path(weapon)

    for idx, url in enumerate(pending_urls, 1):
        print(f"\n[{idx}/{len(pending_urls)}] Downloading & processing: {url}")
        try:
            match_id = parse_match_id(url)
        except ValueError as e:
            print(f"Skipping: {e}")
            record_failure(weapon, url, f"ValueError: {e}")
            continue

        try:
            fpath, title = download_match(url, match_id, precut_dir,
                                          download_options)
        except DownloadError as e:
            print(f"Failed to download {url}, skipping: {e}")
            record_failure(weapon, url, f"download failed: {e}")
            continue
        if fpath is None:
            print(f"Failed to download {url}, skipping.")
            record_failure(weapon, url, "download failed")
            continue

        try:
            count = process_video_file(
                fpath, clips_dir, weapon=weapon, cfg=cfg,
                rejections_path=rejection_log, reason_counts=reason_counts,
                match_id=match_id, source_url=url,
                tournament=parse_tournament(title),
            )
            total_new_clips += count
        except Exception as e:
            print(f"Error processing {match_id}: {e}")
            record_failure(weapon, url, f"{type(e).__name__}: {e}")
            continue
        finally:
            for leftover in (fpath, os.path.join(precut_dir, f"{match_id}.info.json")):
                if os.path.exists(leftover):
                    os.remove(leftover)

        # Log URL as processed only after a successful run
        mark_processed(weapon, url)
        processed_urls.add(url)

        # Gentle delay between matches to keep household network and YouTube happy
        time.sleep(3.0)

    print("\n==========================================")
    print(f"Pipeline complete for {weapon.upper()}!")
    print(f"Generated {total_new_clips} new clips in {clips_dir}.")
    print(f"Total clips now in {clips_dir}: {len(os.listdir(clips_dir))}")
    print("\nOutcome per candidate exchange (this run):")
    print(format_reason_counts(reason_counts))
    print(f"\nRejections of all runs ({rejection_log}):")
    print(format_reason_counts(read_reason_counts(rejection_log)))
    print("==========================================")


def add_cli_args(parser):
    """Register the command line options shared with 2-cut_and_label.py."""
    parser.add_argument("--weapon", choices=["foil", "sabre", "epee"], default="sabre")
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--pre-s", type=float, default=PRE_S,
                        help="seconds kept before the light onset")
    parser.add_argument("--post-s", type=float, default=POST_S,
                        help="seconds kept after the light onset")
    parser.add_argument("--clip-fps", type=float, default=None,
                        help="fps of saved clips (default: native source fps)")
    parser.add_argument("--on-off-unchanged", choices=ON_OFF_UNCHANGED_MODES,
                        default="priority",
                        help="foil On-Off / Off-On with unchanged score: "
                             "'priority' labels the off-target fencer (default), "
                             "'reject' drops the exchange")
    parser.add_argument("--resume-legacy", action="store_true",
                        help="also skip URLs listed in the progress logs of the "
                             "old pipeline (state/processed_*.txt, "
                             "scripts/processed_*.txt); off by default")
    parser.add_argument("--cookies-browser", choices=COOKIE_BROWSERS,
                        default="none",
                        help="browser to read YouTube cookies from (default: none)")
    parser.add_argument("--cookies-file", default=None,
                        help="Netscape cookie file for yt-dlp (overrides "
                             "--cookies-browser)")
    parser.add_argument("--ca-file", default=None,
                        help="PEM file with the CA certificate(s) to trust "
                             "(for example a proxy CA); default: OS "
                             "certificate store")
    parser.add_argument("--no-check-certificates", action="store_true",
                        help="INSECURE: disable TLS certificate verification "
                             "(opt-in; prefer --ca-file)")
    parser.add_argument("--remote-components", action="store_true",
                        help="let yt-dlp download its JS challenge solver "
                             "(ejs:github); off by default")


def config_from_args(args):
    """Build a PipelineConfig from parsed command line options."""
    return PipelineConfig(pre_s=args.pre_s, post_s=args.post_s,
                          clip_fps=args.clip_fps,
                          on_off_unchanged=args.on_off_unchanged)


def download_options_from_args(args):
    """Build DownloadOptions from parsed command line options."""
    return DownloadOptions(cookies_browser=args.cookies_browser,
                           cookies_file=args.cookies_file,
                           remote_components=args.remote_components,
                           ca_file=args.ca_file,
                           no_check_certificates=args.no_check_certificates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    run_pipeline(args.weapon, args.max_matches, config_from_args(args),
                 download_options_from_args(args),
                 resume_legacy=args.resume_legacy)
