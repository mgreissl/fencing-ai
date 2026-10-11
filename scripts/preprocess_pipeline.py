#!/usr/bin/env python3
"""
scripts/preprocess_pipeline.py — Multi-Weapon Preprocessing Pipeline.

Processes match video URLs from {weapon}_videos.txt:
1. Downloads match video in 360p via yt-dlp (with android client fallback).
2. Scans the video once: light state per frame, scoreboard digits at a stride.
3. Auto-labels referee decision (L, R, T) from scoreboard digit OCR changes.
4. Saves clips aligned to the light onset (PRE_S before to POST_S after, at
   the source fps) into data/clips/{weapon}/.
5. Removes match video immediately to conserve disk space.
"""

import argparse
import math
import os
import subprocess as sp
import sys
import time
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


def determine_label(hit_type, left, right, next_left, next_right, weapon="foil"):
    """Determine L, R, or T priority label based on scoreboard change."""
    if hit_type == "On-On":
        if next_left - left == 1 and next_right - right == 0:
            return "L"
        if next_left - left == 0 and next_right - right == 1:
            return "R"
        if next_left - left == 0 and next_right - right == 0:
            return "T"
        if weapon == "epee" and next_left - left == 1 and next_right - right == 1:
            return "T"
    elif hit_type == "On-No":
        if next_left - left == 1 and next_right - right == 0:
            return "L"
    elif hit_type == "No-On":
        if next_left - left == 0 and next_right - right == 1:
            return "R"
    elif hit_type == "On-Off":
        if next_left - left == 1 and next_right - right == 0:
            return "L"
        if next_left - left == 0 and next_right - right == 0:
            return "R"
    elif hit_type == "Off-On":
        if next_left - left == 0 and next_right - right == 1:
            return "R"
        if next_left - left == 0 and next_right - right == 0:
            return "L"
    return None


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
    before_offsets_s: tuple = (-1.0, -0.5)  # score read before the onset
    after_offsets_s: tuple = (1.2, 1.6)     # score read after the onset


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


def score_near(timeline, frame_index):
    """Score sample closest to ``frame_index`` (None if unreadable/absent)."""
    if len(timeline.score_frames) == 0:
        return None
    pos = int(np.argmin(np.abs(timeline.score_frames - frame_index)))
    return timeline.scores[pos]


@dataclass
class Candidate:
    """One detected exchange with its outcome."""

    onset_frame: int
    hit_type: str
    label: str
    score_before: tuple
    score_after: tuple
    window: Window


def analyse_timeline(timeline, weapon, cfg=None):
    """Turn a scanned timeline into labelled candidates (pure, no IO)."""
    cfg = cfg or PipelineConfig()
    fps = timeline.fps
    onsets = find_light_onsets(
        timeline.lit,
        seconds_to_frames(cfg.min_dark_s, fps),
        first_valid=timeline.first_valid,
    )
    candidates = []
    for onset in onsets:
        hit_end = onset + max(1, seconds_to_frames(cfg.hit_window_s, fps))
        hit_type = aggregate_hit_type(
            timeline.left_state[onset:hit_end], timeline.right_state[onset:hit_end]
        )
        if hit_type not in ("On-On", "On-No", "No-On", "On-Off", "Off-On"):
            continue

        before = [score_near(timeline, onset + seconds_to_frames(o, fps))
                  for o in cfg.before_offsets_s]
        after = [score_near(timeline, onset + seconds_to_frames(o, fps))
                 for o in cfg.after_offsets_s]
        if before[0] is None or any(b != before[0] for b in before):
            continue
        if after[0] is None or any(a != after[0] for a in after):
            continue

        label = determine_label(hit_type, before[0][0], before[0][1],
                                after[0][0], after[0][1], weapon=weapon)
        if label is None:
            continue
        window = compute_window(onset, fps, cfg.pre_s, cfg.post_s,
                                total_frames=timeline.n_frames,
                                clip_fps=cfg.clip_fps)
        if window is None:
            continue
        candidates.append(Candidate(onset, hit_type, label, before[0], after[0], window))
    return candidates


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


def process_video_file(vid_path, output_dir, weapon="foil", cfg=None):
    """Cut and label exchanges from a single match video."""
    cfg = cfg or PipelineConfig()
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]

    timeline = scan_video(vid_path, weapon, cfg)
    if timeline is None:
        print(f"Video {vid_name} has no valid fps, skipping.")
        return 0
    if timeline.n_frames < seconds_to_frames(MIN_VIDEO_S, timeline.fps):
        print(f"Video {vid_name} too short ({timeline.n_frames} frames), skipping.")
        return 0

    candidates = analyse_timeline(timeline, weapon, cfg)
    jobs = []
    for n, cand in enumerate(candidates):
        clip_id = f"{cand.label}_{weapon}_{vid_name}_{n}"
        jobs.append((cand.window, os.path.join(output_dir, f"{clip_id}.mp4")))
    saved = write_clips(vid_path, jobs)

    print(f"  Successfully labelled {len(saved)} {weapon.upper()} clips.")
    return len(saved)


def run_pipeline(weapon="sabre", max_matches=None, cfg=None):
    """Main pipeline execution loop."""
    cfg = cfg or PipelineConfig()
    urls_file = os.path.join(BASE_DIR, f"{weapon}_videos.txt")
    if not os.path.exists(urls_file):
        print(f"Error: {urls_file} not found.")
        return

    with open(urls_file) as f:
        all_urls = [line.strip() for line in f if line.strip().startswith("http")]

    progress_log = os.path.join(BASE_DIR, "scripts", f"processed_{weapon}_urls.txt")
    processed_urls = set()
    if os.path.exists(progress_log):
        with open(progress_log) as f:
            processed_urls = set(line.strip() for line in f)

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

    for idx, url in enumerate(pending_urls, 1):
        print(f"\n[{idx}/{len(pending_urls)}] Downloading & processing: {url}")
        # Download in 360p via Firefox cookies + Deno JS challenge solver
        cmd = [
            "yt-dlp",
            "--remote-components", "ejs:github",
            "--cookies-from-browser", "firefox",
            "--sleep-requests", "1.5",
            "-f", "18/134/best[height<=360]",
            "--no-warnings",
            "-o", os.path.join(precut_dir, "%(title)s.%(ext)s"),
            url
        ]
        res = sp.run(cmd)
        if res.returncode != 0:
            print(f"Failed to download {url}, skipping.")
            continue

        # Process downloaded video
        for fname in os.listdir(precut_dir):
            if fname.endswith(".mp4"):
                fpath = os.path.join(precut_dir, fname)
                try:
                    count = process_video_file(fpath, clips_dir, weapon=weapon, cfg=cfg)
                    total_new_clips += count
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                finally:
                    if os.path.exists(fpath):
                        os.remove(fpath)

        # Log URL as processed
        with open(progress_log, "a") as f:
            f.write(url + "\n")
        processed_urls.add(url)

        # Gentle delay between matches to keep household network and YouTube happy
        time.sleep(3.0)

    print("\n==========================================")
    print(f"Pipeline complete for {weapon.upper()}!")
    print(f"Generated {total_new_clips} new clips in {clips_dir}.")
    print(f"Total clips now in {clips_dir}: {len(os.listdir(clips_dir))}")
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


def config_from_args(args):
    """Build a PipelineConfig from parsed command line options."""
    return PipelineConfig(pre_s=args.pre_s, post_s=args.post_s,
                          clip_fps=args.clip_fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    run_pipeline(args.weapon, args.max_matches, config_from_args(args))
