#!/usr/bin/env python3
"""
scripts/preprocess_pipeline.py — Multi-Weapon Preprocessing Pipeline.

Processes match video URLs from {weapon}_videos.txt:
1. Downloads match video in 360p via yt-dlp (with android client fallback).
2. Detects touches via scoreboard light box triggers (red, green, white).
3. Auto-labels referee decision (L, R, T) from scoreboard digit OCR changes.
4. Saves cut, labelled clips directly into data/clips/{weapon}/.
5. Removes match video immediately to conserve disk space.
"""

import argparse
import math
import os
import sys
import cv2
import subprocess as sp
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src import overlay  # noqa: E402

# Overlay geometry/thresholds: single source of truth in configs/overlay/.
OVERLAY = overlay.load_overlay(overlay.DEFAULT_OVERLAY)

# Digit classifier (npz, pickle fallback) for scoreboard numbers 0-15.
SCORE_CLASSIFIER = overlay.load_digit_classifier(cfg=OVERLAY)

FFMPEG_BIN = "ffmpeg"
FPS = 13
# Raw cut geometry in seconds (frame counts depend on the source fps).
JUMP_S = 10.4   # raw cut length and skip after a light
HIDE_S = 8.0    # raw cut starts this long before the light
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


def process_video_file(vid_path, output_dir, weapon="foil"):
    """Cut and label exchanges from a single match video."""
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = validate_source_fps(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps is None:
        print(f"Video {vid_name} has no valid fps, skipping.")
        return 0

    if total_frames < seconds_to_frames(MIN_VIDEO_S, fps):
        print(f"Video {vid_name} too short ({total_frames} frames), skipping.")
        return 0

    jump_frames = seconds_to_frames(JUMP_S, fps)
    hide_frames = seconds_to_frames(HIDE_S, fps)
    cap_end_point = total_frames - jump_frames
    position = seconds_to_frames(INTRO_S, fps)  # Skip intro
    raw_clips = []
    clip_count = 0

    temp_clips_dir = os.path.join(BASE_DIR, f"temp_cut_clips_{weapon}")
    os.makedirs(temp_clips_dir, exist_ok=True)

    # 1. First pass: Identify light triggers and cut raw snippets
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    while position < cap_end_point:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        position += 1

        if frame.shape[:2] != (360, 640):
            frame = cv2.resize(frame, (640, 360))

        # Check for light illumination
        try:
            is_light = overlay.any_light(frame, OVERLAY, weapon)
        except Exception:
            is_light = False

        if is_light:
            out_clip = os.path.join(temp_clips_dir, f"temp_{vid_name}_{clip_count}.mp4")
            clip_count += 1
            raw_clips.append(out_clip)

            # Extract ~10 s starting ~8 s before the light
            clip_start_frame = max(0, position - hide_frames)
            cmd = [
                FFMPEG_BIN, "-y",
                "-ss", str(frame_to_seconds(clip_start_frame, fps)),
                "-i", vid_path,
                "-t", str(JUMP_S),
                "-r", str(FPS),
                "-s", "640x360",
                "-c:v", "libx264",
                "-an",
                out_clip
            ]
            sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

            # Jump forward
            position += jump_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    cap.release()

    if not raw_clips:
        return 0

    # 2. Second pass: Scoreboard verification & auto-labelling
    labelled_count = 0
    for clip_path in raw_clips:
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
            continue

        c_cap = cv2.VideoCapture(clip_path)
        c_frames = []
        while True:
            r, f = c_cap.read()
            if not r or f is None:
                break
            if f.shape[:2] != (360, 640):
                f = cv2.resize(f, (640, 360))
            c_frames.append(f)
        c_cap.release()

        if len(c_frames) < 100:
            continue

        # Check light type at frame 70..100
        hit_type = "No-No"
        for f in c_frames[70:105]:
            lt = check_lights(f, weapon=weapon)
            if lt in ["On-On", "On-Off", "Off-On", "On-No", "No-On"]:
                hit_type = lt
                break

        if hit_type == "No-No":
            continue

        # Score before touch (frames 20..40)
        before_1 = check_score(c_frames[25])
        before_2 = check_score(c_frames[35])
        if before_1 is None or before_1 != before_2:
            continue
        left_1, right_1 = before_1

        left_score, right_score = left_1, right_1

        # Score after touch (around frame 120..130)
        if len(c_frames) < 130:
            continue

        after_1 = check_score(c_frames[120])
        after_2 = check_score(c_frames[125])
        if after_1 is None or after_1 != after_2:
            continue

        next_left, next_right = after_1

        # Determine label
        label = determine_label(hit_type, left_score, right_score, next_left, next_right, weapon=weapon)
        if label in ["L", "R", "T"]:
            # Final clip: 90 frames preceding touch
            frames_to_keep = c_frames[5:95]
            if len(frames_to_keep) == 90:
                clip_id = f"{label}_{weapon}_{vid_name}_{labelled_count}"
                target_path = os.path.join(output_dir, f"{clip_id}.mp4")
                out_writer = cv2.VideoWriter(
                    target_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    FPS,
                    (640, 360)
                )
                for frame_k in frames_to_keep:
                    out_writer.write(frame_k)
                out_writer.release()
                labelled_count += 1

    # Cleanup temp clips
    for p in raw_clips:
        if os.path.exists(p):
            os.remove(p)

    print(f"  Successfully labelled {labelled_count} {weapon.upper()} clips.")
    return labelled_count


def run_pipeline(weapon="sabre", max_matches=None):
    """Main pipeline execution loop."""
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
                    count = process_video_file(fpath, clips_dir, weapon=weapon)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weapon", choices=["foil", "sabre", "epee"], default="sabre")
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(args.weapon, args.max_matches)
