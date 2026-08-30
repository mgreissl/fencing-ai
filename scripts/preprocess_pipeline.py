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
import os
import re
import cv2
import pickle
import subprocess as sp
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Template images for light detection (640x360 coordinates)
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
GREEN_BOX = cv2.imread(os.path.join(RESOURCES_DIR, "greenbox.png"))
RED_BOX = cv2.imread(os.path.join(RESOURCES_DIR, "redbox.png"))
WHITE_BOX = cv2.imread(os.path.join(RESOURCES_DIR, "whitebox.png"))

# Load logistic regression digit classifier (trained for scoreboard numbers 0-15)
CLASSIFIER_PATH = os.path.join(RESOURCES_DIR, "logistic_classifier_0-15.pkl")
with open(CLASSIFIER_PATH, "rb") as fd:
    SCORE_MODEL = pickle.load(fd)
SCORE_MODEL.classes_ = np.arange(16)

FFMPEG_BIN = "ffmpeg"
FPS = 13
JUMP_LENGTH = 260
HIDE_LENGTH = 200


def check_lights(frame, weapon="foil"):
    """Check whether left/right on-target or off-target lights are lit."""
    string = ""
    # Left on target (red)
    if np.sum(abs(frame[330:334, 140:260].astype(int) - RED_BOX.astype(int))) <= 40000:
        string += "On"
    elif weapon == "foil" and np.sum(abs(frame[337:348, 234:250].astype(int) - WHITE_BOX.astype(int))) <= 7000:
        string += "Off"
    else:
        string += "No"

    string += "-"
    # Right on target (green)
    if np.sum(abs(frame[330:334, 380:500].astype(int) - GREEN_BOX.astype(int))) <= 40000:
        string += "On"
    elif weapon == "foil" and np.sum(abs(frame[337:348, 390:406].astype(int) - WHITE_BOX.astype(int))) <= 7000:
        string += "Off"
    else:
        string += "No"

    return string


def check_score(frame):
    """Predict left and right fencer scores from scoreboard crops."""
    left_crop = frame[309:325, 265:285].reshape(1, -1)
    right_crop = frame[309:325, 355:375].reshape(1, -1)
    left_score = int(SCORE_MODEL.predict(left_crop)[0])
    right_score = int(SCORE_MODEL.predict(right_crop)[0])
    return left_score, right_score


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
    cap.release()

    if total_frames < 2500:
        print(f"Video {vid_name} too short ({total_frames} frames), skipping.")
        return 0

    cap_end_point = total_frames - JUMP_LENGTH
    position = 1500  # Skip intro
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
            is_light = (
                (weapon == "foil" and (
                    np.sum(abs(frame[337:348, 234:250].astype(int) - WHITE_BOX.astype(int))) <= 7000 or
                    np.sum(abs(frame[337:348, 390:406].astype(int) - WHITE_BOX.astype(int))) <= 7000
                )) or
                np.sum(abs(frame[330:334, 380:500].astype(int) - GREEN_BOX.astype(int))) <= 40000 or
                np.sum(abs(frame[330:334, 140:260].astype(int) - RED_BOX.astype(int))) <= 40000
            )
        except Exception:
            is_light = False

        if is_light:
            out_clip = os.path.join(temp_clips_dir, f"temp_{vid_name}_{clip_count}.mp4")
            clip_count += 1
            raw_clips.append(out_clip)

            # Extract 260 frames (~10s) starting 200 frames before the light
            clip_start_frame = max(0, position - HIDE_LENGTH)
            cmd = [
                FFMPEG_BIN, "-y",
                "-ss", str(clip_start_frame / 25.0),
                "-i", vid_path,
                "-t", str(JUMP_LENGTH / 25.0),
                "-r", str(FPS),
                "-s", "640x360",
                "-c:v", "libx264",
                "-an",
                out_clip
            ]
            sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

            # Jump forward
            position += JUMP_LENGTH
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
        try:
            left_1, right_1 = check_score(c_frames[25])
            left_2, right_2 = check_score(c_frames[35])
        except Exception:
            continue

        if left_1 != left_2 or right_1 != right_2:
            continue

        left_score, right_score = left_1, right_1

        # Score after touch (around frame 120..130)
        if len(c_frames) < 130:
            continue

        try:
            n_left_1, n_right_1 = check_score(c_frames[120])
            n_left_2, n_right_2 = check_score(c_frames[125])
        except Exception:
            continue

        if n_left_1 != n_left_2 or n_right_1 != n_right_2:
            continue

        next_left, next_right = n_left_1, n_right_1

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
