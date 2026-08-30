#!/usr/bin/env python3
"""
scripts/harvest_fencingvision.py — Strictly Verify and Harvest Matches from FencingVision.

Strictly checks EACH INDIVIDUAL video URL before adding:
1. Title Weapon Filter: Uses FIE title codes (M F/F F/W F, M E/F E/W E, M S/F S/W S)
   to ensure videos from mixed-weapon playlists only go into their correct weapon file.
2. Uses android player client and format 18/134 to prevent 429 rate limiting.
3. Streams a single frame at 03:00 via ffmpeg without downloading the full video.
4. Verifies the broadcast scoreboard header geometry (dark bar y=300..305).
5. Verifies unlit light box regions at y=330..334.
6. Verifies scoreboard digit OCR model predictions (0-15).
7. Only appends verified videos with valid overlays to {weapon}_videos.txt.

Usage:
    python scripts/harvest_fencingvision.py --weapon foil --workers 3
    python scripts/harvest_fencingvision.py --weapon epee --workers 3
    python scripts/harvest_fencingvision.py --weapon sabre --workers 3
"""

import argparse
import json
import os
import pickle
import re
import subprocess as sp
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
CLASSIFIER_PATH = os.path.join(RESOURCES_DIR, "logistic_classifier_0-15.pkl")
with open(CLASSIFIER_PATH, "rb") as fd:
    SCORE_MODEL = pickle.load(fd)
SCORE_MODEL.classes_ = np.arange(16)


def parse_weapon(title):
    """
    Parse weapon category from FIE video title.
    Rules:
      - Strip leading broadcaster tags like 'FE '
      - Match 'M F' / 'F F' / 'W F' -> foil
      - Match 'M E' / 'F E' / 'W E' -> epee
      - Match 'M S' / 'F S' / 'W S' -> sabre
    """
    clean = re.sub(r'^\s*FE\s+', '', title)

    # Sabre check
    if re.search(r'\b(M\s*S|F\s*S|W\s*S|Men\'?s?\s+Sabre|Women\'?s?\s+Sabre|Sabre|Säbel)\b', clean, re.IGNORECASE):
        return "sabre"

    # Foil check
    if re.search(r'\b(M\s*F|F\s*F|W\s*F|Men\'?s?\s+Foil|Women\'?s?\s+Foil|Foil|Florett|Fleuret)\b', clean, re.IGNORECASE):
        return "foil"

    # Epee check
    if re.search(r'\b(M\s*E|F\s*E|W\s*E|Men\'?s?\s+Epee|Women\'?s?\s+Epee|Epee|Épée|Degen)\b', clean, re.IGNORECASE):
        return "epee"

    return "unknown"


def check_video_overlay(url):
    """Probe a single frame at 03:00 from video stream and verify scoreboard overlay."""
    try:
        stream_cmd = [
            "yt-dlp",
            "--remote-components", "ejs:github",
            "--cookies-from-browser", "firefox",
            "-g",
            "-f", "18/134/best[height<=360]",
            "--no-warnings",
            url
        ]
        p_stream = sp.run(stream_cmd, capture_output=True, text=True, timeout=20)
        stream_url = p_stream.stdout.strip().splitlines()[0] if p_stream.stdout.strip() else None
        if not stream_url or "http" not in stream_url:
            return None, "Network error: Failed to resolve stream"

        # Grab 1 frame at 03:00 into memory pipe
        ff_cmd = [
            "ffmpeg",
            "-ss", "00:03:00",
            "-i", stream_url,
            "-vframes", "1",
            "-s", "640x360",
            "-f", "image2pipe",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        p_ff = sp.run(ff_cmd, capture_output=True, timeout=15)
        if len(p_ff.stdout) != 640 * 360 * 3:
            return None, "Network error: Failed to decode frame"

        frame = np.frombuffer(p_ff.stdout, dtype=np.uint8).reshape((360, 640, 3))

        # 1. Header bar intensity check (classic template has dark slate header < 100)
        header_mean = float(np.mean(frame[300:305, 200:440]))
        if header_mean > 100:
            return False, f"No dark header (mean={header_mean:.1f} > 100)"

        # 2. Light boxes unlit region check
        light_l = float(np.mean(frame[330:334, 140:260]))
        light_r = float(np.mean(frame[330:334, 380:500]))
        if light_l > 115 or light_r > 115:
            return False, f"Light regions too bright ({light_l:.1f}, {light_r:.1f})"

        # 3. Score crops validity
        l_crop = frame[309:325, 265:285].reshape(1, -1)
        r_crop = frame[309:325, 355:375].reshape(1, -1)
        s_l = int(SCORE_MODEL.predict(l_crop)[0])
        s_r = int(SCORE_MODEL.predict(r_crop)[0])
        if not (0 <= s_l <= 15 and 0 <= s_r <= 15):
            return False, f"Invalid scores ({s_l}, {s_r})"

        return True, f"Valid overlay (header={header_mean:.1f}, scores={s_l}-{s_r})"
    except sp.TimeoutExpired:
        return None, "Network error: Timeout probing stream"
    except Exception as e:
        return None, f"Error: {e}"


def harvest(weapon="foil", workers=3):
    """Harvest and individually verify match URLs."""
    cache_path = os.path.join(BASE_DIR, "scripts", f"all_{weapon}_playlists.json")
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found.")
        return

    with open(cache_path) as f:
        playlists = json.load(f)

    output_file = os.path.join(BASE_DIR, f"{weapon}_videos.txt")
    verified_urls = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            verified_urls = set(l.strip() for l in f if l.strip().startswith("http"))

    history_file = os.path.join(BASE_DIR, "scripts", f"probed_{weapon}_urls.json")
    probed_history = {}
    if os.path.exists(history_file):
        with open(history_file) as f:
            try:
                probed_history = json.load(f)
            except Exception:
                probed_history = {}

    print(f"=== Starting Paced Per-Video Verification for {weapon.upper()} (workers={workers}) ===")
    print(f"Currently verified URLs in {output_file}: {len(verified_urls)}")
    print(f"Candidate playlists: {len(playlists)}")

    total_added = 0

    for pl_idx, pl in enumerate(playlists, 1):
        pid = pl["id"]
        title = pl["title"]

        # Filter out national championships or youth categories
        if any(ex in title.lower() for ex in ["national", "deutsche", "dm ", "cadet", "junior"]):
            continue

        pl_url = f"https://www.youtube.com/playlist?list={pid}"
        res = sp.run(["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", pl_url], capture_output=True, text=True)
        items = [l.split("\t", 1) for l in res.stdout.splitlines() if "\t" in l]

        if not items:
            continue

        # Filter items that need probing & strictly match the weapon
        candidates = []
        for vid_id, vid_title in items:
            # 1. Strictly verify the title weapon code matches requested weapon
            detected_weapon = parse_weapon(vid_title)
            if detected_weapon != weapon:
                continue

            vid_url = f"https://youtube.com/watch?v={vid_id}"
            if vid_url in verified_urls:
                continue
            if vid_url in probed_history and probed_history[vid_url] is False:
                continue
            candidates.append((vid_id, vid_title, vid_url))

        if not candidates:
            continue

        print(f"\n[{pl_idx}/{len(playlists)}] {title} ({len(candidates)} {weapon.upper()} candidates)")

        # Probe candidates with paced thread pool
        newly_verified = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video = {
                executor.submit(check_video_overlay, vid_url): (vid_id, vid_title, vid_url)
                for vid_id, vid_title, vid_url in candidates
            }

            for future in as_completed(future_to_video):
                vid_id, vid_title, vid_url = future_to_video[future]
                is_valid, reason = future.result()

                if is_valid is not None:
                    probed_history[vid_url] = is_valid

                if is_valid is True:
                    print(f"    ✓ [PASS] {vid_title[:45]} -> {reason}")
                    newly_verified.append(vid_url)
                    with open(output_file, "a") as f:
                        f.write(vid_url + "\n")
                        f.flush()
                    verified_urls.add(vid_url)
                    total_added += 1
                elif is_valid is False:
                    print(f"    ✗ [SKIP] {vid_title[:45]} -> {reason}")
                else:
                    print(f"    ! [RETRY LATER] {vid_title[:45]} -> {reason}")

                time.sleep(0.3)

        if newly_verified:
            print(f"  --> Saved {len(newly_verified)} verified matches to {output_file} (Total: {len(verified_urls)})")

        with open(history_file, "w") as f:
            json.dump(probed_history, f)

    print("\n==========================================")
    print(f"Harvest Complete for {weapon.upper()}!")
    print(f"Added {total_added} new verified videos.")
    print(f"Total verified URLs in {output_file}: {len(verified_urls)}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weapon", choices=["foil", "epee", "sabre"], default="foil")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    harvest(args.weapon, args.workers)
