#!/usr/bin/env python3
"""
1-download_vids.py — Download match videos in 360p directly using yt-dlp.

Usage:
    python 1-download_vids.py --weapon foil --limit 10
    python 1-download_vids.py --weapon sabre
"""

import argparse
import os
import subprocess as sp

DIRECTORIES = ["precut", "final_training_clips", "more_training_data"]
for d in DIRECTORIES:
    os.makedirs(d, exist_ok=True)


def download_video(url, output_dir):
    """Download a match video in 360p mp4 format."""
    cmd = [
        "yt-dlp",
        "-f", "134/bestvideo[height<=360]",
        "--no-warnings",
        "-o", os.path.join(output_dir, "%(title)s.%(ext)s"),
        url
    ]
    res = sp.run(cmd)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download fencing match videos for preprocessing")
    parser.add_argument("--weapon", choices=["foil", "sabre"], default="foil", help="Weapon type")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of videos to download")
    args = parser.parse_args()

    vids_file = f"{args.weapon}_videos.txt"
    if not os.path.exists(vids_file):
        print(f"Error: {vids_file} does not exist.")
        return

    with open(vids_file) as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]

    if args.limit:
        urls = urls[:args.limit]

    print(f"Downloading {len(urls)} videos for {args.weapon} to precut/...")
    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] {url}")
        download_video(url, "precut")

    print("Download complete!")


if __name__ == "__main__":
    main()
