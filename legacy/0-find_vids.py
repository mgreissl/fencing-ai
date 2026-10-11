#!/usr/bin/env python3
"""
0-find_vids.py — Fetch match video URLs from YouTube playlists using yt-dlp or pytube.

Usage:
    python 0-find_vids.py --weapon foil
    python 0-find_vids.py --weapon sabre
"""

import argparse
import subprocess as sp
import os

PLAYLISTS = {
    "foil": [
        "https://www.youtube.com/playlist?list=PL_pQQho0KExyKIiybGuSbwqhJtyMWWXBZ",
        "https://www.youtube.com/playlist?list=PL_pQQho0KExwQU4aN2RxG5sTYK2OKB4Wb",
        "https://www.youtube.com/playlist?list=PL_pQQho0KExx2pdA0cdzmz4UB3hp3CfzX",
    ],
    "sabre": [
        "https://www.youtube.com/playlist?list=PL_pQQho0KExzE6Y6E8w8rL1g8Z8N_e7Lh",
    ]
}


def fetch_playlist_urls(playlist_url):
    """Extract individual video URLs from a YouTube playlist using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "url",
        playlist_url
    ]
    try:
        res = sp.run(cmd, capture_output=True, text=True, check=True)
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception as e:
        print(f"Warning: yt-dlp failed on {playlist_url}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract video URLs from fencing playlists")
    parser.add_argument("--weapon", choices=["foil", "sabre"], default="foil", help="Target weapon")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    out_file = args.output or f"{args.weapon}_videos.txt"
    playlists = PLAYLISTS.get(args.weapon, [])

    existing_urls = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            existing_urls = set(line.strip() for line in f if line.strip().startswith("http"))

    all_urls = list(existing_urls)
    new_count = 0

    for pl in playlists:
        print(f"Fetching: {pl}")
        urls = fetch_playlist_urls(pl)
        for u in urls:
            if u not in existing_urls:
                all_urls.append(u)
                existing_urls.add(u)
                new_count += 1

    with open(out_file, "w") as f:
        for u in all_urls:
            f.write(u + "\n")

    print(f"Done! {out_file} has {len(all_urls)} total URLs (+{new_count} new).")


if __name__ == "__main__":
    main()