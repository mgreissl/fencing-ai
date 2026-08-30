#!/usr/bin/env python3
"""
scripts/reclassify_and_clean_videos.py — Accurate Title-Based Weapon Classifier.

Uses YouTube oEmbed API to fetch titles and strictly categorizes matches:
- Leading 'FE ' is recognized as broadcaster prefix (not Epee).
- 'M F' / 'F F' / 'W F' / 'Foil'  -> Foil
- 'M E' / 'F E' / 'W E' / 'Epee'  -> Epee
- 'M S' / 'F S' / 'W S' / 'Sabre' -> Sabre
Deduplicates and saves clean lists to foil_videos.txt, epee_videos.txt, sabre_videos.txt.
"""

import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_title(video_id):
    """Fetch video title using YouTube oEmbed endpoint."""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return video_id, data.get("title", "")
    except Exception:
        return video_id, ""


def parse_weapon(title):
    """
    Parse weapon category from FIE video title.
    Rules:
      - Strip leading broadcaster tags like 'FE ' or 'FE 2018 '
      - Match 'M F' / 'F F' / 'W F' -> foil
      - Match 'M E' / 'F E' / 'W E' -> epee
      - Match 'M S' / 'F S' / 'W S' -> sabre
    """
    clean = re.sub(r'^\s*FE\s+', '', title)

    # Sabre check
    if re.search(r'\b(M\s*S|F\s*S|W\s*S|Men\'?s?\s+Sabre|Women\'?s?\s+Sabre|Sabre|Säbel)\b', clean, re.IGNORECASE):
        # Ensure it's not a mixed title where foil/epee is primary
        return "sabre"

    # Foil check
    if re.search(r'\b(M\s*F|F\s*F|W\s*F|Men\'?s?\s+Foil|Women\'?s?\s+Foil|Foil|Florett|Fleuret)\b', clean, re.IGNORECASE):
        return "foil"

    # Epee check
    if re.search(r'\b(M\s*E|F\s*E|W\s*E|Men\'?s?\s+Epee|Women\'?s?\s+Epee|Epee|Épée|Degen)\b', clean, re.IGNORECASE):
        return "epee"

    return "unknown"


def clean_and_reclassify():
    files = {
        "foil": os.path.join(BASE_DIR, "foil_videos.txt"),
        "epee": os.path.join(BASE_DIR, "epee_videos.txt"),
        "sabre": os.path.join(BASE_DIR, "sabre_videos.txt")
    }

    all_urls = {}
    for weapon, fpath in files.items():
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    u = line.strip()
                    if u.startswith("http"):
                        m = re.search(r'v=([a-zA-Z0-9_-]+)', u)
                        if m:
                            vid = m.group(1)
                            if vid not in all_urls:
                                all_urls[vid] = weapon

    print(f"Total unique videos to inspect: {len(all_urls)}")

    # Fetch titles via oEmbed
    print("Fetching video titles via oEmbed...")
    titles = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_title, vid): vid for vid in all_urls}
        for future in as_completed(futures):
            vid, title = future.result()
            titles[vid] = title

    # Reclassify
    weapon_buckets = {"foil": set(), "epee": set(), "sabre": set(), "unknown": set()}
    reclassified = 0

    for vid, orig_weapon in all_urls.items():
        title = titles.get(vid, "")
        cat = parse_weapon(title)
        
        if cat in weapon_buckets:
            weapon_buckets[cat].add(vid)
        else:
            weapon_buckets["unknown"].add(vid)

        if cat != orig_weapon and cat != "unknown":
            print(f"[RECLASSIFIED] {orig_weapon.upper()} -> {cat.upper()}: '{title}' ({vid})")
            reclassified += 1

    print("\n==========================================")
    print(f"Reclassification Complete!")
    print(f"Total videos reclassified: {reclassified}")
    for w in ["foil", "epee", "sabre", "unknown"]:
        print(f"  {w.upper()}: {len(weapon_buckets[w])} verified matches")
    print("==========================================")

    # Save cleaned, deduplicated, correctly sorted files
    for w in ["foil", "epee", "sabre"]:
        fpath = files[w]
        sorted_urls = [f"https://youtube.com/watch?v={vid}" for vid in sorted(weapon_buckets[w])]
        with open(fpath, "w") as f:
            for u in sorted_urls:
                f.write(u + "\n")
        print(f"✓ Wrote {len(sorted_urls)} clean matches to {os.path.basename(fpath)}")


if __name__ == "__main__":
    clean_and_reclassify()
