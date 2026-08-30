#!/usr/bin/env python3
"""
2-cut_and_label.py — Automated Hit Cutting, Referee Labelling & Downsampling.

Supports Foil, Sabre, and Epee.

Usage:
    python 2-cut_and_label.py --weapon sabre
    python 2-cut_and_label.py --weapon foil
    python 2-cut_and_label.py --weapon epee
"""

import argparse
from scripts.preprocess_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weapon", choices=["foil", "sabre", "epee"], default="sabre")
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()
    run_pipeline(weapon=args.weapon, max_matches=args.max_matches)
