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
from scripts.preprocess_pipeline import add_cli_args, config_from_args, run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args()
    run_pipeline(
        weapon=args.weapon,
        max_matches=args.max_matches,
        cfg=config_from_args(args),
    )
