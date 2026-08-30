#!/usr/bin/env python3
"""
7-evaluate_AI.py — Evaluate the trained Fencing Referee Model on Test Clips.

Usage:
    python 7-evaluate_AI.py
    python 7-evaluate_AI.py eval.checkpoint=checkpoints/best_model.pt
"""

import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "src.evaluate"] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
