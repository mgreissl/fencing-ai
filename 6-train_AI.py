#!/usr/bin/env python3
"""
6-train_AI.py — Train the modern VideoMAE + LoRA Fencing Referee Model.

Usage:
    python 6-train_AI.py
    python 6-train_AI.py training.epochs=30 training.batch_size=8
"""

import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "src.train"] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
