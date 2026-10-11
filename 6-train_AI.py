#!/usr/bin/env python3
"""6-train_AI.py - deprecated wrapper, use `python -m src train`."""

import sys

from src.cli import main

if __name__ == "__main__":
    print("DEPRECATED: use 'python -m src train' instead of 6-train_AI.py.", file=sys.stderr)
    sys.exit(main(["train", *sys.argv[1:]]))
