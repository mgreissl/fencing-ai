#!/usr/bin/env python3
"""2-cut_and_label.py - deprecated wrapper, use `python -m src preprocess`."""

import sys

from src.cli import main

if __name__ == "__main__":
    print("DEPRECATED: use 'python -m src preprocess' instead of 2-cut_and_label.py.", file=sys.stderr)
    sys.exit(main(["preprocess", *sys.argv[1:]]))
