#!/usr/bin/env python3
"""7-evaluate_AI.py - deprecated wrapper, use `python -m src evaluate`."""

import sys

from src.cli import main

if __name__ == "__main__":
    print("DEPRECATED: use 'python -m src evaluate' instead of 7-evaluate_AI.py.", file=sys.stderr)
    sys.exit(main(["evaluate", *sys.argv[1:]]))
