#!/usr/bin/env python3
"""Evolve a Walker2d-v5 locomotion controller."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymnasium_neat import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("mujoco_walker2d"))
