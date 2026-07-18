#!/usr/bin/env python3
"""Evolve robust continuous LunarLander-v3 control with wind."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gymnasium_neat import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("box2d_lunar_lander_wind"))
