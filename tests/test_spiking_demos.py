"""Headless end-to-end smoke tests for the spiking demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


for script, expected in (
    ("spiking_pattern.py", "spiking_pattern"),
    ("spiking_cartpole.py", "spiking_cartpole"),
    ("spiking_eprop.py", "spiking_eprop"),
):
    result = subprocess.run(
        [sys.executable, str(ROOT / "demos" / script), "--smoke"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["demo"] == expected
    expected_generations = 2 if expected == "spiking_eprop" else 1
    assert payload["generations"] == expected_generations
    if expected == "spiking_eprop":
        assert payload["population"] == 1
        assert payload["optimizer_updates"] == 2
    else:
        assert payload["population"] >= 4
    assert payload["neurons"] > 0
    assert payload["links"] > 0
    assert payload["recorded_spikes"] >= 0


for script, expected in (
    ("xor.py", "xor"),
    ("asteroid_nav.py", "asteroids"),
):
    with tempfile.TemporaryDirectory() as directory:
        command = [
            sys.executable,
            str(ROOT / "demos" / script),
            "--smoke",
            "--spiking",
            "--seed",
            "42",
        ]
        screenshot = Path(directory) / "asteroids-spiking.png"
        if script == "asteroid_nav.py":
            command.extend(("--screenshot", str(screenshot)))
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (
            f"{script} spiking variant failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        assert payload["demo"] == expected
        assert payload["policy"] == "spiking"
        assert payload["generations"] == 1
        assert payload["neurons"] > 0
        assert payload["links"] > 0
        if script == "asteroid_nav.py":
            assert screenshot.stat().st_size > 10_000


physics_wrappers = sorted((ROOT / "demos" / "box2d").glob("*_box2d.py"))
physics_wrappers += sorted((ROOT / "demos" / "mujoco").glob("*_mujoco.py"))
for wrapper in physics_wrappers:
    result = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, f"{wrapper.name} --help failed"
    assert "--spiking" in result.stdout, (
        f"{wrapper.name} does not expose its spiking variant"
    )
