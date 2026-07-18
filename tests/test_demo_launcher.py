#!/usr/bin/env python3
"""Headless tests for the root Tk demo launcher's catalog and commands."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "multineat_demo_launcher", ROOT / "demos.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load demos.py")
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


check(len(launcher.DEMOS) == 23, "launcher should expose 23 unique demos")
check(
    len(launcher.DEMO_BY_ID) == len(launcher.DEMOS),
    "launcher demo IDs must be unique",
)
check(
    {item.family for item in launcher.DEMOS}
    == {"Core", "Spiking", "Games", "Box2D", "MuJoCo"},
    "launcher families are incomplete",
)
for demo in launcher.DEMOS:
    check((ROOT / demo.script).is_file(), f"missing demo script: {demo.script}")
    check(
        (ROOT / demo.requirement).is_file(),
        f"missing requirements file: {demo.requirement}",
    )

runtime = launcher.RuntimeInfo(
    launcher.RuntimeCandidate(Path(sys.executable), None, "test"),
    "test",
    True,
    "test",
)

for demo in launcher.DEMOS:
    command, output = launcher.build_demo_command(
        demo,
        runtime,
        launcher.LaunchOptions(mode="smoke"),
        timestamp="test",
    )
    check(command[0] == sys.executable, "command uses the wrong interpreter")
    check(command[-1] == "--smoke", f"{demo.id} has no safe smoke command")
    check(output is None, "smoke commands should not create run directories")

physics = launcher.DEMO_BY_ID["mujoco_reacher"]
options = launcher.LaunchOptions(
    mode="run",
    population=12,
    generations=3,
    max_steps=17,
    episodes=2,
    workers=4,
    seed=9,
    profile="exploratory",
    render=True,
    plot=True,
    record_video=True,
    output_base=ROOT / "runs",
)
command, output = launcher.build_demo_command(
    physics, runtime, options, timestamp="fixed"
)
expected_pairs = {
    "--population": "12",
    "--generations": "3",
    "--max-steps": "17",
    "--episodes": "2",
    "--workers": "4",
    "--seed": "9",
    "--profile": "exploratory",
    "--render-every": "10",
}
for flag, value in expected_pairs.items():
    index = command.index(flag)
    check(command[index + 1] == value, f"wrong value generated for {flag}")
check("--plot" in command, "plot flag was lost")
check("--record-video" in command, "video flag was lost")
check(output == ROOT / "runs" / "mujoco_reacher-fixed", "wrong output path")

xor_command, _ = launcher.build_demo_command(
    launcher.DEMO_BY_ID["xor"],
    runtime,
    launcher.LaunchOptions(mode="run", render=False),
)
check("--no-show" in xor_command, "headless XOR command should suppress its plot")

spiking_command, _ = launcher.build_demo_command(
    launcher.DEMO_BY_ID["spiking_cartpole"],
    runtime,
    launcher.LaunchOptions(mode="run", render=True, max_steps=33),
)
check("--animate" in spiking_command, "spiking demo should enable animation")
check(
    spiking_command[spiking_command.index("--max-steps") + 1] == "33",
    "spiking Cart-Pole should receive its control horizon",
)

physics_spiking_command, physics_spiking_output = launcher.build_demo_command(
    launcher.DEMO_BY_ID["mujoco_reacher"],
    runtime,
    launcher.LaunchOptions(
        mode="run",
        spiking=True,
        output_base=ROOT / "runs",
    ),
    timestamp="spikes",
)
check(
    "--spiking" in physics_spiking_command,
    "physics demos should expose their spiking policy variant",
)
check(
    physics_spiking_output
    == ROOT / "runs" / "mujoco_reacher-spiking-spikes",
    "spiking runs should use a distinct output folder",
)
spiking_render_modules = launcher.required_modules(
    launcher.DEMO_BY_ID["mujoco_reacher"],
    launcher.LaunchOptions(mode="run", spiking=True, render=True),
)
check(
    {"matplotlib", "networkx"}.issubset(spiking_render_modules),
    "spiking physics rendering dependencies are not preflighted",
)

xor_spiking_command, _ = launcher.build_demo_command(
    launcher.DEMO_BY_ID["xor"],
    runtime,
    launcher.LaunchOptions(mode="smoke", spiking=True),
)
check(
    "--spiking" in xor_spiking_command and "--smoke" in xor_spiking_command,
    "XOR spiking smoke command is incomplete",
)

asteroid_command, _ = launcher.build_demo_command(
    launcher.DEMO_BY_ID["asteroids"],
    runtime,
    launcher.LaunchOptions(mode="run", render=False),
)
check(
    "--headless" in asteroid_command,
    "headless Asteroids command should suppress its window",
)
asteroid_spiking_command, _ = launcher.build_demo_command(
    launcher.DEMO_BY_ID["asteroids"],
    runtime,
    launcher.LaunchOptions(mode="run", render=True, spiking=True),
)
check(
    "--spiking" in asteroid_spiking_command
    and "--headless" not in asteroid_spiking_command,
    "visible Asteroids should launch its live spiking policy panel",
)

print("Demo launcher tests passed.")
