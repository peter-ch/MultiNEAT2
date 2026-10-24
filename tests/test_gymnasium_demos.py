#!/usr/bin/env python3
"""Tests for the standalone Box2D/MuJoCo demos in demos/box2d and demos/mujoco.

Every demo is a self-contained script, so it is exercised as a subprocess:
static checks (compile, no repo-module imports, --help) always run; --inspect
and --smoke run for every task whose physics backend imports.  Exit code 77
skips the dynamic part when gymnasium, numpy or pymultineat is unavailable.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "demos"
BOX2D = sorted((DEMOS / "box2d").glob("*_box2d.py"))
MUJOCO = sorted((DEMOS / "mujoco").glob("*_mujoco.py"))
SCRIPTS = BOX2D + MUJOCO

EXPECTED_BOX2D = {
    "bipedal_walker", "bipedal_walker_hardcore", "car_racing", "car_racing_discrete",
    "car_racing_nofx", "lunar_lander", "lunar_lander_continuous", "lunar_lander_wind",
}
EXPECTED_MUJOCO = {
    "ant", "halfcheetah", "hopper", "humanoid", "humanoid_standup",
    "inverted_double_pendulum", "inverted_pendulum", "pusher", "reacher", "swimmer",
    "walker2d",
}
# Third-party modules a standalone demo may import (besides the stdlib).
ALLOWED_IMPORTS = {"gymnasium", "numpy", "pymultineat", "matplotlib", "imageio"}
FORBIDDEN_NAMES = ("neattools", "spiking_neat", "gymnasium_neat")
# Smoke every policy variant on one discrete, one pixel and one MuJoCo task.
VARIANT_TASKS = {"lunar_lander_box2d.py", "car_racing_box2d.py", "reacher_mujoco.py"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(script: Path, *flags: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.run(
        [sys.executable, str(script), *flags],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def last_json(result: subprocess.CompletedProcess) -> dict:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    check(bool(lines), "no output")
    return json.loads(lines[-1])


# --- Static checks: need nothing but Python ---------------------------------

check(
    {p.name.removesuffix("_box2d.py") for p in BOX2D} == EXPECTED_BOX2D,
    f"unexpected Box2D demos: {[p.name for p in BOX2D]}",
)
check(
    {p.name.removesuffix("_mujoco.py") for p in MUJOCO} == EXPECTED_MUJOCO,
    f"unexpected MuJoCo demos: {[p.name for p in MUJOCO]}",
)
check(not (DEMOS / "gymnasium_neat.py").exists(), "shared gymnasium_neat.py should be gone")

stdlib = set(sys.stdlib_module_names)
for script in SCRIPTS:
    py_compile.compile(str(script), doraise=True)
    source = script.read_text(encoding="utf-8")
    for name in FORBIDDEN_NAMES:
        check(name not in source, f"{script.name} references repo module {name}")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            check(node.level == 0, f"{script.name} uses a relative import")
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            top = module.split(".")[0]
            check(
                top in stdlib or top in ALLOWED_IMPORTS or top == "__future__",
                f"{script.name} imports {module!r}; demos must be self-contained",
            )

for script in SCRIPTS:
    result = run(script, "--help", timeout=30)
    check(result.returncode == 0, f"{script.name} --help failed: {result.stderr}")
    for flag in ("--spiking", "--mcculloch-pitts", "--fast", "--smoke", "--inspect",
                 "--legacy-evolution", "--resume", "--checkpoint-every", "--workers",
                 "--plot", "--record-video", "--output-dir", "--profile"):
        check(flag in result.stdout, f"{script.name} --help does not list {flag}")

listing = subprocess.run(
    [sys.executable, str(DEMOS / "run_gymnasium_suite.py")],
    capture_output=True, text=True, timeout=30, check=False,
)
check(listing.returncode == 0, f"suite listing failed: {listing.stderr}")
check(len(listing.stdout.strip().splitlines()) == len(SCRIPTS), "suite should list every demo")

# --- Dynamic checks: need gymnasium, numpy and pymultineat ------------------

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    import gymnasium  # noqa: F401
    import numpy as np
    import pymultineat  # noqa: F401
except ImportError as error:
    print(f"skipping dynamic demo checks: {error}")
    raise SystemExit(77)


def backend_available(module: str) -> bool:
    try:
        __import__(module)
    except Exception:
        return False
    return True


available = {
    "box2d": backend_available("Box2D"),
    "mujoco": backend_available("mujoco"),
}
runnable = [p for p in SCRIPTS if available[p.parent.name]]
skipped = [p.name for p in SCRIPTS if not available[p.parent.name]]


def load(script: Path):
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script.stem] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


# Encoders/decoders are pure numpy and work without the physics backend.
lander = load(DEMOS / "box2d" / "lunar_lander_box2d.py")
encoded = lander.encode_observation([2.5, -2.5, 0.0, 99.0, 0.0, float("nan"), 0.0, 1.0])
check(np.allclose(encoded, [1.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0]), "bounded scaling")
check(lander.decode_action([-1.0, 0.0, 0.8, 0.1]) == 2, "discrete argmax action")

car = load(DEMOS / "box2d" / "car_racing_box2d.py")
frame = np.zeros((96, 96, 3), dtype=np.uint8)
frame[:, 48:] = 255
pixels = car.encode_observation(frame)
check(len(pixels) == car.NUM_INPUTS - 1 == 144, "CarRacing should pool to 12x12")
check(min(pixels) == -1.0 and max(pixels) == 1.0, "black/white pixels should map to -1/+1")
check(np.allclose(car.decode_action([-1.0, 0.0, 1.0]), [-1.0, 0.5, 1.0]), "box action bounds")

humanoid = load(DEMOS / "mujoco" / "humanoid_mujoco.py")
action = humanoid.decode_action([2.0] * 17)
check(action.shape == (17,) and np.allclose(action, 0.4), "symmetric action bound")
check(np.allclose(humanoid.encode_observation([np.inf, -np.inf, 0.0]), [1, -1, 0]), "tanh")

for script in runnable:
    result = run(script, "--inspect")
    check(result.returncode == 0, f"{script.name} --inspect failed: {result.stderr}")
    shape = json.loads(result.stdout)
    check(shape["network_inputs"] > 1 and shape["network_outputs"] >= 1,
          f"{script.name} reported an empty network")

    variants = [()]
    if script.name in VARIANT_TASKS:
        variants += [("--spiking",), ("--mcculloch-pitts",), ("--legacy-evolution",)]
    for variant in variants:
        result = run(script, "--smoke", *variant)
        check(result.returncode == 0,
              f"{script.name} --smoke {' '.join(variant)} failed: {result.stderr}")
        summary = last_json(result)
        expected_policy = ("mcculloch-pitts" if "--mcculloch-pitts" in variant
                           else "lif" if "--spiking" in variant else "rate")
        check(summary["policy"] == expected_policy, f"{script.name}: wrong policy {summary}")
        check(summary["generations"] == 1 and summary["population"] == 4,
              f"{script.name}: smoke run was not tiny: {summary}")
        check(np.isfinite(summary["best_fitness"]), f"{script.name}: non-finite fitness")

if available["box2d"]:
    script = DEMOS / "box2d" / "lunar_lander_box2d.py"
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "run"
        result = run(script, "--smoke", "--output-dir", str(output))
        check(result.returncode == 0, f"smoke with --output-dir failed: {result.stderr}")
        for name in ("best_genome.txt", "population.state", "metrics.jsonl"):
            check((output / name).is_file(), f"--output-dir did not produce {name}")
        result = run(script, "--smoke", "--resume", str(output))
        check(result.returncode == 0, f"--resume failed: {result.stderr}")
        result = run(script, "--smoke", "--spiking", "--resume", str(output))
        check(result.returncode == 2, "resuming a rate checkpoint as spiking should fail")

if skipped:
    print(f"skipped (physics backend not importable): {', '.join(skipped)}")
print(f"Gymnasium demo tests passed ({len(runnable)} of {len(SCRIPTS)} demos run).")
