#!/usr/bin/env python3
"""List, inspect or smoke-test every standalone Box2D/MuJoCo demo.

Each demo in ``demos/box2d`` and ``demos/mujoco`` is a self-contained script,
so this runner simply executes them as subprocesses:

    python demos/run_gymnasium_suite.py                      # list the demos
    python demos/run_gymnasium_suite.py --inspect            # spaces + network sizes
    python demos/run_gymnasium_suite.py --family box2d --smoke
    python demos/run_gymnasium_suite.py --smoke --spiking    # or --mcculloch-pitts
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

DEMOS = Path(__file__).resolve().parent
FAMILIES = ("box2d", "mujoco")


def demo_scripts(family: str | None = None) -> list[Path]:
    scripts: list[Path] = []
    for name in FAMILIES:
        if family in (None, name):
            scripts += sorted((DEMOS / name).glob(f"*_{name}.py"))
    return scripts


def describe(script: Path) -> tuple[str, str]:
    """Read ENV_ID and DESCRIPTION from a demo without importing it."""

    values = {}
    for node in ast.parse(script.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"ENV_ID", "DESCRIPTION"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values.get("ENV_ID", "?"), values.get("DESCRIPTION", "")


def run(script: Path, flags: list[str], timeout: float) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, str(script), *flags],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout:.0f}s"
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        return False, lines[-1] if lines else f"exit code {result.returncode}"
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if "--inspect" in flags:
        return True, json.dumps(json.loads(result.stdout), sort_keys=True)
    summary = json.loads(lines[-1])
    return True, (
        f"best={summary['best_fitness']:.6g} policy={summary['policy']} "
        f"interactions={summary['environment_interactions']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=FAMILIES, help="limit to one dependency family")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="run every demo with --inspect")
    mode.add_argument("--smoke", action="store_true", help="run every demo with --smoke")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spiking", action="store_true", help="smoke spiking policies")
    parser.add_argument("--mcculloch-pitts", action="store_true",
                        help="smoke McCulloch-Pitts policies")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds per demo")
    args = parser.parse_args()

    scripts = demo_scripts(args.family)
    if not (args.inspect or args.smoke):
        for script in scripts:
            env_id, description = describe(script)
            print(f"{script.relative_to(DEMOS).as_posix():44} {env_id:28} {description}")
        return 0

    failures = []
    for index, script in enumerate(scripts):
        flags = ["--inspect" if args.inspect else "--smoke", "--seed", str(args.seed + index)]
        if args.smoke and args.spiking:
            flags.append("--spiking")
        if args.smoke and args.mcculloch_pitts:
            flags.append("--mcculloch-pitts")
        ok, message = run(script, flags, args.timeout)
        print(f"{'PASS' if ok else 'FAIL'} {script.name}: {message}", flush=True)
        if not ok:
            failures.append(script.name)

    if failures:
        print(f"\n{len(failures)} of {len(scripts)} demos failed: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(scripts)} demos passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
