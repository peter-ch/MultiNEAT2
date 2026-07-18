#!/usr/bin/env python3
"""Inspect or smoke-test the complete Box2D/MuJoCo demo catalog."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gymnasium_neat import catalog, inspect_task, smoke_task


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all registered Gymnasium MultiNEAT demos."
    )
    parser.add_argument(
        "--family",
        choices=("box2d", "mujoco"),
        help="limit the suite to one optional dependency family",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inspect",
        action="store_true",
        help="construct/reset each environment and report policy dimensions",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="evolve four genomes for one three-step generation per task",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configs = list(catalog(args.family))
    if not (args.inspect or args.smoke):
        for config in configs:
            print(f"{config.key:42} {config.env_id:28} {config.description}")
        return 0

    failures = []
    for index, config in enumerate(configs):
        try:
            if args.inspect:
                shape = inspect_task(config, args.seed + index)
                print(
                    json.dumps(
                        {
                            "task": config.key,
                            "environment": config.env_id,
                            **asdict(shape),
                        },
                        sort_keys=True,
                    )
                )
            else:
                result = smoke_task(
                    config.key,
                    seed=args.seed + index,
                    quiet=args.quiet,
                )
                print(
                    f"PASS {config.key} best={result.best_fitness:.6g}",
                    flush=True,
                )
        except Exception as error:  # Continue to report the entire suite.
            failures.append((config.key, str(error)))
            print(f"FAIL {config.key}: {error}", flush=True)

    if failures:
        print(f"\n{len(failures)} of {len(configs)} tasks failed:")
        for task, error in failures:
            print(f"- {task}: {error}")
        return 1
    print(f"\nAll {len(configs)} tasks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
