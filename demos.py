#!/usr/bin/env python3
"""One-click graphical launcher for every MultiNEAT2 demo.

Run this file from the repository root:

    python demos.py

The launcher deliberately imports no optional project dependency. It silently
selects a compatible ``pymultineat`` build, starts the selected demo
immediately, and streams child-process output into the window.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


@dataclass(frozen=True)
class DemoSpec:
    id: str
    title: str
    family: str
    script: str
    environment: str
    description: str
    modules: tuple[str, ...]
    requirement: str
    population: int
    generations: int
    max_steps: int
    episodes: int = 1


PHYSICS_MODULES = ("gymnasium", "numpy")


DEMOS: tuple[DemoSpec, ...] = (
    DemoSpec(
        "xor",
        "XOR",
        "Core",
        "demos/xor.py",
        "Self-contained",
        "The classic NEAT topology-complexification example.",
        ("matplotlib", "networkx"),
        "requirements-visualization.txt",
        150,
        100,
        1,
    ),
    DemoSpec(
        "asteroids",
        "Asteroids navigation",
        "Games",
        "demos/asteroid_nav.py",
        "PyGame",
        "Evolve sensor-driven ships that survive a moving asteroid field.",
        ("pygame", "numba", "numpy"),
        "requirements-games.txt",
        500,
        50,
        30_000,
    ),
    DemoSpec(
        "box2d_lunar_lander",
        "Lunar Lander — discrete",
        "Box2D",
        "demos/box2d/lunar_lander_box2d.py",
        "LunarLander-v3",
        "Choose among four discrete engine actions.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        160,
        300,
        1000,
    ),
    DemoSpec(
        "box2d_lunar_lander_continuous",
        "Lunar Lander — continuous",
        "Box2D",
        "demos/box2d/lunar_lander_continuous_box2d.py",
        "LunarLander-v3",
        "Continuous main and lateral thruster control.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        160,
        300,
        1000,
    ),
    DemoSpec(
        "box2d_lunar_lander_wind",
        "Lunar Lander — wind",
        "Box2D",
        "demos/box2d/lunar_lander_wind_box2d.py",
        "LunarLander-v3",
        "Robust continuous landing with wind and turbulence.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        192,
        400,
        1000,
        3,
    ),
    DemoSpec(
        "box2d_bipedal_walker",
        "Bipedal Walker",
        "Box2D",
        "demos/box2d/bipedal_walker_box2d.py",
        "BipedalWalker-v3",
        "Four-joint walking on normal terrain.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        192,
        600,
        1600,
    ),
    DemoSpec(
        "box2d_bipedal_walker_hardcore",
        "Bipedal Walker — hardcore",
        "Box2D",
        "demos/box2d/bipedal_walker_hardcore_box2d.py",
        "BipedalWalker-v3",
        "Walking over pits, stumps, ladders, and uneven terrain.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        256,
        1000,
        2000,
        2,
    ),
    DemoSpec(
        "box2d_car_racing",
        "Car Racing — continuous",
        "Box2D",
        "demos/box2d/car_racing_box2d.py",
        "CarRacing-v3",
        "Steering, throttle, and brake from compact pooled pixels.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        192,
        500,
        1000,
    ),
    DemoSpec(
        "box2d_car_racing_discrete",
        "Car Racing — discrete",
        "Box2D",
        "demos/box2d/car_racing_discrete_box2d.py",
        "CarRacing-v3",
        "Discrete car control from compact pooled pixels.",
        PHYSICS_MODULES + ("Box2D",),
        "requirements-box2d.txt",
        160,
        500,
        1000,
    ),
    DemoSpec(
        "mujoco_inverted_pendulum",
        "Inverted Pendulum",
        "MuJoCo",
        "demos/mujoco/inverted_pendulum_mujoco.py",
        "InvertedPendulum-v5",
        "Continuous cart-pole balance.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        96,
        200,
        1000,
    ),
    DemoSpec(
        "mujoco_inverted_double_pendulum",
        "Inverted Double Pendulum",
        "MuJoCo",
        "demos/mujoco/inverted_double_pendulum_mujoco.py",
        "InvertedDoublePendulum-v5",
        "Balance a two-link pendulum on a moving cart.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        128,
        350,
        1000,
    ),
    DemoSpec(
        "mujoco_reacher",
        "Reacher",
        "MuJoCo",
        "demos/mujoco/reacher_mujoco.py",
        "Reacher-v5",
        "Move a two-link arm toward randomized targets.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        160,
        600,
        50,
        3,
    ),
    DemoSpec(
        "mujoco_pusher",
        "Pusher",
        "MuJoCo",
        "demos/mujoco/pusher_mujoco.py",
        "Pusher-v5",
        "Use a seven-actuator arm to push an object to its goal.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        224,
        1000,
        100,
        3,
    ),
    DemoSpec(
        "mujoco_half_cheetah",
        "Half Cheetah",
        "MuJoCo",
        "demos/mujoco/halfcheetah_mujoco.py",
        "HalfCheetah-v5",
        "Fast planar locomotion with six actuators.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        192,
        800,
        1000,
    ),
    DemoSpec(
        "mujoco_hopper",
        "Hopper",
        "MuJoCo",
        "demos/mujoco/hopper_mujoco.py",
        "Hopper-v5",
        "Three-actuator hopping and balance.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        192,
        800,
        1000,
    ),
    DemoSpec(
        "mujoco_walker2d",
        "Walker2d",
        "MuJoCo",
        "demos/mujoco/walker2d_mujoco.py",
        "Walker2d-v5",
        "Six-actuator planar biped locomotion.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        224,
        1000,
        1000,
    ),
    DemoSpec(
        "mujoco_swimmer",
        "Swimmer",
        "MuJoCo",
        "demos/mujoco/swimmer_mujoco.py",
        "Swimmer-v5",
        "Two-actuator locomotion through a fluid medium.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        160,
        600,
        1000,
    ),
    DemoSpec(
        "mujoco_ant",
        "Ant",
        "MuJoCo",
        "demos/mujoco/ant_mujoco.py",
        "Ant-v5",
        "Eight-actuator quadruped locomotion.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        256,
        1200,
        1000,
    ),
    DemoSpec(
        "mujoco_humanoid",
        "Humanoid",
        "MuJoCo",
        "demos/mujoco/humanoid_mujoco.py",
        "Humanoid-v5",
        "Seventeen-actuator humanoid locomotion.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        320,
        1600,
        1000,
    ),
    DemoSpec(
        "mujoco_humanoid_standup",
        "Humanoid Standup",
        "MuJoCo",
        "demos/mujoco/humanoid_standup_mujoco.py",
        "HumanoidStandup-v5",
        "Evolve a humanoid that rises from the ground.",
        PHYSICS_MODULES + ("mujoco",),
        "requirements-mujoco.txt",
        320,
        1600,
        1000,
    ),
)

DEMO_BY_ID = {demo.id: demo for demo in DEMOS}


@dataclass(frozen=True)
class RuntimeCandidate:
    python: Path
    module_dir: Path | None
    label: str

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        if self.module_dir is not None:
            existing = environment.get("PYTHONPATH")
            value = str(self.module_dir)
            environment["PYTHONPATH"] = (
                value + os.pathsep + existing if existing else value
            )
        return environment


@dataclass(frozen=True)
class RuntimeInfo:
    candidate: RuntimeCandidate
    version: str
    modern: bool
    details: str

    @property
    def display(self) -> str:
        state = "ready" if self.modern else "outdated"
        return f"{self.candidate.label} · Python {self.version} · {state}"


@dataclass(frozen=True)
class LaunchOptions:
    mode: str = "smoke"
    population: int = 64
    generations: int = 25
    max_steps: int = 500
    episodes: int = 1
    workers: int = 1
    seed: int = 42
    profile: str = "ranked"
    render: bool = False
    plot: bool = False
    record_video: bool = False
    output_base: Path = ROOT / "runs"


def _cache_value(cache: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:[^=]*=(.*)$")
    try:
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
    except OSError:
        pass
    return None


def _probe_runtime(candidate: RuntimeCandidate) -> RuntimeInfo:
    code = r"""
import json
import sys
try:
    import pymultineat as neat
    params = neat.Parameters()
    modern = (
        hasattr(params, "Validate")
        and hasattr(neat.Population, "Deserialize")
        and hasattr(neat.Genome, "Deserialize")
        and hasattr(neat.NeuralNetwork(), "ActivateBatch")
    )
    result = {
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "modern": bool(modern),
        "details": getattr(neat, "__file__", "pymultineat"),
    }
except Exception as error:
    result = {
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "modern": False,
        "details": f"{type(error).__name__}: {error}",
    }
print(json.dumps(result))
"""
    try:
        result = subprocess.run(
            [str(candidate.python), "-c", code],
            cwd=tempfile.gettempdir(),
            env=candidate.environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}
        return RuntimeInfo(
            candidate,
            payload.get("version", "?"),
            bool(payload.get("modern")),
            payload.get("details") or result.stderr.strip() or "probe failed",
        )
    except Exception as error:
        return RuntimeInfo(candidate, "?", False, f"{type(error).__name__}: {error}")


def discover_runtimes() -> list[RuntimeInfo]:
    candidates: list[RuntimeCandidate] = [
        RuntimeCandidate(Path(sys.executable), None, "Current Python")
    ]
    for cache in sorted(ROOT.glob("build*/**/CMakeCache.txt")):
        executable = _cache_value(cache, "Python3_EXECUTABLE")
        if not executable:
            continue
        python = Path(executable)
        if not python.exists():
            continue
        modules = sorted(cache.parent.rglob("pymultineat*.pyd"))
        modules += sorted(cache.parent.rglob("pymultineat*.so"))
        for module in modules:
            try:
                relative = module.parent.relative_to(ROOT)
                label = str(relative).replace("\\", "/")
            except ValueError:
                label = str(module.parent)
            candidates.append(RuntimeCandidate(python, module.parent, label))

    local_modules = list(ROOT.glob("pymultineat*.pyd"))
    local_modules += list(ROOT.glob("pymultineat*.so"))
    if local_modules:
        candidates.append(
            RuntimeCandidate(Path(sys.executable), ROOT, "Repository binary")
        )

    unique: list[RuntimeCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (
            str(candidate.python.resolve()).lower(),
            str(candidate.module_dir.resolve()).lower()
            if candidate.module_dir
            else "",
        )
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    runtimes = [_probe_runtime(candidate) for candidate in unique]
    runtimes.sort(
        key=lambda runtime: (
            not runtime.modern,
            "build" not in runtime.candidate.label.lower(),
            runtime.candidate.label.lower(),
        )
    )
    return runtimes


def probe_modules(
    runtime: RuntimeInfo,
    modules: Iterable[str],
) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(("pymultineat", *modules)))
    code = (
        "import importlib,json\n"
        f"names={names!r}\n"
        "missing=[]\n"
        "for name in names:\n"
        "    try: importlib.import_module(name)\n"
        "    except Exception as error: missing.append([name, str(error)])\n"
        "print(json.dumps(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(runtime.candidate.python), "-c", code],
            cwd=tempfile.gettempdir(),
            env=runtime.candidate.environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            creationflags=CREATE_NO_WINDOW,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        missing = json.loads(lines[-1]) if lines else [["probe", result.stderr]]
        return tuple(item[0] for item in missing)
    except Exception:
        return ("dependency probe",)


def required_modules(spec: DemoSpec, options: LaunchOptions) -> tuple[str, ...]:
    modules = list(spec.modules)
    if options.mode == "run" and options.plot:
        modules.extend(("matplotlib", "networkx"))
    if options.mode == "run" and options.record_video:
        modules.extend(("imageio", "imageio_ffmpeg"))
    return tuple(dict.fromkeys(modules))


def build_demo_command(
    spec: DemoSpec,
    runtime: RuntimeInfo,
    options: LaunchOptions,
    *,
    timestamp: str | None = None,
) -> tuple[list[str], Path | None]:
    command = [
        str(runtime.candidate.python),
        "-u",
        str(ROOT / spec.script),
    ]
    physics = spec.family in {"Box2D", "MuJoCo"}
    if options.mode == "smoke":
        command.append("--smoke")
        return command, None
    if options.mode == "inspect":
        command.append("--inspect" if physics else "--smoke")
        return command, None

    if spec.id == "xor":
        command.extend(
            [
                "--generations",
                str(options.generations),
                "--population",
                str(options.population),
                "--seed",
                str(options.seed),
            ]
        )
        if not options.render:
            command.append("--no-show")
        return command, None

    if spec.id == "asteroids":
        command.extend(
            [
                "--generations",
                str(options.generations),
                "--population",
                str(options.population),
                "--max-trial-steps",
                str(options.max_steps),
                "--seed",
                str(options.seed),
            ]
        )
        if not options.render:
            command.append("--headless")
        return command, None

    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    output = options.output_base / f"{spec.id}-{stamp}"
    command.extend(
        [
            "--generations",
            str(options.generations),
            "--population",
            str(options.population),
            "--max-steps",
            str(options.max_steps),
            "--episodes",
            str(options.episodes),
            "--workers",
            str(options.workers),
            "--seed",
            str(options.seed),
            "--profile",
            options.profile,
            "--output-dir",
            str(output),
            "--checkpoint-every",
            "10",
        ]
    )
    if options.render:
        command.extend(["--render-every", "10"])
    if options.plot:
        command.append("--plot")
    if options.record_video:
        command.append("--record-video")
    return command, output


def _quote_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_self_test() -> int:
    runtimes = discover_runtimes()
    runtime = next((item for item in runtimes if item.modern), None)
    missing_scripts = [
        spec.script for spec in DEMOS if not (ROOT / spec.script).is_file()
    ]
    payload: dict[str, object] = {
        "demo_count": len(DEMOS),
        "missing_scripts": missing_scripts,
        "runtimes": [
            {
                "display": item.display,
                "modern": item.modern,
                "details": item.details,
            }
            for item in runtimes
        ],
    }
    if runtime is None:
        payload["error"] = "No current pymultineat runtime was found"
        print(json.dumps(payload, indent=2))
        return 1

    missing_by_demo = {}
    commands = {}
    for spec in DEMOS:
        options = LaunchOptions(mode="smoke")
        missing = probe_modules(runtime, required_modules(spec, options))
        if missing:
            missing_by_demo[spec.id] = missing
        command, _ = build_demo_command(spec, runtime, options)
        commands[spec.id] = command
    payload["selected_runtime"] = runtime.display
    payload["missing_dependencies"] = missing_by_demo
    payload["commands_generated"] = len(commands)
    print(json.dumps(payload, indent=2))
    return 1 if missing_scripts or missing_by_demo else 0


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as error:
        print(
            "Tk is not available in this Python installation. "
            "Install Python with Tcl/Tk support.",
            file=sys.stderr,
        )
        print(error, file=sys.stderr)
        return 1

    class Launcher:
        BG = "#0b1220"
        PANEL = "#111827"
        PANEL_2 = "#172033"
        TEXT = "#e5e7eb"
        MUTED = "#94a3b8"
        ACCENT = "#38bdf8"
        GREEN = "#34d399"
        RED = "#fb7185"

        def __init__(self, root: "tk.Tk") -> None:
            self.root = root
            self.events: queue.Queue[tuple] = queue.Queue()
            self.active_process: subprocess.Popen[str] | None = None
            self.busy = False
            self.stop_requested = False
            self.last_output: Path | None = None
            self.runtime_infos = discover_runtimes()
            self.runtime_by_display = {
                runtime.display: runtime for runtime in self.runtime_infos
            }
            self.status_by_demo: dict[str, str] = {}

            root.title("MultiNEAT2 Demo Launcher")
            root.geometry("1180x780")
            root.minsize(980, 650)
            root.configure(background=self.BG)
            root.protocol("WM_DELETE_WINDOW", self.close)
            self._configure_style()
            self._variables()
            self._layout()
            self._populate_tree()
            self._select_initial_demo()
            self._select_runtime()
            self.root.after(100, self._drain_events)

        def _configure_style(self) -> None:
            style = ttk.Style(self.root)
            style.theme_use("clam")
            style.configure(".", font=("Segoe UI", 10))
            style.configure("TFrame", background=self.BG)
            style.configure("Panel.TFrame", background=self.PANEL)
            style.configure(
                "TLabel", background=self.BG, foreground=self.TEXT
            )
            style.configure(
                "Panel.TLabel", background=self.PANEL, foreground=self.TEXT
            )
            style.configure(
                "Muted.TLabel", background=self.PANEL, foreground=self.MUTED
            )
            style.configure(
                "Title.TLabel",
                background=self.PANEL,
                foreground=self.TEXT,
                font=("Segoe UI Semibold", 18),
            )
            style.configure(
                "Header.TLabel",
                background=self.BG,
                foreground=self.TEXT,
                font=("Segoe UI Semibold", 21),
            )
            style.configure(
                "Accent.TButton",
                background=self.ACCENT,
                foreground="#082f49",
                font=("Segoe UI Semibold", 10),
                padding=(14, 8),
            )
            style.map(
                "Accent.TButton",
                background=[("active", "#7dd3fc"), ("disabled", "#334155")],
            )
            style.configure("TButton", padding=(10, 7))
            style.configure(
                "Treeview",
                background=self.PANEL,
                fieldbackground=self.PANEL,
                foreground=self.TEXT,
                rowheight=29,
                borderwidth=0,
            )
            style.map(
                "Treeview",
                background=[("selected", "#075985")],
                foreground=[("selected", "#ffffff")],
            )
            style.configure(
                "Treeview.Heading",
                background=self.PANEL_2,
                foreground=self.TEXT,
                relief="flat",
            )
            style.configure(
                "TEntry",
                fieldbackground=self.PANEL_2,
                foreground=self.TEXT,
                insertcolor=self.TEXT,
            )
            style.configure(
                "TCombobox",
                fieldbackground=self.PANEL_2,
                background=self.PANEL_2,
                foreground=self.TEXT,
                arrowcolor=self.TEXT,
            )
            style.map(
                "TCombobox",
                fieldbackground=[("readonly", self.PANEL_2)],
                selectbackground=[("readonly", self.PANEL_2)],
                foreground=[("readonly", self.TEXT)],
                selectforeground=[("readonly", self.TEXT)],
            )
            style.configure(
                "TSpinbox",
                fieldbackground=self.PANEL_2,
                background=self.PANEL_2,
                foreground=self.TEXT,
                arrowcolor=self.TEXT,
            )
            style.configure(
                "TCheckbutton", background=self.PANEL, foreground=self.TEXT
            )
            style.map(
                "TCheckbutton",
                background=[("active", self.PANEL)],
                foreground=[("disabled", "#64748b")],
            )
            style.configure(
                "Horizontal.TProgressbar",
                background=self.ACCENT,
                troughcolor=self.PANEL_2,
            )

        def _variables(self) -> None:
            self.search_var = tk.StringVar()
            self.family_var = tk.StringVar(value="All")
            self.runtime_var = tk.StringVar()
            self.mode_var = tk.StringVar(value="Smoke test")
            self.profile_var = tk.StringVar(value="ranked")
            self.population_var = tk.IntVar(value=64)
            self.generations_var = tk.IntVar(value=25)
            self.steps_var = tk.IntVar(value=500)
            self.episodes_var = tk.IntVar(value=1)
            self.workers_var = tk.IntVar(
                value=max(1, min(8, (os.cpu_count() or 2) - 1))
            )
            self.seed_var = tk.IntVar(value=42)
            self.render_var = tk.BooleanVar(value=False)
            self.plot_var = tk.BooleanVar(value=True)
            self.video_var = tk.BooleanVar(value=False)
            self.output_var = tk.StringVar(value=str(ROOT / "runs"))
            self.demo_title_var = tk.StringVar()
            self.demo_meta_var = tk.StringVar()
            self.demo_description_var = tk.StringVar()
            self.action_status_var = tk.StringVar(value="Ready")

        def _layout(self) -> None:
            header = ttk.Frame(self.root, padding=(22, 16, 22, 10))
            header.pack(fill="x")
            ttk.Label(
                header, text="MultiNEAT2 demos", style="Header.TLabel"
            ).pack(side="left")
            ttk.Label(
                header,
                text=f"{len(DEMOS)} ready-to-run demos",
                foreground=self.MUTED,
            ).pack(side="right")

            main = ttk.Panedwindow(self.root, orient="horizontal")
            main.pack(fill="both", expand=True, padx=18, pady=(0, 12))
            left = ttk.Frame(main, style="Panel.TFrame", padding=12)
            right = ttk.Frame(main, style="Panel.TFrame", padding=18)
            main.add(left, weight=2)
            main.add(right, weight=3)

            filters = ttk.Frame(left, style="Panel.TFrame")
            filters.pack(fill="x", pady=(0, 10))
            search = ttk.Entry(filters, textvariable=self.search_var)
            search.pack(side="left", fill="x", expand=True, padx=(0, 8))
            family = ttk.Combobox(
                filters,
                textvariable=self.family_var,
                values=("All", "Core", "Games", "Box2D", "MuJoCo"),
                state="readonly",
                width=10,
            )
            family.pack(side="right")
            self.search_var.trace_add("write", lambda *_: self._populate_tree())
            family.bind("<<ComboboxSelected>>", lambda _: self._populate_tree())

            tree_frame = ttk.Frame(left, style="Panel.TFrame")
            tree_frame.pack(fill="both", expand=True)
            self.tree = ttk.Treeview(
                tree_frame,
                columns=("family", "status"),
                show="tree headings",
                selectmode="browse",
            )
            self.tree.heading("#0", text="Demo")
            self.tree.heading("family", text="Family")
            self.tree.heading("status", text="Status")
            self.tree.column("#0", width=230, stretch=True)
            self.tree.column("family", width=75, anchor="center")
            self.tree.column("status", width=75, anchor="center")
            scrollbar = ttk.Scrollbar(
                tree_frame, orient="vertical", command=self.tree.yview
            )
            self.tree.configure(yscrollcommand=scrollbar.set)
            self.tree.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            self.tree.bind("<<TreeviewSelect>>", self._on_select)
            self.tree.bind("<Double-1>", lambda _: self.launch())

            title = ttk.Label(
                right, textvariable=self.demo_title_var, style="Title.TLabel"
            )
            title.pack(anchor="w")
            ttk.Label(
                right,
                textvariable=self.demo_meta_var,
                style="Muted.TLabel",
            ).pack(anchor="w", pady=(2, 8))
            ttk.Label(
                right,
                textvariable=self.demo_description_var,
                style="Panel.TLabel",
                wraplength=680,
                justify="left",
            ).pack(anchor="w", fill="x", pady=(0, 14))

            options = ttk.LabelFrame(
                right, text="Run options", padding=12
            )
            options.pack(fill="x")
            labels = (
                ("Mode", self.mode_var, ("Smoke test", "Inspect only", "Full run")),
                (
                    "Algorithm",
                    self.profile_var,
                    ("default", "ranked", "exploratory"),
                ),
            )
            for row, (label, variable, values) in enumerate(labels):
                ttk.Label(options, text=label).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=4
                )
                ttk.Combobox(
                    options,
                    textvariable=variable,
                    values=values,
                    state="readonly",
                    width=18,
                ).grid(row=row, column=1, sticky="ew", pady=4)

            numeric = (
                ("Population", self.population_var, 2, 100_000),
                ("Generations", self.generations_var, 1, 1_000_000),
                ("Max steps", self.steps_var, 1, 1_000_000),
                ("Episodes", self.episodes_var, 1, 1000),
                ("Workers", self.workers_var, 1, max(1, os.cpu_count() or 1)),
                ("Seed", self.seed_var, 0, 2_147_483_647),
            )
            for index, (label, variable, minimum, maximum) in enumerate(numeric):
                row = index % 3
                column = 2 + (index // 3) * 2
                ttk.Label(options, text=label).grid(
                    row=row, column=column, sticky="w", padx=(18, 8), pady=4
                )
                ttk.Spinbox(
                    options,
                    from_=minimum,
                    to=maximum,
                    textvariable=variable,
                    width=11,
                ).grid(row=row, column=column + 1, sticky="ew", pady=4)
            for column in (1, 3, 5):
                options.columnconfigure(column, weight=1)

            toggles = ttk.Frame(options, style="Panel.TFrame")
            toggles.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(10, 2))
            self.render_check = ttk.Checkbutton(
                toggles, text="Open rendering windows", variable=self.render_var
            )
            self.render_check.pack(side="left", padx=(0, 16))
            self.plot_check = ttk.Checkbutton(
                toggles, text="Save summary plot", variable=self.plot_var
            )
            self.plot_check.pack(side="left", padx=(0, 16))
            self.video_check = ttk.Checkbutton(
                toggles, text="Record final video", variable=self.video_var
            )
            self.video_check.pack(side="left")

            output_row = ttk.Frame(right, style="Panel.TFrame")
            output_row.pack(fill="x", pady=10)
            ttk.Label(
                output_row, text="Output folder", style="Panel.TLabel"
            ).pack(side="left")
            ttk.Entry(output_row, textvariable=self.output_var).pack(
                side="left", fill="x", expand=True, padx=10
            )
            ttk.Button(
                output_row, text="Browse", command=self.choose_output
            ).pack(side="right")
            self.open_output_button = ttk.Button(
                output_row,
                text="Open",
                command=self.open_output,
                state="disabled",
            )
            self.open_output_button.pack(side="right", padx=(0, 8))

            actions = ttk.Frame(right, style="Panel.TFrame")
            actions.pack(fill="x", pady=(0, 10))
            self.launch_button = ttk.Button(
                actions,
                text="Launch demo",
                style="Accent.TButton",
                command=self.launch,
            )
            self.launch_button.pack(side="left")
            self.stop_button = ttk.Button(
                actions, text="Stop", command=self.stop, state="disabled"
            )
            self.stop_button.pack(side="left", padx=8)

            log_frame = ttk.LabelFrame(right, text="Live output", padding=6)
            log_frame.pack(fill="both", expand=True)
            self.log = tk.Text(
                log_frame,
                height=12,
                background="#070d19",
                foreground=self.TEXT,
                insertbackground=self.TEXT,
                selectbackground="#075985",
                relief="flat",
                font=("Cascadia Mono", 9),
                wrap="word",
                state="disabled",
            )
            log_scroll = ttk.Scrollbar(
                log_frame, orient="vertical", command=self.log.yview
            )
            self.log.configure(yscrollcommand=log_scroll.set)
            self.log.pack(side="left", fill="both", expand=True)
            log_scroll.pack(side="right", fill="y")

            footer = ttk.Frame(self.root, padding=(20, 0, 20, 12))
            footer.pack(fill="x")
            self.progress = ttk.Progressbar(footer, mode="indeterminate")
            self.progress.pack(side="left", fill="x", expand=True)
            ttk.Label(
                footer, textvariable=self.action_status_var
            ).pack(side="right", padx=(15, 0))

        def _populate_tree(self) -> None:
            selected = self.selected_spec().id if self.tree.selection() else None
            self.tree.delete(*self.tree.get_children())
            query = self.search_var.get().strip().lower()
            family = self.family_var.get()
            for spec in DEMOS:
                if family != "All" and spec.family != family:
                    continue
                haystack = (
                    f"{spec.title} {spec.family} {spec.environment} "
                    f"{spec.description}"
                ).lower()
                if query and query not in haystack:
                    continue
                self.tree.insert(
                    "",
                    "end",
                    iid=spec.id,
                    text=spec.title,
                    values=(spec.family, self.status_by_demo.get(spec.id, "—")),
                )
            if selected and self.tree.exists(selected):
                self.tree.selection_set(selected)

        def _select_initial_demo(self) -> None:
            children = self.tree.get_children()
            if children:
                self.tree.selection_set(children[0])
                self.tree.focus(children[0])
                self._on_select()

        def selected_spec(self) -> DemoSpec:
            selection = self.tree.selection()
            return DEMO_BY_ID[selection[0] if selection else DEMOS[0].id]

        def selected_runtime(self) -> RuntimeInfo | None:
            return self.runtime_by_display.get(self.runtime_var.get())

        def options(self) -> LaunchOptions:
            mode = {
                "Smoke test": "smoke",
                "Inspect only": "inspect",
                "Full run": "run",
            }[self.mode_var.get()]
            return LaunchOptions(
                mode=mode,
                population=max(2, self.population_var.get()),
                generations=max(1, self.generations_var.get()),
                max_steps=max(1, self.steps_var.get()),
                episodes=max(1, self.episodes_var.get()),
                workers=max(1, self.workers_var.get()),
                seed=max(0, self.seed_var.get()),
                profile=self.profile_var.get(),
                render=self.render_var.get(),
                plot=self.plot_var.get(),
                record_video=self.video_var.get(),
                output_base=Path(self.output_var.get()).expanduser().resolve(),
            )

        def _on_select(self, _event=None) -> None:
            spec = self.selected_spec()
            self.demo_title_var.set(spec.title)
            self.demo_meta_var.set(f"{spec.family}  ·  {spec.environment}")
            self.demo_description_var.set(spec.description)
            self.population_var.set(spec.population)
            self.generations_var.set(spec.generations)
            self.steps_var.set(spec.max_steps)
            self.episodes_var.set(spec.episodes)
            physics = spec.family in {"Box2D", "MuJoCo"}
            self.plot_check.configure(state="normal" if physics else "disabled")
            self.video_check.configure(state="normal" if physics else "disabled")
            if not physics:
                self.plot_var.set(False)
                self.video_var.set(False)

        def _select_runtime(self) -> None:
            modern = next(
                (runtime for runtime in self.runtime_infos if runtime.modern),
                None,
            )
            if modern:
                self.runtime_var.set(modern.display)
            elif self.runtime_infos:
                self.runtime_var.set(self.runtime_infos[0].display)

        def _append_log(self, text: str) -> None:
            self.log.configure(state="normal")
            self.log.insert("end", text)
            if not text.endswith("\n"):
                self.log.insert("end", "\n")
            self.log.see("end")
            self.log.configure(state="disabled")

        def _emit(self, *event) -> None:
            self.events.put(tuple(event))

        def _set_busy(self, busy: bool, status: str = "Ready") -> None:
            self.busy = busy
            self.action_status_var.set(status)
            self.launch_button.configure(state="disabled" if busy else "normal")
            self.stop_button.configure(state="normal" if busy else "disabled")
            if busy:
                self.progress.start(12)
            else:
                self.progress.stop()

        def _drain_events(self) -> None:
            try:
                while True:
                    event = self.events.get_nowait()
                    kind = event[0]
                    if kind == "log":
                        self._append_log(event[1])
                    elif kind == "busy":
                        self._set_busy(event[1], event[2])
                    elif kind == "error":
                        messagebox.showerror(event[1], event[2], parent=self.root)
                    elif kind == "info":
                        messagebox.showinfo(event[1], event[2], parent=self.root)
                    elif kind == "demo_status":
                        self.status_by_demo[event[1]] = event[2]
                        self._populate_tree()
                    elif kind == "output":
                        self.last_output = Path(event[1])
                        self.open_output_button.configure(state="normal")
            except queue.Empty:
                pass
            self.root.after(100, self._drain_events)

        def _run_streaming(
            self,
            command: Sequence[str],
            runtime: RuntimeInfo | None,
        ) -> int:
            environment = (
                runtime.candidate.environment()
                if runtime
                else os.environ.copy()
            )
            self._emit("log", f"\n> {_quote_command(command)}\n")
            try:
                process = subprocess.Popen(
                    [str(part) for part in command],
                    cwd=str(ROOT),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                )
                self.active_process = process
                assert process.stdout is not None
                for line in process.stdout:
                    self._emit("log", line)
                return process.wait()
            except Exception as error:
                self._emit("log", f"Launcher error: {type(error).__name__}: {error}")
                return 1
            finally:
                self.active_process = None

        def launch(self) -> None:
            if self.busy:
                return
            runtime = self.selected_runtime()
            if runtime is None or not runtime.modern:
                messagebox.showerror(
                    "Current extension required",
                    "No current pymultineat build was found. Build the project "
                    "once, then run this launcher again.",
                    parent=self.root,
                )
                return
            spec = self.selected_spec()
            self._set_busy(True, f"Starting {spec.title}…")
            self.stop_requested = False
            threading.Thread(
                target=self._launch_worker,
                args=(spec, self.options(), runtime),
                daemon=True,
            ).start()

        def _launch_worker(
            self,
            spec: DemoSpec,
            options: LaunchOptions,
            runtime: RuntimeInfo,
        ) -> None:
            command, output = build_demo_command(spec, runtime, options)
            if output:
                self._emit("output", str(output))
            self._emit("busy", True, f"Running {spec.title}…")
            exit_code = self._run_streaming(command, runtime)
            if exit_code == 0:
                self._emit("demo_status", spec.id, "Passed")
                self._emit("busy", False, f"{spec.title} finished")
            elif self.stop_requested:
                self._emit("busy", False, f"{spec.title} stopped")
            else:
                self._emit("demo_status", spec.id, "Failed")
                self._emit("busy", False, f"Exited with code {exit_code}")
                self._emit(
                    "error",
                    f"{spec.title} stopped",
                    f"The demo exited with code {exit_code}. See the live "
                    "output for details.",
                )

        def stop(self) -> None:
            process = self.active_process
            if process is None or process.poll() is not None:
                return
            self.stop_requested = True
            self._append_log(f"Stopping process {process.pid}…")
            try:
                if os.name == "nt":
                    subprocess.run(
                        [
                            "taskkill",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                        ],
                        capture_output=True,
                        creationflags=CREATE_NO_WINDOW,
                    )
                else:
                    process.terminate()
            except OSError as error:
                self._append_log(f"Could not stop process: {error}")

        def choose_output(self) -> None:
            selected = filedialog.askdirectory(
                parent=self.root,
                initialdir=self.output_var.get(),
                title="Choose demo output folder",
            )
            if selected:
                self.output_var.set(selected)

        def open_output(self) -> None:
            path = self.last_output or Path(self.output_var.get())
            path.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])

        def close(self) -> None:
            if self.active_process is not None:
                if not messagebox.askyesno(
                    "Stop running demo?",
                    "A demo is still running. Stop it and close the launcher?",
                    parent=self.root,
                ):
                    return
                self.stop()
            self.root.destroy()

    root = tk.Tk()
    Launcher(root)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MultiNEAT2 graphical demo launcher"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="developer check for runtimes, packages, scripts, and commands",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list demos without opening the GUI",
    )
    args = parser.parse_args()
    if args.list:
        for spec in DEMOS:
            print(f"{spec.family:7}  {spec.title:32}  {spec.environment}")
        return 0
    if args.self_test:
        return run_self_test()
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
