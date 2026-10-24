# MultiNEAT 3 examples

The easiest way to start any example is the root graphical launcher:

```sh
python demos.py
```

It silently selects the current `pymultineat` build, starts the chosen demo
immediately, offers safe smoke/inspect/full-run modes, and shows live output.
Package installation stays outside the launcher. The commands below remain
available for automation and advanced use.

Evolution demos now use reliability-first requests by default. Stochastic
tasks race genomes on 2/4/8/16 shared cases, validate five finalists on 32
fresh cases, and certify the winner on 64 unseen cases. Gymnasium JSONL output
includes environment interactions, lower-20% return, success rate, archive
coverage, wall time, and held-out certification statistics. A result reports
`validated=true` only when its one-sided 95% Wilson lower bound reaches the
configured target. Use `--legacy-evolution` for the historical scalar-fitness
mechanics, and see
[`docs/RELIABLE_EVOLUTION.md`](../docs/RELIABLE_EVOLUTION.md) for the API.

## 3D HyperNEAT example

`hyperneat_3d.py` evolves a CPPN, discovers hidden positions with a real 3D
ES-HyperNEAT octree, materializes directed axon lengths and conduction delays,
and renders the physical substrate. It can also demonstrate fixed-coordinate
3D HyperNEAT:

```sh
python demos/hyperneat_3d.py --smoke --mcculloch-pitts
python demos/hyperneat_3d.py --generations 30 --mcculloch-pitts
python demos/hyperneat_3d.py --fixed --spiking
```

## Spiking examples

Three self-contained examples exercise spiking evolution, online learning, and
the live visualization suite without Box2D or MuJoCo:

| Script | Task |
| --- | --- |
| `spiking_pattern.py` | Evolve a mixed LIF/adaptive-LIF network that separates synchronous from alternating spike trains |
| `spiking_cartpole.py` | Evolve a Poisson-encoded spiking Cart-Pole controller in a built-in physics environment |
| `spiking_eprop.py` | Train an adaptive spiking circuit online with e-prop, surrogate gradients, and AdamW |

All three have bounded smoke modes and visualization output:

```sh
python demos/spiking_pattern.py --smoke
python demos/spiking_pattern.py --generations 80 --animate
python demos/spiking_cartpole.py --smoke
python demos/spiking_cartpole.py --generations 100 --animate
python demos/spiking_eprop.py --smoke
python demos/spiking_eprop.py --epochs 60
python demos/spiking_pattern.py --smoke --mcculloch-pitts
python demos/spiking_cartpole.py --smoke --mcculloch-pitts
python demos/spiking_eprop.py --smoke --mcculloch-pitts
```

`--mcculloch-pitts` selects canonical binary threshold neurons with an
absolute inhibitory veto. Thresholds, refractory periods, topology, weights,
delays, and whether each neuron uses the veto remain evolvable. The e-prop
example uses its surrogate-gradient path for these nondifferentiable units.

The animation shows the live phenotype, a moving spike raster, membrane
traces, and the task environment on the same clock. See
[`docs/SPIKING.md`](../docs/SPIKING.md) for the solver and evolution API.

## Spiking variants of the existing demos

XOR, Asteroids, and every Box2D and MuJoCo controller also have opt-in
spiking variants. The original rate-network behavior remains the default;
add `--spiking` for adaptive-LIF/LIF or `--mcculloch-pitts` for binary
threshold neurons through the same NEAT engine:

```sh
python demos/xor.py --spiking --generations 100
python demos/asteroid_nav.py --spiking --generations 25
python demos/box2d/lunar_lander_box2d.py \
  --spiking --generations 300 --output-dir runs/lander --plot --record-video
python demos/mujoco/inverted_pendulum_mujoco.py --smoke --spiking
python demos/xor.py --smoke --mcculloch-pitts
python demos/run_gymnasium_suite.py \
  --family mujoco --smoke --mcculloch-pitts
```

Each spiking policy maps normalized observations to deterministic seeded
Poisson rates, advances the SNN for several solver steps per environment
action, and decodes filtered output firing rates back into the existing
action adapters. Use `--spiking-steps`, `--spiking-time-step`,
`--spiking-input-rate`, and `--spiking-output-rate` to tune that temporal
interface for the Gymnasium tasks.

Spiking Asteroids adds a live side panel with output-rate
meters, active topology, and a moving 200 ms raster, then automatically
replays the best policy after a finite visible run. `--screenshot FILE` saves
the first replay frame. Spiking XOR produces a four-panel phenotype, raster,
membrane, and response view.

The graphical launcher exposes all these modes through its **Spiking policy
variant** and **McCulloch-Pitts neurons** switches.
For Box2D and MuJoCo, **Show physics GUI** opens the native viewer immediately;
visible launcher runs use one evaluator so the window stays live. Clear it to
launch directly in physics-only fast mode with the configured worker count.

Every example is a single standalone file: it depends only on
`pymultineat`, the Python standard library, and third-party packages, so it
can be copied out of the repository and read top to bottom. The Box2D and
MuJoCo scripts share one structure and accept the same command-line options;
they differ only in their task constants and observation/action encoding.

## Installation

Build or install `pymultineat`, then install one or both simulator families:

```sh
python -m pip install -r requirements-box2d.txt
python -m pip install -r requirements-mujoco.txt
```

Install `requirements-demos.txt` to enable every simulator, plots, and
ImageIO/FFmpeg video support. `gymnasium[box2d]` installs a prebuilt Box2D
wheel; install SWIG first only if pip has to build Box2D from source.

## Quick start

Every script supports an environment-only inspection and a real, short
evolution smoke test:

```sh
python demos/box2d/lunar_lander_box2d.py --inspect
python demos/box2d/lunar_lander_box2d.py --smoke
python demos/mujoco/inverted_pendulum_mujoco.py --smoke
```

`--inspect` and `--smoke` are intentionally headless diagnostics; each prints
a one-line JSON summary. Omit them to run the visible demo.

Start a practical run with explicit output and parallel evaluation:

```sh
python demos/box2d/bipedal_walker_box2d.py \
  --generations 600 --population 192 --workers 8 \
  --profile ranked --output-dir runs/bipedal --plot \
  --checkpoint-every 10
```

Training always runs headless. Unless `--fast` (or its `--headless` alias)
is given, the current champion is replayed in the simulator's native window
every `--render-every` generations (default 1) and once at the end:

```sh
python demos/box2d/bipedal_walker_box2d.py --fast
python demos/mujoco/walker2d_mujoco.py --fast
```

Parallel workers are always headless; replays happen in the main process.

Resume the complete population, innovation database, species, and RNG state:

```sh
python demos/box2d/bipedal_walker_box2d.py \
  --resume runs/bipedal/population.state \
  --generations 200 --output-dir runs/bipedal
```

An output directory holds `best_genome.txt`, `population.state`,
`metrics.jsonl`, and, when requested, `fitness.png` (`--plot`) and `best.mp4`
(`--record-video`, which records the best policy at the end of a run).

## Example catalog

### Box2D

| Script | Environment and variation |
| --- | --- |
| `lunar_lander_box2d.py` | LunarLander-v3, discrete |
| `lunar_lander_continuous_box2d.py` | LunarLander-v3, continuous |
| `lunar_lander_wind_box2d.py` | Continuous lander with wind and turbulence |
| `bipedal_walker_box2d.py` | BipedalWalker-v3, normal terrain |
| `bipedal_walker_hardcore_box2d.py` | BipedalWalker-v3, hardcore terrain |
| `car_racing_box2d.py` | CarRacing-v3, continuous pooled-pixel control |
| `car_racing_discrete_box2d.py` | CarRacing-v3, discrete pooled-pixel control |
| `car_racing_nofx_box2d.py` | CarRacing-v3, continuous, explicitly without domain randomization |

### MuJoCo

The suite covers all eleven Gymnasium MuJoCo tasks:

| Script | Environment |
| --- | --- |
| `inverted_pendulum_mujoco.py` | InvertedPendulum-v5 |
| `inverted_double_pendulum_mujoco.py` | InvertedDoublePendulum-v5 |
| `reacher_mujoco.py` | Reacher-v5 |
| `pusher_mujoco.py` | Pusher-v5 |
| `halfcheetah_mujoco.py` | HalfCheetah-v5 |
| `hopper_mujoco.py` | Hopper-v5 |
| `walker2d_mujoco.py` | Walker2d-v5 |
| `swimmer_mujoco.py` | Swimmer-v5 |
| `ant_mujoco.py` | Ant-v5 |
| `humanoid_mujoco.py` | Humanoid-v5 |
| `humanoid_standup_mujoco.py` | HumanoidStandup-v5 |

## Common features

- Runtime inference of observation and action dimensions.
- Correct `terminated or truncated` episode handling.
- Finite-bound-aware continuous action mapping and discrete argmax policies.
- Bounded vector observation scaling (Box2D) and `tanh` squashing (MuJoCo).
- Compact grayscale area pooling for CarRacing images.
- Common case IDs and disjoint training/validation/test seed streams.
- Successive-halving evaluation and held-out Wilson certification.
- Persistent worker environments for parallel evaluation.
- Native Box2D/MuJoCo champion replays, or physics-only `--fast` training.
- Raw negative fitness support via robust rank-based allocation.
- Full population checkpoints, best-genome files, JSONL metrics, plots, video,
  and deterministic smoke runs.
- Default, rank-selection, and exploration-heavy algorithm profiles.

List, inspect, or smoke-test a whole family (each demo runs as its own
process):

```sh
python demos/run_gymnasium_suite.py
python demos/run_gymnasium_suite.py --family box2d --inspect
python demos/run_gymnasium_suite.py --family mujoco --smoke
python demos/run_gymnasium_suite.py --family box2d --smoke --spiking
python demos/run_gymnasium_suite.py --family box2d --smoke --mcculloch-pitts
```
