# Physics-control demos

The Box2D and MuJoCo examples use one shared, tested trainer:
`demos/gymnasium_neat.py`. Every task-specific file is directly runnable and
accepts the same command-line options.

## Installation

Build or install `pymultineat`, then install one or both simulator families:

```sh
python -m pip install -r requirements-box2d.txt
python -m pip install -r requirements-mujoco.txt
```

Install `requirements-demos.txt` to enable every simulator, plots, and
ImageIO/FFmpeg video support. The Box2D installation follows Gymnasium's
recommendation to install SWIG before the Box2D extra.

## Quick start

Every script supports an environment-only inspection and a real, short
evolution smoke test:

```sh
python demos/box2d/lunar_lander_box2d.py --inspect
python demos/box2d/lunar_lander_box2d.py --smoke
python demos/mujoco/inverted_pendulum_mujoco.py --smoke
```

Start a practical run with explicit output and parallel evaluation:

```sh
python demos/box2d/bipedal_walker_box2d.py \
  --generations 600 --population 192 --workers 8 \
  --profile ranked --output-dir runs/bipedal --plot \
  --checkpoint-every 10
```

Resume the complete population, innovation database, species, and RNG state:

```sh
python demos/box2d/bipedal_walker_box2d.py \
  --resume runs/bipedal/population.state \
  --generations 200 --output-dir runs/bipedal
```

Current MultiNEAT2 builds write `population.state`. For compatibility, a
binding too old to round-trip the full state writes `population.legacy`
through its original `Population.Save` API instead.

Use `--record-video` to record the best policy at the end of a run or
`--render-every N` for periodic interactive evaluation. Rendering is always
performed in the main process.

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
| `car_racing_nofx_box2d.py` | Compatibility alias for lightweight racing |

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

`bipedal_walker_mujoco.py` remains as a compatibility alias for Walker2d.

## Shared features

- Runtime inference of observation and action dimensions.
- Correct `terminated or truncated` episode handling.
- Finite-bound-aware continuous action mapping and discrete argmax policies.
- Recursive vector/tuple/dictionary observation encoding.
- Compact grayscale area pooling for CarRacing images.
- Common episode seeds for fair comparisons within each generation.
- Persistent worker environments for parallel evaluation.
- Raw negative fitness support via MultiNEAT2's stable fitness scaling.
- Full population checkpoints, best-genome files, JSONL metrics, plots, video,
  and deterministic smoke runs.
- Default, rank-selection, and exploration-heavy algorithm profiles.

List, inspect, or smoke-test a whole family:

```sh
python demos/run_gymnasium_suite.py
python demos/run_gymnasium_suite.py --family box2d --inspect
python demos/run_gymnasium_suite.py --family mujoco --smoke
```
