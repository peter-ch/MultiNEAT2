#!/usr/bin/env python3
"""Fast tests for the shared Box2D/MuJoCo example machinery."""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "demos"
sys.path.insert(0, str(DEMOS))

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:
    raise SystemExit(77)

import gymnasium_neat as demo  # noqa: E402

if demo.neat is None:
    raise SystemExit(77)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


box2d_tasks = list(demo.catalog("box2d"))
mujoco_tasks = list(demo.catalog("mujoco"))
check(len(box2d_tasks) == 7, "Box2D catalog should contain seven tasks")
check(len(mujoco_tasks) == 11, "MuJoCo catalog should contain eleven tasks")
check(
    {task.env_id for task in mujoco_tasks}
    == {
        "Ant-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
        "InvertedDoublePendulum-v5",
        "InvertedPendulum-v5",
        "Pusher-v5",
        "Reacher-v5",
        "Swimmer-v5",
        "Walker2d-v5",
    },
    "MuJoCo catalog should cover every official v5 task",
)

vector_space = spaces.Box(
    low=np.array([-2.0, -np.inf, 0.0]),
    high=np.array([2.0, np.inf, 10.0]),
    dtype=np.float64,
)
vector_encoder = demo.ObservationEncoder(vector_space)
encoded = vector_encoder.encode(np.array([2.0, np.inf, 5.0]))
check(encoded.shape == (3,), "vector encoder changed dimensionality")
check(np.all(np.isfinite(encoded)), "vector encoder emitted non-finite values")
check(np.all(encoded <= 1.0) and np.all(encoded >= -1.0), "encoding out of range")
check(np.allclose(encoded[[0, 2]], [1.0, 0.0]), "bounded scaling is wrong")

image = np.zeros((24, 30, 3), dtype=np.uint8)
image[:, 15:, :] = 255
image_encoder = demo.ObservationEncoder(
    spaces.Box(0, 255, shape=image.shape, dtype=np.uint8),
    image_size=(4, 5),
)
image_values = image_encoder.encode(image)
check(image_values.shape == (20,), "image pooling dimensions are wrong")
check(image_values.min() == -1.0, "black pixels should normalize to -1")
check(image_values.max() == 1.0, "white pixels should normalize to 1")

discrete = demo.ActionAdapter(spaces.Discrete(4))
check(discrete.output_size == 4, "discrete output size is wrong")
check(discrete.convert([-1.0, 0.0, 0.8, 0.1]) == 2, "argmax action is wrong")

continuous_space = spaces.Box(
    low=np.array([-2.0, 0.0, -1.0], dtype=np.float32),
    high=np.array([2.0, 10.0, 1.0], dtype=np.float32),
)
continuous = demo.ActionAdapter(continuous_space)
action = continuous.convert([-1.0, 0.0, 1.0])
check(
    np.allclose(action, [-2.0, 5.0, 1.0]),
    "continuous bound mapping is wrong",
)
check(continuous_space.contains(action), "adapter emitted an invalid action")


class TinyControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=np.array([-2.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0], dtype=np.float32),
        )
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0.25, -0.5, 1.0]), {}

    def step(self, action):
        check(self.action_space.contains(action), "policy action left its space")
        self.steps += 1
        observation = np.array([action[0], action[1], self.steps])
        reward = 1.0 - float(np.square(action).mean())
        return observation, reward, self.steps >= 3, False, {}


tiny_config = demo.TaskConfig(
    key="test_tiny",
    family="test",
    env_id="TinyControl-v0",
    description="In-process integration environment.",
    recurrent=False,
)
tiny_shape = demo.TaskShape(3, 2, "Box(3)", "Box(2)")
population = demo.create_population(
    tiny_shape,
    tiny_config,
    population_size=6,
    seed=17,
    profile="ranked",
    initial_connectivity="full",
)
tiny_env = TinyControlEnv()
fitnesses = []
for species in population.m_Species:
    for genome in species.m_Individuals:
        fitness = demo.evaluate_on_environment(
            genome,
            tiny_env,
            tiny_config,
            episode_seeds=(1, 2),
            max_steps=4,
            activation_steps=2,
        )
        check(np.isfinite(fitness), "demo evaluation produced non-finite fitness")
        genome.SetFitness(fitness)
        genome.SetEvaluated()
        fitnesses.append(fitness)
check(len(fitnesses) == 6, "not every demo genome was evaluated")
population.Epoch()
if hasattr(population, "Validate"):
    valid, error = population.Validate()
    check(valid, f"demo evolution produced invalid population: {error}")
tiny_env.close()

spiking_population = demo.create_population(
    tiny_shape,
    tiny_config,
    population_size=4,
    seed=23,
    profile="default",
    initial_connectivity="full",
    spiking=True,
)
spiking_env = TinyControlEnv()
spiking_settings = demo.SpikingPolicySettings(
    simulation_steps=4,
    time_step=0.001,
    input_rate_hz=200.0,
    output_rate_hz=200.0,
)
spiking_fitnesses = []
for species in spiking_population.m_Species:
    for genome in species.m_Individuals:
        phenotype = demo.neat.NeuralNetwork()
        genome.BuildPhenotype(phenotype)
        check(phenotype.IsSpiking(), "spiking demo built a rate phenotype")
        policy = demo.SpikingPolicy(phenotype, spiking_settings)
        check(
            np.allclose(
                policy.encode([-1.0, 0.0, 1.0, np.inf]),
                [0.0, 100.0, 200.0, 100.0],
            ),
            "spiking observation-to-rate encoding is wrong",
        )
        try:
            policy.encode([0.0])
        except ValueError:
            pass
        else:
            raise AssertionError("spiking policy accepted the wrong input size")
        fitness = demo.evaluate_on_environment(
            genome,
            spiking_env,
            tiny_config,
            episode_seeds=(3,),
            max_steps=3,
            activation_steps=1,
            spiking_settings=spiking_settings,
        )
        check(
            np.isfinite(fitness),
            "spiking demo evaluation produced non-finite fitness",
        )
        genome.SetFitness(fitness)
        genome.SetEvaluated()
        spiking_fitnesses.append(fitness)
check(len(spiking_fitnesses) == 4, "not every spiking genome was evaluated")
spiking_population.Epoch()
spiking_env.close()

wrappers = list((DEMOS / "box2d").glob("*_box2d.py"))
wrappers += list((DEMOS / "mujoco").glob("*_mujoco.py"))
check(len(wrappers) == 20, "expected eight Box2D and twelve MuJoCo scripts")
for wrapper in wrappers:
    py_compile.compile(str(wrapper), doraise=True)

print("Gymnasium demo tests passed.")
