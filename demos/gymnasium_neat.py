#!/usr/bin/env python3
"""Reusable MultiNEAT trainer for Gymnasium Box2D and MuJoCo tasks.

The task-specific scripts in ``demos/box2d`` and ``demos/mujoco`` are thin
entry points around this module.  Keeping environment interaction here makes
the examples small while ensuring that seeding, action conversion,
termination handling, multiprocessing, and checkpointing behave consistently.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from spiking_neat import (
    SpikingPolicy,
    SpikingPolicySettings,
    configure_spiking_genome,
    configure_spiking_parameters,
)

# Running ``python demos/...`` places the script directory, rather than the
# repository root, first on sys.path.  Add the root so an in-tree extension
# build/copy and ``neattools.py`` are discoverable without installation.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # The catalog and --help remain usable without Gymnasium.
    gym = None
    spaces = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pymultineat as neat
except ImportError:
    neat = None


@dataclass(frozen=True)
class TaskConfig:
    """Configuration that distinguishes one Gymnasium control task."""

    key: str
    family: str
    env_id: str
    description: str
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    default_population: int = 128
    default_generations: int = 300
    default_max_steps: int = 1000
    default_episodes: int = 1
    image_size: tuple[int, int] | None = None
    crop_bottom: int = 0
    sparse_seed: bool = False
    recurrent: bool = True


def _task(
    key: str,
    family: str,
    env_id: str,
    description: str,
    **kwargs: Any,
) -> TaskConfig:
    return TaskConfig(key, family, env_id, description, **kwargs)


TASKS: dict[str, TaskConfig] = {
    # Box2D
    "box2d_lunar_lander": _task(
        "box2d_lunar_lander",
        "box2d",
        "LunarLander-v3",
        "Discrete four-engine lunar landing.",
        env_kwargs={"continuous": False},
        default_population=160,
        default_generations=300,
        default_max_steps=1000,
        recurrent=False,
    ),
    "box2d_lunar_lander_continuous": _task(
        "box2d_lunar_lander_continuous",
        "box2d",
        "LunarLander-v3",
        "Continuous main and lateral thruster control.",
        env_kwargs={"continuous": True},
        default_population=160,
        default_generations=300,
        default_max_steps=1000,
        recurrent=False,
    ),
    "box2d_lunar_lander_wind": _task(
        "box2d_lunar_lander_wind",
        "box2d",
        "LunarLander-v3",
        "Robust continuous landing with wind and turbulence.",
        env_kwargs={
            "continuous": True,
            "enable_wind": True,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        },
        default_population=192,
        default_generations=400,
        default_max_steps=1000,
        default_episodes=3,
    ),
    "box2d_bipedal_walker": _task(
        "box2d_bipedal_walker",
        "box2d",
        "BipedalWalker-v3",
        "Four-joint walking on normal terrain.",
        env_kwargs={"hardcore": False},
        default_population=192,
        default_generations=600,
        default_max_steps=1600,
    ),
    "box2d_bipedal_walker_hardcore": _task(
        "box2d_bipedal_walker_hardcore",
        "box2d",
        "BipedalWalker-v3",
        "Four-joint walking over hardcore obstacles.",
        env_kwargs={"hardcore": True},
        default_population=256,
        default_generations=1000,
        default_max_steps=2000,
        default_episodes=2,
    ),
    "box2d_car_racing": _task(
        "box2d_car_racing",
        "box2d",
        "CarRacing-v3",
        "Continuous steering, throttle, and brake from pooled pixels.",
        env_kwargs={"continuous": True},
        default_population=192,
        default_generations=500,
        default_max_steps=1000,
        image_size=(12, 12),
        crop_bottom=12,
        sparse_seed=True,
    ),
    "box2d_car_racing_discrete": _task(
        "box2d_car_racing_discrete",
        "box2d",
        "CarRacing-v3",
        "Discrete car control from pooled pixels.",
        env_kwargs={"continuous": False},
        default_population=160,
        default_generations=500,
        default_max_steps=1000,
        image_size=(12, 12),
        crop_bottom=12,
        sparse_seed=True,
    ),
    # MuJoCo
    "mujoco_ant": _task(
        "mujoco_ant",
        "mujoco",
        "Ant-v5",
        "Eight-actuator quadruped locomotion.",
        default_population=256,
        default_generations=1200,
        default_max_steps=1000,
        sparse_seed=True,
    ),
    "mujoco_half_cheetah": _task(
        "mujoco_half_cheetah",
        "mujoco",
        "HalfCheetah-v5",
        "Fast planar locomotion with six actuators.",
        default_population=192,
        default_generations=800,
        default_max_steps=1000,
    ),
    "mujoco_hopper": _task(
        "mujoco_hopper",
        "mujoco",
        "Hopper-v5",
        "Three-actuator hopping and balance.",
        default_population=192,
        default_generations=800,
        default_max_steps=1000,
    ),
    "mujoco_humanoid": _task(
        "mujoco_humanoid",
        "mujoco",
        "Humanoid-v5",
        "Seventeen-actuator humanoid locomotion.",
        default_population=320,
        default_generations=1600,
        default_max_steps=1000,
        sparse_seed=True,
    ),
    "mujoco_humanoid_standup": _task(
        "mujoco_humanoid_standup",
        "mujoco",
        "HumanoidStandup-v5",
        "Seventeen-actuator humanoid stand-up task.",
        default_population=320,
        default_generations=1600,
        default_max_steps=1000,
        sparse_seed=True,
    ),
    "mujoco_inverted_pendulum": _task(
        "mujoco_inverted_pendulum",
        "mujoco",
        "InvertedPendulum-v5",
        "Continuous cart-pole balancing.",
        default_population=96,
        default_generations=200,
        default_max_steps=1000,
        recurrent=False,
    ),
    "mujoco_inverted_double_pendulum": _task(
        "mujoco_inverted_double_pendulum",
        "mujoco",
        "InvertedDoublePendulum-v5",
        "Continuous double-pendulum balancing.",
        default_population=128,
        default_generations=350,
        default_max_steps=1000,
        recurrent=False,
    ),
    "mujoco_pusher": _task(
        "mujoco_pusher",
        "mujoco",
        "Pusher-v5",
        "Seven-actuator arm pushing an object to a goal.",
        default_population=224,
        default_generations=1000,
        default_max_steps=100,
        default_episodes=3,
    ),
    "mujoco_reacher": _task(
        "mujoco_reacher",
        "mujoco",
        "Reacher-v5",
        "Two-actuator arm reaching a randomized target.",
        default_population=160,
        default_generations=600,
        default_max_steps=50,
        default_episodes=3,
    ),
    "mujoco_swimmer": _task(
        "mujoco_swimmer",
        "mujoco",
        "Swimmer-v5",
        "Two-actuator fluid locomotion.",
        default_population=160,
        default_generations=600,
        default_max_steps=1000,
    ),
    "mujoco_walker2d": _task(
        "mujoco_walker2d",
        "mujoco",
        "Walker2d-v5",
        "Six-actuator planar biped locomotion.",
        default_population=224,
        default_generations=1000,
        default_max_steps=1000,
    ),
}


def require_dependencies() -> None:
    """Raise an actionable error when an optional demo dependency is absent."""

    missing = []
    if np is None:
        missing.append("numpy")
    if gym is None:
        missing.append("gymnasium")
    if neat is None:
        missing.append("pymultineat")
    if missing:
        raise RuntimeError(
            "Missing demo dependencies: "
            + ", ".join(missing)
            + ". Build pymultineat and install requirements-box2d.txt or "
            "requirements-mujoco.txt."
        )


class ObservationEncoder:
    """Convert Gymnasium observations to compact, finite NEAT inputs."""

    def __init__(
        self,
        observation_space: Any,
        image_size: tuple[int, int] | None = None,
        crop_bottom: int = 0,
    ) -> None:
        require_dependencies()
        self.space = observation_space
        self.image_size = image_size
        self.crop_bottom = max(0, int(crop_bottom))

    def encode(self, observation: Any) -> "np.ndarray":
        values = self._encode(self.space, observation)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    def _encode(self, space: Any, observation: Any) -> "np.ndarray":
        if isinstance(space, spaces.Dict):
            chunks = [
                self._encode(space.spaces[key], observation[key])
                for key in sorted(space.spaces)
            ]
            return np.concatenate(chunks) if chunks else np.empty(0)

        if isinstance(space, spaces.Tuple):
            chunks = [
                self._encode(subspace, value)
                for subspace, value in zip(space.spaces, observation)
            ]
            return np.concatenate(chunks) if chunks else np.empty(0)

        if isinstance(space, spaces.Discrete):
            result = np.full(space.n, -1.0, dtype=np.float64)
            result[int(observation) - int(space.start)] = 1.0
            return result

        if isinstance(space, spaces.MultiDiscrete):
            chunks = []
            starts = getattr(space, "start", np.zeros_like(space.nvec))
            for value, count, start in zip(
                np.asarray(observation).flat,
                np.asarray(space.nvec).flat,
                np.asarray(starts).flat,
            ):
                chunk = np.full(int(count), -1.0, dtype=np.float64)
                chunk[int(value) - int(start)] = 1.0
                chunks.append(chunk)
            return np.concatenate(chunks)

        if isinstance(space, spaces.MultiBinary):
            return np.asarray(observation, dtype=np.float64).reshape(-1) * 2.0 - 1.0

        if not isinstance(space, spaces.Box):
            return np.asarray(observation, dtype=np.float64).reshape(-1)

        array = np.asarray(observation)
        if self.image_size is not None and array.ndim in (2, 3):
            return self._encode_image(array)

        values = np.asarray(array, dtype=np.float64)
        low = np.broadcast_to(np.asarray(space.low, dtype=np.float64), values.shape)
        high = np.broadcast_to(np.asarray(space.high, dtype=np.float64), values.shape)
        bounded = np.isfinite(low) & np.isfinite(high) & (high > low)
        bounded_values = (
            2.0 * (values[bounded] - low[bounded])
            / (high[bounded] - low[bounded])
            - 1.0
        )
        scaled = np.tanh(values)
        scaled = np.asarray(scaled)
        scaled[bounded] = bounded_values
        return np.clip(scaled.reshape(-1), -1.0, 1.0)

    def _encode_image(self, image: "np.ndarray") -> "np.ndarray":
        values = np.asarray(image, dtype=np.float64)
        if self.crop_bottom and values.shape[0] > self.crop_bottom:
            values = values[: -self.crop_bottom]
        if values.ndim == 3:
            if values.shape[-1] >= 3:
                values = (
                    values[..., 0] * 0.299
                    + values[..., 1] * 0.587
                    + values[..., 2] * 0.114
                )
            else:
                values = values.mean(axis=-1)

        target_height, target_width = self.image_size
        y_edges = np.linspace(0, values.shape[0], target_height + 1, dtype=int)
        x_edges = np.linspace(0, values.shape[1], target_width + 1, dtype=int)
        pooled = np.empty((target_height, target_width), dtype=np.float64)
        for row in range(target_height):
            for column in range(target_width):
                block = values[
                    y_edges[row] : y_edges[row + 1],
                    x_edges[column] : x_edges[column + 1],
                ]
                pooled[row, column] = float(block.mean()) if block.size else 0.0

        if np.issubdtype(image.dtype, np.integer) or pooled.max(initial=0.0) > 1.5:
            pooled = pooled / 127.5 - 1.0
        else:
            pooled = pooled * 2.0 - 1.0
        return np.clip(pooled.reshape(-1), -1.0, 1.0)


class ActionAdapter:
    """Convert neural-network outputs into valid Gymnasium actions."""

    def __init__(self, action_space: Any) -> None:
        require_dependencies()
        self.space = action_space

    @property
    def output_size(self) -> int:
        if isinstance(self.space, spaces.Discrete):
            return int(self.space.n)
        if isinstance(self.space, spaces.MultiDiscrete):
            return int(np.asarray(self.space.nvec).sum())
        if isinstance(self.space, spaces.MultiBinary):
            return int(np.prod(self.space.shape))
        if isinstance(self.space, spaces.Box):
            return int(np.prod(self.space.shape))
        raise TypeError(f"Unsupported action space: {type(self.space).__name__}")

    def convert(self, outputs: Sequence[float]) -> Any:
        values = np.nan_to_num(
            np.asarray(outputs, dtype=np.float64),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        if values.size < self.output_size:
            raise ValueError(
                f"Network returned {values.size} outputs; "
                f"{self.output_size} are required"
            )

        if isinstance(self.space, spaces.Discrete):
            return int(self.space.start) + int(np.argmax(values[: self.space.n]))

        if isinstance(self.space, spaces.MultiDiscrete):
            action = []
            offset = 0
            starts = getattr(self.space, "start", np.zeros_like(self.space.nvec))
            for count, start in zip(
                np.asarray(self.space.nvec).flat,
                np.asarray(starts).flat,
            ):
                action.append(
                    int(start)
                    + int(np.argmax(values[offset : offset + int(count)]))
                )
                offset += int(count)
            return np.asarray(action, dtype=self.space.dtype).reshape(
                self.space.shape
            )

        if isinstance(self.space, spaces.MultiBinary):
            return (values[: self.output_size] >= 0.0).astype(
                self.space.dtype
            ).reshape(self.space.shape)

        normalized = np.clip(values[: self.output_size], -1.0, 1.0).reshape(
            self.space.shape
        )
        low = np.asarray(self.space.low, dtype=np.float64)
        high = np.asarray(self.space.high, dtype=np.float64)
        action = normalized.copy()
        both = np.isfinite(low) & np.isfinite(high)
        action[both] = low[both] + (normalized[both] + 1.0) * 0.5 * (
            high[both] - low[both]
        )
        lower_only = np.isfinite(low) & ~np.isfinite(high)
        upper_only = ~np.isfinite(low) & np.isfinite(high)
        action[lower_only] = low[lower_only] + np.exp(
            np.minimum(normalized[lower_only], 0.0)
        )
        action[upper_only] = high[upper_only] - np.exp(
            np.minimum(-normalized[upper_only], 0.0)
        )
        return np.asarray(action, dtype=self.space.dtype)


@dataclass(frozen=True)
class TaskShape:
    input_size: int
    output_size: int
    observation_shape: str
    action_shape: str


def make_environment(config: TaskConfig, render_mode: str | None = None) -> Any:
    require_dependencies()
    kwargs = dict(config.env_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    try:
        return gym.make(config.env_id, **kwargs)
    except Exception as error:
        requirement = (
            "requirements-box2d.txt"
            if config.family == "box2d"
            else "requirements-mujoco.txt"
        )
        raise RuntimeError(
            f"Could not create {config.env_id!r} for {config.key}. "
            f"Install {requirement}. Original error: {error}"
        ) from error


def inspect_task(config: TaskConfig, seed: int = 1) -> TaskShape:
    """Create and reset an environment, returning its NEAT policy dimensions."""

    env = make_environment(config)
    try:
        observation, _ = env.reset(seed=seed)
        encoder = ObservationEncoder(
            env.observation_space,
            config.image_size,
            config.crop_bottom,
        )
        input_size = int(encoder.encode(observation).size)
        output_size = ActionAdapter(env.action_space).output_size
        return TaskShape(
            input_size=input_size,
            output_size=output_size,
            observation_shape=repr(env.observation_space),
            action_shape=repr(env.action_space),
        )
    finally:
        env.close()


def _network_action(
    network: Any,
    observation: Any,
    encoder: ObservationEncoder,
    adapter: ActionAdapter,
    activation_steps: int,
    spiking_policy: SpikingPolicy | None = None,
) -> Any:
    inputs = encoder.encode(observation).tolist()
    inputs.append(1.0)
    if spiking_policy is not None:
        return adapter.convert(spiking_policy.step_signed(inputs))
    network.Input(inputs)
    if hasattr(network, "ActivateSteps"):
        network.ActivateSteps(activation_steps, True)
    else:
        for _ in range(activation_steps):
            network.Activate()
    return adapter.convert(network.Output())


def evaluate_on_environment(
    genome: Any,
    env: Any,
    config: TaskConfig,
    episode_seeds: Sequence[int],
    max_steps: int,
    activation_steps: int = 2,
    spiking_settings: SpikingPolicySettings | None = None,
) -> float:
    """Evaluate a genome using an already-created environment."""

    encoder = ObservationEncoder(
        env.observation_space,
        config.image_size,
        config.crop_bottom,
    )
    adapter = ActionAdapter(env.action_space)
    network = neat.NeuralNetwork()
    genome.BuildPhenotype(network)
    spiking_policy = (
        SpikingPolicy(network, spiking_settings)
        if spiking_settings is not None
        else None
    )
    rewards = []

    for episode_seed in episode_seeds:
        if spiking_policy is None:
            network.Flush()
        else:
            spiking_policy.reset(int(episode_seed))
        observation, _ = env.reset(seed=int(episode_seed))
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(int(episode_seed))
        total_reward = 0.0
        for _ in range(max_steps):
            action = _network_action(
                network,
                observation,
                encoder,
                adapter,
                activation_steps,
                spiking_policy,
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            numeric_reward = float(reward)
            if not math.isfinite(numeric_reward):
                numeric_reward = -1_000_000.0
                terminated = True
            total_reward += numeric_reward
            if terminated or truncated:
                break
        rewards.append(total_reward)

    fitness = float(np.mean(rewards))
    return fitness if math.isfinite(fitness) else -1_000_000.0


_WORKER_ENV = None
_WORKER_CONFIG = None
_WORKER_MAX_STEPS = 0
_WORKER_ACTIVATION_STEPS = 1
_WORKER_SPIKING_SETTINGS = None


def _initialize_worker(
    config: TaskConfig,
    max_steps: int,
    activation_steps: int,
    spiking_settings: SpikingPolicySettings | None,
) -> None:
    global _WORKER_ENV
    global _WORKER_CONFIG
    global _WORKER_MAX_STEPS
    global _WORKER_ACTIVATION_STEPS
    global _WORKER_SPIKING_SETTINGS
    _WORKER_CONFIG = config
    _WORKER_MAX_STEPS = max_steps
    _WORKER_ACTIVATION_STEPS = activation_steps
    _WORKER_SPIKING_SETTINGS = spiking_settings
    _WORKER_ENV = make_environment(config)


def _worker_evaluate(payload: tuple[Any, Sequence[int]]) -> float:
    genome, episode_seeds = payload
    return evaluate_on_environment(
        genome,
        _WORKER_ENV,
        _WORKER_CONFIG,
        episode_seeds,
        _WORKER_MAX_STEPS,
        _WORKER_ACTIVATION_STEPS,
        _WORKER_SPIKING_SETTINGS,
    )


def configure_parameters(
    population_size: int,
    recurrent: bool,
    profile: str,
    spiking: bool = False,
    mcculloch_pitts: bool = False,
) -> Any:
    """Return modern but conservative parameters for control problems."""

    params = neat.Parameters()
    params.PopulationSize = population_size
    params.DynamicCompatibility = population_size >= 8
    params.MinSpecies = 1
    params.MaxSpecies = max(1, min(12, population_size // 4))
    params.CompatTreshold = 3.0
    params.YoungAgeTreshold = 15
    params.OldAgeTreshold = 50
    params.SpeciesMaxStagnation = 30
    params.SurvivalRate = 0.25
    params.EliteFraction = min(0.1, 1.0 / population_size)
    params.AllowClones = False
    params.AllowLoops = recurrent
    params.RecurrentProb = 0.1 if recurrent else 0.0
    params.SplitRecurrent = recurrent
    params.SplitLoopedRecurrent = recurrent

    params.OverallMutationRate = 0.8
    params.MutateWeightsProb = 0.9
    params.MutateWeightsSevereProb = 0.1
    params.WeightMutationRate = 0.9
    params.WeightReplacementRate = 0.1
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 2.0
    params.MinWeight = -5.0
    params.MaxWeight = 5.0
    params.MutateAddNeuronProb = 0.02
    params.MutateAddLinkProb = 0.08
    params.MutateRemLinkProb = 0.01
    params.MutateRemSimpleNeuronProb = 0.005
    params.CrossoverRate = 0.6
    params.MultipointCrossoverRate = 0.5

    activation_probabilities = (
        "ActivationFunction_SignedSigmoid_Prob",
        "ActivationFunction_UnsignedSigmoid_Prob",
        "ActivationFunction_Tanh_Prob",
        "ActivationFunction_TanhCubic_Prob",
        "ActivationFunction_SignedStep_Prob",
        "ActivationFunction_UnsignedStep_Prob",
        "ActivationFunction_SignedGauss_Prob",
        "ActivationFunction_UnsignedGauss_Prob",
        "ActivationFunction_Abs_Prob",
        "ActivationFunction_SignedSine_Prob",
        "ActivationFunction_UnsignedSine_Prob",
        "ActivationFunction_Linear_Prob",
        "ActivationFunction_Relu_Prob",
        "ActivationFunction_Softplus_Prob",
    )
    for name in activation_probabilities:
        if hasattr(params, name):
            setattr(params, name, 0.0)
    params.ActivationFunction_Tanh_Prob = 1.0
    params.MutateNeuronActivationTypeProb = 0.02

    if spiking:
        configure_spiking_parameters(
            neat,
            params,
            recurrent=recurrent,
            enable_stdp=False,
            neuron_model=(
                "mcculloch-pitts" if mcculloch_pitts else "lif"
            ),
        )

    # These operators are additive in MultiNEAT2. Attribute checks retain
    # compatibility with binaries built from older MultiNEAT releases.
    if profile == "ranked" and hasattr(params, "ParentSelectionMode"):
        params.ParentSelectionMode = neat.RANK_LINEAR
        params.RankSelectionPressure = 1.7
    elif profile == "exploratory" and hasattr(params, "ParentSelectionMode"):
        params.ParentSelectionMode = neat.TOURNAMENT
        params.TournamentSize = 4
        params.MutateAddNeuronProb = 0.04
        params.MutateAddLinkProb = 0.12
        if hasattr(params, "AdaptiveMutationStart"):
            params.AdaptiveMutationStart = 15
            params.AdaptiveMutationRate = 0.1
            params.AdaptiveMutationMaxFactor = 3.0

    if hasattr(params, "FitnessScaling"):
        params.FitnessScaling = neat.SIGMA_FITNESS_SCALING
        params.FitnessSigmaScale = 2.0
    if hasattr(params, "OffspringAllocation"):
        params.OffspringAllocation = neat.STOCHASTIC_REMAINDER
    if hasattr(params, "SpeciesRepresentativeSelection"):
        params.SpeciesRepresentativeSelection = neat.MEDOID_REPRESENTATIVE
        params.RepresentativeSelectionCandidates = 24
    if hasattr(params, "RequireEvaluatedGenomes"):
        params.RequireEvaluatedGenomes = True
        params.RejectNonFiniteFitness = True

    if hasattr(params, "Validate"):
        valid, message = params.Validate()
        if not valid:
            raise ValueError(f"Invalid demo parameters: {message}")
    return params


def create_population(
    shape: TaskShape,
    config: TaskConfig,
    population_size: int,
    seed: int,
    profile: str,
    initial_connectivity: str,
    spiking: bool = False,
    mcculloch_pitts: bool = False,
) -> Any:
    recurrent = config.recurrent or profile == "exploratory"
    params = configure_parameters(
        population_size,
        recurrent,
        profile,
        spiking,
        mcculloch_pitts,
    )
    initial = neat.GenomeInitStruct()
    initial.NumInputs = shape.input_size + 1
    initial.NumOutputs = shape.output_size
    initial.NumHidden = 0
    initial.NumLayers = 0
    initial.SeedType = neat.GenomeSeedType.PERCEPTRON
    initial.HiddenActType = neat.TANH
    initial.OutputActType = neat.TANH
    if spiking:
        configure_spiking_genome(
            neat,
            initial,
            neuron_model=(
                "mcculloch-pitts" if mcculloch_pitts else "lif"
            ),
        )
    use_sparse_seed = shape.output_size > 1 and (
        initial_connectivity == "sparse"
        or (initial_connectivity == "auto" and config.sparse_seed)
    )
    initial.FS_NEAT = use_sparse_seed
    initial.FS_NEAT_links = max(1, shape.output_size)

    prototype = neat.Genome(params, initial)
    return neat.Population(prototype, params, True, 1.0, seed)


@dataclass
class TrainOptions:
    task: str
    generations: int
    population: int
    max_steps: int
    episodes: int
    workers: int
    seed: int
    activation_steps: int
    profile: str
    initial_connectivity: str
    spiking: bool = False
    mcculloch_pitts: bool = False
    spiking_steps: int = 8
    spiking_time_step: float = 0.001
    spiking_input_rate: float = 200.0
    spiking_output_rate: float = 200.0
    output_dir: Path | None = None
    checkpoint_every: int = 0
    resume: Path | None = None
    render_every: int = 0
    record_video: bool = False
    plot: bool = False
    quiet: bool = False


@dataclass
class TrainResult:
    task: str
    generations: int
    best_fitness: float
    best_genome: Any
    history: list[dict[str, Any]]


def _spiking_settings(options: TrainOptions) -> SpikingPolicySettings | None:
    if not options.spiking:
        return None
    return SpikingPolicySettings(
        simulation_steps=options.spiking_steps,
        time_step=options.spiking_time_step,
        input_rate_hz=options.spiking_input_rate,
        output_rate_hz=options.spiking_output_rate,
    )


def _genomes(population: Any) -> list[Any]:
    return [
        genome
        for species in population.m_Species
        for genome in species.m_Individuals
    ]


def _copy_genome(genome: Any) -> Any:
    if hasattr(neat.Genome, "Deserialize") and hasattr(genome, "Serialize"):
        return neat.Genome.Deserialize(genome.Serialize())
    return pickle.loads(pickle.dumps(genome))


def _save_genome(genome: Any, path: Path) -> Path:
    """Save a genome with the best format exposed by the loaded binding."""

    if hasattr(genome, "Save"):
        genome.Save(str(path))
        return path
    if hasattr(genome, "Serialize"):
        path.write_text(genome.Serialize(), encoding="utf-8")
        return path
    pickle_path = path.with_suffix(".pickle")
    with pickle_path.open("wb") as stream:
        pickle.dump(genome, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return pickle_path


def _save_population(population: Any, path: Path) -> Path:
    """Save a resumable population, including a fallback for old bindings."""

    if hasattr(population, "SaveState") and hasattr(neat.Population, "Deserialize"):
        population.SaveState(str(path))
        return path
    if hasattr(population, "Save"):
        legacy_path = path.with_suffix(".legacy")
        population.Save(str(legacy_path))
        return legacy_path
    pickle_path = path.with_suffix(".pickle")
    with pickle_path.open("wb") as stream:
        pickle.dump(population, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return pickle_path


def _load_population(path: Path) -> Any:
    if path.suffix == ".legacy":
        return neat.Population(str(path))
    if path.suffix == ".pickle":
        with path.open("rb") as stream:
            return pickle.load(stream)
    try:
        return neat.Population(str(path))
    except Exception:
        fallback = path.with_suffix(".pickle")
        if not fallback.exists():
            raise
        with fallback.open("rb") as stream:
            return pickle.load(stream)


def _write_run_metadata(
    options: TrainOptions,
    config: TaskConfig,
    shape: TaskShape,
) -> None:
    if options.output_dir is None:
        return
    options.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "options": {
            **asdict(options),
            "output_dir": str(options.output_dir),
            "resume": str(options.resume) if options.resume else None,
        },
        "task": {**asdict(config), "env_kwargs": dict(config.env_kwargs)},
        "shape": asdict(shape),
    }
    (options.output_dir / "run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_metrics(output_dir: Path | None, row: Mapping[str, Any]) -> None:
    if output_dir is None:
        return
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _render_or_record(
    genome: Any,
    config: TaskConfig,
    options: TrainOptions,
    generation: int,
    record: bool,
) -> float:
    if options.spiking and not record:
        return _show_spiking_rollout(genome, config, options)
    render_mode = "rgb_array" if record else "human"
    env = make_environment(config, render_mode)
    if record:
        if options.output_dir is None:
            raise ValueError("--record-video requires --output-dir")
        video_dir = options.output_dir / "videos" / f"generation-{generation}"
        video_dir.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            env.close()
            raise RuntimeError(
                "--record-video requires ImageIO/FFmpeg; install "
                "requirements-demos.txt"
            ) from error
        path = video_dir / f"{config.key}.mp4"
        encoder = ObservationEncoder(
            env.observation_space,
            config.image_size,
            config.crop_bottom,
        )
        adapter = ActionAdapter(env.action_space)
        network = neat.NeuralNetwork()
        genome.BuildPhenotype(network)
        recorder = None
        if options.spiking:
            from neattools import SpikingRecorder

            recorder = SpikingRecorder(network)
            spiking_policy = SpikingPolicy(
                network,
                _spiking_settings(options),
                recorder=recorder,
            )
        else:
            spiking_policy = None
        if spiking_policy is None:
            network.Flush()
        else:
            spiking_policy.reset(
                options.seed + generation * options.episodes
            )
        writer = None
        total_reward = 0.0
        try:
            writer = imageio.get_writer(
                str(path),
                fps=30,
                codec="libx264",
                quality=8,
            )
            observation, _ = env.reset(
                seed=options.seed + generation * options.episodes
            )
            frame = env.render()
            if frame is not None:
                writer.append_data(np.asarray(frame))
            for _ in range(options.max_steps):
                action = _network_action(
                    network,
                    observation,
                    encoder,
                    adapter,
                    options.activation_steps,
                    spiking_policy,
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                frame = env.render()
                if frame is not None:
                    writer.append_data(np.asarray(frame))
                if terminated or truncated:
                    break
            if recorder is not None:
                recorder.save(video_dir / f"{config.key}-spikes.npz")
            return total_reward
        except Exception as error:
            raise RuntimeError(f"Could not record {path}: {error}") from error
        finally:
            if writer is not None:
                writer.close()
            env.close()
    try:
        return evaluate_on_environment(
            genome,
            env,
            config,
            [options.seed + generation * options.episodes],
            options.max_steps,
            options.activation_steps,
            _spiking_settings(options),
        )
    finally:
        env.close()


def _record_spiking_rollout(
    genome: Any,
    config: TaskConfig,
    options: TrainOptions,
    *,
    max_steps: int | None = None,
) -> tuple[Any, Any, Any | None, float]:
    """Replay one policy with aligned environment and SNN recording."""

    from neattools import SpikingRecorder

    env = make_environment(config, "rgb_array")
    try:
        encoder = ObservationEncoder(
            env.observation_space,
            config.image_size,
            config.crop_bottom,
        )
        adapter = ActionAdapter(env.action_space)
        network = neat.NeuralNetwork()
        genome.BuildPhenotype(network)
        recorder = SpikingRecorder(network)
        settings = _spiking_settings(options)
        if settings is None:
            raise ValueError("spiking replay requires spiking options")
        policy = SpikingPolicy(network, settings, recorder=recorder)
        policy.reset(options.seed)
        observation, _ = env.reset(seed=options.seed)
        final_frame = env.render()
        total_reward = 0.0
        horizon = min(
            options.max_steps,
            max_steps if max_steps is not None else options.max_steps,
        )
        for _ in range(horizon):
            inputs = encoder.encode(observation).tolist()
            inputs.append(1.0)
            action = adapter.convert(policy.step_signed(inputs))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            frame = env.render()
            if frame is not None:
                final_frame = np.asarray(frame)
            if terminated or truncated:
                break
        return network, recorder, final_frame, total_reward
    finally:
        env.close()


def _show_spiking_rollout(
    genome: Any,
    config: TaskConfig,
    options: TrainOptions,
) -> float:
    """Show environment, phenotype, raster, and membranes for one replay."""

    try:
        import matplotlib.pyplot as plt
        from neattools import (
            DrawSpikingNetwork,
            PlotMembraneTraces,
            PlotSpikeRaster,
        )
    except ImportError as error:
        raise RuntimeError(
            "spiking rendering requires requirements-visualization.txt"
        ) from error

    network, recorder, frame, reward = _record_spiking_rollout(
        genome,
        config,
        options,
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        constrained_layout=True,
    )
    if frame is not None:
        axes[0, 0].imshow(frame)
    axes[0, 0].set_title(f"{config.env_id} · reward={reward:.3g}")
    axes[0, 0].axis("off")
    DrawSpikingNetwork(
        network,
        ax=axes[0, 1],
        title="Live spiking phenotype",
        show=False,
    )
    PlotSpikeRaster(
        recorder,
        ax=axes[1, 0],
        title="Policy spike trains",
        show=False,
    )
    output_neurons = list(
        range(
            network.NumInputs(),
            network.NumInputs() + network.NumOutputs(),
        )
    )
    PlotMembraneTraces(
        recorder,
        ax=axes[1, 1],
        neurons=output_neurons,
        title="Output membrane potentials",
        show=False,
    )
    plt.show()
    return reward


def _save_plot(
    result: TrainResult,
    output_dir: Path,
    config: TaskConfig,
    options: TrainOptions,
) -> None:
    try:
        import matplotlib

        if not os.environ.get("DISPLAY") and os.name != "nt":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        demos_dir = Path(__file__).resolve().parent
        if str(demos_dir.parent) not in sys.path:
            sys.path.insert(0, str(demos_dir.parent))
        from neattools import DrawGenome, DrawSpikingNetwork, PlotSpikeRaster
    except ImportError as error:
        raise RuntimeError(
            "--plot requires requirements-visualization.txt"
        ) from error

    if options.spiking:
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(16, 11),
            constrained_layout=True,
        )
        fitness_axis = axes[0, 0]
        environment_axis = axes[0, 1]
        genome_axis = axes[1, 0]
        raster_axis = axes[1, 1]
    else:
        figure, (fitness_axis, genome_axis) = plt.subplots(
            1,
            2,
            figsize=(14, 6),
        )
    background = "#0f172a"
    panel = "#111827"
    foreground = "#e5e7eb"
    figure.patch.set_facecolor(background)
    fitness_axis.set_facecolor(panel)
    fitness_axis.plot(
        [row["generation"] for row in result.history],
        [row["best_fitness"] for row in result.history],
        label="generation best",
        marker="o",
    )
    fitness_axis.plot(
        [row["generation"] for row in result.history],
        [row["mean_fitness"] for row in result.history],
        label="population mean",
        alpha=0.7,
        marker="o",
    )
    fitness_axis.set(
        title=f"{result.task} fitness",
        xlabel="Generation",
        ylabel="Fitness",
    )
    fitness_axis.title.set_color(foreground)
    fitness_axis.xaxis.label.set_color(foreground)
    fitness_axis.yaxis.label.set_color(foreground)
    fitness_axis.tick_params(colors=foreground)
    for spine in fitness_axis.spines.values():
        spine.set_color("#64748b")
    fitness_axis.grid(color="#94a3b8", alpha=0.2)
    legend = fitness_axis.legend(facecolor=panel, edgecolor="#64748b")
    for text in legend.get_texts():
        text.set_color(foreground)
    if len(result.history) == 1:
        generation = result.history[0]["generation"]
        fitness_axis.set_xlim(generation - 0.5, generation + 0.5)
        values = (
            result.history[0]["best_fitness"],
            result.history[0]["mean_fitness"],
        )
        padding = max(1.0, abs(max(values) - min(values)) * 0.2)
        fitness_axis.set_ylim(min(values) - padding, max(values) + padding)
    if options.spiking:
        network, recorder, final_frame, replay_reward = (
            _record_spiking_rollout(
                result.best_genome,
                config,
                options,
                max_steps=min(options.max_steps, 250),
            )
        )
        if final_frame is not None:
            environment_axis.imshow(final_frame)
        environment_axis.set_title(
            f"{config.env_id} replay · reward={replay_reward:.3g}",
            color=foreground,
        )
        environment_axis.set_facecolor(panel)
        environment_axis.axis("off")
        DrawSpikingNetwork(
            network,
            ax=genome_axis,
            title="Best evolved spiking phenotype",
            show=False,
        )
        PlotSpikeRaster(
            recorder,
            ax=raster_axis,
            title="Policy spike trains",
            show=False,
        )
        recorder.save(output_dir / "spiking_trace.npz")
    else:
        DrawGenome(
            result.best_genome,
            ax=genome_axis,
            layout="topology",
            with_edge_labels=False,
            show=False,
        )
        genome_axis.set_title("Best evolved topology")
        figure.tight_layout()
    figure.savefig(output_dir / "summary.png", dpi=160)
    plt.close(figure)


def train(options: TrainOptions) -> TrainResult:
    """Run a complete evolutionary experiment."""

    require_dependencies()
    if options.task not in TASKS:
        raise KeyError(f"Unknown task {options.task!r}")
    if options.generations < 1 or options.population < 2:
        raise ValueError("generations must be >= 1 and population must be >= 2")
    if options.max_steps < 1 or options.episodes < 1:
        raise ValueError("max_steps and episodes must be >= 1")
    if options.activation_steps < 1:
        raise ValueError("activation_steps must be >= 1")
    if options.spiking:
        _spiking_settings(options).validate()
    if (options.plot or options.record_video) and options.output_dir is None:
        raise ValueError("--plot and --record-video require --output-dir")

    config = TASKS[options.task]
    shape = inspect_task(config, options.seed)
    _write_run_metadata(options, config, shape)

    if options.resume:
        population = _load_population(options.resume)
        best = population.GetBestGenome()
        if (
            best.NumInputs() != shape.input_size + 1
            or best.NumOutputs() != shape.output_size
        ):
            raise ValueError(
                "Checkpoint policy dimensions do not match the selected task"
            )
        resumed_network = neat.NeuralNetwork()
        best.BuildPhenotype(resumed_network)
        if bool(resumed_network.IsSpiking()) != options.spiking:
            raise ValueError(
                "Checkpoint policy type does not match --spiking selection"
            )
    else:
        population = create_population(
            shape,
            config,
            options.population,
            options.seed,
            options.profile,
            options.initial_connectivity,
            options.spiking,
            options.mcculloch_pitts,
        )

    start_generation = int(population.GetGeneration())
    serial_env = None
    pool = None
    if options.workers > 1:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(
            options.workers,
            initializer=_initialize_worker,
            initargs=(
                config,
                options.max_steps,
                options.activation_steps,
                _spiking_settings(options),
            ),
        )
    else:
        serial_env = make_environment(config)

    history = []
    best_fitness = -math.inf
    best_genome = None
    started = time.perf_counter()
    try:
        for offset in range(options.generations):
            generation = start_generation + offset
            genomes = _genomes(population)
            episode_seeds = tuple(
                options.seed
                + generation * options.episodes
                + episode
                for episode in range(options.episodes)
            )
            if pool is not None:
                fitnesses = pool.map(
                    _worker_evaluate,
                    [(genome, episode_seeds) for genome in genomes],
                )
            else:
                fitnesses = [
                    evaluate_on_environment(
                        genome,
                        serial_env,
                        config,
                        episode_seeds,
                        options.max_steps,
                        options.activation_steps,
                        _spiking_settings(options),
                    )
                    for genome in genomes
                ]

            for genome, fitness in zip(genomes, fitnesses):
                genome.SetFitness(float(fitness))
                genome.SetEvaluated()

            best_index = int(np.argmax(fitnesses))
            generation_best = float(fitnesses[best_index])
            if generation_best > best_fitness:
                best_fitness = generation_best
                best_genome = _copy_genome(genomes[best_index])
                if options.output_dir is not None:
                    _save_genome(
                        best_genome,
                        options.output_dir / "best_genome.txt",
                    )

            elapsed = time.perf_counter() - started
            row = {
                "generation": generation,
                "best_fitness": generation_best,
                "best_ever": best_fitness,
                "mean_fitness": float(np.mean(fitnesses)),
                "median_fitness": float(np.median(fitnesses)),
                "species": len(population.m_Species),
                "genomes": len(genomes),
                "elapsed_seconds": elapsed,
                "spiking": options.spiking,
                "neuron_model": (
                    "mcculloch-pitts"
                    if options.mcculloch_pitts
                    else ("lif" if options.spiking else "rate")
                ),
            }
            history.append(row)
            _append_metrics(options.output_dir, row)
            if not options.quiet:
                print(
                    f"[{config.key}] generation={generation} "
                    f"best={generation_best:.6g} "
                    f"mean={row['mean_fitness']:.6g} "
                    f"species={row['species']} elapsed={elapsed:.2f}s",
                    flush=True,
                )

            if options.render_every and (
                (offset + 1) % options.render_every == 0
            ):
                _render_or_record(
                    best_genome,
                    config,
                    options,
                    generation,
                    record=False,
                )

            population.Epoch()
            if (
                options.output_dir is not None
                and options.checkpoint_every
                and (offset + 1) % options.checkpoint_every == 0
            ):
                _save_population(
                    population,
                    options.output_dir / "population.state",
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if serial_env is not None:
            serial_env.close()

    if best_genome is None:
        raise RuntimeError("Evolution produced no evaluated genome")
    if options.output_dir is not None:
        _save_population(population, options.output_dir / "population.state")
    result = TrainResult(
        task=config.key,
        generations=options.generations,
        best_fitness=best_fitness,
        best_genome=best_genome,
        history=history,
    )
    if options.record_video:
        _render_or_record(
            best_genome,
            config,
            options,
            start_generation + options.generations - 1,
            record=True,
        )
    if options.plot:
        if options.output_dir is None:
            raise ValueError("--plot requires --output-dir")
        _save_plot(
            result,
            options.output_dir,
            config,
            options,
        )
    return result


def smoke_task(
    task_key: str,
    seed: int = 7,
    quiet: bool = False,
    spiking: bool = False,
    mcculloch_pitts: bool = False,
) -> TrainResult:
    """Run a tiny real environment/evolution integration test."""

    spiking = spiking or mcculloch_pitts
    return train(
        TrainOptions(
            task=task_key,
            generations=1,
            population=4,
            max_steps=3,
            episodes=1,
            workers=1,
            seed=seed,
            activation_steps=1,
            profile="default",
            initial_connectivity="sparse",
            spiking=spiking,
            mcculloch_pitts=mcculloch_pitts,
            spiking_steps=4,
            quiet=quiet,
        )
    )


def catalog(family: str | None = None) -> Iterable[TaskConfig]:
    for config in TASKS.values():
        if family is None or config.family == family:
            yield config


def _parser(default_task: str | None) -> argparse.ArgumentParser:
    config = TASKS.get(default_task) if default_task else None
    parser = argparse.ArgumentParser(
        description=(
            config.description
            if config
            else "Train MultiNEAT policies on Gymnasium physics tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default=default_task,
        required=default_task is None,
        help="registered task configuration",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=config.default_generations if config else None,
    )
    parser.add_argument(
        "--population",
        type=int,
        default=config.default_population if config else None,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.default_max_steps if config else None,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.default_episodes if config else None,
        help="common-seed episodes per genome and generation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel evaluator processes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--activation-steps",
        type=int,
        default=2,
        help="network propagation steps per environment step",
    )
    parser.add_argument(
        "--spiking",
        action="store_true",
        help="evolve a Poisson-encoded spiking policy variant",
    )
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="use evolvable McCulloch-Pitts neurons in the spiking variant",
    )
    parser.add_argument(
        "--spiking-steps",
        type=int,
        default=8,
        help="SNN solver steps per environment action",
    )
    parser.add_argument(
        "--spiking-time-step",
        type=float,
        default=0.001,
        help="SNN solver time step in seconds",
    )
    parser.add_argument(
        "--spiking-input-rate",
        type=float,
        default=200.0,
        help="maximum Poisson observation rate in hertz",
    )
    parser.add_argument(
        "--spiking-output-rate",
        type=float,
        default=200.0,
        help="filtered output rate mapped to a full action",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "ranked", "exploratory"),
        default="ranked",
        help="evolutionary operator preset",
    )
    parser.add_argument(
        "--initial-connectivity",
        choices=("auto", "full", "sparse"),
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--render-every", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="construct the environment and print inferred policy dimensions",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one generation of four genomes for three environment steps",
    )
    return parser


def main(default_task: str | None = None) -> int:
    parser = _parser(default_task)
    args = parser.parse_args()
    if args.mcculloch_pitts:
        args.spiking = True
    config = TASKS[args.task]
    try:
        if args.inspect:
            shape = inspect_task(config, args.seed)
            print(
                json.dumps(
                    {
                        "task": asdict(config),
                        "shape": asdict(shape),
                    },
                    indent=2,
                    default=str,
                )
            )
            return 0
        if args.smoke:
            result = smoke_task(
                args.task,
                args.seed,
                args.quiet,
                args.spiking,
                args.mcculloch_pitts,
            )
        else:
            generations = (
                args.generations
                if args.generations is not None
                else config.default_generations
            )
            population = (
                args.population
                if args.population is not None
                else config.default_population
            )
            max_steps = (
                args.max_steps
                if args.max_steps is not None
                else config.default_max_steps
            )
            episodes = (
                args.episodes
                if args.episodes is not None
                else config.default_episodes
            )
            result = train(
                TrainOptions(
                    task=args.task,
                    generations=generations,
                    population=population,
                    max_steps=max_steps,
                    episodes=episodes,
                    workers=args.workers,
                    seed=args.seed,
                    activation_steps=args.activation_steps,
                    profile=args.profile,
                    initial_connectivity=args.initial_connectivity,
                    spiking=args.spiking,
                    mcculloch_pitts=args.mcculloch_pitts,
                    spiking_steps=args.spiking_steps,
                    spiking_time_step=args.spiking_time_step,
                    spiking_input_rate=args.spiking_input_rate,
                    spiking_output_rate=args.spiking_output_rate,
                    output_dir=args.output_dir,
                    checkpoint_every=args.checkpoint_every,
                    resume=args.resume,
                    render_every=args.render_every,
                    record_video=args.record_video,
                    plot=args.plot,
                    quiet=args.quiet,
                )
            )
        print(
            f"completed task={result.task} generations={result.generations} "
            f"best_fitness={result.best_fitness:.6g} "
            f"policy={'mcculloch-pitts' if args.mcculloch_pitts else ('lif' if args.spiking else 'rate')}",
            flush=True,
        )
        return 0
    except (RuntimeError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
