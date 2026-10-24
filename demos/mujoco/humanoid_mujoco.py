#!/usr/bin/env python3
"""Evolve a MultiNEAT controller for Humanoid-v5 (humanoid walking).

Observation: 348 unbounded values (pose, velocities, inertia, actuator and
contact forces), squashed into [-1, 1] with tanh.
Action: 17 actuator commands in [-0.4, 0.4].
The large input layer starts sparsely connected (FS-NEAT).

Self-contained: needs only pymultineat, numpy and gymnasium[mujoco]
(pip install -r requirements-mujoco.txt).

Examples:
    python demos/mujoco/humanoid_mujoco.py --smoke
    python demos/mujoco/humanoid_mujoco.py --fast --output-dir runs/humanoid
    python demos/mujoco/humanoid_mujoco.py --fast --workers 8 --plot \\
        --output-dir runs/humanoid
    python demos/mujoco/humanoid_mujoco.py --fast --output-dir runs/humanoid \\
        --resume runs/humanoid
    python demos/mujoco/humanoid_mujoco.py --spiking --fast
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ``python demos/<family>/<file>.py`` puts only this file's directory on
# sys.path; append the repository root so an in-tree pymultineat build is found.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

try:
    import gymnasium as gym
    import numpy as np
except ImportError:  # --help still works; main() explains what is missing.
    gym = np = None
try:
    import pymultineat as neat
except ImportError:
    neat = None

# --- Task configuration ------------------------------------------------------

TASK = "mujoco_humanoid"
ENV_ID = "Humanoid-v5"
DESCRIPTION = "Seventeen-actuator humanoid locomotion."
REQUIREMENTS = "requirements-mujoco.txt"
POPULATION = 320
GENERATIONS = 1600
MAX_STEPS = 1000
EPISODES = 1
RECURRENT = True
SPARSE_SEED = True
SUCCESS_THRESHOLD: float | None = None


def make_env(render_mode: str | None = None) -> Any:
    """Create the environment, explaining how to install a missing backend."""

    try:
        return gym.make(ENV_ID, render_mode=render_mode)
    except Exception as error:
        raise RuntimeError(
            f"Could not create {ENV_ID} ({error}). "
            f"Install the physics backend: pip install -r {REQUIREMENTS}"
        ) from error


# --- Observation encoding and action mapping ---------------------------------

OBS_SIZE = 348
NUM_ACTIONS = 17  # abdomen, hips, knees, shoulders and elbows
ACTION_BOUND = 0.4  # every actuator accepts [-0.4, 0.4]
NUM_INPUTS = OBS_SIZE + 1  # + a constant bias input
NUM_OUTPUTS = NUM_ACTIONS


def encode_observation(observation: Any) -> list[float]:
    """Joint positions and velocities are unbounded: squash them with tanh."""

    values = np.tanh(np.asarray(observation, dtype=np.float64))
    return np.nan_to_num(values, nan=0.0).tolist()


def decode_action(outputs: Any) -> Any:
    """Scale each output from [-1, 1] onto [-ACTION_BOUND, ACTION_BOUND]."""

    values = np.asarray(outputs, dtype=np.float64)[:NUM_OUTPUTS]
    values = np.clip(np.nan_to_num(values, nan=0.0), -1.0, 1.0)
    return (values * ACTION_BOUND).astype(np.float32)


# --- Policies: a rate network or a Poisson-encoded spiking network -----------


@dataclass(frozen=True)
class SpikingSettings:
    steps: int = 8  # SNN solver steps per environment action
    time_step: float = 0.001  # seconds
    input_rate: float = 200.0  # Poisson rate (Hz) for an input of +1
    output_rate: float = 200.0  # filtered output rate (Hz) mapped to +1


class Policy:
    """Build a genome's phenotype and turn observations into actions."""

    def __init__(self, genome: Any, activation_steps: int, spiking: SpikingSettings | None):
        self.network = neat.NeuralNetwork()
        genome.BuildPhenotype(self.network)
        self.activation_steps = activation_steps
        self.spiking = spiking
        if spiking is not None:
            if not self.network.IsSpiking():
                raise ValueError("--spiking requires a spiking genome")
            self.network.SetSpikingInputMode(neat.POISSON_RATE_INPUT)
            self.network.SetSpikingOutputMode(neat.FILTERED_SPIKE_OUTPUT)
            self.network.SetSpikingTimeStep(spiking.time_step)
            self.network.EnableSTDP(False)

    def reset(self, seed: int) -> None:
        self.network.Flush()
        if self.spiking is not None:
            self.network.SeedSpiking(int(seed))

    def act(self, observation: Any) -> Any:
        inputs = encode_observation(observation) + [1.0]
        if self.spiking is None:
            self.network.Input(inputs)
            self.network.ActivateSteps(self.activation_steps, True)
            return decode_action(self.network.Output())
        rates = [0.5 * (value + 1.0) * self.spiking.input_rate for value in inputs]
        outputs: list[float] = []
        for _ in range(self.spiking.steps):
            outputs = list(self.network.StepSpiking(rates, self.spiking.time_step))
        unsigned = np.clip(np.asarray(outputs) / self.spiking.output_rate, 0.0, 1.0)
        return decode_action(2.0 * unsigned - 1.0)


def run_episode(policy: Policy, env: Any, seed: int, max_steps: int,
                on_frame: Callable[[Any], None] | None = None) -> float:
    """Play one seeded episode and return its total reward."""

    policy.reset(seed)
    observation, _ = env.reset(seed=int(seed))
    total = 0.0
    for _ in range(max_steps):
        observation, reward, terminated, truncated, _ = env.step(policy.act(observation))
        if on_frame is not None:
            on_frame(env.render())
        if not math.isfinite(float(reward)):
            return -1_000_000.0
        total += float(reward)
        if terminated or truncated:
            break
    return total


def evaluate(genome: Any, env: Any, seeds: tuple[int, ...], settings: tuple) -> float:
    """Mean return over seeded episodes; settings = (max_steps, activation_steps, spiking)."""

    max_steps, activation_steps, spiking = settings
    policy = Policy(genome, activation_steps, spiking)
    fitness = float(np.mean([run_episode(policy, env, s, max_steps) for s in seeds]))
    return fitness if math.isfinite(fitness) else -1_000_000.0


# --- Serial or multiprocess evaluation ---------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(settings: tuple) -> None:
    _WORKER["env"] = make_env()
    _WORKER["settings"] = settings


def _worker_evaluate(job: tuple[Any, tuple[int, ...]]) -> float:
    genome, seeds = job
    return evaluate(genome, _WORKER["env"], seeds, _WORKER["settings"])


class Evaluator:
    """Score (genome, seeds) jobs in-process or on a spawn-based worker pool."""

    def __init__(self, workers: int, settings: tuple) -> None:
        self.settings = settings
        self.pool = self.env = None
        if workers > 1:
            context = multiprocessing.get_context("spawn")
            self.pool = context.Pool(workers, initializer=_init_worker, initargs=(settings,))
        else:
            self.env = make_env()

    def __call__(self, jobs: list[tuple[Any, tuple[int, ...]]]) -> list[float]:
        if self.pool is not None:
            return [float(score) for score in self.pool.map(_worker_evaluate, jobs)]
        return [evaluate(genome, self.env, seeds, self.settings) for genome, seeds in jobs]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        if self.env is not None:
            self.env.close()


def case_seed(base_seed: int, case_id: int) -> int:
    """Map a reliability case id onto a disjoint 31-bit environment seed."""

    mask = (1 << 64) - 1
    value = (int(case_id) + 0x9E3779B97F4A7C15 + int(base_seed)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return int((value ^ (value >> 31)) & 0x7FFFFFFF)


# --- NEAT configuration ------------------------------------------------------


def make_parameters(args: argparse.Namespace) -> Any:
    legacy = args.legacy_evolution
    recurrent = RECURRENT or args.profile == "exploratory"
    params = neat.Parameters()
    if legacy:
        params.ConfigureLegacyV2()
    else:
        params.ReliabilityTarget = args.reliability_target
    for name in dir(params):  # tanh neurons only (set below), rare activation mutation
        if name.startswith("ActivationFunction_") and name.endswith("_Prob"):
            setattr(params, name, 0.0)
    settings: dict[str, Any] = {
        # population and speciation
        "PopulationSize": args.population, "DynamicCompatibility": args.population >= 8,
        "MinSpecies": 1, "MaxSpecies": max(1, min(12, args.population // 4)),
        "CompatTreshold": 3.0, "YoungAgeTreshold": 15, "OldAgeTreshold": 50,
        "SpeciesMaxStagnation": 30, "SurvivalRate": 0.25,
        "EliteFraction": min(0.1, 1.0 / args.population), "AllowClones": False,
        # topology
        "AllowLoops": recurrent, "RecurrentProb": 0.1 if recurrent else 0.0,
        "SplitRecurrent": recurrent, "SplitLoopedRecurrent": recurrent,
        # mutation and crossover
        "OverallMutationRate": 0.8, "MutateWeightsProb": 0.9, "MutateWeightsSevereProb": 0.1,
        "WeightMutationRate": 0.9, "WeightReplacementRate": 0.1,
        "WeightMutationMaxPower": 0.5, "WeightReplacementMaxPower": 2.0,
        "MinWeight": -5.0, "MaxWeight": 5.0,
        "MutateAddNeuronProb": 0.02, "MutateAddLinkProb": 0.08,
        "MutateRemLinkProb": 0.01, "MutateRemSimpleNeuronProb": 0.005,
        "CrossoverRate": 0.6, "MultipointCrossoverRate": 0.5,
        "ActivationFunction_Tanh_Prob": 1.0, "MutateNeuronActivationTypeProb": 0.02,
    }
    for name, value in settings.items():
        setattr(params, name, value)

    if args.spiking:  # control-oriented evolvable SNN preset
        if args.mcculloch_pitts:
            params.ConfigureMcCullochPitts(True, False)
        else:
            params.ConfigureSpiking(False)
        settings = {
            "DontUseBiasNeuron": True, "RecurrentProb": 0.2 if recurrent else 0.0,
            "MinWeight": -12.0, "MaxWeight": 12.0,
            "WeightMutationMaxPower": 1.0, "WeightReplacementMaxPower": 6.0,
            "MinSpikingTimeConstant": 0.005, "MaxSpikingTimeConstant": 0.04,
            "MinSpikeRateTimeConstant": 0.02, "MaxSpikeRateTimeConstant": 0.08,
            "MinSpikeThreshold": 0.4, "MaxSpikeThreshold": 1.4,
            "MinSynapticDelay": 0.0, "MaxSynapticDelay": 0.012,
            "InitialSTDPEnabledProb": 0.0,
        }
        for name, value in settings.items():
            setattr(params, name, value)

    settings = {  # selection; the "ranked" profile only changes legacy evolution
        "FitnessScaling": (neat.SIGMA_FITNESS_SCALING if legacy
                           else neat.LINEAR_RANK_FITNESS_SCALING),
        "OffspringAllocation": neat.STOCHASTIC_REMAINDER,
        "SpeciesRepresentativeSelection": neat.MEDOID_REPRESENTATIVE,
        "RepresentativeSelectionCandidates": 24,
        "RequireEvaluatedGenomes": True, "RejectNonFiniteFitness": True,
    }
    if legacy:
        settings["FitnessSigmaScale"] = 2.0
    if args.smoke:  # tiny reliability budget so CI smoke runs take seconds
        settings.update(EvaluationMaxSamples=2, ValidationSamples=2, CertificationSamples=4)
    if args.profile == "ranked" and legacy:
        settings.update(ParentSelectionMode=neat.RANK_LINEAR, RankSelectionPressure=1.7)
    elif args.profile == "exploratory":
        settings.update(
            ParentSelectionMode=neat.TOURNAMENT, TournamentSize=4,
            MutateAddNeuronProb=0.04, MutateAddLinkProb=0.12, AdaptiveMutationStart=15,
            AdaptiveMutationRate=0.1, AdaptiveMutationMaxFactor=3.0,
        )
    for name, value in settings.items():
        setattr(params, name, value)
    valid, message = params.Validate()
    if not valid:
        raise ValueError(f"Invalid NEAT parameters: {message}")
    return params


def make_population(args: argparse.Namespace) -> Any:
    params = make_parameters(args)
    initial = neat.GenomeInitStruct()
    initial.NumInputs = NUM_INPUTS
    initial.NumOutputs = NUM_OUTPUTS
    initial.NumHidden = initial.NumLayers = 0
    initial.SeedType = neat.GenomeSeedType.PERCEPTRON
    initial.HiddenActType = initial.OutputActType = neat.TANH
    if args.mcculloch_pitts:
        initial.HiddenActType = initial.OutputActType = neat.MCCULLOCH_PITTS
    elif args.spiking:
        initial.HiddenActType = neat.SPIKING_ADAPTIVE_LIF
        initial.OutputActType = neat.SPIKING_LIF
    sparse = args.initial_connectivity == "sparse" or (
        args.initial_connectivity == "auto" and SPARSE_SEED
    )
    initial.FS_NEAT = sparse and NUM_OUTPUTS > 1
    initial.FS_NEAT_links = NUM_OUTPUTS
    return neat.Population(neat.Genome(params, initial), params, True, 1.0, args.seed)


def load_population(path: Path, args: argparse.Namespace) -> Any:
    if path.is_dir():
        path = path / "population.state"
    population = neat.Population(str(path))
    best = population.GetBestGenome()
    if best.NumInputs() != NUM_INPUTS or best.NumOutputs() != NUM_OUTPUTS:
        raise ValueError(f"{path} does not match {ENV_ID}'s network shape")
    network = neat.NeuralNetwork()
    best.BuildPhenotype(network)
    if bool(network.IsSpiking()) != args.spiking:
        raise ValueError("checkpoint policy type does not match --spiking")
    return population


# --- Training loop -----------------------------------------------------------


def submit_requests(population: Any, requests: Any, evaluator: Evaluator, args) -> list[float]:
    """Score reliability-first evaluation requests and submit the results."""

    batch = list(requests)
    scores = evaluator([(population.GetEvaluationGenome(r), (case_seed(args.seed, r.case_id),))
                        for r in batch])
    threshold = args.success_threshold
    for request, score in zip(batch, scores):
        success = threshold is not None and score >= threshold
        population.SubmitEvaluation(request.genome_id, request.case_id, score,
                                    threshold is not None, success)
    return scores


def save_checkpoint(population: Any, best_genome: Any, output_dir: Path | None) -> None:
    if output_dir is not None:
        population.SaveState(str(output_dir / "population.state"))
        if best_genome is not None:
            best_genome.Save(str(output_dir / "best_genome.txt"))


def train(args: argparse.Namespace, spiking: SpikingSettings | None) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    population = load_population(args.resume, args) if args.resume else make_population(args)
    legacy = population.m_Parameters.EvolutionModeSetting == neat.LEGACY_V2
    evaluator = Evaluator(args.workers, (args.max_steps, args.activation_steps, spiking))
    preview_env = None
    first_generation = int(population.GetGeneration())
    best_genome, best_fitness = None, -math.inf
    history: list[dict[str, Any]] = []
    interactions = 0
    started = time.perf_counter()
    try:
        for offset in range(args.generations):
            generation = first_generation + offset
            scores: list[float] = []
            if legacy:  # every genome plays the same seeded episodes
                genomes = [g for s in population.m_Species for g in s.m_Individuals]
                seeds = tuple(args.seed + generation * args.episodes + e
                              for e in range(args.episodes))
                fitnesses = evaluator([(genome, seeds) for genome in genomes])
                for genome, fitness in zip(genomes, fitnesses):
                    genome.SetFitness(fitness)
                    genome.SetEvaluated()
                interactions += len(genomes) * len(seeds)
                champion = genomes[int(np.argmax(fitnesses))]
            else:  # the population decides which genome plays which case
                requests = population.GetEvaluationRequests()
                while requests:
                    scores += submit_requests(population, requests, evaluator, args)
                    requests = population.GetEvaluationRequests()
                interactions += len(scores)
                genomes = [g for s in population.m_Species for g in s.m_Individuals]
                fitnesses = [float(genome.GetFitness()) for genome in genomes]
                champion = population.GetProvisionalChampion()
            if float(champion.GetFitness()) > best_fitness:
                best_fitness = float(champion.GetFitness())
                best_genome = neat.Genome.Deserialize(champion.Serialize())
                if output_dir is not None:
                    best_genome.Save(str(output_dir / "best_genome.txt"))

            threshold = args.success_threshold
            row: dict[str, Any] = {
                "generation": generation,
                "best_fitness": float(champion.GetFitness()),
                "best_ever": best_fitness,
                "mean_fitness": float(np.mean(fitnesses)),
                "median_fitness": float(np.median(fitnesses)),
                "success_rate": (float(np.mean(np.asarray(scores) >= threshold))
                                 if scores and threshold is not None else None),
                "species": len(population.m_Species),
                "environment_interactions": interactions,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            history.append(row)
            if not args.quiet:
                print(f"[{TASK}] generation={generation} best={row['best_fitness']:.6g} "
                      f"mean={row['mean_fitness']:.6g} species={row['species']} "
                      f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)

            if not args.fast and args.render_every and (offset + 1) % args.render_every == 0:
                preview_env = preview_env or make_env("human")
                policy = Policy(best_genome, args.activation_steps, spiking)
                run_episode(policy, preview_env, args.seed, args.max_steps)

            if offset + 1 == args.generations and not legacy:
                # Re-test the leading candidates on fresh cases and certify one.
                requests = population.ValidateCandidates(5)
                while requests:
                    interactions += len(submit_requests(population, requests, evaluator, args))
                    requests = population.GetEvaluationRequests()
                champion = population.GetValidatedChampion()
                best_genome = neat.Genome.Deserialize(champion.genome.Serialize())
                best_fitness = float(champion.validation_summary.mean)
                test = champion.certification_summary
                row.update(validated=bool(champion.validated), validation_mean=best_fitness,
                           test_mean=float(test.mean), test_success_rate=float(test.success_rate),
                           success_lower_bound=float(test.success_lower_bound),
                           environment_interactions=interactions)
            population.Epoch()  # a saved checkpoint resumes at the next generation

            if output_dir is not None:
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                if args.checkpoint_every and (offset + 1) % args.checkpoint_every == 0:
                    save_checkpoint(population, best_genome, output_dir)
    except KeyboardInterrupt:
        save_checkpoint(population, best_genome, output_dir)
        raise
    finally:
        evaluator.close()
        if preview_env is not None:
            preview_env.close()

    save_checkpoint(population, best_genome, output_dir)
    return {
        "best_genome": best_genome,
        "best_fitness": best_fitness,
        "history": history,
        "legacy": legacy,
        "validated": bool(history[-1].get("validated", False)),
        "success_lower_bound": float(history[-1].get("success_lower_bound", 0.0)),
        "environment_interactions": interactions,
    }


# --- Inspection, plotting, video and replay ----------------------------------


def inspect_task(seed: int) -> dict[str, Any]:
    env = make_env()
    try:
        observation, _ = env.reset(seed=seed)
        space = env.action_space
        outputs = int(space.n) if hasattr(space, "n") else int(np.prod(space.shape))
        inputs = len(encode_observation(observation)) + 1
        if (inputs, outputs) != (NUM_INPUTS, NUM_OUTPUTS):
            raise RuntimeError(f"{ENV_ID} needs a {inputs}x{outputs} network, not "
                               f"{NUM_INPUTS}x{NUM_OUTPUTS}; gymnasium version mismatch?")
        return {
            "demo": TASK,
            "env_id": ENV_ID,
            "observation_space": repr(env.observation_space),
            "action_space": repr(env.action_space),
            "observation_shape": list(np.shape(observation)),
            "network_inputs": NUM_INPUTS,
            "network_outputs": NUM_OUTPUTS,
            "defaults": {"population": POPULATION, "generations": GENERATIONS,
                         "max_steps": MAX_STEPS, "episodes": EPISODES,
                         "recurrent": RECURRENT, "sparse_seed": SPARSE_SEED,
                         "success_threshold": SUCCESS_THRESHOLD},
        }
    finally:
        env.close()


def save_plot(history: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = [row["generation"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(generations, [r["best_fitness"] for r in history], marker="o",
              label="generation best")
    axis.plot(generations, [r["mean_fitness"] for r in history], marker=".",
              label="population mean")
    if SUCCESS_THRESHOLD is not None:
        axis.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", label="success")
    axis.set(title=f"{ENV_ID} fitness", xlabel="Generation", ylabel="Episode return")
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def play(policy: Policy, seed: int, max_steps: int, video: Path | None = None) -> float:
    """Replay one episode in a window, or record it to an MP4 file."""

    env = make_env("rgb_array" if video else "human")
    try:
        if video is None:
            return run_episode(policy, env, seed, max_steps)
        import imageio.v2 as imageio

        fps = env.metadata.get("render_fps", 30)
        with imageio.get_writer(str(video), fps=fps, macro_block_size=1) as writer:
            return run_episode(policy, env, seed, max_steps, writer.append_data)
    finally:
        env.close()


# --- Command line ------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{ENV_ID}: {DESCRIPTION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add = parser.add_argument
    add("--generations", type=int, default=GENERATIONS)
    add("--population", type=int, default=POPULATION)
    add("--max-steps", type=int, default=MAX_STEPS)
    add("--episodes", type=int, default=EPISODES,
        help="episodes per genome (legacy evolution; reliable mode schedules cases itself)")
    add("--workers", type=int, default=1, help="parallel evaluator processes")
    add("--seed", type=int, default=42)
    add("--activation-steps", type=int, default=2, help="network propagation steps per env step")
    add("--profile", choices=("default", "ranked", "exploratory"), default="ranked",
        help="operator preset (ranked only affects --legacy-evolution)")
    add("--initial-connectivity", choices=("auto", "full", "sparse"), default="auto")
    add("--legacy-evolution", action="store_true", help="MultiNEAT 2 scalar-fitness evolution")
    add("--reliability-target", type=float, default=0.95,
        help="success lower bound required for validated=true")
    add("--success-threshold", type=float, default=SUCCESS_THRESHOLD,
        help="episode return counted as a success")
    add("--spiking", action="store_true", help="evolve an adaptive-LIF/LIF spiking policy")
    add("--mcculloch-pitts", action="store_true",
        help="spiking policy built from McCulloch-Pitts neurons")
    add("--spiking-steps", type=int, default=8, help="SNN solver steps per action")
    add("--spiking-time-step", type=float, default=0.001, help="SNN solver step (seconds)")
    add("--spiking-input-rate", type=float, default=200.0, help="Poisson rate (Hz) for input +1")
    add("--spiking-output-rate", type=float, default=200.0, help="output rate (Hz) mapped to +1")
    add("--output-dir", type=Path, help="save best genome, metrics.jsonl and checkpoints here")
    add("--checkpoint-every", type=int, default=10, metavar="N",
        help="save population.state every N generations (needs --output-dir)")
    add("--resume", type=Path, help="population.state (or its directory) to continue from")
    add("--fast", "--headless", dest="fast", action="store_true",
        help="no rendering: no champion previews and no final replay")
    add("--render-every", type=int, default=1, metavar="N",
        help="show the champion every N generations (0 = only the final replay)")
    add("--plot", action="store_true", help="save fitness.png to --output-dir (matplotlib)")
    add("--record-video", action="store_true", help="save best.mp4 to --output-dir (imageio)")
    add("--quiet", action="store_true", help="no per-generation progress lines")
    add("--inspect", action="store_true", help="print spaces and network size as JSON, then exit")
    add("--smoke", action="store_true", help="tiny headless run for CI; prints one JSON line")
    args = parser.parse_args(argv)
    args.spiking = args.spiking or args.mcculloch_pitts
    if args.smoke:
        args.generations, args.population, args.max_steps = 1, 4, 3
        args.episodes, args.workers, args.activation_steps = 1, 1, 1
        args.spiking_steps, args.initial_connectivity = 4, "sparse"
        args.fast = args.quiet = True
    if min(args.generations, args.max_steps, args.episodes, args.workers,
           args.activation_steps, args.spiking_steps) < 1 or args.population < 2:
        parser.error("counts must be >= 1 (population >= 2)")
    if not all(math.isfinite(v) and v > 0.0 for v in (
            args.spiking_time_step, args.spiking_input_rate, args.spiking_output_rate)):
        parser.error("spiking time step and rates must be positive")
    if (args.plot or args.record_video) and args.output_dir is None:
        parser.error("--plot and --record-video require --output-dir")
    return args


def require(module: str, install_hint: str) -> None:
    try:
        __import__(module)
    except ImportError:
        raise RuntimeError(f"{module} is required: {install_hint}") from None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if np is None or gym is None:
            raise RuntimeError(f"numpy and gymnasium are required: pip install -r {REQUIREMENTS}")
        if neat is None:
            raise RuntimeError("pymultineat is not importable: build it or add it to PYTHONPATH")
        if args.inspect:
            print(json.dumps(inspect_task(args.seed), indent=2))
            return 0
        if args.plot:
            require("matplotlib", "pip install -r requirements-visualization.txt")
        if args.record_video:
            require("imageio_ffmpeg", "pip install -r requirements-demos.txt")
        spiking = None
        if args.spiking:
            spiking = SpikingSettings(args.spiking_steps, args.spiking_time_step,
                                      args.spiking_input_rate, args.spiking_output_rate)
        started = time.perf_counter()
        result = train(args, spiking)
        best = result["best_genome"]
        if args.plot:
            save_plot(result["history"], args.output_dir / "fitness.png")
        if args.record_video:
            policy = Policy(best, args.activation_steps, spiking)
            play(policy, args.seed, args.max_steps, video=args.output_dir / "best.mp4")
        print(json.dumps({
            "demo": TASK,
            "env_id": ENV_ID,
            "policy": "mcculloch-pitts" if args.mcculloch_pitts
            else ("lif" if args.spiking else "rate"),
            "evolution_mode": "legacy-v2" if result["legacy"] else "reliable-neat",
            "generations": len(result["history"]),
            "population": args.population,
            "best_fitness": result["best_fitness"],
            "validated": result["validated"],
            "success_lower_bound": result["success_lower_bound"],
            "environment_interactions": result["environment_interactions"],
            "neurons": best.NumNeurons(),
            "links": best.NumLinks(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "output_dir": str(args.output_dir) if args.output_dir else None,
        }), flush=True)
        if not args.fast:
            policy = Policy(best, args.activation_steps, spiking)
            reward = play(policy, args.seed, args.max_steps)
            print(f"[{TASK}] replay reward={reward:.6g}", flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"[{TASK}] interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
