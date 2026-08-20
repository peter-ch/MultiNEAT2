#!/usr/bin/env python3
"""Evolve and visualize a spiking controller for a self-contained cart-pole."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pymultineat as neat  # noqa: E402


SNN_DT = 0.001
SNN_STEPS_PER_CONTROL = 10
CONTROL_DT = SNN_DT * SNN_STEPS_PER_CONTROL


@dataclass
class CartPole:
    x: float = 0.0
    x_velocity: float = 0.0
    angle: float = 0.035
    angle_velocity: float = 0.0

    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    force_scale: float = 10.0
    track_limit: float = 2.4
    angle_limit: float = 12.0 * math.pi / 180.0

    def reset(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.x = rng.uniform(-0.04, 0.04)
        self.x_velocity = rng.uniform(-0.02, 0.02)
        self.angle = rng.uniform(-0.04, 0.04)
        self.angle_velocity = rng.uniform(-0.02, 0.02)

    def step(self, normalized_force: float) -> bool:
        force = self.force_scale * max(-1.0, min(1.0, normalized_force))
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.half_length
        sine = math.sin(self.angle)
        cosine = math.cos(self.angle)
        temporary = (
            force
            + pole_mass_length * self.angle_velocity**2 * sine
        ) / total_mass
        angle_acceleration = (
            self.gravity * sine - cosine * temporary
        ) / (
            self.half_length
            * (4.0 / 3.0 - self.pole_mass * cosine**2 / total_mass)
        )
        x_acceleration = (
            temporary
            - pole_mass_length * angle_acceleration * cosine / total_mass
        )
        self.x += CONTROL_DT * self.x_velocity
        self.x_velocity += CONTROL_DT * x_acceleration
        self.angle += CONTROL_DT * self.angle_velocity
        self.angle_velocity += CONTROL_DT * angle_acceleration
        return (
            abs(self.x) < self.track_limit
            and abs(self.angle) < self.angle_limit
        )

    def rates(self) -> list[float]:
        values = (
            self.x / self.track_limit,
            math.tanh(self.x_velocity),
            self.angle / self.angle_limit,
            math.tanh(self.angle_velocity),
        )
        baseline = 2.0
        scale = 180.0
        rates = []
        for value in values:
            clipped = max(-1.0, min(1.0, value))
            rates.extend(
                (
                    baseline + scale * max(0.0, clipped),
                    baseline + scale * max(0.0, -clipped),
                )
            )
        return rates


def parameters(
    population_size: int,
    mcculloch_pitts: bool = False,
) -> neat.Parameters:
    params = neat.Parameters()
    if mcculloch_pitts:
        params.ConfigureMcCullochPitts(True, False)
    else:
        params.ConfigureSpiking(False)
    params.PopulationSize = population_size
    params.DontUseBiasNeuron = True
    params.DynamicCompatibility = True
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.CompatTreshold = 2.5
    params.MaxCompatTreshold = 20.0
    params.OverallMutationRate = 0.9
    params.MutateWeightsProb = 0.55
    params.MutateAddNeuronProb = 0.06
    params.MutateAddLinkProb = 0.10
    params.MutateRemLinkProb = 0.01
    params.MinWeight = -15.0
    params.MaxWeight = 15.0
    params.WeightMutationMaxPower = 1.5
    params.MinSpikingTimeConstant = 0.006
    params.MaxSpikingTimeConstant = 0.04
    params.MinSpikeThreshold = 0.4
    params.MaxSpikeThreshold = 1.5
    params.MinSynapticDelay = 0.0
    params.MaxSynapticDelay = 0.008
    return params


def seed_genome(
    params: neat.Parameters,
    mcculloch_pitts: bool = False,
) -> neat.Genome:
    init = neat.GenomeInitStruct()
    init.NumInputs = 8
    init.NumOutputs = 2
    init.NumHidden = 4
    init.NumLayers = 1
    init.SeedType = neat.LAYERED
    init.HiddenActType = (
        neat.MCCULLOCH_PITTS
        if mcculloch_pitts
        else neat.SPIKING_ADAPTIVE_LIF
    )
    init.OutputActType = (
        neat.MCCULLOCH_PITTS if mcculloch_pitts else neat.SPIKING_LIF
    )
    return neat.Genome(params, init)


def build_network(genome: neat.Genome, seed: int) -> neat.NeuralNetwork:
    network = neat.NeuralNetwork()
    genome.BuildPhenotype(network)
    network.SetSpikingInputMode(neat.POISSON_RATE_INPUT)
    network.SetSpikingOutputMode(neat.FILTERED_SPIKE_OUTPUT)
    network.SetSpikingTimeStep(SNN_DT)
    network.SeedSpiking(seed)
    network.EnableSpikeRecording(False)
    return network


def evaluate(
    genome: neat.Genome,
    max_steps: int,
    episode_seed: int,
) -> float:
    environment = CartPole()
    environment.reset(episode_seed)
    network = build_network(genome, episode_seed + 101)
    fitness = 0.0
    for _ in range(max_steps):
        rates = environment.rates()
        output = [0.0, 0.0]
        for _ in range(SNN_STEPS_PER_CONTROL):
            output = network.StepSpiking(rates)
        force = math.tanh((output[1] - output[0]) / 50.0)
        alive = environment.step(force)
        centered = 1.0 - min(1.0, abs(environment.x) / environment.track_limit)
        upright = 1.0 - min(
            1.0, abs(environment.angle) / environment.angle_limit
        )
        fitness += 0.25 + 0.25 * centered + 0.5 * upright
        if not alive:
            break
    energy = sum(neuron.m_spike_count for neuron in network.m_neurons)
    return max(1.0e-6, fitness - 0.0002 * energy)


def evolve(
    generations: int,
    population_size: int,
    max_steps: int,
    seed: int,
    mcculloch_pitts: bool = False,
) -> tuple[neat.Genome, list[float]]:
    params = parameters(population_size, mcculloch_pitts)
    population = neat.Population(
        seed_genome(params, mcculloch_pitts),
        params,
        True,
        6.0,
        seed,
    )
    best = None
    history = []
    for generation in range(generations):
        generation_best = None
        for species in population.m_Species:
            for genome in species.m_Individuals:
                fitness = evaluate(
                    genome,
                    max_steps,
                    seed + generation * 997,
                )
                genome.SetFitness(fitness)
                genome.SetEvaluated()
                if generation_best is None or fitness > generation_best.GetFitness():
                    generation_best = neat.Genome.Deserialize(genome.Serialize())
        assert generation_best is not None
        history.append(generation_best.GetFitness())
        if best is None or generation_best.GetFitness() > best.GetFitness():
            best = generation_best
        if generation + 1 < generations:
            population.Epoch()
    assert best is not None
    return best, history


def replay(
    genome: neat.Genome,
    max_steps: int,
    seed: int,
):
    from neattools import SpikingRecorder

    environment = CartPole()
    environment.reset(seed)
    network = build_network(genome, seed + 101)
    network.EnableSpikeRecording(True)
    recorder = SpikingRecorder(network)
    alive_steps = 0
    for _ in range(max_steps):
        rates = environment.rates()
        output = [0.0, 0.0]
        for _ in range(SNN_STEPS_PER_CONTROL):
            output = recorder.step(rates, SNN_DT)
        force = math.tanh((output[1] - output[0]) / 50.0)
        if not environment.step(force):
            break
        alive_steps += 1
    return environment, network, recorder, alive_steps


def draw_cartpole(axis, environment: CartPole, force: float = 0.0) -> None:
    from matplotlib.patches import Rectangle

    cart_width = 0.42
    cart_height = 0.20
    pole_length = environment.half_length * 2.0
    axis.axhline(0.0, color="#94a3b8", linewidth=2)
    axis.add_patch(
        Rectangle(
            (environment.x - cart_width / 2.0, 0.0),
            cart_width,
            cart_height,
            color="#38bdf8",
        )
    )
    pivot_x = environment.x
    pivot_y = cart_height
    tip_x = pivot_x + pole_length * math.sin(environment.angle)
    tip_y = pivot_y + pole_length * math.cos(environment.angle)
    axis.plot((pivot_x, tip_x), (pivot_y, tip_y), color="#fb7185", linewidth=5)
    axis.scatter((pivot_x,), (pivot_y,), color="#e2e8f0", s=45, zorder=3)
    axis.arrow(
        environment.x,
        -0.15,
        force * 0.35,
        0.0,
        width=0.015,
        color="#facc15",
        length_includes_head=True,
    )
    axis.set_xlim(-environment.track_limit - 0.3, environment.track_limit + 0.3)
    axis.set_ylim(-0.35, 1.35)
    axis.set_aspect("equal")
    axis.set_title(
        f"Cart-pole · x={environment.x:+.2f} · "
        f"angle={math.degrees(environment.angle):+.1f}°",
        color="#e2e8f0",
    )
    axis.set_facecolor("#0f172a")
    axis.axis("off")


def visualize(
    best: neat.Genome,
    max_steps: int,
    seed: int,
    output: Path | None,
    animate: bool,
) -> None:
    import matplotlib.pyplot as plt
    from neattools import (
        AnimateSpikingNetwork,
        DrawSpikingNetwork,
        PlotFiringRates,
        PlotSpikeRaster,
        SpikingRecorder,
    )

    if animate:
        environment = CartPole()
        environment.reset(seed)
        network = build_network(best, seed + 101)
        network.EnableSpikeRecording(True)
        recorder = SpikingRecorder(network)
        state = {"force": 0.0, "alive": True, "control": 0}

        def inputs(frame: int, _time: float) -> list[float]:
            if frame > 0 and frame % SNN_STEPS_PER_CONTROL == 0:
                output = network.OutputFilteredSpikes()
                state["force"] = math.tanh(
                    (output[1] - output[0]) / 50.0
                )
                state["alive"] = environment.step(state["force"])
                state["control"] += 1
                if not state["alive"]:
                    environment.reset(seed + state["control"])
                    network.Flush()
                    network.SeedSpiking(seed + 101 + state["control"])
                    state["alive"] = True
            return environment.rates()

        def draw_environment(axis, _frame, _network, _recorder) -> None:
            draw_cartpole(axis, environment, state["force"])

        animation = AnimateSpikingNetwork(
            network,
            inputs,
            recorder=recorder,
            frames=max_steps * SNN_STEPS_PER_CONTROL,
            interval=15,
            time_step=SNN_DT,
            window_seconds=0.5,
            trace_neurons=[
                network.NumInputs(),
                network.NumInputs() + 1,
            ],
            environment_draw=draw_environment,
            show=output is None,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(output, fps=60)
        return

    environment, network, recorder, _ = replay(best, max_steps, seed)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    DrawSpikingNetwork(network, ax=axes[0, 0], show=False)
    draw_cartpole(axes[0, 1], environment)
    PlotSpikeRaster(recorder, ax=axes[1, 0], show=False)
    PlotFiringRates(
        recorder,
        ax=axes[1, 1],
        neurons=[network.NumInputs(), network.NumInputs() + 1],
        show=False,
    )
    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evolve a spiking cart-pole controller."
    )
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=140)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="evolve the same controller with McCulloch-Pitts neurons",
    )
    args = parser.parse_args()
    if args.smoke:
        args.generations = 1
        args.population = 4
        args.max_steps = 20
        args.no_show = True
        args.animate = False
    best, history = evolve(
        args.generations,
        args.population,
        args.max_steps,
        args.seed,
        args.mcculloch_pitts,
    )
    _, _, recorder, survived = replay(
        best,
        args.max_steps,
        args.seed + 1234,
    )
    print(
        json.dumps(
            {
                "demo": "spiking_cartpole",
                "neuron_model": (
                    "mcculloch-pitts"
                    if args.mcculloch_pitts
                    else "lif"
                ),
                "generations": args.generations,
                "population": args.population,
                "max_steps": args.max_steps,
                "best_fitness": best.GetFitness(),
                "survived_steps": survived,
                "neurons": best.NumNeurons(),
                "links": best.NumLinks(),
                "recorded_spikes": len(recorder.events),
                "history": history,
            }
        )
    )
    if not args.no_show or args.output is not None:
        visualize(
            best,
            args.max_steps,
            args.seed + 1234,
            args.output,
            args.animate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
