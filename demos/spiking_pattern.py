#!/usr/bin/env python3
"""Evolve a spiking network that detects synchronous spike patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pymultineat as neat  # noqa: E402


DT = 0.001
PATTERN_STEPS = 80


def make_patterns() -> list[tuple[str, list[list[float]], float]]:
    patterns = []
    definitions = (
        ("synchronous", (10, 30, 50), (10, 30, 50), 1.0),
        ("near synchronous", (12, 32, 52), (14, 34, 54), 1.0),
        ("alternating", (10, 30, 50), (20, 40, 60), 0.0),
        ("reverse alternating", (20, 40, 60), (10, 30, 50), 0.0),
    )
    for name, first, second, target in definitions:
        sequence = [[0.0, 0.0] for _ in range(PATTERN_STEPS)]
        for step in first:
            sequence[step][0] = 1.0
        for step in second:
            sequence[step][1] = 1.0
        patterns.append((name, sequence, target))
    return patterns


PATTERNS = make_patterns()


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
    params.MaxSpecies = 8
    params.CompatTreshold = 2.5
    params.MaxCompatTreshold = 20.0
    params.OverallMutationRate = 0.9
    params.MutateAddNeuronProb = 0.08
    params.MutateAddLinkProb = 0.12
    params.MutateRemLinkProb = 0.01
    params.MutateWeightsProb = 0.55
    params.WeightMutationRate = 0.9
    params.WeightMutationMaxPower = 1.0
    params.MinWeight = -12.0
    params.MaxWeight = 12.0
    params.MinSynapticDelay = 0.0
    params.MaxSynapticDelay = 0.012
    params.MinSpikingTimeConstant = 0.005
    params.MaxSpikingTimeConstant = 0.03
    params.MinSpikeThreshold = 0.4
    params.MaxSpikeThreshold = 1.4
    return params


def seed_genome(
    params: neat.Parameters,
    mcculloch_pitts: bool = False,
) -> neat.Genome:
    init = neat.GenomeInitStruct()
    init.NumInputs = 2
    init.NumOutputs = 1
    init.NumHidden = 2
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


def build_network(genome: neat.Genome) -> neat.NeuralNetwork:
    network = neat.NeuralNetwork()
    genome.BuildPhenotype(network)
    network.SetSpikingInputMode(neat.BINARY_SPIKE_INPUT)
    network.SetSpikingOutputMode(neat.SPIKE_OUTPUT)
    network.SetSpikingTimeStep(DT)
    network.EnableSpikeRecording(False)
    return network


def evaluate(genome: neat.Genome) -> float:
    network = build_network(genome)
    score = 0.0
    total_output_spikes = 0
    for _, sequence, target in PATTERNS:
        network.Flush()
        before = network.m_neurons[network.NumInputs()].m_spike_count
        for sample in sequence:
            network.StepSpiking(sample)
        after = network.m_neurons[network.NumInputs()].m_spike_count
        output_spikes = int(after - before)
        total_output_spikes += output_spikes
        response = min(output_spikes / 3.0, 1.0)
        score += 1.0 - abs(target - response)
    # Reward correct temporal discrimination and mildly penalize excess firing.
    return max(1.0e-6, score * score - 0.002 * total_output_spikes)


def evolve(
    generations: int,
    population_size: int,
    seed: int,
    mcculloch_pitts: bool = False,
) -> tuple[neat.Genome, list[float]]:
    params = parameters(population_size, mcculloch_pitts)
    population = neat.Population(
        seed_genome(params, mcculloch_pitts),
        params,
        True,
        5.0,
        seed,
    )
    best = None
    history = []
    for generation in range(generations):
        generation_best = None
        for species in population.m_Species:
            for genome in species.m_Individuals:
                fitness = evaluate(genome)
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


def replay(best: neat.Genome):
    from neattools import SpikingRecorder

    network = build_network(best)
    network.EnableSpikeRecording(True)
    recorder = SpikingRecorder(network)
    for _, sequence, _ in PATTERNS:
        for sample in sequence:
            recorder.step(sample, DT)
        for _ in range(20):
            recorder.step([0.0, 0.0], DT)
    return network, recorder


def visualize(
    best: neat.Genome,
    output: Path | None,
    animate: bool,
) -> None:
    import matplotlib.pyplot as plt
    from neattools import (
        AnimateSpikingNetwork,
        DrawSpikingNetwork,
        PlotFiringRates,
        PlotSpikeRaster,
    )

    network, recorder = replay(best)
    if animate:
        name, sequence, target = PATTERNS[0]
        network.Flush()
        network.EnableSpikeRecording(True)
        cursor = {"step": 0}

        def inputs(frame: int, _time: float) -> list[float]:
            cursor["step"] = frame % len(sequence)
            return sequence[cursor["step"]]

        def environment(axis, frame, _network, _recorder) -> None:
            step = frame % len(sequence)
            axis.set_xlim(0, PATTERN_STEPS)
            axis.set_ylim(-0.6, 1.6)
            axis.scatter(
                [index for index, value in enumerate(sequence) if value[0]],
                [1.0] * sum(value[0] > 0 for value in sequence),
                marker="|",
                s=120,
                label="input A",
            )
            axis.scatter(
                [index for index, value in enumerate(sequence) if value[1]],
                [0.0] * sum(value[1] > 0 for value in sequence),
                marker="|",
                s=120,
                label="input B",
            )
            axis.axvline(step, color="white", alpha=0.8)
            axis.set_yticks((0, 1), ("B", "A"))
            axis.set_title(
                f"{name} · target={target:g}",
                color="#e2e8f0",
            )
            axis.set_xlabel("Pattern step", color="#e2e8f0")
            axis.tick_params(colors="#94a3b8")
            axis.set_facecolor("#0f172a")

        animation = AnimateSpikingNetwork(
            network,
            inputs,
            frames=PATTERN_STEPS * 4,
            interval=20,
            time_step=DT,
            window_seconds=0.08,
            environment_draw=environment,
            show=output is None,
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(output, fps=50)
        return

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    DrawSpikingNetwork(network, ax=axes[0], show=False)
    PlotSpikeRaster(recorder, ax=axes[1], show=False)
    PlotFiringRates(
        recorder,
        ax=axes[2],
        neurons=[network.NumInputs()],
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
        description="Evolve a temporal spike-pattern detector."
    )
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="evolve the same detector with McCulloch-Pitts neurons",
    )
    args = parser.parse_args()
    if args.smoke:
        args.generations = 1
        args.population = 6
        args.no_show = True
        args.animate = False
    random.seed(args.seed)
    best, history = evolve(
        args.generations,
        args.population,
        args.seed,
        args.mcculloch_pitts,
    )
    network, recorder = replay(best)
    payload = {
        "demo": "spiking_pattern",
        "neuron_model": (
            "mcculloch-pitts" if args.mcculloch_pitts else "lif"
        ),
        "generations": args.generations,
        "population": args.population,
        "best_fitness": best.GetFitness(),
        "neurons": best.NumNeurons(),
        "links": best.NumLinks(),
        "recorded_spikes": len(recorder.events),
        "history": history,
    }
    print(json.dumps(payload))
    if not args.no_show or args.output is not None:
        visualize(best, args.output, args.animate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
