#!/usr/bin/env python3
"""Evolve a CPPN that builds a physical three-dimensional neural substrate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pymultineat as neat  # noqa: E402


def make_substrate(neuron_model: str, es_hyperneat: bool) -> neat.Substrate:
    inputs = [[-1.0, -0.5, -1.0], [0.0, 0.0, 0.0]]
    hidden = (
        []
        if es_hyperneat
        else [[-0.25, 0.5, 0.5], [0.25, -0.5, -0.5]]
    )
    outputs = [[1.0, 0.5, 1.0]]
    substrate = neat.Substrate(inputs, hidden, outputs)
    substrate.m_query_weights_only = True
    substrate.m_with_distance = False
    substrate.m_allow_input_hidden_links = True
    substrate.m_allow_input_output_links = True
    substrate.m_allow_hidden_hidden_links = True
    substrate.m_allow_hidden_output_links = True
    substrate.m_max_weight_and_bias = 1.0
    substrate.m_max_connection_length = 4.0
    substrate.m_use_spatial_distance_for_delays = True
    substrate.m_conduction_velocity = 4.0
    activation = {
        "rate": neat.TANH,
        "lif": neat.SPIKING_LIF,
        "mcculloch-pitts": neat.MCCULLOCH_PITTS,
    }[neuron_model]
    substrate.m_hidden_nodes_activation = activation
    substrate.m_output_nodes_activation = activation
    return substrate


def make_cppn(
    substrate: neat.Substrate,
    population_size: int,
    es_hyperneat: bool,
) -> tuple[neat.Parameters, neat.Genome]:
    parameters = neat.Parameters()
    parameters.PopulationSize = population_size
    parameters.RequireEvaluatedGenomes = True
    parameters.RejectNonFiniteFitness = True
    parameters.InitialDepth = 1
    parameters.MaxDepth = 1
    parameters.IterationLevel = 0
    parameters.DivisionThreshold = 0.0
    parameters.VarianceThreshold = 10.0
    parameters.BandThreshold = 0.01
    parameters.Width = 2.0
    parameters.Height = 2.0
    parameters.Depth = 2.0
    parameters.Qtree_X = 0.0
    parameters.Qtree_Y = 0.0
    parameters.Qtree_Z = 0.0

    initial = neat.GenomeInitStruct()
    initial.NumInputs = substrate.GetMinCPPNInputs()
    initial.NumOutputs = substrate.GetMinCPPNOutputs()
    initial.OutputActType = neat.LINEAR if es_hyperneat else neat.TANH
    cppn = neat.Genome(parameters, initial)
    if es_hyperneat:
        links = cppn.m_LinkGenes
        for link in links:
            link.SetWeight(1.0)
        cppn.m_LinkGenes = links
    return parameters, cppn


def build_phenotype(
    cppn: neat.Genome,
    substrate: neat.Substrate,
    parameters: neat.Parameters,
    es_hyperneat: bool,
) -> neat.NeuralNetwork:
    network = neat.NeuralNetwork()
    if es_hyperneat:
        cppn.BuildESHyperNEATPhenotype(network, substrate, parameters)
    else:
        cppn.BuildHyperNEATPhenotype(network, substrate)
    return network


def phenotype_fitness(network: neat.NeuralNetwork) -> float:
    if not network.m_connections:
        return -1_000_000.0
    total_length = sum(link.m_length for link in network.m_connections)
    hidden = sum(neuron.m_type == neat.HIDDEN for neuron in network.m_neurons)
    return float(len(network.m_connections) + hidden - 0.02 * total_length)


def evolve(
    substrate: neat.Substrate,
    parameters: neat.Parameters,
    seed_cppn: neat.Genome,
    generations: int,
    seed: int,
    es_hyperneat: bool,
) -> tuple[neat.Genome, neat.NeuralNetwork, list[float]]:
    population = neat.Population(
        seed_cppn, parameters, False, 1.0, seed
    )
    champion = seed_cppn
    champion_network = build_phenotype(
        champion, substrate, parameters, es_hyperneat
    )
    champion_fitness = -math.inf
    history = []
    for generation in range(generations):
        for species in population.m_Species:
            for genome in species.m_Individuals:
                network = None
                try:
                    network = build_phenotype(
                        genome, substrate, parameters, es_hyperneat
                    )
                    fitness = phenotype_fitness(network)
                except (RuntimeError, ValueError):
                    fitness = -1_000_000.0
                genome.SetFitness(fitness)
                genome.SetEvaluated()
                if network is not None and fitness > champion_fitness:
                    champion = genome
                    champion_network = network
                    champion_fitness = fitness
        history.append(champion_fitness)
        if generation + 1 < generations:
            population.Epoch()
    return champion, champion_network, history


# Colors for the 3D plot (dark theme).
BACKGROUND = "#0f172a"
FOREGROUND = "#e2e8f0"
MUTED = "#94a3b8"
NEURON_STYLE = {  # neuron type -> (color, marker)
    neat.INPUT: ("#22c55e", "s"),
    neat.BIAS: ("#facc15", "h"),
    neat.HIDDEN: ("#38bdf8", "o"),
    neat.OUTPUT: ("#fb7185", "D"),
}


def draw_spatial_network_3d(network: neat.NeuralNetwork, title: str):
    """Draw neurons and directed axons at their real substrate positions.

    Blue axons are excitatory, red inhibitory; thickness and opacity grow
    with the absolute weight. Returns the 3D axis so callers can save it.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    figure = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = figure.add_subplot(111, projection="3d")
    neurons = network.m_neurons
    positions = [
        np.asarray((neuron.m_x, neuron.m_y, neuron.m_z), dtype=float)
        for neuron in neurons
    ]
    connections = network.m_connections
    max_abs_weight = max((abs(c.m_weight) for c in connections), default=1.0)
    for connection in connections:
        source = positions[connection.m_source_neuron_idx]
        direction = positions[connection.m_target_neuron_idx] - source
        if np.linalg.norm(direction) == 0.0:
            continue  # self-loops have no visible extent in 3D
        weight = connection.m_weight
        magnitude = (
            math.sqrt(abs(weight) / max_abs_weight) if max_abs_weight else 0.0
        )
        ax.quiver(
            *source,
            *direction,
            color=(
                "#2563eb" if weight > 0.0
                else "#ef4444" if weight < 0.0
                else "#64748b"
            ),
            linewidth=0.6 + 4.4 * magnitude,
            alpha=0.25 + 0.7 * magnitude,
            arrow_length_ratio=0.08,
        )

    for neuron_type, (color, marker) in NEURON_STYLE.items():
        indices = [i for i, neuron in enumerate(neurons) if neuron.m_type == neuron_type]
        if not indices:
            continue
        coordinates = np.asarray([positions[i] for i in indices])
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            marker=marker,
            s=[105 if neurons[i].m_spike else 70 for i in indices],
            c=color,
            edgecolors=FOREGROUND,
            linewidths=0.8,
            depthshade=False,
        )
    for index, position in enumerate(positions):
        ax.text(*position, f" {index}", color=FOREGROUND, fontsize=8)

    ax.set_title(title, color=FOREGROUND)
    for set_label, name in (
        (ax.set_xlabel, "x"),
        (ax.set_ylabel, "y"),
        (ax.set_zlabel, "z"),
    ):
        set_label(name, color=FOREGROUND)
    ax.set_facecolor(BACKGROUND)
    figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=MUTED)
    return ax


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="use a fixed 3D HyperNEAT substrate instead of ES-HyperNEAT",
    )
    parser.add_argument(
        "--spiking",
        action="store_true",
        help="place LIF neurons in the generated 3D phenotype",
    )
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="place McCulloch-Pitts neurons in the generated 3D phenotype",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.generations < 1 or args.population < 2:
        parser.error("generations must be >= 1 and population must be >= 2")
    if args.smoke:
        args.generations = 1
        args.population = 4
        args.no_show = True

    neuron_model = (
        "mcculloch-pitts"
        if args.mcculloch_pitts
        else ("lif" if args.spiking else "rate")
    )
    es_hyperneat = not args.fixed
    substrate = make_substrate(neuron_model, es_hyperneat)
    parameters, seed_cppn = make_cppn(
        substrate, args.population, es_hyperneat
    )
    _, network, history = evolve(
        substrate,
        parameters,
        seed_cppn,
        args.generations,
        args.seed,
        es_hyperneat,
    )
    if neuron_model != "rate":
        network.SetSpikingInputMode(neat.BINARY_SPIKE_INPUT)
        network.SetSpikingOutputMode(neat.SPIKE_OUTPUT)
        network.StepSpiking([1.0] * network.NumInputs(), 0.001)

    hidden_z = [
        neuron.m_z
        for neuron in network.m_neurons
        if neuron.m_type == neat.HIDDEN
    ]
    payload = {
        "demo": "hyperneat_3d",
        "algorithm": "es-hyperneat" if es_hyperneat else "hyperneat",
        "neuron_model": neuron_model,
        "generations": args.generations,
        "population": args.population,
        "neurons": len(network.m_neurons),
        "hidden_neurons": len(hidden_z),
        "links": len(network.m_connections),
        "nonzero_hidden_z": any(abs(value) > 0.0 for value in hidden_z),
        "total_axon_length": sum(
            link.m_length for link in network.m_connections
        ),
        "maximum_delay": max(
            (link.m_synaptic_delay for link in network.m_connections),
            default=0.0,
        ),
        "best_fitness": history[-1],
    }
    print(json.dumps(payload, sort_keys=True))

    if args.output is not None or not args.no_show:
        axis = draw_spatial_network_3d(
            network,
            title=(
                f"3D {'ES-' if es_hyperneat else ''}HyperNEAT · "
                f"{neuron_model}"
            ),
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            axis.figure.savefig(args.output, dpi=160, bbox_inches="tight")
        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
