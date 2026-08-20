#!/usr/bin/env python
# python_xor_demo.py

import argparse
import json
from pathlib import Path
import sys
import time

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import pymultineat as pnt  # noqa: E402
from neattools import DrawGenome  # noqa: E402
from spiking_neat import (  # noqa: E402
    SpikingPolicy,
    SpikingPolicySettings,
    configure_spiking_genome,
    configure_spiking_parameters,
)

# Define the XOR training data.
# Note that the genome is initialized with three inputs: two (the XOR inputs)
# plus an extra bias input (set to 1.0). Hence each input vector has three values.
xor_data = [
    ([0.0, 0.0, 1.0], 0.0),
    ([0.0, 1.0, 1.0], 1.0),
    ([1.0, 0.0, 1.0], 1.0),
    ([1.0, 1.0, 1.0], 0.0)
]

# Define a function to evaluate the fitness of a genome on XOR.
def xor_test(
    genome,
    *,
    spiking: bool = False,
    settings: SpikingPolicySettings | None = None,
    seed: int = 1,
):
    # Create a neural network object.
    nn = pnt.NeuralNetwork()
    # ‘BuildPhenotype’ constructs the network from the genome.
    genome.BuildPhenotype(nn)
    total_error = 0.0
    policy = SpikingPolicy(nn, settings) if spiking else None

    # For each XOR sample, clear the network, provide the inputs and propagate.
    for sample, (inputs, expected) in enumerate(xor_data):
        if policy is None:
            nn.Flush()              # reset activations
            nn.Input(inputs)        # the genome expects 3 inputs (2 data + bias)
            # Two steps allow signals to propagate through hidden topology.
            nn.Activate()
            nn.Activate()
            output = nn.Output()[0]
        else:
            policy.reset(seed + sample)
            signed_inputs = [2.0 * value - 1.0 for value in inputs[:2]]
            signed_inputs.append(1.0)
            output = policy.step_unsigned(signed_inputs)[0]
        total_error += abs(expected - output)

    # In this example the fitness is defined as the square of (4 – total error)
    fitness = (4.0 - total_error) ** 2
    return fitness


def visualize_spiking_xor(
    genome,
    settings: SpikingPolicySettings,
    seed: int,
    output_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt
    from neattools import (
        DEFAULT_THEME,
        DrawSpikingNetwork,
        PlotMembraneTraces,
        PlotSpikeRaster,
        SpikingRecorder,
    )

    network = pnt.NeuralNetwork()
    genome.BuildPhenotype(network)
    recorder = SpikingRecorder(network)
    policy = SpikingPolicy(network, settings, recorder=recorder)
    policy.reset(seed)
    predictions = []
    for inputs, _ in xor_data:
        signed_inputs = [2.0 * value - 1.0 for value in inputs[:2]]
        signed_inputs.append(1.0)
        predictions.append(policy.step_unsigned(signed_inputs)[0])
        for _ in range(4):
            policy.step_unsigned([-1.0, -1.0, 1.0])

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        constrained_layout=True,
    )
    DrawSpikingNetwork(
        network,
        ax=axes[0, 0],
        title="Evolved spiking XOR phenotype",
        show=False,
    )
    PlotSpikeRaster(
        recorder,
        ax=axes[0, 1],
        title="XOR spike trains",
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
        ax=axes[1, 0],
        neurons=output_neurons,
        title="Output membrane potential",
        show=False,
    )
    labels = ("00", "01", "10", "11")
    expected = [value for _, value in xor_data]
    positions = list(range(len(labels)))
    response_axis = axes[1, 1]
    response_axis.set_facecolor(DEFAULT_THEME.background)
    response_axis.tick_params(colors=DEFAULT_THEME.muted)
    response_axis.grid(
        axis="y",
        color=DEFAULT_THEME.grid,
        alpha=0.3,
        zorder=0,
    )
    for spine in response_axis.spines.values():
        spine.set_color(DEFAULT_THEME.grid)
    axes[1, 1].bar(
        [position - 0.18 for position in positions],
        expected,
        width=0.36,
        label="target",
        color=DEFAULT_THEME.hidden_color,
        zorder=2,
    )
    axes[1, 1].bar(
        [position + 0.18 for position in positions],
        predictions,
        width=0.36,
        label="filtered spike response",
        color=DEFAULT_THEME.output_color,
        zorder=2,
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_xlabel("XOR input")
    axes[1, 1].set_ylabel("Normalized response")
    axes[1, 1].set_title("Temporal XOR responses")
    axes[1, 1].title.set_color(DEFAULT_THEME.foreground)
    axes[1, 1].xaxis.label.set_color(DEFAULT_THEME.foreground)
    axes[1, 1].yaxis.label.set_color(DEFAULT_THEME.foreground)
    legend = axes[1, 1].legend(
        facecolor=DEFAULT_THEME.background,
        edgecolor=DEFAULT_THEME.grid,
        framealpha=0.88,
    )
    for text in legend.get_texts():
        text.set_color(DEFAULT_THEME.foreground)
    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="MultiNEAT XOR example")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--spiking",
        action="store_true",
        help="evolve a Poisson-encoded spiking XOR variant",
    )
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="use evolvable McCulloch-Pitts neurons in the spiking variant",
    )
    parser.add_argument(
        "--spiking-steps",
        type=int,
        default=32,
        help="SNN solver steps per XOR sample",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the final genome visualization",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one small headless generation",
    )
    args = parser.parse_args()
    if args.mcculloch_pitts:
        args.spiking = True
    neuron_model = (
        "mcculloch-pitts" if args.mcculloch_pitts else "lif"
    )
    if args.smoke:
        args.generations = 1
        args.population = 10
        args.no_show = True
        if args.spiking:
            args.spiking_steps = 4

    # Create and customize MultiNEAT parameters.
    params = pnt.Parameters()
    params.PopulationSize = args.population
    params.DynamicCompatibility = True
    params.NormalizeGenomeSize = False
    params.WeightDiffCoeff = 0.1
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 15
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.3
    params.ArchiveEnforcement = False
    params.MutateWeightsProb = 0.25
    params.WeightMutationMaxPower = 0.5
    params.WeightReplacementMaxPower = 8.0
    params.MutateWeightsSevereProb = 0.0
    params.WeightMutationRate = 0.85
    params.WeightReplacementRate = 0.2
    params.MaxWeight = 8
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.0
    params.MinActivationA = 4.9
    params.MaxActivationA = 4.9
    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 0.0
    params.CrossoverRate = 0.0
    params.MultipointCrossoverRate = 0.0
    params.SurvivalRate = 0.2
    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0
    params.AllowLoops = False
    params.AllowClones = False
    if args.spiking:
        configure_spiking_parameters(
            pnt,
            params,
            recurrent=False,
            enable_stdp=False,
            neuron_model=neuron_model,
        )

    # Create a GenomeInitStruct.
    # Here we specify 3 inputs (2 XOR plus bias), 1 output, no hidden nodes,
    # and use the PERCEPTRON seed type with UNSIGNED_SIGMOID activation functions.
    init_struct = pnt.GenomeInitStruct()
    init_struct.NumInputs = 3
    init_struct.NumOutputs = 1
    init_struct.NumHidden = 0
    init_struct.SeedType = pnt.GenomeSeedType.PERCEPTRON
    init_struct.HiddenActType = pnt.UNSIGNED_SIGMOID
    init_struct.OutputActType = pnt.UNSIGNED_SIGMOID
    if args.spiking:
        configure_spiking_genome(
            pnt, init_struct, neuron_model=neuron_model
        )

    # Create a prototype genome using the parameters and initialization struct.
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population.
    seed = args.seed if args.seed is not None else int(time.time())
    pop = pnt.Population(genome_prototype, params, True, 1.0, seed)
    spiking_settings = SpikingPolicySettings(
        simulation_steps=args.spiking_steps,
    )

    for gen in range(args.generations):
        # Evaluate all genomes in every species on the XOR task.
        for species in pop.m_Species:
            for i in range(len(species.m_Individuals)):
                genome = species.m_Individuals[i]
                fitness = xor_test(
                    genome,
                    spiking=args.spiking,
                    settings=spiking_settings,
                    seed=seed,
                )
                genome.SetFitness(fitness)
                genome.SetEvaluated()

        # Get the best genome in the population.
        bestGenome = pop.GetBestGenome()
        bestFitness = bestGenome.GetFitness()
        print("Generation: {}, Best Fitness: {}".format(gen, bestFitness), flush=True)

        # Advance one generation.
        pop.Epoch()

    print(
        json.dumps(
            {
                "demo": "xor",
                "policy": (
                    "mcculloch-pitts"
                    if args.mcculloch_pitts
                    else ("spiking" if args.spiking else "rate")
                ),
                "generations": args.generations,
                "population": args.population,
                "best_fitness": bestFitness,
                "neurons": bestGenome.NumNeurons(),
                "links": bestGenome.NumLinks(),
            }
        ),
        flush=True,
    )

    if not args.no_show or args.output is not None:
        if args.spiking:
            visualize_spiking_xor(
                bestGenome,
                spiking_settings,
                seed,
                args.output,
            )
        else:
            DrawGenome(bestGenome)


if __name__ == "__main__":
    main()
