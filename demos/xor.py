#!/usr/bin/env python
"""Evolve a network that solves XOR, with optional spiking variants.

This file is self-contained: besides the standard library it only needs
``pymultineat`` (and matplotlib/networkx for the final picture).

    python demos/xor.py                     # classic rate-coded NEAT
    python demos/xor.py --spiking           # Poisson-encoded LIF neurons
    python demos/xor.py --mcculloch-pitts   # evolvable McCulloch-Pitts neurons
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

# Allow running from a source checkout where pymultineat is built in-tree.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import pymultineat as pnt  # noqa: E402

# Define the XOR training data.
# Note that the genome is initialized with three inputs: two (the XOR inputs)
# plus an extra bias input (set to 1.0). Hence each input vector has three values.
xor_data = [
    ([0.0, 0.0, 1.0], 0.0),
    ([0.0, 1.0, 1.0], 1.0),
    ([1.0, 0.0, 1.0], 1.0),
    ([1.0, 1.0, 1.0], 0.0)
]


# ---------------------------------------------------------------------------
# Spiking policy: Poisson-rate input encoding, filtered-spike output decoding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpikingPolicySettings:
    """Temporal encoding and decoding settings for an evolved SNN."""

    simulation_steps: int = 8
    time_step: float = 0.001
    input_rate_hz: float = 200.0
    output_rate_hz: float = 200.0

    def validate(self) -> None:
        if self.simulation_steps < 1:
            raise ValueError("spiking simulation_steps must be positive")
        values = (self.time_step, self.input_rate_hz, self.output_rate_hz)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("spiking rates and time step must be finite and positive")


def configure_spiking_parameters(params: pnt.Parameters, neuron_model: str) -> None:
    """Apply a feed-forward evolvable SNN preset (no STDP) to ``params``."""

    if neuron_model == "mcculloch-pitts":
        params.ConfigureMcCullochPitts(True, False)
    else:
        params.ConfigureSpiking(False)
    params.DontUseBiasNeuron = True
    params.AllowLoops = False
    params.RecurrentProb = 0.0
    params.SplitRecurrent = False
    params.SplitLoopedRecurrent = False
    params.MinWeight = -12.0
    params.MaxWeight = 12.0
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 6.0
    params.MinSpikingTimeConstant = 0.005
    params.MaxSpikingTimeConstant = 0.04
    params.MinSpikeRateTimeConstant = 0.02
    params.MaxSpikeRateTimeConstant = 0.08
    params.MinSpikeThreshold = 0.4
    params.MaxSpikeThreshold = 1.4
    params.MinSynapticDelay = 0.0
    params.MaxSynapticDelay = 0.012
    params.InitialSTDPEnabledProb = 0.0


def configure_spiking_genome(init: pnt.GenomeInitStruct, neuron_model: str) -> None:
    """Select the stateful neuron model used by the seed genome."""

    if neuron_model == "mcculloch-pitts":
        init.HiddenActType = pnt.MCCULLOCH_PITTS
        init.OutputActType = pnt.MCCULLOCH_PITTS
    else:
        init.HiddenActType = pnt.SPIKING_ADAPTIVE_LIF
        init.OutputActType = pnt.SPIKING_LIF


class SpikingPolicy:
    """Drive a spiking phenotype with rate-coded inputs, read rates back."""

    def __init__(self, network, settings: SpikingPolicySettings, *, recorder=None):
        settings.validate()
        if not network.IsSpiking():
            raise ValueError("SpikingPolicy requires a spiking phenotype")
        self.network = network
        self.settings = settings
        self.recorder = recorder
        network.SetSpikingInputMode(pnt.POISSON_RATE_INPUT)
        network.SetSpikingOutputMode(pnt.FILTERED_SPIKE_OUTPUT)
        network.SetSpikingTimeStep(settings.time_step)
        network.EnableSTDP(False)

    def reset(self, seed: int) -> None:
        self.network.Flush()
        self.network.SeedSpiking(int(seed))
        if self.recorder is not None:
            self.recorder.clear()

    def step_unsigned(self, values: Sequence[float]) -> list[float]:
        """Map inputs in [-1, 1] to Poisson rates; return outputs in [0, 1]."""

        if len(values) != self.network.NumInputs():
            raise ValueError(
                f"Spiking policy expected {self.network.NumInputs()} inputs, "
                f"received {len(values)}"
            )
        rates = []
        for value in values:
            numeric = float(value)
            if not math.isfinite(numeric):
                numeric = 0.0
            normalized = min(1.0, max(-1.0, numeric))
            rates.append(0.5 * (normalized + 1.0) * self.settings.input_rate_hz)
        outputs: list[float] = []
        for _ in range(self.settings.simulation_steps):
            if self.recorder is None:
                outputs = list(self.network.StepSpiking(rates, self.settings.time_step))
            else:
                outputs = self.recorder.step(rates, self.settings.time_step)
        return [
            min(1.0, max(0.0, output / self.settings.output_rate_hz))
            for output in outputs
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
    policy = SpikingPolicy(nn, settings or SpikingPolicySettings()) if spiking else None

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


# ---------------------------------------------------------------------------
# Visualization (matplotlib and networkx are imported only when drawing)
# ---------------------------------------------------------------------------
BACKGROUND = "#0f172a"
FOREGROUND = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#334155"
NEURON_STYLE = {  # neuron type -> (color, marker, legend label)
    pnt.INPUT: ("#22c55e", "s", "Input"),
    pnt.BIAS: ("#facc15", "h", "Bias"),
    pnt.HIDDEN: ("#38bdf8", "o", "Hidden"),
    pnt.OUTPUT: ("#fb7185", "D", "Output"),
}


def edge_style(weight: float, max_abs_weight: float) -> tuple[str, float, float]:
    """Blue for excitatory, red for inhibitory; width/alpha grow with |w|."""

    magnitude = math.sqrt(abs(weight) / max_abs_weight) if max_abs_weight else 0.0
    color = "#2563eb" if weight > 0.0 else "#ef4444" if weight < 0.0 else "#64748b"
    return color, 0.6 + 4.4 * magnitude, 0.25 + 0.7 * magnitude


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=FOREGROUND)
    ax.set_xlabel(xlabel, color=FOREGROUND)
    ax.set_ylabel(ylabel, color=FOREGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def dark_legend(ax, **kwargs) -> None:
    legend = ax.legend(facecolor=BACKGROUND, edgecolor=GRID, framealpha=0.88, **kwargs)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)


def layered_positions(graph) -> dict[int, tuple[float, float]]:
    """Inputs on top, outputs at the bottom, hidden neurons by depth."""

    import networkx as nx

    feed_forward = nx.DiGraph()
    feed_forward.add_nodes_from(graph)
    feed_forward.add_edges_from(
        (source, target)
        for source, target, data in graph.edges(data=True)
        if not data["recurrent"] and source != target
    )
    # Condensing strongly connected components keeps unmarked cycles drawable.
    condensed = nx.condensation(feed_forward)
    depth: dict[int, int] = {}
    for component in nx.topological_sort(condensed):
        depth[component] = max(
            (depth[parent] + 1 for parent in condensed.predecessors(component)),
            default=0,
        )
    mapping = condensed.graph["mapping"]
    layer = {node: depth[mapping[node]] for node in graph}
    kinds = nx.get_node_attributes(graph, "type")
    for node in graph:
        if kinds[node] in (pnt.INPUT, pnt.BIAS):
            layer[node] = 0
    deepest = max((layer[n] for n in graph if kinds[n] != pnt.OUTPUT), default=0)
    for node in graph:
        if kinds[node] == pnt.OUTPUT:
            layer[node] = max(layer[node], deepest + 1)

    rows: dict[int, list[int]] = {}
    for node in sorted(graph):
        rows.setdefault(layer[node], []).append(node)
    bottom = max(1, max(rows))
    return {
        node: ((order + 1) / (len(nodes) + 1), 1.0 - level / bottom)
        for level, nodes in rows.items()
        for order, node in enumerate(nodes)
    }


def draw_genome(genome) -> object:
    """Draw a NEAT genome: node shape/color by type, edges by weight sign."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    graph = nx.DiGraph()
    for neuron in genome.m_NeuronGenes:
        graph.add_node(neuron.m_ID, type=neuron.m_Type)
    for link in genome.m_LinkGenes:
        graph.add_edge(
            link.m_FromNeuronID,
            link.m_ToNeuronID,
            weight=link.m_Weight,
            recurrent=bool(link.m_IsRecurrent),
        )
    positions = layered_positions(graph)
    figure, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    style_axis(ax, "")

    max_abs_weight = max((abs(d["weight"]) for *_, d in graph.edges(data=True)), default=1.0)
    for recurrent in (False, True):
        edges = [
            (s, t, d) for s, t, d in graph.edges(data=True)
            if (d["recurrent"] or s == t) == recurrent
        ]
        if not edges:
            continue
        styles = [edge_style(d["weight"], max_abs_weight) for *_, d in edges]
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(s, t) for s, t, _ in edges],
            edge_color=[style[0] for style in styles],
            width=[style[1] for style in styles],
            alpha=[style[2] for style in styles],
            style="dashed" if recurrent else "solid",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.22" if recurrent else "arc3,rad=0.02",
            min_source_margin=8,
            min_target_margin=10,
            ax=ax,
        )

    handles = []
    for neuron_type, (color, marker, label) in NEURON_STYLE.items():
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == neuron_type]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_color=color, node_shape=marker,
            node_size=650, linewidths=1.2, edgecolors=FOREGROUND, ax=ax,
        )
        handles.append(Line2D([], [], marker=marker, linestyle="", markerfacecolor=color,
                              markeredgecolor=FOREGROUND, markersize=9, label=label))
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color=FOREGROUND, ax=ax)

    recurrent_links = sum(d["recurrent"] for *_, d in graph.edges(data=True))
    ax.set_title(
        f"Genome {genome.GetID()}  |  fitness {genome.GetFitness():.6g}  |  "
        f"{graph.number_of_nodes()} neurons · {graph.number_of_edges()} links"
        f" · {recurrent_links} recurrent",
        color=FOREGROUND, fontsize=12, pad=14,
    )
    handles += [
        Line2D([], [], color="#2563eb", label="Positive"),
        Line2D([], [], color="#ef4444", label="Negative"),
        Line2D([], [], color=MUTED, linestyle="dashed", label="Recurrent"),
    ]
    legend = ax.legend(handles=handles, loc="best", frameon=False, fontsize=8, ncol=4)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)
    ax.margins(0.16)
    ax.axis("off")
    return figure


class SpikingRecorder:
    """Step a spiking network and keep the traces needed for plotting."""

    def __init__(self, network) -> None:
        self.network = network
        self.times: list[float] = []
        self.membrane: list[list[float]] = []

    def clear(self) -> None:
        self.times.clear()
        self.membrane.clear()

    def step(self, inputs: Sequence[float], time_step: float) -> list[float]:
        output = self.network.StepSpiking(list(inputs), time_step)
        self.times.append(float(self.network.SpikingTime()))
        self.membrane.append(
            [float(neuron.m_membrane_potential) for neuron in self.network.m_neurons]
        )
        return list(output)


def draw_spiking_network(network, ax, title: str) -> None:
    """Draw a phenotype: spiking neurons are enlarged, others show potential."""

    import networkx as nx

    neurons = network.m_neurons
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(neurons)))
    levels: dict[float, list[int]] = {}
    for index, neuron in enumerate(neurons):
        level = float(neuron.m_split_y)
        if neuron.m_type in (pnt.INPUT, pnt.BIAS):
            level = 0.0
        elif neuron.m_type == pnt.OUTPUT:
            level = 1.0
        levels.setdefault(level, []).append(index)
    positions = {
        index: ((order + 1.0) / (len(indices) + 1.0), level)
        for level, indices in levels.items()
        for order, index in enumerate(indices)
    }

    connections = network.m_connections
    max_abs_weight = max((abs(c.m_weight) for c in connections), default=1.0)
    for connection in connections:
        source = connection.m_source_neuron_idx
        target = connection.m_target_neuron_idx
        color, width, alpha = edge_style(connection.m_weight, max_abs_weight)
        curved = connection.m_recur_flag or source == target
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(source, target)], width=width,
            edge_color=color, alpha=alpha,
            style="dotted" if connection.m_stdp_enabled else "solid",
            connectionstyle="arc3,rad=0.22" if curved else "arc3,rad=0.02",
            arrows=True, arrowsize=12, ax=ax,
        )
    for neuron_type, (color, marker, _) in NEURON_STYLE.items():
        nodes = [i for i, neuron in enumerate(neurons) if neuron.m_type == neuron_type]
        if not nodes:
            continue
        spiking = [bool(neurons[i].m_spike) for i in nodes]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes, node_shape=marker, node_color=color,
            node_size=[950 if s else 650 for s in spiking],
            edgecolors=[FOREGROUND if s else GRID for s in spiking],
            linewidths=[3.0 if s else 1.0 for s in spiking],
            ax=ax,
        )
    labels = {
        index: f"{index}\n★" if neuron.m_spike else f"{index}\n{neuron.m_membrane_potential:.2f}"
        for index, neuron in enumerate(neurons)
    }
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=8,
                            font_color=FOREGROUND, ax=ax)
    style_axis(ax, title)
    ax.axis("off")


def plot_spike_raster(network, ax, title: str) -> None:
    """Scatter every recorded input and neuron spike on a shared time axis."""

    events = network.GetSpikeHistory()
    for is_input, label, color in (
        (True, "Input spike", NEURON_STYLE[pnt.INPUT][0]),
        (False, "Neuron spike", NEURON_STYLE[pnt.OUTPUT][0]),
    ):
        selected = [event for event in events if bool(event.input) == is_input]
        if selected:
            ax.scatter(
                [event.time for event in selected],
                [event.neuron_index for event in selected],
                marker="|", s=80, linewidths=1.5, color=color, label=label,
            )
    style_axis(ax, title, "Time (s)", "Neuron")
    ax.grid(axis="x", color=GRID, alpha=0.35)
    if len(events):
        dark_legend(ax, loc="upper right")


def plot_membrane_traces(recorder: SpikingRecorder, ax, neurons: Sequence[int], title: str) -> None:
    colors = ("#2563eb", NEURON_STYLE[pnt.HIDDEN][0], NEURON_STYLE[pnt.OUTPUT][0])
    for order, neuron in enumerate(neurons):
        ax.plot(
            recorder.times,
            [sample[neuron] for sample in recorder.membrane],
            label=f"neuron {neuron}",
            color=colors[order % len(colors)],
            alpha=0.9,
        )
    style_axis(ax, title, "Time (s)", "Potential")
    ax.grid(color=GRID, alpha=0.3)
    if neurons and recorder.times:
        dark_legend(ax, loc="upper right", ncol=2)


def show_or_save(figure, output_path: Path | None) -> None:
    import matplotlib.pyplot as plt

    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        plt.close(figure)


def visualize_spiking_xor(
    genome,
    settings: SpikingPolicySettings,
    seed: int,
    output_path: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    network = pnt.NeuralNetwork()
    genome.BuildPhenotype(network)
    network.EnableSpikeRecording(True)
    recorder = SpikingRecorder(network)
    policy = SpikingPolicy(network, settings, recorder=recorder)
    policy.reset(seed)
    predictions = []
    for inputs, _ in xor_data:
        signed_inputs = [2.0 * value - 1.0 for value in inputs[:2]]
        signed_inputs.append(1.0)
        predictions.append(policy.step_unsigned(signed_inputs)[0])
        # A short quiet gap separates the four input patterns in the plots.
        for _ in range(4):
            policy.step_unsigned([-1.0, -1.0, 1.0])

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    draw_spiking_network(network, axes[0, 0], "Evolved spiking XOR phenotype")
    plot_spike_raster(network, axes[0, 1], "XOR spike trains")
    output_neurons = list(
        range(network.NumInputs(), network.NumInputs() + network.NumOutputs())
    )
    plot_membrane_traces(recorder, axes[1, 0], output_neurons, "Output membrane potential")

    labels = ("00", "01", "10", "11")
    expected = [value for _, value in xor_data]
    positions = list(range(len(labels)))
    response_axis = axes[1, 1]
    response_axis.grid(axis="y", color=GRID, alpha=0.3, zorder=0)
    response_axis.bar(
        [position - 0.18 for position in positions], expected, width=0.36,
        label="target", color=NEURON_STYLE[pnt.HIDDEN][0], zorder=2,
    )
    response_axis.bar(
        [position + 0.18 for position in positions], predictions, width=0.36,
        label="filtered spike response", color=NEURON_STYLE[pnt.OUTPUT][0], zorder=2,
    )
    response_axis.set_xticks(positions, labels)
    response_axis.set_ylim(0.0, 1.05)
    style_axis(response_axis, "Temporal XOR responses", "XOR input", "Normalized response")
    dark_legend(response_axis)
    show_or_save(figure, output_path)


def main():
    parser = argparse.ArgumentParser(description="MultiNEAT XOR example")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--legacy-evolution",
        action="store_true",
        help="use exact MultiNEAT 2 scalar-fitness evolution",
    )
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
    parser.add_argument("--output", type=Path, help="save the final figure here")
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
    if args.legacy_evolution:
        params.ConfigureLegacyV2()
    else:
        params.DeterministicEvaluation = not args.spiking
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
        configure_spiking_parameters(params, neuron_model)

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
        configure_spiking_genome(init_struct, neuron_model)

    # Create a prototype genome using the parameters and initialization struct.
    genome_prototype = pnt.Genome(params, init_struct)

    # Create the initial population.
    seed = args.seed if args.seed is not None else int(time.time())
    pop = pnt.Population(genome_prototype, params, True, 1.0, seed)
    spiking_settings = SpikingPolicySettings(
        simulation_steps=args.spiking_steps,
    )

    for gen in range(args.generations):
        if args.legacy_evolution:
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
        else:
            requests = pop.GetEvaluationRequests()
            while requests:
                for request in requests:
                    genome = pop.GetEvaluationGenome(request)
                    case_seed = int(
                        (seed ^ request.case_id ^
                         (request.case_id >> 32)) & 0x7FFFFFFF
                    )
                    fitness = xor_test(
                        genome,
                        spiking=args.spiking,
                        settings=spiking_settings,
                        seed=case_seed,
                    )
                    pop.SubmitEvaluation(
                        request.genome_id,
                        request.case_id,
                        fitness,
                        True,
                        fitness >= 15.9,
                    )
                requests = pop.GetEvaluationRequests()

        # Get the best genome in the population.
        bestGenome = (
            pop.GetBestGenome()
            if args.legacy_evolution
            else pop.GetProvisionalChampion()
        )
        bestFitness = bestGenome.GetFitness()
        print("Generation: {}, Best Fitness: {}".format(gen, bestFitness), flush=True)

        # Advance one generation.
        if gen + 1 < args.generations:
            pop.Epoch()

    validated = False
    success_lower_bound = 0.0
    if not args.legacy_evolution:
        requests = pop.ValidateCandidates(5)
        while requests:
            for request in requests:
                genome = pop.GetEvaluationGenome(request)
                case_seed = int(
                    (seed ^ request.case_id ^
                     (request.case_id >> 32)) & 0x7FFFFFFF
                )
                fitness = xor_test(
                    genome,
                    spiking=args.spiking,
                    settings=spiking_settings,
                    seed=case_seed,
                )
                pop.SubmitEvaluation(
                    request.genome_id,
                    request.case_id,
                    fitness,
                    True,
                    fitness >= 15.9,
                )
            requests = pop.GetEvaluationRequests()
        champion = pop.GetValidatedChampion()
        bestGenome = champion.genome
        bestFitness = champion.validation_summary.mean
        validated = champion.validated
        success_lower_bound = (
            champion.certification_summary.success_lower_bound
        )

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
                "validated": validated,
                "success_lower_bound": success_lower_bound,
                "evolution_mode": (
                    "legacy-v2"
                    if args.legacy_evolution
                    else "reliable-neat"
                ),
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
            show_or_save(draw_genome(bestGenome), args.output)


if __name__ == "__main__":
    main()
