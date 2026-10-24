#!/usr/bin/env python3
"""Train an adaptive spiking network with online e-prop."""

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


DT = 0.001
SEQUENCE_STEPS = 20


def build_network(mcculloch_pitts: bool = False) -> neat.NeuralNetwork:
    network = neat.NeuralNetwork()

    input_neuron = neat.Neuron()
    input_neuron.m_type = neat.INPUT

    output_neuron = neat.Neuron()
    output_neuron.m_type = neat.OUTPUT
    output_neuron.m_activation_function_type = (
        neat.MCCULLOCH_PITTS
        if mcculloch_pitts
        else neat.SPIKING_LIF
    )
    output_neuron.m_timeconst = 0.01
    output_neuron.m_spike_threshold = 0.5
    output_neuron.m_refractory_period = 0.0

    hidden_neuron = neat.Neuron()
    hidden_neuron.m_type = neat.HIDDEN
    hidden_neuron.m_activation_function_type = (
        neat.MCCULLOCH_PITTS
        if mcculloch_pitts
        else neat.SPIKING_ADAPTIVE_LIF
    )
    hidden_neuron.m_timeconst = 0.01
    hidden_neuron.m_spike_threshold = 0.5
    hidden_neuron.m_refractory_period = 0.0
    hidden_neuron.m_adaptation_time_constant = 0.1
    hidden_neuron.m_adaptation_increment = 0.01

    input_link = neat.Connection()
    input_link.m_source_neuron_idx = 0
    input_link.m_target_neuron_idx = 2
    input_link.m_weight = 0.1
    input_link.m_synaptic_time_constant = 0.001

    output_link = neat.Connection()
    output_link.m_source_neuron_idx = 2
    output_link.m_target_neuron_idx = 1
    output_link.m_weight = 2.0
    output_link.m_synaptic_time_constant = 0.001

    network.AddNeuron(input_neuron)
    network.AddNeuron(output_neuron)
    network.AddNeuron(hidden_neuron)
    network.AddConnection(input_link)
    network.AddConnection(output_link)
    network.SetInputOutputDimensions(1, 1)
    network.SetSpikingInputMode(neat.BINARY_SPIKE_INPUT)
    network.SetSpikingOutputMode(neat.SPIKE_OUTPUT)
    network.SetSpikingTimeStep(DT)
    network.EnableSpikeRecording(False)
    network.Flush()
    return network


def build_learner(network: neat.NeuralNetwork, seed: int) -> neat.EPropLearner:
    config = neat.EPropConfig()
    config.learning_rate = 0.2
    config.optimizer = neat.EPROP_ADAMW
    config.feedback_mode = neat.EPROP_SYMMETRIC_FEEDBACK
    config.surrogate = neat.EPROP_FAST_SIGMOID
    config.surrogate_scale = 5.0
    config.surrogate_dampening = 0.5
    config.gradient_clip_norm = 10.0
    config.min_weight = -12.0
    config.max_weight = 12.0
    config.update_interval = SEQUENCE_STEPS
    config.random_seed = seed
    learner = neat.EPropLearner(config)
    learner.Initialize(network)
    return learner


def train(
    epochs: int,
    seed: int,
    mcculloch_pitts: bool = False,
) -> tuple[neat.NeuralNetwork, neat.EPropLearner, list[float]]:
    network = build_network(mcculloch_pitts)
    learner = build_learner(network, seed)
    inputs = [[1.0] for _ in range(SEQUENCE_STEPS)]
    targets = [[1.0] for _ in range(SEQUENCE_STEPS)]
    losses = []
    for _ in range(epochs):
        result = learner.TrainSequence(
            network,
            inputs,
            targets,
            DT,
            True,
            True,
        )
        losses.append(result.mean_loss)
    return network, learner, losses


def replay(network: neat.NeuralNetwork) -> neat.NeuralNetwork:
    """Run a copy of the trained network once, recording every spike."""

    replay_network = neat.NeuralNetwork.Deserialize(network.Serialize())
    replay_network.Flush()
    replay_network.EnableSpikeRecording(True)
    for _ in range(SEQUENCE_STEPS):
        replay_network.StepSpiking([1.0], DT)
    return replay_network


# ---------------------------------------------------------------------------
# Visualization (matplotlib and networkx are imported only when drawing)
# ---------------------------------------------------------------------------
BACKGROUND = "#0f172a"
FOREGROUND = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#334155"
NEURON_STYLE = {  # neuron type -> (color, marker)
    neat.INPUT: ("#22c55e", "s"),
    neat.BIAS: ("#facc15", "h"),
    neat.HIDDEN: ("#38bdf8", "o"),
    neat.OUTPUT: ("#fb7185", "D"),
}


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=FOREGROUND)
    ax.set_xlabel(xlabel, color=FOREGROUND)
    ax.set_ylabel(ylabel, color=FOREGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=MUTED)


def draw_spiking_network(network: neat.NeuralNetwork, ax, title: str) -> None:
    """Draw a phenotype: spiking neurons are enlarged, others show potential.

    Blue links are excitatory, red inhibitory; width grows with |weight|.
    """

    import networkx as nx

    neurons = network.m_neurons
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(neurons)))
    levels: dict[float, list[int]] = {}
    for index, neuron in enumerate(neurons):
        level = float(neuron.m_split_y)
        if neuron.m_type in (neat.INPUT, neat.BIAS):
            level = 0.0
        elif neuron.m_type == neat.OUTPUT:
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
        weight = connection.m_weight
        magnitude = math.sqrt(abs(weight) / max_abs_weight) if max_abs_weight else 0.0
        curved = connection.m_recur_flag or source == target
        nx.draw_networkx_edges(
            graph, positions, edgelist=[(source, target)],
            width=0.6 + 4.4 * magnitude, alpha=0.25 + 0.7 * magnitude,
            edge_color="#2563eb" if weight > 0 else "#ef4444" if weight < 0 else "#64748b",
            style="dotted" if connection.m_stdp_enabled else "solid",
            connectionstyle="arc3,rad=0.22" if curved else "arc3,rad=0.02",
            arrows=True, arrowsize=12, ax=ax,
        )
    for neuron_type, (color, marker) in NEURON_STYLE.items():
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


def plot_spike_raster(network: neat.NeuralNetwork, ax, title: str) -> None:
    """Scatter every recorded input and neuron spike on a shared time axis."""

    events = network.GetSpikeHistory()
    for is_input, label, color in (
        (True, "Input spike", NEURON_STYLE[neat.INPUT][0]),
        (False, "Neuron spike", NEURON_STYLE[neat.OUTPUT][0]),
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
        legend = ax.legend(loc="upper right", facecolor=BACKGROUND,
                           edgecolor=GRID, framealpha=0.88)
        for text in legend.get_texts():
            text.set_color(FOREGROUND)


def visualize(
    network: neat.NeuralNetwork,
    losses: list[float],
    output: Path | None,
) -> None:
    import matplotlib.pyplot as plt

    replay_network = replay(network)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )
    axes[0].plot(
        range(1, len(losses) + 1),
        losses,
        color="#38bdf8",
        linewidth=2,
    )
    style_axis(axes[0], "Online e-prop learning", "Training epoch", "Mean temporal loss")
    axes[0].grid(color=GRID, alpha=0.6)
    plot_spike_raster(replay_network, axes[1], "Learned spike response")
    draw_spiking_network(replay_network, axes[2], "Trained phenotype")
    if output is None:
        plt.show()
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recurrently stateful SNN with online "
            "eligibility propagation."
        )
    )
    parser.add_argument(
        "--epochs",
        "--generations",
        dest="epochs",
        type=int,
        default=60,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mcculloch-pitts",
        action="store_true",
        help="train an equivalent McCulloch-Pitts circuit",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.smoke:
        args.epochs = 2
        args.no_show = True

    network, learner, losses = train(
        args.epochs, args.seed, args.mcculloch_pitts
    )
    replay_network = replay(network)
    payload = {
        "demo": "spiking_eprop",
        "neuron_model": (
            "mcculloch-pitts" if args.mcculloch_pitts else "lif"
        ),
        "generations": args.epochs,
        "population": 1,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "optimizer_updates": learner.OptimizerStep(),
        "neurons": len(network.m_neurons),
        "links": len(network.m_connections),
        "weights": [
            network.GetConnectionByIndex(index).m_weight
            for index in range(len(network.m_connections))
        ],
        "recorded_spikes": len(replay_network.GetSpikeHistory()),
        "output_spikes": int(
            replay_network.GetNeuronByIndex(1).m_spike_count
        ),
    }
    print(json.dumps(payload))
    if not args.no_show or args.output is not None:
        visualize(network, losses, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
