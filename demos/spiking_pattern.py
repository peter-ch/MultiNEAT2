#!/usr/bin/env python3
"""Evolve a spiking network that detects synchronous spike patterns."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
import random
import sys
from typing import Callable, Sequence


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
    legacy_evolution: bool = False,
) -> neat.Parameters:
    params = neat.Parameters()
    if legacy_evolution:
        params.ConfigureLegacyV2()
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
    if not legacy_evolution:
        params.DeterministicEvaluation = True
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
    legacy_evolution: bool = False,
) -> tuple[neat.Genome, list[float], bool, float]:
    params = parameters(
        population_size, mcculloch_pitts, legacy_evolution
    )
    population = neat.Population(
        seed_genome(params, mcculloch_pitts),
        params,
        True,
        5.0,
        seed,
    )
    best = None
    validated = False
    success_lower_bound = 0.0
    history = []
    for generation in range(generations):
        generation_best = None
        if legacy_evolution:
            for species in population.m_Species:
                for genome in species.m_Individuals:
                    fitness = evaluate(genome)
                    genome.SetFitness(fitness)
                    genome.SetEvaluated()
                    if (
                        generation_best is None
                        or fitness > generation_best.GetFitness()
                    ):
                        generation_best = neat.Genome.Deserialize(
                            genome.Serialize()
                        )
        else:
            requests = population.GetEvaluationRequests()
            while requests:
                for request in requests:
                    genome = population.GetEvaluationGenome(request)
                    fitness = evaluate(genome)
                    population.SubmitEvaluation(
                        request.genome_id,
                        request.case_id,
                        fitness,
                        True,
                        fitness >= 15.5,
                    )
                requests = population.GetEvaluationRequests()
            generation_best = population.GetProvisionalChampion()
        assert generation_best is not None
        history.append(generation_best.GetFitness())
        if best is None or generation_best.GetFitness() > best.GetFitness():
            best = generation_best
        if generation + 1 < generations:
            population.Epoch()
    if not legacy_evolution:
        requests = population.ValidateCandidates(5)
        while requests:
            for request in requests:
                genome = population.GetEvaluationGenome(request)
                fitness = evaluate(genome)
                population.SubmitEvaluation(
                    request.genome_id,
                    request.case_id,
                    fitness,
                    True,
                    fitness >= 15.5,
                )
            requests = population.GetEvaluationRequests()
        champion = population.GetValidatedChampion()
        best = champion.genome
        validated = champion.validated
        success_lower_bound = (
            champion.certification_summary.success_lower_bound
        )
    assert best is not None
    return best, history, validated, success_lower_bound


# ---------------------------------------------------------------------------
# Recording and visualization (matplotlib/networkx are imported only here)
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
TRACE_COLORS = ("#2563eb", "#38bdf8", "#fb7185")


class SpikingRecorder:
    """Step a spiking network and keep membrane and firing-rate traces.

    Spike events themselves are kept by the network (``EnableSpikeRecording``)
    and read back through :attr:`events`.
    """

    def __init__(self, network: neat.NeuralNetwork) -> None:
        self.network = network
        self.times: list[float] = []
        self.membrane: list[list[float]] = []
        self.filtered_rates: list[list[float]] = []

    @property
    def events(self) -> list:
        return list(self.network.GetSpikeHistory())

    def step(self, inputs: Sequence[float], time_step: float) -> list[float]:
        output = self.network.StepSpiking(list(inputs), time_step)
        neurons = self.network.m_neurons
        self.times.append(float(self.network.SpikingTime()))
        self.membrane.append([float(n.m_membrane_potential) for n in neurons])
        self.filtered_rates.append([float(n.m_rate_trace) for n in neurons])
        return list(output)


def style_axis(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=FOREGROUND)
    ax.set_xlabel(xlabel, color=FOREGROUND)
    ax.set_ylabel(ylabel, color=FOREGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.figure.patch.set_facecolor(BACKGROUND)
    ax.tick_params(colors=MUTED)


def dark_legend(ax, **kwargs) -> None:
    legend = ax.legend(facecolor=BACKGROUND, edgecolor=GRID, framealpha=0.88, **kwargs)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)


def draw_spiking_network(network: neat.NeuralNetwork, ax, title: str | None = None) -> None:
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
    if title is None:
        active = sum(bool(neuron.m_spike) for neuron in neurons)
        title = f"Spiking phenotype · t={network.SpikingTime():.3f}s · {active} active"
    style_axis(ax, title)
    ax.axis("off")


def plot_spike_raster(
    network: neat.NeuralNetwork,
    ax,
    window: tuple[float, float] | None = None,
) -> None:
    """Scatter recorded input and neuron spikes on a shared time axis."""

    events = network.GetSpikeHistory()
    if window is not None:
        events = [e for e in events if window[0] <= e.time <= window[1]]
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
    if window is not None:
        ax.set_xlim(*window)
    style_axis(ax, "Spike trains", "Time (s)", "Neuron")
    ax.grid(axis="x", color=GRID, alpha=0.35)
    if events:
        dark_legend(ax, loc="upper right")


def plot_traces(
    recorder: SpikingRecorder,
    ax,
    neurons: Sequence[int],
    quantity: str = "membrane",
    window: tuple[float, float] | None = None,
) -> None:
    """Plot membrane potentials or filtered firing rates of chosen neurons."""

    samples = recorder.membrane if quantity == "membrane" else recorder.filtered_rates
    first = 0 if window is None else bisect.bisect_left(recorder.times, window[0])
    times = recorder.times[first:]
    for order, neuron in enumerate(neurons):
        ax.plot(
            times,
            [sample[neuron] for sample in samples[first:]],
            label=f"neuron {neuron}",
            color=TRACE_COLORS[order % len(TRACE_COLORS)],
            alpha=0.9,
        )
    if window is not None:
        ax.set_xlim(*window)
    if quantity == "membrane":
        style_axis(ax, "Membrane potentials", "Time (s)", "Potential")
    else:
        style_axis(ax, "Filtered firing rates", "Time (s)", "Filtered rate (Hz)")
    ax.grid(color=GRID, alpha=0.3)
    if neurons and times:
        dark_legend(ax, loc="upper right", ncol=2)


def animate_spiking_network(
    network: neat.NeuralNetwork,
    recorder: SpikingRecorder,
    input_provider: Callable[[int, float], Sequence[float]],
    *,
    frames: int,
    interval: int,
    time_step: float,
    window_seconds: float,
    environment_draw: Callable[[object, int], None],
    trace_neurons: Sequence[int] | None = None,
):
    """Animate topology, a scrolling spike raster, traces and the task.

    Each frame advances the network by one solver step with
    ``input_provider(frame, time)``. Keep the returned animation alive.
    """

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    (network_axis, environment_axis), (raster_axis, membrane_axis) = axes
    if trace_neurons is None:
        trace_neurons = list(range(network.NumInputs(), len(network.m_neurons)))

    def update(frame: int) -> list:
        recorder.step(input_provider(frame, network.SpikingTime()), time_step)
        now = float(network.SpikingTime())
        window = (max(0.0, now - window_seconds), now + 1.0e-12)
        for axis in axes.flat:
            axis.clear()
        draw_spiking_network(network, network_axis)
        plot_spike_raster(network, raster_axis, window)
        plot_traces(recorder, membrane_axis, trace_neurons, "membrane", window)
        environment_draw(environment_axis, frame)
        return []

    return FuncAnimation(
        figure, update, frames=frames, interval=interval,
        blit=False, cache_frame_data=False,
    )


def replay(best: neat.Genome):
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

    network, recorder = replay(best)
    if animate:
        name, sequence, target = PATTERNS[0]
        network.Flush()
        network.EnableSpikeRecording(True)

        def inputs(frame: int, _time: float) -> list[float]:
            return sequence[frame % len(sequence)]

        def environment(axis, frame: int) -> None:
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
            style_axis(axis, f"{name} · target={target:g}", "Pattern step")

        animation = animate_spiking_network(
            network,
            SpikingRecorder(network),
            inputs,
            frames=PATTERN_STEPS * 4,
            interval=20,
            time_step=DT,
            window_seconds=0.08,
            environment_draw=environment,
        )
        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(output, fps=50)
        return

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    draw_spiking_network(network, axes[0])
    plot_spike_raster(network, axes[1])
    plot_traces(recorder, axes[2], [network.NumInputs()], "rate")
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
        "--legacy-evolution",
        action="store_true",
        help="use exact MultiNEAT 2 scalar-fitness evolution",
    )
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
    best, history, validated, success_lower_bound = evolve(
        args.generations,
        args.population,
        args.seed,
        args.mcculloch_pitts,
        args.legacy_evolution,
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
        "evolution_mode": (
            "legacy-v2" if args.legacy_evolution else "reliable-neat"
        ),
        "validated": validated,
        "success_lower_bound": success_lower_bound,
    }
    print(json.dumps(payload))
    if not args.no_show or args.output is not None:
        visualize(best, args.output, args.animate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
