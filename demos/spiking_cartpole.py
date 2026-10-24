#!/usr/bin/env python3
"""Evolve and visualize a spiking controller for a self-contained cart-pole."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
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
    legacy_evolution: bool = False,
) -> tuple[neat.Genome, list[float], bool, float]:
    params = parameters(
        population_size, mcculloch_pitts, legacy_evolution
    )
    population = neat.Population(
        seed_genome(params, mcculloch_pitts),
        params,
        True,
        6.0,
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
                    fitness = evaluate(
                        genome,
                        max_steps,
                        seed + generation * 997,
                    )
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
                    case_seed = int(
                        (seed ^ request.case_id ^
                         (request.case_id >> 32)) & 0x7FFFFFFF
                    )
                    fitness = evaluate(genome, max_steps, case_seed)
                    population.SubmitEvaluation(
                        request.genome_id,
                        request.case_id,
                        fitness,
                        True,
                        fitness >= 0.9 * max_steps,
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
                case_seed = int(
                    (seed ^ request.case_id ^
                     (request.case_id >> 32)) & 0x7FFFFFFF
                )
                fitness = evaluate(genome, max_steps, case_seed)
                population.SubmitEvaluation(
                    request.genome_id,
                    request.case_id,
                    fitness,
                    True,
                    fitness >= 0.9 * max_steps,
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

    def clear(self) -> None:
        """Drop the traces, e.g. after ``network.Flush()`` restarts time at 0."""

        self.times.clear()
        self.membrane.clear()
        self.filtered_rates.clear()

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


def replay(
    genome: neat.Genome,
    max_steps: int,
    seed: int,
):
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

    if animate:
        environment = CartPole()
        environment.reset(seed)
        network = build_network(best, seed + 101)
        network.EnableSpikeRecording(True)
        recorder = SpikingRecorder(network)
        state = {"force": 0.0, "alive": True, "control": 0}

        def inputs(frame: int, _time: float) -> list[float]:
            # The SNN runs SNN_STEPS_PER_CONTROL solver steps per physics step.
            if frame > 0 and frame % SNN_STEPS_PER_CONTROL == 0:
                output = network.OutputFilteredSpikes()
                state["force"] = math.tanh(
                    (output[1] - output[0]) / 50.0
                )
                state["alive"] = environment.step(state["force"])
                state["control"] += 1
                if not state["alive"]:
                    # Restart the episode; Flush() also resets SNN time to 0.
                    environment.reset(seed + state["control"])
                    network.Flush()
                    network.SeedSpiking(seed + 101 + state["control"])
                    recorder.clear()
                    state["alive"] = True
            return environment.rates()

        def draw_environment(axis, _frame: int) -> None:
            draw_cartpole(axis, environment, state["force"])

        animation = animate_spiking_network(
            network,
            recorder,
            inputs,
            frames=max_steps * SNN_STEPS_PER_CONTROL,
            interval=15,
            time_step=SNN_DT,
            window_seconds=0.5,
            trace_neurons=[
                network.NumInputs(),
                network.NumInputs() + 1,
            ],
            environment_draw=draw_environment,
        )
        if output is None:
            plt.show()
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            animation.save(output, fps=60)
        return

    environment, network, recorder, _ = replay(best, max_steps, seed)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    draw_spiking_network(network, axes[0, 0])
    draw_cartpole(axes[0, 1], environment)
    plot_spike_raster(network, axes[1, 0])
    plot_traces(
        recorder,
        axes[1, 1],
        [network.NumInputs(), network.NumInputs() + 1],
        "rate",
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
        "--legacy-evolution",
        action="store_true",
        help="use exact MultiNEAT 2 scalar-fitness evolution",
    )
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
    best, history, validated, success_lower_bound = evolve(
        args.generations,
        args.population,
        args.max_steps,
        args.seed,
        args.mcculloch_pitts,
        args.legacy_evolution,
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
                "evolution_mode": (
                    "legacy-v2"
                    if args.legacy_evolution
                    else "reliable-neat"
                ),
                "validated": validated,
                "success_lower_bound": success_lower_bound,
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
