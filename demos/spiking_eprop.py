#!/usr/bin/env python3
"""Train an adaptive spiking network with online e-prop."""

from __future__ import annotations

import argparse
import json
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


def replay(network: neat.NeuralNetwork):
    from neattools import SpikingRecorder

    replay_network = neat.NeuralNetwork.Deserialize(network.Serialize())
    replay_network.Flush()
    replay_network.EnableSpikeRecording(True)
    recorder = SpikingRecorder(replay_network)
    for _ in range(SEQUENCE_STEPS):
        recorder.step([1.0], DT)
    return replay_network, recorder


def visualize(
    network: neat.NeuralNetwork,
    losses: list[float],
    output: Path | None,
) -> None:
    import matplotlib.pyplot as plt
    from neattools import DrawSpikingNetwork, PlotSpikeRaster

    replay_network, recorder = replay(network)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#0f172a")
    axes[0].set_facecolor("#0f172a")
    axes[0].plot(
        range(1, len(losses) + 1),
        losses,
        color="#38bdf8",
        linewidth=2,
    )
    axes[0].set_xlabel("Training epoch", color="#e2e8f0")
    axes[0].set_ylabel("Mean temporal loss", color="#e2e8f0")
    axes[0].set_title("Online e-prop learning", color="#e2e8f0")
    axes[0].tick_params(colors="#94a3b8")
    axes[0].grid(color="#334155", alpha=0.6)
    PlotSpikeRaster(recorder, ax=axes[1], show=False)
    axes[1].set_title("Learned spike response")
    DrawSpikingNetwork(replay_network, ax=axes[2], show=False)
    axes[2].set_title("Trained phenotype")
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
    replay_network, recorder = replay(network)
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
        "recorded_spikes": len(recorder.events),
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
