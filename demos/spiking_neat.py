#!/usr/bin/env python3
"""Shared spiking-policy support for the MultiNEAT demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class SpikingPolicySettings:
    """Temporal encoding and decoding settings for an evolved SNN policy."""

    simulation_steps: int = 8
    time_step: float = 0.001
    input_rate_hz: float = 200.0
    output_rate_hz: float = 200.0

    def validate(self) -> None:
        values = (
            self.time_step,
            self.input_rate_hz,
            self.output_rate_hz,
        )
        if self.simulation_steps < 1:
            raise ValueError("spiking simulation_steps must be positive")
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("spiking rates and time step must be finite and positive")


def configure_spiking_parameters(
    neat: Any,
    params: Any,
    *,
    recurrent: bool,
    enable_stdp: bool = False,
) -> Any:
    """Apply a control-oriented evolvable SNN preset."""

    if not hasattr(params, "ConfigureSpiking"):
        raise RuntimeError(
            "This demo requires a pymultineat build with spiking support"
        )
    params.ConfigureSpiking(enable_stdp)
    params.DontUseBiasNeuron = True
    params.AllowLoops = recurrent
    params.RecurrentProb = 0.2 if recurrent else 0.0
    params.SplitRecurrent = recurrent
    params.SplitLoopedRecurrent = recurrent
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
    params.InitialSTDPEnabledProb = 0.1 if enable_stdp else 0.0
    return params


def configure_spiking_genome(neat: Any, initial: Any) -> Any:
    """Select adaptive hidden neurons and LIF output neurons."""

    initial.HiddenActType = neat.SPIKING_ADAPTIVE_LIF
    initial.OutputActType = neat.SPIKING_LIF
    return initial


class SpikingPolicy:
    """Stateful Poisson-encoded, filtered-rate-decoded SNN policy."""

    def __init__(
        self,
        network: Any,
        settings: SpikingPolicySettings,
        *,
        recorder: Any | None = None,
    ) -> None:
        settings.validate()
        if not network.IsSpiking():
            raise ValueError("SpikingPolicy requires a spiking phenotype")
        self.network = network
        self.settings = settings
        self.recorder = recorder
        network.SetSpikingInputMode(network_input_mode(network))
        network.SetSpikingOutputMode(network_output_mode(network))
        network.SetSpikingTimeStep(settings.time_step)
        network.EnableSTDP(False)

    def reset(self, seed: int) -> None:
        self.network.Flush()
        self.network.SeedSpiking(int(seed))
        if self.recorder is not None:
            self.recorder.clear(reset_network=False)

    def encode(self, values: Sequence[float]) -> list[float]:
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
            rates.append(
                0.5 * (normalized + 1.0) * self.settings.input_rate_hz
            )
        return rates

    def step_rates(self, values: Sequence[float]) -> list[float]:
        rates = self.encode(values)
        outputs: list[float] = []
        for _ in range(self.settings.simulation_steps):
            if self.recorder is None:
                outputs = list(
                    self.network.StepSpiking(rates, self.settings.time_step)
                )
            else:
                outputs = self.recorder.step(rates, self.settings.time_step)
        return outputs

    def step_unsigned(self, values: Sequence[float]) -> list[float]:
        return [
            min(1.0, max(0.0, output / self.settings.output_rate_hz))
            for output in self.step_rates(values)
        ]

    def step_signed(self, values: Sequence[float]) -> list[float]:
        return [
            2.0 * value - 1.0
            for value in self.step_unsigned(values)
        ]


def network_input_mode(network: Any) -> Any:
    """Return the binding's Poisson-rate enum without importing it globally."""

    module = __import__(type(network).__module__)
    return module.POISSON_RATE_INPUT


def network_output_mode(network: Any) -> Any:
    """Return the binding's filtered-spike enum."""

    module = __import__(type(network).__module__)
    return module.FILTERED_SPIKE_OUTPUT
