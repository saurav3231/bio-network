"""Simulation orchestration and spike recording.

The scheduler advances the network one millisecond at a time, injects a
stimulus, and hands any resulting synaptic current to the following step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.synapses import RandomSynapses

StimulusFn = Callable[[float, int], np.ndarray]


@dataclass
class SpikeRecording:
    """Record of every spike produced by a simulation.

    Args:
        times_ms: spike times in milliseconds (one entry per spike).
        indices: neuron indices of each spike.
        n_neurons: total number of neurons in the simulated network.
        duration_ms: simulation duration; used to convert counts to rates.
        is_excitatory: per-neuron excitatory mask, when available, used by
            visualizations to color excitatory vs inhibitory spikes.
    """

    times_ms: np.ndarray
    indices: np.ndarray
    n_neurons: int
    duration_ms: float = 0.0
    is_excitatory: np.ndarray | None = None

    def mean_rates_hz(self) -> np.ndarray:
        """Return the per-neuron mean firing rate in Hz.

        Rate is the number of spikes divided by the simulation duration, so
        silent neurons report 0 Hz.
        """
        if self.duration_ms:
            duration_s = self.duration_ms / 1000.0
        elif self.times_ms.size:
            duration_s = (self.times_ms.max() + 1.0) / 1000.0
        else:
            duration_s = 1.0
        counts = np.bincount(self.indices, minlength=self.n_neurons)
        return counts / duration_s


def simulate(
    population: IzhikevichPopulation,
    synapses: RandomSynapses,
    T_ms: float = 1000.0,
    stimulus_fn: StimulusFn | None = None,
    seed: int = 42,
) -> SpikeRecording:
    """Run the network for ``T_ms`` milliseconds.

    Args:
        population: the spiking neuron population to advance.
        synapses: the synaptic weight matrix driving the population.
        T_ms: simulation duration in milliseconds.
        stimulus_fn: optional ``stimulus_fn(t_ms, n_neurons)`` returning the
            input current array for every neuron at that millisecond. If None,
            a default thalamic noise drive is used: ``5 * randn`` for
            excitatory neurons and ``2 * randn`` for inhibitory neurons, as in
            Izhikevich (2003).
        seed: random seed for the default stimulus.

    Returns:
        A ``SpikeRecording`` of all spikes. Synaptic current from spikes at
        time ``t`` is added to the stimulus at time ``t + 1``.
    """
    rng = np.random.default_rng(seed)
    n_neurons = population.v.size
    n_excitatory = population.n_excitatory
    steps = round(T_ms)

    times_list: list[float] = []
    indices_list: list[int] = []
    synaptic = np.zeros(n_neurons)

    for t in range(steps):
        if stimulus_fn is not None:
            current = np.asarray(stimulus_fn(float(t), n_neurons), dtype=float)
        else:
            current = np.zeros(n_neurons)
            current[:n_excitatory] = 5.0 * rng.standard_normal(n_excitatory)
            current[n_excitatory:] = 2.0 * rng.standard_normal(n_neurons - n_excitatory)
        current = current + synaptic

        fired = population.step(current)
        if fired.size:
            times_list.extend([float(t)] * int(fired.size))
            indices_list.extend(int(i) for i in fired)
            synaptic = synapses.deliver(fired)
        else:
            synaptic = np.zeros(n_neurons)

    return SpikeRecording(
        times_ms=np.asarray(times_list, dtype=float),
        indices=np.asarray(indices_list, dtype=np.int64),
        n_neurons=n_neurons,
        duration_ms=float(T_ms),
        is_excitatory=population.is_excitatory.copy(),
    )
