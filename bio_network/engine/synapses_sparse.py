"""Event-driven sparse synaptic engine with axonal transmission delays.

This module provides a drop-in alternative to :class:`RandomSynapses` that
honors sparsity and conduction delays. A synapse only matters when a spike
arrives, and spikes take 1-20 ms to travel along an axon, so instead of
evaluating every potential connection every millisecond we only touch the
connections of the neurons that actually spiked.

Why this matters: the dense engine in :mod:`bio_network.engine.synapses`
computes every synapse every millisecond, including the 95 %+ that carry no
spike. Real brains are sparse and event-driven. This yields (a) large savings
in RAM and compute, and (b) richer, more realistic dynamics.

Reference:
    Izhikevich, E. M. (2006). Polychronization: computation with spikes.
    Neural Computation, 18(2), 245--282.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_OUT_DEGREE = 100
_DEFAULT_MAX_DELAY = 20  # ms, per the range of excitatory axonal delays.


class SparseSynapses:
    """A sparse, event-driven synaptic engine with axonal transmission delays.

    Each neuron projects to a fixed number of distinct post-synaptic targets,
    stored in CSR-style flat arrays (``targets``, ``weights``, ``delays``, plus
    per-neuron ``offsets``). No dense ``N x N`` matrix is stored anywhere.

    Dale's principle is preserved: the outgoing weights of excitatory neurons
    are uniform in ``[0, 0.5]`` and those of inhibitory neurons are uniform in
    ``[-1, 0]``. Delays follow Izhikevich (2006): excitatory synapses are
    uniform integer in ``1..20`` ms and inhibitory synapses are fixed at 1 ms.

    Spike delivery is O(fired x out_degree), never O(n^2): a driven spike is
    fanned out to its post-synaptic targets through a preallocated ring buffer
    of shape ``(max_delay, n_neurons)``.
    """

    def __init__(
        self,
        n_excit: int = 800,
        n_inhib: int = 200,
        out_degree: int = _DEFAULT_OUT_DEGREE,
        seed: int = 42,
        max_delay: int = _DEFAULT_MAX_DELAY,
    ) -> None:
        """Initialize the sparse synaptic graph.

        Args:
            n_excit: number of excitatory neurons.
            n_inhib: number of inhibitory neurons.
            out_degree: number of distinct post-synaptic targets per neuron.
            seed: random seed; a given seed always builds the identical graph.
            max_delay: the ring buffer length, in milliseconds.
        """
        self.n_excit = n_excit
        self.n_inhib = n_inhib
        self.out_degree = out_degree
        self.max_delay = max_delay

        n_neurons = n_excit + n_inhib
        self.n_neurons = n_neurons
        self._n_synapses = n_neurons * out_degree

        self.is_excitatory = np.zeros(n_neurons, dtype=bool)
        self.is_excitatory[:n_excit] = True

        rng = np.random.default_rng(seed)

        # CSR-style flat arrays. offsets[i:i+1] addresses the outgoing edges of
        # neuron i: targets / weights / delays share the same row span.
        self.offsets = np.linspace(0, self._n_synapses, n_neurons + 1, dtype=np.int64)
        self.targets = np.empty(self._n_synapses, dtype=np.int32)
        self.weights = np.empty(self._n_synapses, dtype=np.float64)
        self.delays = np.empty(self._n_synapses, dtype=np.int32)

        for i in range(n_neurons):
            start = int(self.offsets[i])
            end = int(self.offsets[i + 1])
            self.targets[start:end] = rng.choice(
                n_neurons, size=out_degree, replace=False
            )
            if i < n_excit:
                # excitatory pre-synaptic neuron: positive weights.
                self.weights[start:end] = rng.uniform(0.0, 0.5, size=out_degree)
                # delays 1..20 integer ms.
                self.delays[start:end] = rng.integers(1, max_delay + 1, size=out_degree)
            else:
                # inhibitory pre-synaptic neuron: negative weights.
                self.weights[start:end] = rng.uniform(-1.0, 0.0, size=out_degree)
                # fixed 1 ms delay.
                self.delays[start:end] = 1

        # Pre-allocation of the ring buffer for axonal transmission.
        self._queue = np.zeros((max_delay, n_neurons), dtype=np.float32)

    # -- public graph size ------------------------------------------------
    @property
    def n_synapses(self) -> int:
        """Total number of stored synapses (``n_neurons * out_degree``)."""
        return self._n_synapses

    # ---- event-driven delivery ------------------------------------------
    def deliver(self, fired: np.ndarray, t: int) -> None:
        """Deliver the outgoing weights of the spikes fired at time ``t``.

        For each fired pre-synaptic neuron, each of its outgoing weights is
        added into the ring-buffer slot in which the spike will arrive,
        ``(t + delay) % max_delay``. Work is proportional to ``len(fired) *
        out_degree`` and is never quadratic in the population.

        Args:
            fired: indices of pre-synaptic neurons that spiked at time ``t``.
            t: the current simulated time step (milliseconds).
        """
        if fired.size == 0:
            return
        offsets = self.offsets
        targets = self.targets
        weights = self.weights
        delays = self.delays
        queue = self._queue
        max_delay = self.max_delay

        for i in fired:
            start = int(offsets[i])
            end = int(offsets[i + 1])
            slot = (t + delays[start:end]) % max_delay
            row = targets[start:end]
            # targets are distinct within a neuron, so (slot, target) pairs are
            # distinct within one neuron; sequential loop iterations accumulate
            # across neurons correctly.
            queue[slot, row] += weights[start:end]

    def currents(self, t: int) -> np.ndarray:
        """Return the post-synaptic current scheduled to arrive at time ``t``.

        Reads slot ``t % max_delay``, zeroes it for the next use of that slot,
        and returns a copy so callers may mutate it freely.

        Args:
            t: the current time step (milliseconds).

        Returns:
            The per-neuron synaptic input current for time ``t``.
        """
        slot = t % self.max_delay
        row = self._queue[slot].copy()
        self._queue[slot] = 0.0
        return row
