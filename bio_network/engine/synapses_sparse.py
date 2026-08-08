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
_TRACE_NEVER = -1.0e18
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

    The ``gain`` parameter scales the excitatory weights at construction. In a
    recurrent network the mean firing rate is controlled by the balance of
    excitation and inhibition, and that balance depends on the number of
    inputs each neuron receives (its fan-in). In balanced network theory
    (van Vreeswijk & Sompolinsky, 1998) neurons remain in a fluctuating
    regime only when excitation and inhibition track the fan-in; halving the
    fan-in without re-scaling weights pushes the network toward a low-gain,
    quieter regime. ``gain`` re-establishes the excitatory drive lost by the
    reduced fan-in of sparse connectivity (out_degree << N) so a sparse
    engine can reproduce the dense engine's mean rate.
    """

    def __init__(
        self,
        n_excit: int = 800,
        n_inhib: int = 200,
        out_degree: int = _DEFAULT_OUT_DEGREE,
        seed: int = 42,
        max_delay: int = _DEFAULT_MAX_DELAY,
        gain: float = 1.0,
    ) -> None:
        """Initialize the sparse synaptic graph.

        Args:
            n_excit: number of excitatory neurons.
            n_inhib: number of inhibitory neurons.
            out_degree: number of distinct post-synaptic targets per neuron.
            seed: random seed; a given seed always builds the identical graph.
            max_delay: the ring buffer length, in milliseconds.
            gain: multiplicative scale applied to excitatory weights. The
                neutral value is 1.0 (identity). At ``out_degree=100`` a gain
                of ``10.0`` reproduces the dense baseline mean rate (see
                ``docs/M1_5_RESULTS.md`` for the calibration sweep).
        """
        self.n_excit = n_excit
        self.n_inhib = n_inhib
        self.out_degree = out_degree
        self.max_delay = max_delay
        self.gain = float(gain)

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
                self.weights[start:end] = (
                    rng.uniform(0.0, 0.5, size=out_degree) * self.gain
                )
                # delays 1..20 integer ms.
                self.delays[start:end] = rng.integers(1, max_delay + 1, size=out_degree)
            else:
                # inhibitory pre-synaptic neuron: negative weights.
                self.weights[start:end] = rng.uniform(-1.0, 0.0, size=out_degree)
                # fixed 1 ms delay.
                self.delays[start:end] = 1

        # Pre-allocation of the ring buffer for axonal transmission.
        self._queue = np.zeros((max_delay, n_neurons), dtype=np.float32)

        # STDP (M2) state. Everything here is lazily allocated the first time
        # `enable_learning()` is called, so the ordinary sparse engine (used by
        # the M1.5 benchmarks and the 50k memory test) pays nothing for it.
        self._learning_enabled = False
        self._a_plus = 0.1
        self._a_minus = 0.12
        self._tau_plus = 20.0
        self._tau_minus = 20.0
        self._syn_trace = None
        self._syn_last = None
        self._post_trace = None
        self._post_last = None
        self._in_syn = None
        self._in_offsets = None
        self._arrivals = None

    # -- plasticity helpers -----------------------------------------------
    def enable_learning(
        self,
        a_plus: float = 0.1,
        a_minus: float = 0.12,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ) -> None:
        """Enable STDP and lazily allocate the plasticity state.

        Excitatory weights are plastic; inhibitory weights are frozen (see
        ``docs/ARCHITECTURE.md``). Rule follows Bi & Poo (1998) with the stable
        range (normalized) bounds of Song, Miller & Abbott (2000).

        Args:
            a_plus: potentiation amplitude (LTP occurs on causal order).
            A_minus: depression amplitude (LTD occurs on causal order).
                Slightly larger than ``A_plus`` (``0.12 > 0.1``) so depression
                balances potentiation and keeps the network stable.
            tau_plus: LTP time constant, in ms (Bi & Poo window).
            tau_minus: LTD time constant, in ms.
        """
        self._a_plus = float(a_plus)
        self._a_minus = float(a_minus)
        self._tau_plus = float(tau_plus)
        self._tau_minus = float(tau_minus)
        self._learning_enabled = True

        if self._syn_trace is None:
            self._allocate_plasticity()

    def _allocate_plasticity(self) -> None:
        """Allocate the STDP trace/arrival state (idempotent helper)."""
        n_syn = self._n_synapses
        n_neurons = self.n_neurons
        self._syn_trace = np.zeros(n_syn, dtype=np.float64)
        self._syn_last = np.full(n_syn, _TRACE_NEVER)
        self._post_trace = np.zeros(n_neurons, dtype=np.float64)
        self._post_last = np.full(n_neurons, _TRACE_NEVER)
        self._arrivals = [[] for _ in range(self.max_delay)]

        # Reverse (incoming) adjacency for LTP: the incoming synapses of each
        # neuron are the plastic excitatory synapses that target it.
        incoming: list[list[int]] = [[] for _ in range(n_neurons)]
        targets = self.targets
        n_pooled = self.n_excit * self.out_degree  # flat excitatory block
        for s in range(n_pooled):
            incoming[int(targets[s])].append(s)
        lengths = np.fromiter((len(b) for b in incoming), dtype=np.int64)
        self._in_offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        flat = np.empty(int(self._in_offsets[-1]), dtype=np.int64)
        for j, bucket in enumerate(incoming):
            start = int(self._in_offsets[j])
            flat[start : start + len(bucket)] = bucket
        self._in_syn = flat

    # -- state snapshot / restore -----------------------------------------
    def save_state(self) -> dict:
        """Return a deep copy of every mutable quantity needed to resume.

        Captures the plastic weights, the axonal-delay ring buffer, the STDP
        traces (if learning was enabled) and the arrival ledger. The static
        graph (``targets``/``delays``/``offsets``) never changes after
        construction, so it is not copied. Used to fork two experimental arms
        from an identical network state (see E4b in the M4 benchmark).

        Returns:
            A dict that can be passed back to :meth:`load_state`.
        """
        state = {
            "weights": self.weights.copy(),
            "_queue": self._queue.copy(),
            "_learning_enabled": self._learning_enabled,
            "_a_plus": self._a_plus,
            "_a_minus": self._a_minus,
            "_tau_plus": self._tau_plus,
            "_tau_minus": self._tau_minus,
        }
        if self._syn_trace is not None:
            state["_syn_trace"] = self._syn_trace.copy()
            state["_syn_last"] = self._syn_last.copy()
            state["_post_trace"] = self._post_trace.copy()
            state["_post_last"] = self._post_last.copy()
            state["_arrivals"] = [
                [arr.copy() for arr in bucket] for bucket in self._arrivals
            ]
            state["_in_syn"] = self._in_syn.copy()
            state["_in_offsets"] = self._in_offsets.copy()
        return state

    def load_state(self, state: dict) -> None:
        """Restore every mutable quantity captured by :meth:`save_state`.

        The network graph (``targets``/``delays``/``offsets``) is *not*
        touched, so the state must come from a synapses object with the same
        topology (e.g. built from the same seed). After loading, the engine
        produces bit-identical future spikes to the arm that saved it.

        Args:
            state: the dict returned by :meth:`save_state`.
        """
        self.weights[...] = state["weights"]
        self._queue[...] = state["_queue"]
        self._learning_enabled = state["_learning_enabled"]
        self._a_plus = state["_a_plus"]
        self._a_minus = state["_a_minus"]
        self._tau_plus = state["_tau_plus"]
        self._tau_minus = state["_tau_minus"]
        if "_syn_trace" in state:
            if self._syn_trace is None:
                self._allocate_plasticity()
            self._syn_trace[...] = state["_syn_trace"]
            self._syn_last[...] = state["_syn_last"]
            self._post_trace[...] = state["_post_trace"]
            self._post_last[...] = state["_post_last"]
            self._arrivals = [
                [arr.copy() for arr in bucket] for bucket in state["_arrivals"]
            ]
            self._in_syn[...] = state["_in_syn"]
            self._in_offsets[...] = state["_in_offsets"]

    # -- public graph size ------------------------------------------------
    @property
    def n_synapses(self) -> int:
        """Total number of stored synapses (``n_neurons * out_degree``)."""
        return self._n_synapses

    # ---- event-driven delivery ------------------------------------------
    def deliver(self, fired: np.ndarray, t: int, learn: bool = False) -> None:
        """Deliver the outgoing weights of the spikes fired at time ``t``.

        For each fired pre-synaptic neuron, each of its outgoing weights is
        added into the ring-buffer slot in which the spike will arrive,
        ``(t + delay) % max_delay``. Work is proportional to ``len(fired) *
        out_degree`` and is never quadratic in the population.

        When ``learn`` is True the fired synapses are also booked into the
        arrival ledger so the LTD/plasticity updates can run on the exact
        millisecond each spike arrives (``currents`` processes them).

        Args:
            fired: indices of pre-synaptic neurons that spiked at time ``t``.
            t: the current simulated time step (milliseconds).
            learn: record arrival events for STDP (must follow
                ``enable_learning``).
        """
        if fired.size == 0:
            return
        offsets = self.offsets
        targets = self.targets
        weights = self.weights
        delays = self.delays
        queue = self._queue
        max_delay = self.max_delay
        t = int(t)

        for i in fired:
            start = int(offsets[i])
            end = int(offsets[i + 1])
            slot = (t + delays[start:end]) % max_delay
            row = targets[start:end]
            # targets are distinct within a neuron, so (slot, target) pairs are
            # distinct within one neuron; sequential loop iterations accumulate
            # across neurons correctly.
            queue[slot, row] += weights[start:end]

        if learn and self._learning_enabled:
            # Book every fired EXCITATORY synapse for arrival-time LTD,
            # bucketed by the ring-buffer slot it will arrive in. Plasticity is
            # excitatory-only (see ARCHITECTURE.md); inhibitory synapses are
            # frozen, so an inhibitory neuron firing must never enqueue an
            # arrival event for its (negative) outgoing weights.
            blocks = [
                np.arange(int(offsets[i]), int(offsets[i + 1]))
                for i in fired
                if i < self.n_excit and int(offsets[i + 1]) > int(offsets[i])
            ]
            ids = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
            if ids.size:
                slot_arr = (t + delays[ids]) % max_delay
                for s in np.unique(slot_arr):
                    self._arrivals[int(s)].append(ids[slot_arr == s])

    def currents(self, t: int, learn: bool = False) -> np.ndarray:
        """Return the post-synaptic current scheduled to arrive at time ``t``.

        Reads slot ``t % max_delay``, zeroes it for the next use of that slot,
        and returns a copy so callers may mutate it freely.

        When ``learn`` is True and ``enable_learning`` has been called, every
        synapse arriving at time ``t`` first applies its LTD update (weights
        are depressed by the post-synaptic trace) and marks its pre-synaptic
        trace, which is then used for LTP when the post-synaptic neuron fires.

        Args:
            t: the current time step (milliseconds).
            learn: apply the arrival-time half of STDP.

        Returns:
            The per-neuron synaptic input current for time ``t``.
        """
        t = int(t)
        slot = t % self.max_delay
        row = self._queue[slot].copy()
        self._queue[slot] = 0.0

        if learn and self._learning_enabled:
            arrivals = self._arrivals[slot]
            if arrivals:
                syn = np.concatenate(arrivals)
                arrivals.clear()
                post = self.targets[syn]
                dt_post = t - self._post_last[post]
                traces = np.where(
                    self._post_last[post] == _TRACE_NEVER,
                    0.0,
                    self._post_trace[post]
                    * np.exp(-np.maximum(dt_post, 0.0) / self._tau_minus),
                )
                w = self.weights[syn] - self._a_minus * traces
                np.clip(w, 0.0, 1.0, out=w)
                self.weights[syn] = w
                dt_syn = t - self._syn_last[syn]
                syn_trace = np.where(
                    self._syn_last[syn] == _TRACE_NEVER,
                    0.0,
                    self._syn_trace[syn]
                    * np.exp(-np.maximum(dt_syn, 0.0) / self._tau_plus),
                )
                self._syn_trace[syn] = syn_trace + 1.0
                self._syn_last[syn] = t
        return row

    def on_firing(self, fired: np.ndarray, t: int, learn: bool = False) -> None:
        """Apply the LTP half of STDP after the neurons in ``fired`` spike.

        For every plastic (excitatory) synapse targeting a neuron that spiked
        at time ``t``, the weight is potentiated in proportion to how much
        pre-synaptic activity arrived at that synapse recently (its trace).
        Post-synaptic traces are then refreshed for the next LTD window.

        Args:
            fired: indices of the neurons that spiked at time ``t``.
            t: the current simulated time step (milliseconds).
            learn: apply plasticity. When False (e.g. after ``freeze_at_ms``)
                no weights change and post-synaptic traces are not refreshed.
        """
        if fired.size == 0 or not self._learning_enabled or not learn:
            return
        t = int(t)
        in_syn = self._in_syn
        in_offsets = self._in_offsets

        # Gather the (disjoint) incoming excitatory synapse blocks of every
        # neuron that fired, then potentiate them in one vectorized pass.
        blocks = [
            in_syn[int(in_offsets[post]) : int(in_offsets[post + 1])] for post in fired
        ]
        syn_ids = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        if syn_ids.size:
            dt = t - self._syn_last[syn_ids]
            trace = np.where(
                self._syn_last[syn_ids] == _TRACE_NEVER,
                0.0,
                self._syn_trace[syn_ids]
                * np.exp(-np.maximum(dt, 0.0) / self._tau_plus),
            )
            w = self.weights[syn_ids] + self._a_plus * trace
            np.clip(w, 0.0, 1.0, out=w)
            self.weights[syn_ids] = w

        # Refresh the post-synaptic trace of every neuron that fired.
        dt_post = t - self._post_last[fired]
        post_trace = np.where(
            self._post_last[fired] == _TRACE_NEVER,
            0.0,
            self._post_trace[fired]
            * np.exp(-np.maximum(dt_post, 0.0) / self._tau_minus),
        )
        self._post_trace[fired] = post_trace + 1.0
        self._post_last[fired] = t
