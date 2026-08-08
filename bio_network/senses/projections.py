"""Fixed or plastic input projection from retina channels onto neurons.

The retina encodes images into ``(t_ms, pixel_index)`` timetables. This module
maps each of the retina's ``n_pixels`` channels onto a fixed, random subset of
the recurrent population ("fan-out"): every channel drives ``fanout`` distinct
neurons, chosen once with the constructor seed.

M3 (v1) froze this pathway: the projection was **never touched by STDP**, so
the only plasticity lived in the recurrent ``SparseSynapses`` engine and a
neuron's receptive field was a static random mix of pixels. M3.2 makes the
sensory cable itself learnable: with ``plastic=True`` the projection carries
per-channel input synapses ``w_in in [0, 1]`` (initialized uniform 0.2-0.4)
that STDP potentiates/depresses on the exact same arrival-time causality and
trace bookkeeping as the recurrent rule (tau 20 ms, A+ 0.10, A- 0.12; Song,
Miller & Abbott bounds). This mirrors the retinogeniculate/visual-cortex
plasticity of the developmental critical period and is the input-layer
plasticity Diehl & Cook (2015) identify as the source of digit-shaped
receptive fields.

With ``plastic=False`` the projection is byte-for-byte identical to the M3 v1
class: no weight array, constant pulse drive, deterministic from the seed.

M3.3 adds the second biological missing regulator observed in M3.2: **synaptic
scaling** (Turrigiano et al. 1998). The M3.2 arm demonstrated that pure STDP
without a thermostat starves the input pathway: LTD (A- 0.12, applied on every
arrival) outruns LTP (A+ 0.10, which needs a neuron to actually spike), so
``w_in`` drains toward zero and the population collapses to ~0.1 Hz. Synaptic
scaling is the real neuron's answer -- a slow, multiplicative, per-neuron
renormalization of the *post-synaptic* side: after each training image window
each neuron's input-weight column is rescaled so its total is pinned to
``C = n_in_per_neuron * 0.30``. The total input power a neuron can inject is
now constant (no death spiral), while the *distribution* of that power across
its winning/experimental inputs stays free: pixels that reliably drive causal
spikes keep and grow their share; everything else is squeezed. This is the
classic Turrigiano homeostasis: preserving total synaptic strength per neuron
while letting correlation structure be re-allocated. It is gated to the
training phase only (assignment/test leaves the weights untouched), and it is
disabled with ``plastic=False`` (the frozen v1 pathway).
"""

from __future__ import annotations

import numpy as np

_TRACE_NEVER = -1.0e18

# Synaptic-scaling constant (Turrigiano et al. 1998 style): per-neuron input
# weight sum is pinned to ``n_in_per_neuron * SCALING_TARGET_PER_EDGE``. With
# STDP initials uniform in [0.2, 0.4] (mean 0.30) this is exactly the arm's
# initial total power, so the homeostasis is *shape free*: only relative
# sharing of that fixed power can change.
SCALING_TARGET_PER_EDGE = 0.30


class InputProjection:
    """A channel -> neuron fan-out map, optionally with plastic input synapses.

    ``targets[p, k]`` is the ``k``-th neuron targeted (driven) by pixel
    channel ``p``. The topology is fixed at construction and identical from the
    seed; only the per-edge ``weights`` (when ``plastic=True``) change over
    time.

    Args:
        n_pixels: number of retina channels (e.g. 784 for a 28x28).
        n_neurons: total recurrent population size (excitatory + inhibitory).
        n_excitatory: size of the excitatory sub-population (default: same as
            ``n_neurons``). Only used when ``excitatory_only`` is set.
        fanout: how many distinct random neurons each pixel drives.
        seed: random seed; a given seed always draws the same targets.
        plastic: when False (default) the projection is the frozen v1 pathway
            (constant unit drive, no weight state). When True it carries
            learnable ``w_in`` input synapses driven by STDP during training.
        competition_gain: drive sharpening exponent on ``w_in`` (1.0 identity).
        synaptic_scaling: M3.3 per-window Turrigiano-style renormalization.
        excitatory_only: structural constraint (M3.3) -- draw input targets
            exclusively from the excitatory population so no input synapses
            land on inhibitory neurons.
    """

    def __init__(
        self,
        n_pixels: int,
        n_neurons: int,
        n_excitatory: int | None = None,
        fanout: int = 20,
        seed: int = 42,
        plastic: bool = False,
        competition_gain: float = 1.0,
        synaptic_scaling: bool = False,
        excitatory_only: bool = False,
    ) -> None:
        self.n_pixels = int(n_pixels)
        self.n_neurons = int(n_neurons)
        if n_excitatory is None:
            n_excitatory = self.n_neurons
        self.n_excitatory = int(n_excitatory)
        self.fanout = int(fanout)
        self.seed = seed
        self.plastic = bool(plastic)
        self.competition_gain = float(competition_gain)
        self.synaptic_scaling = bool(synaptic_scaling)
        self.excitatory_only = bool(excitatory_only)

        if self.n_excitatory > self.n_neurons:
            raise ValueError("n_excitatory cannot exceed n_neurons")
        pool = self.n_excitatory if self.excitatory_only else self.n_neurons
        if self.fanout > pool:
            raise ValueError("fanout cannot exceed the target pool size")
        if self.excitatory_only and self.n_excitatory <= 0:
            raise ValueError("excitatory_only=True requires n_excitatory >= 1")

        rng = np.random.default_rng(seed)
        # Each pixel channel targets a distinct set of `fanout` neurons.
        self.targets = np.stack(
            [
                rng.choice(pool, size=self.fanout, replace=False)
                for _ in range(self.n_pixels)
            ]
        ).astype(np.int64)

        if self.plastic:
            self._allocate_plastic_state(rng)

    # -- plastic state (only when plastic=True) ----------------------------
    def _allocate_plastic_state(self, rng: np.random.Generator) -> None:
        """Allocate the input-synapse weights and STDP trace/ledger state."""
        self.weights = rng.uniform(0.2, 0.4, size=(self.n_pixels, self.fanout))
        # Flat, aligned with targets.reshape(-1) via edge = pixel*fanout + k.
        self._weights_flat = self.weights.reshape(-1)

        # Per-edge pre-trace and per-neuron post-trace, mirroring the recurrent
        # engine's exact lazy-exponential bookkeeping.
        self._syn_trace = np.zeros(self.n_pixels * self.fanout, dtype=np.float64)
        self._syn_last = np.full(
            self.n_pixels * self.fanout, _TRACE_NEVER, dtype=np.float64
        )
        self._post_trace = np.zeros(self.n_neurons, dtype=np.float64)
        self._post_last = np.full(self.n_neurons, _TRACE_NEVER, dtype=np.float64)
        self._learning = False

        # Reverse (incoming) adjacency: which input edges target each neuron,
        # used to potentiate a neuron's input synapses when it fires (LTP).
        incoming: list[list[int]] = [[] for _ in range(self.n_neurons)]
        flat = self.targets.reshape(-1)
        for e in range(flat.size):
            incoming[int(flat[e])].append(e)
        lengths = np.fromiter((len(b) for b in incoming), dtype=np.int64)
        self._in_offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        flat_edges = np.empty(int(self._in_offsets[-1]), dtype=np.int64)
        for j, bucket in enumerate(incoming):
            start = int(self._in_offsets[j])
            flat_edges[start : start + len(bucket)] = bucket
        self._in_edges = flat_edges
        self._refresh_homeo()

    def set_learning(self, active: bool) -> None:
        """Enable/disable STDP on the input synapses (training phase only)."""
        if self.plastic:
            self._learning = bool(active)

    def _homeo_scale(self) -> np.ndarray:
        """Per-neuron multiplicative scale keeping each neuron's total input power.

        Returns a cached vector over neurons ``target / sum_of_its_w_in`` where
        the target is the number of incoming edges (mean 1.0 per edge). The
        scale is refreshed after every STDP update, so the plastic arm's drive
        always matches the frozen v1 arm's unit-weight power and STDP only
        re-allocates which inputs a neuron attends to.
        """
        if not hasattr(self, "_homeo_cache"):
            self._refresh_homeo()
        return self._homeo_cache

    def _incoming_sums(self) -> np.ndarray:
        """Per-neuron sum of incoming weight — robust to empty buckets."""
        weights = self._weights_flat[self._in_edges]
        neuron_ids = np.repeat(
            np.arange(self.n_neurons, dtype=np.int64),
            np.diff(self._in_offsets).astype(np.int64),
        )
        return np.bincount(neuron_ids, weights=weights, minlength=self.n_neurons)

    def _refresh_homeo(self) -> None:
        """Recompute and cache the per-neuron homeostatic scale."""
        n_neurons = self.n_neurons
        per_edges = np.diff(self._in_offsets)
        scale = np.ones(n_neurons, dtype=np.float64)
        if self._in_edges.size:
            sums = self._incoming_sums()
            target = per_edges.astype(np.float64)
            nz = sums > 0
            scale[nz] = target[nz] / sums[nz]
        self._homeo_cache = scale

    # -- drive --------------------------------------------------------------
    def edge_ids(self, pixels: np.ndarray) -> np.ndarray:
        """Flat input-synapse ids for a per-pixel event set.

        Returns the flat edge index ``pixel * fanout + k`` for every
        (pixel, target) edge, aligned position-for-position with
        :meth:`drive_neurons`.
        """
        px = np.asarray(pixels, dtype=np.int64)
        if px.size == 0:
            return np.empty(0, dtype=np.int64)
        if px.max() >= self.n_pixels:
            raise ValueError(f"pixel {px.max()} >= n_pixels {self.n_pixels}")
        edges = px[:, None] * self.fanout + np.arange(self.fanout)[None, :]
        return edges.reshape(-1)

    def drive_neurons(self, pixels: np.ndarray) -> np.ndarray:
        """Expand a per-pixel event set into the neurons those pixels drive.

        Args:
            pixels: array of flat pixel indices (from a retina timetable).

        Returns:
            Flat array of neuron indices, one per (pixel, fanout) edge.
        """
        px = np.asarray(pixels, dtype=np.int64)
        if px.size == 0:
            return np.empty(0, dtype=np.int64)
        if px.max() >= self.n_pixels:
            raise ValueError(f"pixel {px.max()} >= n_pixels {self.n_pixels}")
        out = self.targets[px].reshape(-1)
        return np.asarray(out, dtype=np.int64)

    def drive_weights(self, pixels: np.ndarray) -> np.ndarray:
        """Per-edge input weights for a pixel event set (plastic mode).

        Returns the same shape as :meth:`drive_neurons`: for a frozen
        (``plastic=False``) projection this is a vector of ones (the v1 unit
        drive); for a plastic projection it is the current ``w_in`` raised to
        ``competition_gain`` (a gain > 1 sharpens the strongest edges and
        weakens the weakest, promoting a winner-take-most readout; gain 1.0 is
        the identity and keeps two-arm runs identical).
        """
        if not self.plastic:
            px = np.asarray(pixels, dtype=np.int64)
            n_edges = px.size * self.fanout
            return np.ones(n_edges, dtype=np.float64)
        w = self._weights_flat[self.edge_ids(pixels)]
        if self.competition_gain != 1.0:
            w = np.power(w, self.competition_gain)
        # Homeostatic rescale: each neuron's total input power equals its fan-in
        # count (mean 1.0 / edge), so the drive seen by the population is the
        # same as the frozen v1 arm's unit weights at every step.
        w = w * self._homeo_scale()[self.drive_neurons(pixels)]
        return w

    def fan_in_stats(self) -> tuple[float, float]:
        """Return (mean, std) number of input edges converging per neuron."""
        counts = np.bincount(self.targets.reshape(-1), minlength=self.n_neurons).astype(
            float
        )
        return float(counts.mean()), float(counts.std())

    # -- synaptic scaling (Turrigiano et al. 1998) -------------------------
    def synaptic_scale(self) -> None:
        """M3.3 homeostatic renormalization of each neuron's input weights.

        After every training image window, each neuron's incoming ``w_in``
        weights are rescaled multiplicatively so their sum equals
        ``C = n_in_per_neuron * 0.30`` (``SCALING_TARGET_PER_EDGE``). Because
        the STDP initials are uniform in [0.2, 0.4] (mean 0.30) this ``C`` is
        exactly the neuron's initial total power: the homeostasis is neutral on
        day one and afterwards only *reallocates* correlation structure within
        a constant power budget. Winning pixels that drive causal spikes grow
        their share; losing pixels are squeezed. Gated to the training phase
        (``_learning`` must be on) and to ``plastic`` / ``synaptic_scaling``
        arms only; assignment/test never rescale.
        """
        if not self.plastic or not self.synaptic_scaling:
            return
        if not self._learning:
            return
        n_neurons = self.n_neurons
        if self._in_edges.size == 0:
            return

        per_neuron = np.diff(self._in_offsets).astype(np.float64)
        sums = self._incoming_sums()
        target = per_neuron * SCALING_TARGET_PER_EDGE
        # Fixed-point multiplicative renormalization that hits the per-neuron
        # sum target C while keeping every weight within [0, 1]: repeatedly
        # scale, then clamp, then rescale (a clipped edge drops out of the
        # budget and the remainder is re-shared).
        for _ in range(64):
            scale = np.ones(n_neurons, dtype=np.float64)
            nz = sums > 0
            scale[nz] = target[nz] / sums[nz]
            edges_expanded = np.repeat(
                np.arange(n_neurons, dtype=np.int64), per_neuron.astype(np.int64)
            )
            self._weights_flat[self._in_edges] *= scale[edges_expanded]
            self._weights_flat[self._in_edges] = np.clip(
                self._weights_flat[self._in_edges], 0.0, 1.0
            )
            sums = self._incoming_sums()
            if np.allclose(sums, target, atol=1e-9):
                break
        self._refresh_homeo()

    # -- input STDP ---------------------------------------------------------
    def on_input_arrival(self, pixels: np.ndarray, t: int, learn: bool) -> None:
        """Arrival-side STDP: a pixel spike landing on its neurons at time ``t``.

        Every arriving input synapse is depressed by the post-synaptic trace of
        the neuron it drives (a neuron that fired recently weakens inputs that
        arrive late -- non-causal) and then its pre-trace is refreshed. This is
        the exact counterpart of the recurrent engine's arrival bookkeeping.

        Args:
            pixels: flat pixel indices whose spikes arrive at time ``t``.
            t: the arrival millisecond.
            learn: whether plasticity is active (False during frozen phases).
        """
        if not self.plastic or not learn or not self._learning:
            return
        pixels = np.asarray(pixels, dtype=np.int64)
        if pixels.size == 0:
            return
        edges = self.edge_ids(pixels)
        post = self.drive_neurons(pixels)

        dt_post = t - self._post_last[post]
        traces = np.where(
            self._post_last[post] == _TRACE_NEVER,
            0.0,
            self._post_trace[post] * np.exp(-np.maximum(dt_post, 0.0) / 20.0),
        )
        w = self._weights_flat[edges] - 0.12 * traces
        np.clip(w, 0.0, 1.0, out=w)
        self._weights_flat[edges] = w

        dt_syn = t - self._syn_last[edges]
        syn_trace = np.where(
            self._syn_last[edges] == _TRACE_NEVER,
            0.0,
            self._syn_trace[edges] * np.exp(-np.maximum(dt_syn, 0.0) / 20.0),
        )
        self._syn_trace[edges] = syn_trace + 1.0
        self._syn_last[edges] = t
        self._refresh_homeo()

    def on_neurons_fired(self, fired: np.ndarray, t: int, learn: bool) -> None:
        """Firing-side STDP: neurons that spiked potentiate their input synapses.

        For every fired neuron, all of its incoming input synapses are
        potentiated by the pre-synaptic trace (causal: input arrived shortly
        before the spike), then the neuron's post-trace is refreshed.

        Args:
            fired: indices of the neurons that spiked at time ``t``.
            t: the spike time (milliseconds).
            learn: whether plasticity is active.
        """
        if not self.plastic or not learn or not self._learning:
            return
        fired = np.asarray(fired, dtype=np.int64)
        if fired.size == 0:
            return
        blocks = [
            self._in_edges[
                int(self._in_offsets[post]) : int(self._in_offsets[post + 1])
            ]
            for post in fired
        ]
        edges = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
        if edges.size:
            dt_syn = t - self._syn_last[edges]
            syn_trace = np.where(
                self._syn_last[edges] == _TRACE_NEVER,
                0.0,
                self._syn_trace[edges] * np.exp(-np.maximum(dt_syn, 0.0) / 20.0),
            )
            w = self._weights_flat[edges] + 0.10 * syn_trace
            np.clip(w, 0.0, 1.0, out=w)
            self._weights_flat[edges] = w

        dt_post = t - self._post_last[fired]
        post_trace = np.where(
            self._post_last[fired] == _TRACE_NEVER,
            0.0,
            self._post_trace[fired] * np.exp(-np.maximum(dt_post, 0.0) / 20.0),
        )
        self._post_trace[fired] = post_trace + 1.0
        self._post_last[fired] = t
        self._refresh_homeo()
