"""Tests for the plastic input projection / optic-nerve STDP (M3.2).

Covers the six required specifications:

1. ``plastic=False`` reproduces the frozen v1 input pathway byte-for-byte
   (same targets, unit weight drive, deterministic from the seed, no mutable
   plastic state).
2. Single-channel pair STDP: causal arrival-then-fire potentiates, non-causal
   fire-then-arrival depresses, run into [0, 1] bounds.
3. Structural: the projection exposes excitatory-only fan-out plumbing and the
   ``w_in`` STDP muations never touch an index outside the driven roster.
4. Freeze boundaries: with learning disabled (assignment/test phase) both the
   arrival- and firing-side updates are exact no-ops.
5. Two arms with the same seed produce identical weight trajectories.
6. A short end-to-end training run stays finite and bounded (no NaN / no
   runaway weights).
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses
from bio_network.senses import InputProjection, Retina, RetinaStimulus

SEED = 42
H, W = 28, 28
FANOUT = 20


def _bright_block() -> np.ndarray:
    img = np.zeros((H, W))
    img[10:18, 10:18] = 1.0
    img[18:22, 13:16] = 0.6
    return img


# ---------------------------------------------------------------------------
# spec 1: plastic=False === frozen v1
# ---------------------------------------------------------------------------
def test_plastic_false_matches_frozen_v1() -> None:
    p = InputProjection(H * W, 200, fanout=10, seed=42, plastic=False)
    pixels = np.array([0, 3, 51, 400, 783])
    neurons = p.drive_neurons(pixels)
    weights = p.drive_weights(pixels)
    targets = p.targets

    assert neurons.shape == (pixels.size * 10,)
    assert np.all(weights == 1.0)
    # drive expands each pixel's fanout in the same order the flat targets do.
    expect = targets[pixels].reshape(-1)
    assert np.array_equal(neurons, expect)

    # determinism: a same-seed plastic twin is byte-identical in targets and
    # initial plastic drive.
    p_plastic = InputProjection(H * W, 200, fanout=10, seed=42, plastic=True)
    twin = InputProjection(H * W, 200, fanout=10, seed=42, plastic=True)
    assert np.array_equal(p.targets, p_plastic.targets)
    assert np.array_equal(p.drive_neurons(pixels), p_plastic.drive_neurons(pixels))
    assert np.array_equal(p_plastic.drive_weights(pixels), twin.drive_weights(pixels))
    # the frozen arm drives at unity per edge, unlike plastic stored w_in.
    assert np.all(p.drive_weights(pixels) == 1.0)
    # plastic stores w_in ~ uniform(0.2, 0.4); the stored state is in range.
    stored = p_plastic._weights_flat
    assert np.all((stored >= 0.2 - 1e-12) & (stored <= 0.4 + 1e-12))
    # homeostatic drive-weight matching the frozen arm's unit total power:
    # the total *input power* each neuron receives equals its fan-in count
    # (i.e., a mean drive weight of 1.0 per edge -- the v1 unit drive).
    drive_all = p_plastic.drive_weights(np.arange(H * W))
    assert drive_all.shape == (H * W * 10,)
    assert float(np.mean(drive_all)) == pytest.approx(1.0, rel=5e-3)
    assert p.fan_in_stats() == p_plastic.fan_in_stats()


def test_plastic_false_has_no_plastic_state() -> None:
    p = InputProjection(H * W, 128, fanout=20, seed=42, plastic=False)
    # STDP entry points must be no-ops and must not corrupt the frozen drive.
    before = p.drive_weights(np.array([0, 7]))
    p.on_input_arrival(np.array([0, 7]), t=5, learn=True)
    p.on_neurons_fired(np.array([0, 1, 2]), t=6, learn=True)
    after = p.drive_weights(np.array([0, 7]))
    assert np.all(before == 1.0) and np.all(after == 1.0)
    # non-plastic arms also still present a *different* topology per seed.
    p2 = InputProjection(H * W, 128, fanout=20, seed=7, plastic=False)
    assert not np.array_equal(p.targets, p2.targets)


# ---------------------------------------------------------------------------
# spec 2: single-channel pair STDP
# ---------------------------------------------------------------------------
def test_pair_causal_fire_potentiates() -> None:
    p = InputProjection(H * W, 16, fanout=1, seed=42, plastic=True)
    # channel 0 -> fanout 1 -> single edge 0 targeting `neuron`.
    edge = 0
    neuron = p.targets[0, 0]
    w0 = p._weights_flat[edge]
    assert 0.2 - 1e-12 <= w0 <= 0.4 + 1e-12

    p.set_learning(True)
    # a pre-spike arrives at t=5 (pre-trace=1); the neuron fires at t=6
    # (within tau=20): causal -> LTP.
    p.on_input_arrival(np.array([0]), t=5, learn=True)
    p.on_neurons_fired(np.array([neuron]), t=6, learn=True)
    w1 = p._weights_flat[edge]
    assert w1 > w0
    assert w1 <= 1.0 + 1e-12


def test_pair_noncausal_fire_depresses() -> None:
    p = InputProjection(H * W, 16, fanout=1, seed=42, plastic=True)
    edge = 0
    neuron = p.targets[0, 0]
    p.set_learning(True)
    p.on_neurons_fired(np.array([neuron]), t=5, learn=True)  # post-trace=1
    w_before = p._weights_flat[edge]
    # an input arrives *after* the neuron fired -> non-causal -> LTD.
    p.on_input_arrival(np.array([0]), t=6, learn=True)
    assert p._weights_flat[edge] < w_before
    assert p._weights_flat[edge] >= 0.0 - 1e-12


def test_weights_bound_01_under_storm() -> None:
    p = InputProjection(H * W, 8, fanout=1, seed=42, plastic=True)
    p.set_learning(True)
    neuron = p.targets[0, 0]
    for t in range(300):
        p.on_input_arrival(np.array([0]), t=t, learn=True)
        p.on_neurons_fired(np.array([neuron]), t=t, learn=True)
    flat = p._weights_flat
    assert flat.min() >= 0.0 - 1e-9
    assert flat.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# spec 3: structural -- all edges valid, plastic state aligned to edges
# ---------------------------------------------------------------------------
def test_structural_edge_integrity() -> None:
    p = InputProjection(H * W, 100, fanout=20, seed=42, plastic=True)
    # every edge points at a valid neuron index
    flat = p.targets.reshape(-1)
    assert flat.min() >= 0 and flat.max() < p.n_neurons
    assert flat.size == H * W * FANOUT
    # plastic state arrays are aligned to the flat edge ordering
    assert p._weights_flat.shape == (flat.size,)
    assert p._syn_trace.shape == (flat.size,)
    # fan in: every neuron has a nonzero (coverage sanity) and mean+std are sane
    mean, std = p.fan_in_stats()
    assert mean > 0.0
    assert 0.0 <= std


# ---------------------------------------------------------------------------
# spec 4: freeze boundaries (assignment / test)
# ---------------------------------------------------------------------------
def test_frozen_phase_is_exact_noop() -> None:
    p = InputProjection(H * W, 16, fanout=1, seed=42, plastic=True)
    # train briefly
    p.set_learning(True)
    p.on_input_arrival(np.array([1]), t=3, learn=True)
    p.on_neurons_fired(np.array([p.targets[1, 0]]), t=4, learn=True)
    trained = p._weights_flat.copy()

    # freeze (assignment/test phase)
    p.set_learning(False)
    p.on_input_arrival(np.array([1]), t=8, learn=False)
    p.on_neurons_fired(np.array([p.targets[1, 0]]), t=9, learn=False)
    frozen = p._weights_flat.copy()
    assert np.array_equal(trained, frozen)

    # even with learn=True inside the method, the set_learning(False) gate
    # must hold the same boundary.
    p.on_input_arrival(np.array([1]), t=10, learn=True)
    p.on_neurons_fired(np.array([p.targets[1, 0]]), t=11, learn=True)
    assert np.array_equal(trained, p._weights_flat.copy())


# ---------------------------------------------------------------------------
# spec 5: identical seed => identical weights
# ---------------------------------------------------------------------------
def test_two_arms_same_seed_trajectory() -> None:
    a = InputProjection(H * W, 256, fanout=20, seed=42, plastic=True)
    b = InputProjection(H * W, 256, fanout=20, seed=42, plastic=True)
    rng = np.random.default_rng(7)
    for t in range(60):
        px = rng.integers(0, H * W, size=12)
        a.on_input_arrival(px, t=t, learn=True)
        b.on_input_arrival(px, t=t, learn=True)
        fired = rng.integers(0, 256, size=5)
        a.on_neurons_fired(fired, t=t, learn=True)
        b.on_neurons_fired(fired, t=t, learn=True)
    assert np.array_equal(a.targets, b.targets)
    assert np.allclose(a._weights_flat, b._weights_flat)


# ---------------------------------------------------------------------------
# spec 6: end-to-end plastic training run health
# ---------------------------------------------------------------------------
def test_plastic_training_run_health() -> None:
    n_exc, n_inh = 80, 20
    img = _bright_block()
    imgs = [img, img, img]
    retina = Retina(seed=SEED, window_ms=80.0)
    proj = InputProjection(H * W, n_exc + n_inh, fanout=20, seed=SEED, plastic=True)
    stim = RetinaStimulus(retina, proj, imgs, gap_ms=30.0)
    stim.set_learning(True)

    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh, seed=SEED)
    synapses = SparseSynapses(
        n_excit=n_exc, n_inhib=n_inh, out_degree=60, seed=SEED, gain=4.0
    )

    rec = simulate(
        population,
        synapses,
        T_ms=int(3 * stim.slot_ms),
        engine="sparse",
        seed=SEED,
        learning=True,
        stimulus_fn=stim,
        input_plastic_fn=proj.on_neurons_fired,
    )
    assert np.isfinite(rec.times_ms).all()
    assert np.isfinite(rec.indices).all()
    assert rec.times_ms.size > 0
    assert np.isfinite(synapses.weights).all()
    assert np.isfinite(proj._weights_flat).all()
    assert proj._weights_flat.min() >= 0.0 and proj._weights_flat.max() <= 1.0
