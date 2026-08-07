"""Tests for spike-timing-dependent plasticity (M2).

The tests exercise both the microscopic causality of the rule (a tiny, fully
controlled two-neuron rig) and the mesoscopic behaviour of a running network
(bounds, reproducibility, stability, and the frozen-inhibitory guarantee).
The biomedical rule follows Bi & Poo (1998) with the normalized hard bounds
of Song, Miller & Abbott (2000).
"""

from __future__ import annotations

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses

N_EXC = 800
N_INH = 200
SEED = 42


def make_sparse(*, gain: float = 1.0, seed: int = SEED) -> SparseSynapses:
    return SparseSynapses(
        n_excit=N_EXC,
        n_inhib=N_INH,
        out_degree=N_EXC + N_INH,
        seed=seed,
        gain=gain,
    )


def make_small(*, seed: int = SEED) -> tuple[IzhikevichPopulation, SparseSynapses]:
    """A smaller network for fast network-level learning tests."""
    n_exc, n_inh = 120, 30
    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh, seed=seed)
    synapses = SparseSynapses(
        n_excit=n_exc,
        n_inhib=n_inh,
        out_degree=60,
        seed=seed,
        gain=1.0,
    )
    return population, synapses


def make_rig(weight: float = 0.5, delay: int = 1) -> SparseSynapses:
    """A fully controlled two-neuron rig: neuron 0 excites neuron 1.

    Returns an engine on which the caller drives the exact spike times, so
    every STDP update can be checked by hand.
    """
    rig = SparseSynapses(n_excit=2, n_inhib=0, out_degree=1, seed=1, gain=1.0)
    h0s = int(rig.offsets[0])
    rig.targets[h0s] = 1  # neuron 0 -> neuron 1
    rig.weights[h0s] = weight
    rig.delays[h0s] = delay
    rig.enable_learning()
    return rig


def _lt_syn_before_post(gap: float) -> float:
    """LTP delta: pre fires, then the post fires ``gap`` ms after arrival."""
    rig = make_rig()
    w0 = float(rig.weights[0])
    rig.deliver(np.array([0], dtype=np.int64), t=10, learn=True)  # pre fires
    rig.currents(11, learn=True)  # pre arrives at 11 ms
    rig.on_firing(np.array([1], dtype=np.int64), t=11 + gap, learn=True)  # post
    return float(rig.weights[0]) - w0


def _lt_post_first(gap: float) -> float:
    """LTD delta: post fires, then the pre spike arrives ``gap`` ms later."""
    rig = make_rig()
    w0 = float(rig.weights[0])
    rig.on_firing(np.array([1], dtype=np.int64), t=10, learn=True)  # post fires
    rig.deliver(np.array([0], dtype=np.int64), t=10 + gap - 1, learn=True)
    rig.currents(10 + gap, learn=True)  # pre arrives gap ms after post
    return float(rig.weights[0]) - w0


def test_causal_order_potentiates() -> None:
    """Pre before post within the LTP window strengthens the synapse."""
    assert _lt_syn_before_post(0.0) > 0.0
    assert _lt_syn_before_post(10.0) > 0.0
    assert _lt_syn_before_post(0.0) > _lt_syn_before_post(10.0)


def test_non_causal_order_depresses() -> None:
    """Post before pre within the LTD window weakens the synapse."""
    assert _lt_post_first(0.0) < 0.0
    assert _lt_post_first(10.0) < 0.0
    assert _lt_post_first(0.0) < _lt_post_first(10.0)


def test_causal_signs_are_opposite() -> None:
    """The causal pre-before-post rule potentiates; the reverse depresses."""
    assert _lt_syn_before_post(10.0) > 0.0
    assert _lt_post_first(10.0) < 0.0


def test_gate_dependence() -> None:
    """The 10 ms window produces a bigger update than the 25 ms window."""
    assert abs(_lt_syn_before_post(10.0)) > abs(_lt_syn_before_post(25.0))
    assert abs(_lt_post_first(10.0)) > abs(_lt_post_first(25.0))


def test_weights_stay_bounded() -> None:
    """Excitatory weights stay clamped to [0, 1] (Song-Miller-Abbott bounds)."""
    population, synapses = make_small()
    simulate(population, synapses, T_ms=1500, engine="sparse", learning=True)
    n_exc_syn = synapses.n_excit * synapses.out_degree
    exc = synapses.weights[:n_exc_syn]
    assert np.all(exc >= 0.0)
    assert np.all(exc <= 1.0)


def test_no_learning_is_bit_identical_to_baseline() -> None:
    """With ``learning=False`` the weights never change at all."""
    population, synapses = make_small()
    before = synapses.weights.copy()
    simulate(population, synapses, T_ms=800, engine="sparse", learning=False)
    np.testing.assert_array_equal(synapses.weights, before)


def test_reproducible_learning_is_deterministic() -> None:
    """Two fresh learning runs with the same seed yield identical weights."""
    weights = []
    for _ in range(2):
        population, synapses = make_small()
        simulate(
            population, synapses, T_ms=800, seed=SEED, engine="sparse", learning=True
        )
        weights.append(synapses.weights.copy())
    np.testing.assert_array_equal(weights[0], weights[1])


def test_inhibitory_weights_are_frozen() -> None:
    """STDP only touches excitatory synapses; inhibitory weights are frozen."""
    population, synapses = make_small()
    in_syn = synapses.n_excit * synapses.out_degree
    before = synapses.weights[in_syn:].copy()
    simulate(population, synapses, T_ms=1500, engine="sparse", learning=True)
    np.testing.assert_array_equal(synapses.weights[in_syn:], before)


def test_stable_learning_rate_no_nan() -> None:
    """Per-second firing rates stay in the active regime without NaNs."""
    population, synapses = make_small()
    rec = simulate(population, synapses, T_ms=2000, engine="sparse", learning=True)
    assert np.all(np.isfinite(population.v))
    assert np.all(np.isfinite(population.u))
    rates = rec.mean_rates_hz()[:120]
    assert rates.size > 0
    assert 0.0 <= float(rates.mean()) <= 60.0
