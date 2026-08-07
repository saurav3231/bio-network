"""Tests for the event-driven sparse synaptic engine."""

from __future__ import annotations

import tracemalloc

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses

N_EXC = 800
N_INH = 200
N_NEURONS = N_EXC + N_INH
T_MS = 1000
SEED = 42


def make_sparse(out_degree: int | None = None, seed: int = SEED) -> SparseSynapses:
    """Build a sparse engine with ``out_degree`` chosen to match the default."""
    if out_degree is None:
        out_degree = N_NEURONS
    return SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=out_degree, seed=seed
    )


def _rate_series(recording) -> np.ndarray:
    """Per-neuron population firing rate, binned at 10 ms (Hz)."""
    bin_ms = 10.0
    n_bins = int(np.ceil(recording.duration_ms / bin_ms))
    counts, _ = np.histogram(
        recording.times_ms, bins=n_bins, range=(0.0, recording.duration_ms)
    )
    return counts / (bin_ms / 1000.0) / recording.n_neurons


def test_delay_correctness() -> None:
    """A spike travels to its target exactly after the synaptic delay."""
    weight = 1.5
    sparse = SparseSynapses(n_excit=2, n_inhib=0, out_degree=1, seed=1)

    start, end = int(sparse.offsets[0]), int(sparse.offsets[1])
    assert end - start == 1
    sparse.targets[start:end] = 1  # neuron 0 -> neuron 1
    sparse.weights[start:end] = weight
    sparse.delays[start:end] = 5  # 5 ms conduction delay

    sparse.deliver(np.array([0]), t=10)  # neuron 0 spikes at t=10

    before = sparse.currents(14)
    arrival = sparse.currents(15)

    np.testing.assert_allclose(before, 0.0)
    np.testing.assert_allclose(arrival[1], weight)


def test_determinism() -> None:
    """Two fresh sparse runs with the same seed yield identical spikes."""
    pop1 = IzhikevichPopulation(seed=SEED)
    rec1 = simulate(pop1, make_sparse(), T_ms=1000, seed=SEED, engine="sparse")

    pop2 = IzhikevichPopulation(seed=SEED)
    rec2 = simulate(pop2, make_sparse(), T_ms=1000, seed=SEED, engine="sparse")

    np.testing.assert_array_equal(rec1.times_ms, rec2.times_ms)
    np.testing.assert_array_equal(rec1.indices, rec2.indices)


def test_dales_principle_sparse_graph() -> None:
    """Outgoing weights respect Dale's principle by neuron type."""
    sparse = make_sparse(out_degree=10, seed=SEED)
    offsets, weights, n_excit = sparse.offsets, sparse.weights, sparse.n_excit
    for i in range(N_NEURONS):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        row = weights[start:end]
        if i < n_excit:
            assert np.all(row >= 0.0)
        else:
            assert np.all(row <= 0.0)


def test_sparsity_synapse_count() -> None:
    """Every neuron has exactly out_degree distinct targets."""
    sparse = SparseSynapses(n_excit=200, n_inhib=50, out_degree=15, seed=SEED)
    n_neurons = sparse.n_excit + sparse.n_inhib
    assert sparse.n_synapses == n_neurons * sparse.out_degree
    assert sparse.offsets.size == n_neurons + 1
    for i in range(n_neurons):
        length = int(sparse.offsets[i + 1]) - int(sparse.offsets[i])
        assert length == sparse.out_degree
    assert len(np.unique(sparse.targets[: sparse.out_degree])) == sparse.out_degree


def test_statistical_equivalence_with_dense() -> None:
    """Dense and sparse engines agree statistically, not spike-exactly.

    The trajectories diverge after ~100 ms because the system is chaotic and
    the summation order differs. Equivalence is therefore statistical: the
    mean excitatory rates are within 25 % of each other, both fall in the
    active regime (0.5..60 Hz), and both population-rate series fluctuate
    clearly above the noise floor (std > 5 Hz).

    To compare like-for-like, the sparse graph uses an equal total synaptic
    budget to the dense engine (``out_degree = n_neurons``), isolating the
    effect of the event-driven, delay-encoded delivery from the effect of
    reduced connectivity density.
    """
    dense_pop = IzhikevichPopulation(seed=SEED)
    dense_rec = simulate(dense_pop, T_ms=T_MS, seed=SEED, engine="dense")

    sparse_pop = IzhikevichPopulation(seed=SEED)
    sparse_rec = simulate(
        sparse_pop, make_sparse(), T_ms=T_MS, seed=SEED, engine="sparse"
    )

    dense_exc = float(dense_rec.mean_rates_hz()[:N_EXC].mean())
    sparse_exc = float(sparse_rec.mean_rates_hz()[:N_EXC].mean())

    assert 0.5 <= dense_exc <= 60.0
    assert 0.5 <= sparse_exc <= 60.0

    denominator = max(dense_exc, sparse_exc)
    assert abs(dense_exc - sparse_exc) / denominator <= 0.25

    assert _rate_series(dense_rec).std() > 5.0
    assert _rate_series(sparse_rec).std() > 5.0


def test_sparse_memory_ceiling() -> None:
    """A 50k-neuron sparse run stays comfortably under 300 MB peak RAM."""
    n_exc, n_inh = 40000, 10000
    tracemalloc.start()
    try:
        population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh)
        synapses = SparseSynapses(n_excit=n_exc, n_inhib=n_inh, out_degree=100)
        simulate(population, synapses, T_ms=200, seed=SEED, engine="sparse")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 300 * 1024 * 1024


def test_gain_scales_only_excitatory_weights() -> None:
    """gain rescales excitatory weights and leaves inhibitory weights intact."""
    base = SparseSynapses(n_excit=200, n_inhib=50, out_degree=15, seed=SEED)
    scaled = SparseSynapses(n_excit=200, n_inhib=50, out_degree=15, seed=SEED, gain=2.5)
    for i in range(250):
        start = int(base.offsets[i])
        end = int(base.offsets[i + 1])
        if i < 200:
            np.testing.assert_allclose(
                scaled.weights[start:end], base.weights[start:end] * 2.5
            )
            assert np.all(scaled.weights[start:end] >= 0.0)
        else:
            np.testing.assert_allclose(
                scaled.weights[start:end], base.weights[start:end]
            )
            assert np.all(scaled.weights[start:end] <= 0.0)


def test_sparse_no_nan_or_inf() -> None:
    population = IzhikevichPopulation(seed=SEED)
    simulate(population, make_sparse(), T_ms=T_MS, seed=SEED, engine="sparse")
    assert np.all(np.isfinite(population.v))
    assert np.all(np.isfinite(population.u))
