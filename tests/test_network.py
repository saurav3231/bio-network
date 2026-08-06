"""Integration tests for the full M1 network."""

from __future__ import annotations

import time

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses import RandomSynapses


def make_default_network():
    return IzhikevichPopulation(seed=42), RandomSynapses(seed=42)


def test_run_completes_under_60s() -> None:
    population, synapses = make_default_network()
    start = time.perf_counter()
    recording = simulate(population, synapses, T_ms=1000, seed=42)
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0
    assert recording.n_neurons == 1000


def test_network_is_active_but_not_saturated() -> None:
    population, synapses = make_default_network()
    recording = simulate(population, synapses, T_ms=1000, seed=42)

    assert recording.times_ms.size > 0
    mean_rate = float(recording.mean_rates_hz().mean())
    assert 0.5 < mean_rate < 100.0


def test_reproducible_with_same_seed() -> None:
    population1, synapses1 = make_default_network()
    recording1 = simulate(population1, synapses1, T_ms=500, seed=42)

    population2, synapses2 = make_default_network()
    recording2 = simulate(population2, synapses2, T_ms=500, seed=42)

    np.testing.assert_array_equal(recording1.times_ms, recording2.times_ms)
    np.testing.assert_array_equal(recording1.indices, recording2.indices)
