"""Tests for the Izhikevich spiking neuron population."""

from __future__ import annotations

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation


def run_constant_input(
    n_excitatory: int,
    n_inhibitory: int,
    I_value: float,
    T_ms: int = 1000,
    seed: int = 42,
):
    """Run a population with constant input and return spikes, and max v."""
    population = IzhikevichPopulation(
        n_excitatory=n_excitatory,
        n_inhibitory=n_inhibitory,
        seed=seed,
    )
    spike_times: list[int] = []
    max_voltage = -np.inf
    for t in range(T_ms):
        fired = population.step(np.full(population.v.size, I_value))
        if fired.size:
            spike_times.extend([t] * int(fired.size))
        max_voltage = max(max_voltage, float(population.v.max()))
    return population, np.asarray(spike_times), max_voltage


def test_single_rs_neuron_fires_regularly() -> None:
    population, spike_times, _ = run_constant_input(1, 0, 10.0)
    assert population.n_excitatory == 1
    assert population.n_inhibitory == 0
    assert len(spike_times) >= 3

    isis = np.diff(spike_times)[1:]  # intervals after the first spike
    assert isis.size >= 2
    cv = float(np.std(isis) / np.mean(isis))
    assert cv < 0.3


def test_reset_correctness_max_voltage() -> None:
    _, _, max_voltage = run_constant_input(1, 0, 10.0)
    assert max_voltage < 40.0


def test_no_nan_or_inf() -> None:
    population, _, _ = run_constant_input(1, 0, 10.0)
    assert np.all(np.isfinite(population.v))
    assert np.all(np.isfinite(population.u))
    assert not np.isnan(population.v).any()
    assert not np.isnan(population.u).any()


def test_excitatory_mask() -> None:
    population = IzhikevichPopulation(n_excitatory=3, n_inhibitory=2)
    np.testing.assert_array_equal(
        population.is_excitatory, np.array([True, True, True, False, False])
    )
