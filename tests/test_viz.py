"""Tests for the visualization helpers using the Agg backend."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bio_network.engine.scheduler import SpikeRecording
from bio_network.viz.raster import plot_population_rate, plot_raster, plot_voltage_trace


def make_recording() -> SpikeRecording:
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0.0, 1000.0, size=500))
    indices = rng.integers(0, 1000, size=500)
    excitatory = np.zeros(1000, dtype=bool)
    excitatory[:800] = True
    return SpikeRecording(
        times_ms=times,
        indices=indices,
        n_neurons=1000,
        duration_ms=1000.0,
        is_excitatory=excitatory,
    )


def test_plot_raster_returns_axes() -> None:
    ax = plot_raster(make_recording())
    assert ax is not None
    plt.close(ax.figure)


def test_plot_population_rate_returns_axes() -> None:
    ax = plot_population_rate(make_recording())
    assert ax is not None
    plt.close(ax.figure)


def test_plot_population_rate_default_args() -> None:
    ax = plot_population_rate(make_recording(), bin_ms=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_voltage_trace_1d() -> None:
    trace = -65.0 + 95.0 * np.sin(np.linspace(0, 30, 1000))
    ax = plot_voltage_trace(trace, neuron_index=0)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_voltage_trace_2d() -> None:
    history = -65.0 * np.ones((1000, 200))
    ax = plot_voltage_trace(history, neuron_index=0)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_raster_empty_recording() -> None:
    recording = SpikeRecording(
        times_ms=np.array([]),
        indices=np.array([]),
        n_neurons=1000,
        duration_ms=1000.0,
    )
    ax = plot_raster(recording)
    assert ax is not None
    plt.close(ax.figure)
