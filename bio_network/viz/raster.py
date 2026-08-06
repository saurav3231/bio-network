"""Spike raster and population activity visualization.

All functions use the matplotlib Agg-compatible API and never call
``plt.show()``; they return the ``matplotlib.axes.Axes`` they drew on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bio_network.engine.scheduler import SpikeRecording


def _excitatory_mask(recording: SpikeRecording) -> np.ndarray:
    """Return the per-neuron excitatory mask for a recording."""
    if recording.is_excitatory is not None:
        return recording.is_excitatory
    mask = np.zeros(recording.n_neurons, dtype=bool)
    n_excitatory = max(recording.n_neurons - 200, 0)
    mask[:n_excitatory] = True
    return mask


def plot_raster(
    recording: SpikeRecording,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a spike raster: excitatory spikes dark, inhibitory spikes red.

    Args:
        recording: the spike recording to plot.
        ax: optional axes to draw on; a new one is created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    times = recording.times_ms
    indices = recording.indices
    if times.size == 0:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron index")
        return ax

    excitatory = _excitatory_mask(recording)[indices]
    ax.scatter(times[excitatory], indices[excitatory], s=2, c="black", linewidths=0)
    ax.scatter(times[~excitatory], indices[~excitatory], s=2, c="red", linewidths=0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_xlim(times.min(), times.max())
    return ax


def plot_population_rate(
    recording: SpikeRecording,
    bin_ms: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the smoothed population firing rate (Hz per neuron) over time.

    Args:
        recording: the spike recording to plot.
        bin_ms: width of the rate histogram bin in milliseconds.
        ax: the axes to draw on; a new one is created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    if recording.times_ms.size == 0:
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Population rate (Hz)")
        return ax

    duration = recording.duration_ms or (recording.times_ms.max() + 1.0)
    n_bins = max(1, int(np.ceil(duration / bin_ms)))
    counts, _ = np.histogram(recording.times_ms, bins=n_bins, range=(0.0, duration))
    rate_hz = counts / (bin_ms / 1000.0) / recording.n_neurons

    if n_bins >= 5:
        kernel = np.hamming(5)
        kernel = kernel / kernel.sum()
        rate_hz = np.convolve(rate_hz, kernel, mode="same")

    centers = (np.arange(n_bins) + 0.5) * bin_ms
    ax.plot(centers, rate_hz)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Population rate (Hz)")
    return ax


def plot_voltage_trace(
    population_v_history: np.ndarray,
    neuron_index: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the membrane potential trace of a single neuron.

    Args:
        population_v_history: membrane potential history of shape
            ``(n_neurons, n_steps)``, or ``(n_steps,)`` for a single neuron.
        neuron_index: which neuron to plot when the history is 2-D.
        ax: the axes to draw on; a new one is created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    trace = np.asarray(population_v_history)
    if trace.ndim == 2:
        trace = trace[neuron_index]
    ax.plot(trace)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential (mV)")
    return ax
