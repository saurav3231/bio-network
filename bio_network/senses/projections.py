"""Fixed, non-plastic input projection from retina channels onto neurons.

The retina encodes images into ``(t_ms, pixel_index)`` timetables. This module
maps each of the retina's ``n_pixels`` channels onto a fixed, random subset of
the recurrent population ("fan-out"): every channel drives ``fanout`` distinct
neurons, chosen once with the constructor seed and, crucially, **never touched
by STDP**. Plasticity lives only in the recurrent ``SparseSynapses`` engine, so
the figure the network learns is a genuine lateral effect of the input wiring,
not a learned input matrix (that is the planned v2 Diehl-Cook upgrade).
"""

from __future__ import annotations

import numpy as np


class InputProjection:
    """A frozen (non-plastic) channel -> neuron fan-out map.

    ``targets[p, k]`` is the ``k``-th neuron targeted (driven) by pixel
    channel ``p``. The map is fixed at construction: identical from the seed,
    and weights/structure never change afterwards.
    """

    def __init__(
        self,
        n_pixels: int,
        n_neurons: int,
        fanout: int = 20,
        seed: int = 42,
    ) -> None:
        """Build the projection.

        Args:
            n_pixels: number of retina channels (e.g. 784 for a 28x28).
            n_neurons: total recurrent population size (excitatory +
                inhibitory).
            fanout: how many distinct random neurons each pixel drives.
            seed: random seed; a given seed always draws the same targets.
        """
        self.n_pixels = int(n_pixels)
        self.n_neurons = int(n_neurons)
        self.fanout = int(fanout)
        self.seed = seed

        if self.fanout > self.n_neurons:
            raise ValueError("fanout cannot exceed n_neurons")

        rng = np.random.default_rng(seed)
        # Each pixel channel targets a distinct set of `fanout` neurons.
        self.targets = np.stack(
            [
                rng.choice(self.n_neurons, size=self.fanout, replace=False)
                for _ in range(self.n_pixels)
            ]
        ).astype(np.int64)

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

    def fan_in_stats(self) -> tuple[float, float]:
        """Return (mean, std) number of input edges converging per neuron."""
        counts = np.bincount(self.targets.reshape(-1), minlength=self.n_neurons).astype(
            float
        )
        return float(counts.mean()), float(counts.std())
