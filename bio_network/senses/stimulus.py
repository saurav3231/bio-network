"""Stimulus presenter: turn a retina timetable into a scheduler stimulus.

The scheduler's ``simulate(..., stimulus_fn=t)`` expects a callable
``stimulus_fn(t_ms, n_neurons) -> np.ndarray`` of input currents. This module
provides ``RetinaStimulus``, which encodes a batch of images with a ``Retina``,
routes their spikes through an ``InputProjection``, and presents them one after
another: each image fills one ``window_ms + gap_ms`` slot. Spikes inside the
window become brief strong current pulses on the projected neurons; the gap is
silent so the network's spiking between presentations can settle.
"""

from __future__ import annotations

import numpy as np

from bio_network.senses.projections import InputProjection
from bio_network.senses.retina import Retina

_PULSE_AMP = 20.0
_PULSE_WIDTH_MS = 3.0


class RetinaStimulus:
    """Present a list of images through a retina + fixed projection.

    The object is a drop-in ``stimulus_fn`` for ``scheduler.simulate``. It is
    deterministic end to end: the retina timeline, projection, and pulse
    schedule all derive from fixed seeds.
    """

    def __init__(
        self,
        retina: Retina,
        projection: InputProjection,
        images: list[np.ndarray],
        gap_ms: float = 150.0,
        pulse_amp: float = _PULSE_AMP,
        pulse_width_ms: float = _PULSE_WIDTH_MS,
    ) -> None:
        """Wire a retina and projection to a batch of images.

        Args:
            retina: image encoder defining ``window_ms`` and spike timings.
            projection: fixed non-plastic pixel -> neuron fan-out.
            images: list of ``(H, W)`` float arrays (all standardized to the
                retina's ``image_shape``).
            gap_ms: silent interval after each image window, so the network can
                settle between presentations.
            pulse_amp: injected current on each driven neuron per pixel spike.
            pulse_width_ms: duration (ms) of each injected pulse.
        """
        self.retina = retina
        self.projection = projection
        self.gap_ms = float(gap_ms)
        self.pulse_amp = float(pulse_amp)
        self.pulse_width_ms = float(pulse_width_ms)
        self.slot_ms = retina.window_ms + self.gap_ms

        self.timetables = [retina.encode(np.asarray(img)) for img in images]

    # -- scheduling ------------------------------------------------------ --
    def __len__(self) -> int:
        """Number of image presentations this stimulus covers."""
        return len(self.timetables)

    def slot_boundaries(self, slot: int) -> tuple[float, float]:
        """Return the ``(t0, t1)`` wall-clock window of presentation ``slot``."""
        t0 = slot * self.slot_ms
        return t0, t0 + self.retina.window_ms

    def __call__(self, t: float, n_neurons: int) -> np.ndarray:
        """Return the input-current vector at simulated time ``t``.

        The current is all-zero except during the ``pulse_width_ms`` following
        each pixel-spike, where the pixel's fan-out neurons receive a strong
        brief pulse. Because the drive is purely epoch-localized the network
        only receives input while an image is being presented.
        """
        t0 = int(t)
        slot = int(t0 // self.slot_ms)
        if slot >= len(self.timetables):
            return np.zeros(n_neurons)

        rel = t0 - slot * self.slot_ms
        if not (0 <= rel < self.retina.window_ms):
            return np.zeros((n_neurons,))

        table = self.timetables[slot]
        if table.size == 0:
            return np.zeros((n_neurons,))

        # Pulses active at this millisecond: events whose spike time falls in
        # [rel-pulse_width, rel] (a spike at `tt` cuts).
        tt = table[:, 0]
        active = (tt <= rel) & (tt > rel - self.pulse_width_ms)
        if not active.any():
            return np.zeros((n_neurons,))

        pixels = table[active, 1].astype(np.int64)
        neurons = self.projection.drive_neurons(pixels)

        current = np.zeros((n_neurons,))
        np.add.at(current, neurons, self.pulse_amp)
        return current
