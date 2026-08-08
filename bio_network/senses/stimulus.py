"""Stimulus presenter: turn a retina timetable into a scheduler stimulus.

The scheduler's ``simulate(..., stimulus_fn=t)`` expects a callable
``stimulus_fn(t_ms, n_neurons) -> np.ndarray`` of input currents. This module
provides ``RetinaStimulus``, which encodes a batch of images with a ``Retina``,
routes their spikes through an ``InputProjection``, and presents them one after
another: each image fills one ``window_ms + gap_ms`` slot. Spikes inside the
window become brief strong current pulses on the projected neurons; the gap is
silent so the network's spiking between presentations can settle.

When the projection is plastic (M3.2), the pulse amplitude is scaled by each
edge's ``w_in`` and the projection's arrival-side STDP is triggered on the exact
arrival of each pixel spike; the firing-side half is driven by the scheduler's
``input_plastic_fn`` callback (see ``scheduler.simulate``).
"""

from __future__ import annotations

import numpy as np

from bio_network.senses.projections import InputProjection
from bio_network.senses.retina import Retina

_PULSE_AMP = 20.0
_PULSE_WIDTH_MS = 3.0


class RetinaStimulus:
    """Present a list of images through a retina + input projection.

    The object is a drop-in ``stimulus_fn`` for ``scheduler.simulate``. It is
    deterministic end to end: the retina timeline, projection, and pulse
    schedule all derive from fixed seeds.

    The ``projection`` may be plastic (M3.2). In that case the per-edge input
    weight ``w_in`` scales the pulse and, when learning is active (set by
    ``set_learning``), each pixel spike triggers arrival-side STDP on the exact
    millisecond it lands.
    """

    def __init__(
        self,
        retina: Retina,
        projection: InputProjection,
        images: list[np.ndarray],
        gap_ms: float = 150.0,
        pulse_amp: float = _PULSE_AMP,
        pulse_width_ms: float = _PULSE_WIDTH_MS,
        ambient_drive: float = 0.0,
    ) -> None:
        """Wire a retina and projection to a batch of images.

        Args:
            retina: image encoder defining ``window_ms`` and spike timings.
            projection: pixel -> neuron fan-out (fixed or plastic).
            images: list of ``(H, W)`` float arrays (all standardized to the
                retina's ``image_shape``).
            gap_ms: silent interval after each image window, so the network can
                settle between presentations.
            pulse_amp: injected current on each driven neuron per pixel spike.
            pulse_width_ms: duration (ms) of each injected pulse.
            ambient_drive: M3.4 dial -- constant tonic current applied to *every*
                excitatory neuron during each image window, but only while
                learning is on (the ``_learning`` training-phase gate). Measured
                in "mV-equivalent Izhikevich step units": the engine update is
                ``v += 0.5*((0.04v^2+5v+140-u)+I)``, so ``ambient_drive=1`` adds
                a +0.5 mV push per ms, i.e. a modest wake-up blood current
                rather than a spike-scale (20x) pulse. Off for frozen
                assignment/test/probe phases by the training-only gate.
        """
        self.retina = retina
        self.projection = projection
        self.gap_ms = float(gap_ms)
        self.pulse_amp = float(pulse_amp)
        self.pulse_width_ms = float(pulse_width_ms)
        self.ambient_drive = float(ambient_drive)
        self.slot_ms = retina.window_ms + self.gap_ms
        self._learning = False

        self.timetables = [retina.encode(np.asarray(img)) for img in images]

    # -- learning toggle (M3.2) --------------------------------------------
    def set_learning(self, active: bool) -> None:
        """Enable/disable input-projection STDP (training phase only)."""
        self._learning = bool(active)
        self.projection.set_learning(active)

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
        each pixel-spike, where the pixel's fan-out neurons receive a brief
        pulse. Each pulse's amplitude is ``pulse_amp * w_in[edge]`` (with unit
        weights for a frozen projection, so plastic=False reproduces v1).
        Because the drive is purely epoch-localized the network only receives
        input while an image is being presented.
        """
        t0 = int(t)
        slot = int(t0 // self.slot_ms)
        if slot >= len(self.timetables):
            return np.zeros(n_neurons)

        # M3.3: at the start of each new slot the previous training window is
        # over -- apply per-neuron synaptic scaling (Turrigiano) if on.
        rel = t0 - slot * self.slot_ms
        if rel == 0 and slot > 0 and self._learning:
            self.projection.synaptic_scale()
        if not (0 <= rel < self.retina.window_ms):
            return np.zeros((n_neurons,))

        # M3.4: constant tonic "morning-coffee" drive over the image window, on
        # all excitatory neurons, but ONLY during the training phase (learning
        # gate). Frozen assessment (assignment/test, probe) never gets it, so
        # wakefulness is measured on the real image drive alone.
        current = np.zeros((n_neurons,))
        if self._learning and self.ambient_drive:
            n_exc = min(self.projection.n_excitatory, n_neurons)
            current[:n_exc] += self.ambient_drive

        table = self.timetables[slot]
        if table.size == 0:
            return current

        # Spikes active at this millisecond (a spike at `tt` injects a pulse in
        # ``[tt, tt+pulse_width_ms)``).
        tt = table[:, 0]
        active = (tt <= rel) & (tt > rel - self.pulse_width_ms)
        if not active.any():
            return current

        # M3.2: a spike *starts* its pulse at ``rel`` only on the single
        # rounded time of the spike itself; that is the arrival instant for
        # STDP (post-synaptic neuron causality is measured at this ms).
        starts = np.round(tt).astype(np.int64) == rel
        new_pixels = table[active & starts, 1].astype(np.int64)
        self.projection.on_input_arrival(new_pixels, rel, self._learning)

        pixels = table[active, 1].astype(np.int64)
        neurons = self.projection.drive_neurons(pixels)
        weights = self.projection.drive_weights(pixels)

        np.add.at(current, neurons, self.pulse_amp * weights)
        return current
