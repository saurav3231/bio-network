"""Sensory front-ends: turning images into spike trains.

The retina encodes a pixel image into a ``(t_ms, pixel_index)`` spike timetable.
An input projection maps the retina's channels onto the recurrent population via
a fixed, random fan-out (no plasticity on this pathway); the presenter turns the
timetable into a millisecond stimulus for the scheduler. See
``docs/ARCHITECTURE.md`` for the honest framing and the M3 results in
``docs/M3_RESULTS.md``.
"""

from bio_network.senses.projections import InputProjection
from bio_network.senses.readout import LabelsReadout
from bio_network.senses.retina import Retina
from bio_network.senses.stimulus import RetinaStimulus

__all__ = ["InputProjection", "LabelsReadout", "Retina", "RetinaStimulus"]
