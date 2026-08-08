"""Fast one-shot episodic memory and offline replay.

The episodic store records experienced spike patterns verbatim (hippocampus
analog, abstracted); the replay engine turns stored episodes into timetables
of re-activations used by the sleep-phase scheduler to consolidate them into
the recurrent weights. See ``docs/ARCHITECTURE.md`` for the honest framing.
"""

from bio_network.memory.episodic import EpisodicStore
from bio_network.memory.replay import ReplayEngine

__all__ = ["EpisodicStore", "ReplayEngine"]
