"""One-shot episodic memory store (hippocampus-like).

A verbatim record of experienced spike patterns. Recording is immediate and
exact: the set of neurons that spiked and their firing times relative to the
onset of the episode are stored as-is and returned as-is.

This is an *abstraction* of hippocampal fast storage, not a neural model of
it. Real hippocampus (CA3) stores new memories quickly and replays them later
during sleep; here we model the "quick verbatim capture" half with a bounded
buffer. See ``docs/ARCHITECTURE.md`` for the honest framing.
"""

from __future__ import annotations

from collections import deque

import numpy as np

DEFAULT_CAPACITY = 32


class EpisodicStore:
    """A bounded, one-shot store of verbatim spike episodes.

    Each episode is an ordered set of spikes: ``neuron_ids`` (which neurons
    fired) with ``rel_times_ms`` (milliseconds since episode onset). Recording
    is O(n) and exact -- ``record -> get`` is a bit-for-bit round trip.

    Capacity is bounded (FIFO). When the store is full the *oldest* episode is
    evicted first. Rationale: the episodic buffer models ongoing wake
    experience; preserving the most recent episodes mirrors the natural loss
    of older, unreplayed episodes, and bounds memory usage. Replays that
    matter are scheduled from episodes still present.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._episodes: dict[int, dict[str, object]] = {}
        self._order: deque[int] = deque()
        self._next_id = 0

    def record(
        self,
        tag: str,
        neuron_ids: np.ndarray,
        rel_times_ms: np.ndarray,
    ) -> int:
        """Record an episode verbatim and return its id.

        Args:
            tag: a human-readable label (e.g. ``"P1"``).
            neuron_ids: neuron indices that fired during the episode.
            rel_times_ms: firing times relative to episode onset (ms), same
                length as ``neuron_ids``.

        Returns:
            The integer id of the stored episode.
        """
        neuron_ids = np.asarray(neuron_ids, dtype=np.int64)
        rel_times_ms = np.asarray(rel_times_ms, dtype=np.float64)
        if neuron_ids.ndim != 1 or rel_times_ms.ndim != 1:
            raise ValueError("neuron_ids and rel_times_ms must be 1-D arrays")
        if neuron_ids.size != rel_times_ms.size:
            raise ValueError("neuron_ids and rel_times_ms must have equal length")

        # Store a canonical, sorted-by-time view so replay is chronological.
        order = np.argsort(rel_times_ms, kind="stable")
        eid = self._next_id
        self._next_id += 1
        self._episodes[eid] = {
            "id": eid,
            "tag": str(tag),
            "neuron_ids": neuron_ids[order].copy(),
            "rel_times_ms": rel_times_ms[order].copy(),
        }
        self._order.append(eid)

        if len(self._order) > self.capacity:
            evicted = self._order.popleft()
            del self._episodes[evicted]
        return eid

    def get(self, episode_id: int) -> dict[str, object]:
        """Return a copy of ``episode_id`` (verbatim round trip)."""
        ep = self._episodes[episode_id]
        return {
            "id": ep["id"],
            "tag": ep["tag"],
            "neuron_ids": ep["neuron_ids"].copy(),
            "rel_times_ms": ep["rel_times_ms"].copy(),
        }

    def all(self) -> list[int]:
        """Return the ids of all stored episodes, newest last (insertion order)."""
        return list(self._order)

    def clear(self) -> None:
        """Remove every stored episode."""
        self._episodes.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._order)
