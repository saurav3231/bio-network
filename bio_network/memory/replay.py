"""Offline replay of stored episodes during a simulated sleep phase.

An episode can be *replayed* back into the network as a timetable of
re-activations: each stored (neuron, time) spike becomes a strong brief
current pulse at the scheduled time. Learning (STDP) stays active during
replay, so replayed experience is consolidated into the recurrent weights.

Biology note: hippocampal replay during sleep is typically time-compressed --
CA3 replays the waking sequence much faster than it was experienced (Wilson &
McNaughton 1994; Diba & Buzsaki 2007; Rasch & Born 2013). We therefore support
a ``compression`` factor: replay at ``1/compression`` of the original
inter-spike intervals. The default is ``compression=1.0`` (simplest), and the
tests exercise ``compression=5.0`` explicitly.
"""

from __future__ import annotations

import numpy as np

from bio_network.memory.episodic import EpisodicStore


class ReplayEngine:
    """Schedules stored episodes as replay timetables.

    A *timetable* is an ``(n_events, 2)`` array whose rows are
    ``(time_ms, neuron_id)``: at ``time_ms`` the neuron ``neuron_id`` should
    receive a strong current pulse. Replay does not inject weights directly;
    it only re-activates neurons, and learning acts on whatever fires.
    """

    def __init__(self, store: EpisodicStore) -> None:
        self.store = store

    def schedule(
        self,
        episode_id: int,
        start_ms: float = 0.0,
        compression: float = 1.0,
    ) -> np.ndarray:
        """Build the ``(time_ms, neuron_id)`` timetable for one replay.

        The episode's relative spike times ``rel_times_ms`` are shifted by
        ``start_ms`` and divided by ``compression``; the neuron identities are
        preserved verbatim.

        Args:
            episode_id: episode to replay.
            start_ms: absolute time (ms) at which the episode begins.
            compression: replay speed-up factor (inter-spike intervals become
                ``1/compression`` of the original). Must be > 0.

        Returns:
            Array of shape ``(n_spikes, 2)`` with columns ``time_ms`` and
            ``neuron_id``, sorted by time.
        """
        if compression <= 0:
            raise ValueError("compression must be > 0")
        ep = self.store.get(episode_id)
        times = (
            start_ms + np.asarray(ep["rel_times_ms"], dtype=np.float64) / compression
        )
        neurons = np.asarray(ep["neuron_ids"], dtype=np.int64)
        order = np.argsort(times, kind="stable")
        return np.column_stack((times[order], neurons[order]))

    def plan(
        self,
        episode_ids: list[int],
        start_ms: float = 0.0,
        gap_ms: float = 0.0,
        compression: float = 1.0,
    ) -> list[np.ndarray]:
        """Repeatedly replay several episodes into a sleep-time plan.

        Each episode in ``episode_ids`` is scheduled once, each separated by
        ``gap_ms`` from the previous one. The result is the list of timetables
        the scheduler folds into the sleep phase.

        Args:
            episode_ids: episodes to replay, in order.
            start_ms: absolute start of the first replay.
            gap_ms: delay between the *end* of one replay and the *start* of
                the next.
            compression: replay speed-up applied to every episode.

        Returns:
            List of timetables, one per episode, spanning roughly
            ``start_ms + sum(duration_i + gap)``.
        """
        tables: list[np.ndarray] = []
        cursor = start_ms
        for eid in episode_ids:
            table = self.schedule(eid, start_ms=float(cursor), compression=compression)
            tables.append(table)
            if table.size:
                cursor = float(table[:, 0].max()) + gap_ms
            else:
                cursor += gap_ms
        return tables
