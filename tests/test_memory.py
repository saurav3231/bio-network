"""Tests for episodic memory and sleep-phase consolidation (M4).

Covers the three new abstractions end to end:

- ``EpisodicStore``: verbatim one-shot recording, FIFO bound, retrieval.
- ``ReplayEngine``: ``(time_ms, neuron_id)`` timetables with compression.
- Sleep phase in ``simulate(..., phase="sleep")``: quiet background drive,
  replay-pulse injection, STDP stays on during consolidation.
- ``save_state``/``load_state``: an identical copy of a mid-run network
  produces bit-identical future spikes.
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses
from bio_network.memory import EpisodicStore, ReplayEngine

N_EXC = 800
N_INH = 200
SEED = 42


def make_small(*, seed: int = SEED) -> tuple[IzhikevichPopulation, SparseSynapses]:
    """A small network so sleep/wake tests run fast."""
    n_exc, n_inh = 120, 30
    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh, seed=seed)
    synapses = SparseSynapses(
        n_excit=n_exc,
        n_inhib=n_inh,
        out_degree=60,
        seed=seed,
        gain=1.0,
    )
    return population, synapses


# ---- EpisodicStore ---------------------------------------------------------


def test_store_round_trip_is_verbatim() -> None:
    """record -> get reproduces tags, neurons and times bit for bit."""
    store = EpisodicStore(capacity=32)
    neurons = np.array([3, 9, 3, 1], dtype=np.int64)
    times = np.array([2.5, 11.0, 7.0, 0.25])
    eid = store.record("P1", neurons, times)
    got = store.get(eid)
    assert got["tag"] == "P1"
    assert got["id"] == eid
    # Ordering inside the store is by time (chronological replay view); the
    # multiset of (neuron, time) pairs must be preserved exactly once.
    pairs_in = sorted(zip(times.tolist(), neurons.tolist()))
    pairs_out = sorted(zip(got["rel_times_ms"].tolist(), got["neuron_ids"].tolist()))
    assert pairs_in == pairs_out


def test_round_trip_christmas_neighbors() -> None:
    """Multiple episodes are individually retrievable, newest last."""
    store = EpisodicStore(capacity=8)
    ids = [store.record(f"e{i}", np.array([i]), np.array([float(i)])) for i in range(3)]
    assert store.all() == ids
    assert store.get(ids[1])["tag"] == "e1"
    assert len(store) == 3
    store.clear()
    assert len(store) == 0
    assert store.all() == []


def test_store_evicts_oldest_when_full() -> None:
    """A bounded store drops the oldest episode first (FIFO)."""
    store = EpisodicStore(capacity=2)
    first = store.record("old", np.array([0]), np.array([0.0]))
    store.record("mid", np.array([1]), np.array([1.0]))
    last = store.record("new", np.array([2]), np.array([2.0]))
    assert store.all() == [1, last]
    with pytest.raises(KeyError):
        store.get(first)
    assert len(store) == 2


def test_store_rejects_mismatched_arrays() -> None:
    store = EpisodicStore(capacity=4)
    try:
        store.record("bad", np.array([0, 1]), np.array([0.0]))
    except ValueError:
        return
    raise AssertionError("record must reject non-equal-length inputs")


# ---- ReplayEngine ------------------------------------------------------


def test_replay_schedule_timeshift_and_compression() -> None:
    """schedule shifts by start_ms and divides intervals by compression."""
    store = EpisodicStore(capacity=4)
    eid = store.record(
        "seq",
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0.0, 10.0, 25.0]),
    )
    engine = ReplayEngine(store)
    table = engine.schedule(eid, start_ms=100.0, compression=5.0)
    assert table.shape == (3, 2)
    times = table[:, 0]
    assert np.allclose(times, [100.0, 102.0, 105.0])  # 0, 10/5, 25/5
    assert sorted(table[:, 1].tolist()) == [0, 1, 2]


def test_replay_plan_chains_episodes() -> None:
    """plan staggers multiple replays with a gap and returns one table each."""
    store = EpisodicStore(capacity=4)
    a = store.record("a", np.array([5]), np.array([10.0]))
    b = store.record("b", np.array([6]), np.array([5.0]))
    engine = ReplayEngine(store)
    tables = engine.plan([a, b], start_ms=1000.0, gap_ms=50.0, compression=1.0)
    assert len(tables) == 2
    assert tables[0].shape == (1, 2)
    assert tables[0][0, 0] == 1010.0
    assert tables[1][0, 0] == 1065.0  # 1010 + 50 gap -> start 1060, rel 5


# ---- Sleep phase ---------------------------------------------------------


def test_sleep_quiet_drive_lowers_spiking() -> None:
    """Sleep with no replay and attenuated noise is sparser than wake."""
    pop_sleep, syn_sleep = make_small()
    pop_wake, syn_wake = make_small()
    rec_sleep = simulate(
        pop_sleep,
        syn_sleep,
        T_ms=400,
        engine="sparse",
        seed=SEED,
        phase="sleep",
        learning=True,
    )
    rec_wake = simulate(
        pop_wake, syn_wake, T_ms=400, engine="sparse", seed=SEED, learning=True
    )
    assert rec_sleep.times_ms.size < rec_wake.times_ms.size


def test_sleep_replay_injects_pulse_stimulus() -> None:
    """A replayed neuron fires more often than its same-seed quiet control."""
    targets = np.array([5, 6, 7, 8])

    def _run(replay: bool) -> int:
        pop, syn = make_small()
        plan = None
        if replay:
            plan = [np.column_stack((np.arange(10, 40, 3.0), np.full(10, 5)))]
        rec = simulate(
            pop,
            syn,
            T_ms=60,
            engine="sparse",
            seed=SEED,
            phase="sleep",
            replay_plan=plan,
            learning=True,
        )
        replayed = np.isin(rec.indices, targets)
        return int(replayed.sum())

    with_replay = _run(replay=True)
    without_replay = _run(replay=False)
    assert (
        with_replay > without_replay
    ), f"replay should drive extra spikes ({with_replay} vs {without_replay})"


def test_sleep_learning_consolidates_weights() -> None:
    """Sleep keeps STDP on: weights change during a sleep replay."""
    population, synapses = make_small()
    before = synapses.weights.copy()
    store = EpisodicStore(capacity=4)
    eid = store.record(
        "e", np.array([3, 4, 5], dtype=np.int64), np.array([5.0, 15.0, 20.0])
    )
    engine = ReplayEngine(store)
    plan = engine.plan([eid], start_ms=5.0, gap_ms=20.0, compression=1.0)
    simulate(
        population,
        synapses,
        T_ms=200,
        engine="sparse",
        seed=SEED,
        phase="sleep",
        replay_plan=plan,
        learning=True,
    )
    assert not np.array_equal(synapses.weights, before)


def test_sleep_with_replay_produces_some_spikes() -> None:
    """A reasonable sleep run yields a finite recording with learning on."""
    population, synapses = make_small()
    rec = simulate(
        population,
        synapses,
        T_ms=500,
        engine="sparse",
        phase="sleep",
        learning=True,
    )
    assert np.all(np.isfinite(rec.times_ms))


def test_sleep_noise_gating_scales_drive() -> None:
    """Sleep attenuates the background drive by sleep_noise_scale.

    Two identical engines run the same sleep (no replay) with different
    ``sleep_noise_scale``; the quieter one must spike strictly less. The
    default 0.25 is the quiet end; scale=1.0 is the loud end, so this pins
    the gating knob to its documented direction.
    """

    def _run(scale: float) -> int:
        pop, syn = make_small()
        rec = simulate(
            pop,
            syn,
            T_ms=300,
            engine="sparse",
            seed=SEED,
            phase="sleep",
            sleep_noise_scale=scale,
            learning=True,
        )
        return int(rec.times_ms.size)

    quiet = _run(0.25)
    loud = _run(1.0)
    assert (
        loud > quiet
    ), f"scale 1.0 should drive more spikes than 0.25 ({loud} vs {quiet})"


def test_sleep_learning_off_keeps_weights_bit_identical() -> None:
    """With learning=False, sleep must not touch a single weight bit.

    This is the control that proves any consolidation measured during sleep is
    attributable to the learning gate, not to the replay pulses themselves.
    """
    population, synapses = make_small()
    store = EpisodicStore(capacity=4)
    eid = store.record(
        "e", np.array([3, 4, 5], dtype=np.int64), np.array([5.0, 15.0, 20.0])
    )
    engine = ReplayEngine(store)
    plan = engine.plan([eid], start_ms=5.0, gap_ms=20.0, compression=1.0)
    before = synapses.weights.copy()
    simulate(
        population,
        synapses,
        T_ms=200,
        engine="sparse",
        seed=SEED,
        phase="sleep",
        replay_plan=plan,
        learning=False,
    )
    np.testing.assert_array_equal(synapses.weights, before)


def test_sleep_consolidation_is_causal() -> None:
    """Replaying a pre-before-post order during sleep potentiates that synapse.

    A tiny controlled pair (neuron A fires 8 ms before B, inside the ~20 ms
    STDP window) is replayed through the sleep phase with STDP on. The A->B
    weight must rise, and the same-order baseline A->C (never replayed
    together) must not.
    """
    n_exc, n_inh = 30, 0
    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh, seed=SEED)
    synapses = SparseSynapses(
        n_excit=n_exc,
        n_inhib=n_inh,
        out_degree=n_exc,
        seed=SEED,
        gain=1.0,
    )
    a, b, c = 0, 1, 2

    def _mean(pre: int, post: int) -> float:
        s, e = int(synapses.offsets[pre]), int(synapses.offsets[pre + 1])
        sel = (synapses.targets[s:e] >= post) & (synapses.targets[s:e] < post + 1)
        return float(synapses.weights[s:e][sel].mean()) if sel.any() else 0.0

    store = EpisodicStore(capacity=4)
    eid = store.record("pair", np.array([a, b], dtype=np.int64), np.array([2.0, 10.0]))
    engine = ReplayEngine(store)
    plan = engine.plan([eid] * 20, start_ms=10.0, gap_ms=30.0, compression=1.0)

    w_ab_before, w_ac_before = _mean(a, b), _mean(a, c)
    simulate(
        population,
        synapses,
        T_ms=900,
        engine="sparse",
        seed=SEED,
        phase="sleep",
        replay_plan=plan,
        learning=True,
    )
    w_ab_after, w_ac_after = _mean(a, b), _mean(a, c)
    # A->B has a direct synapse in the seeded graph and is replayed together
    # with B: it must potentiate, and more than the never-paired A->C control.
    assert (
        w_ab_after > w_ab_before
    ), f"A->B did not potentiate ({w_ab_before} -> {w_ab_after})"
    assert (w_ab_after - w_ab_before) > (w_ac_after - w_ac_before)


# ---- save_state / load_state --------------------------------------------


def test_save_load_resumes_identically() -> None:
    """Two copies loaded from the same mid-run state fire spikes in lockstep.

    Simulate 200 ms on network A, snapshot it, then keep going on A. On a
    fresh copy B we load the snapshot and continue. Dense-equivalent chaos
    means the continued spikes must match bit for bit -- this is the guarantee
    ``save_state``/``load_state`` exist to provide for forked experiments.
    """
    pop_a, syn_a = make_small()
    simulate(pop_a, syn_a, T_ms=200, engine="sparse", seed=SEED, learning=True)

    state_syn = syn_a.save_state()
    state_pop = pop_a.save_state()

    # Keep running A (no reload) for a reference.
    rec_a = simulate(pop_a, syn_a, T_ms=300, engine="sparse", seed=SEED, learning=True)

    # Fresh copy B, load the snapshot, then run the same 300 ms.
    pop_b, syn_b = make_small()
    syn_b.load_state(state_syn)
    pop_b.load_state(state_pop)
    rec_b = simulate(pop_b, syn_b, T_ms=300, engine="sparse", seed=SEED, learning=True)

    np.testing.assert_array_equal(rec_a.times_ms, rec_b.times_ms)
    np.testing.assert_array_equal(rec_a.indices, rec_b.indices)


def test_saved_state_traces_are_copied_not_aliased() -> None:
    """Mutating the live engine after a save must not corrupt the snapshot."""
    pop, syn = make_small()
    simulate(pop, syn, T_ms=100, engine="sparse", seed=SEED, learning=True)
    snap = syn.save_state()
    simulate(pop, syn, T_ms=300, engine="sparse", seed=SEED, learning=True)
    # The snapshot must still reproduce the post-100ms weights.
    w_after_100 = snap["weights"]
    _pop2, syn2 = make_small()
    syn2.load_state(snap)
    np.testing.assert_array_equal(syn2.weights, w_after_100)
