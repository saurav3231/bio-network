"""Tests for the M3.4 "morning coffee" bounded parameter tuning.

Specifications covered:

1. **L2 ambient drive guards**: the constant tonic current reaches every
   excitatory neuron, and ONLY inside an image window (gap stays silent) and
   ONLY during the training phase (frozen assignment/test gets nothing).
2. **L2 units**: ``ambient_drive`` is an "mV-equivalent of the Izhikevich
   step" -- the engine integrates ``v += 0.5*((0.04 v^2 + 5 v + 140 - u) + I)``
   per half-step, and the stimulus injects ``+ambient_drive`` on the I
   channel (so the membrane receives a persistent ``+0.5 * ambient`` mV push
   per 0.5 ms half-step).
3. **L1 scaling target C**: ``synaptic_scale`` clamps each neuron's incoming
   sum to ``n_in_per_neuron * C``; C is configurable (0.30 / 0.60 / 1.00) and
   honored while the default 0.30 stays the exact M3.3 behavior.
4. **Determinism**: every lever config binds a fixed seed to an identical
   spike train.
5. **Integration health**: a short ARM-C-style run (scaling C + ambient) stays
   finite, fires the network, and reproduces the same config's spike count.
"""

from __future__ import annotations

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses
from bio_network.senses import InputProjection, Retina, RetinaStimulus

SEED = 42
H, W = 28, 28
FANOUT = 20
N_EXC, N_INH = 80, 20
AMBIENT = 2.0
SCALE = 0.60


def _bright_block() -> np.ndarray:
    img = np.zeros((H, W))
    img[10:18, 10:18] = 1.0
    img[18:22, 13:16] = 0.6
    return img


def _per_neuron_sums(p: InputProjection) -> np.ndarray:
    neuron_ids = np.repeat(
        np.arange(p.n_neurons, dtype=np.int64),
        np.diff(p._in_offsets).astype(np.int64),
    )
    return np.bincount(
        neuron_ids, weights=p._weights_flat[p._in_edges], minlength=p.n_neurons
    )


def _make_stim(
    ambient: float = AMBIENT, scale: float = SCALE, n_imgs: int = 4
) -> RetinaStimulus:
    retina = Retina(seed=SEED, window_ms=80.0)
    proj = InputProjection(
        H * W,
        N_EXC + N_INH,
        n_excitatory=N_EXC,
        fanout=FANOUT,
        seed=SEED,
        plastic=True,
        synaptic_scaling=True,
        scaling_target=scale,
        excitatory_only=True,
    )
    imgs = [_bright_block() for _ in range(n_imgs)]
    return RetinaStimulus(retina, proj, imgs, gap_ms=30.0, ambient_drive=ambient)


# ---------------------------------------------------------------------------
# spec 1 + 2: ambient drive window/phase guards + units
# ---------------------------------------------------------------------------
def test_ambient_reaches_all_excitatory_only() -> None:
    stim = _make_stim()
    stim.set_learning(True)
    # a ms well inside the first window, before the per-pixel pulses start
    cur = stim(10, N_EXC + N_INH)
    assert cur.shape == (N_EXC + N_INH,)
    assert np.all(cur[:N_EXC] >= AMBIENT)
    assert np.allclose(cur[:N_EXC], AMBIENT)
    assert np.allclose(cur[N_EXC:], 0.0)


def test_ambient_units_are_izhikevich_step_push() -> None:
    """ambient_drive lands on the I channel; each half-step sees +0.5*a mV."""
    stim = _make_stim(ambient=1.0)
    stim.set_learning(True)
    cur = stim(10, N_EXC + N_INH)
    assert np.allclose(cur[:N_EXC], 1.0)
    assert stim.ambient_drive == 1.0


def test_ambient_off_in_gap() -> None:
    stim = _make_stim()
    stim.set_learning(True)
    # 80 ms window + 30 ms gap => t=100 is in the gap of slot 0
    cur = stim(100, N_EXC + N_INH)
    assert np.allclose(cur, 0.0)


def test_ambient_off_when_frozen() -> None:
    stim = _make_stim()
    stim.set_learning(False)
    cur = stim(10, N_EXC + N_INH)
    assert np.allclose(cur, 0.0)
    # learning back on but still inside a gap: still silent
    stim.set_learning(True)
    cur_gap = stim(100, N_EXC + N_INH)
    assert np.allclose(cur_gap, 0.0)


def test_ambient_zero_default_is_m33_behavior() -> None:
    stim = _make_stim(ambient=0.0)
    stim.set_learning(True)
    cur = stim(10, N_EXC + N_INH)
    assert np.allclose(cur, 0.0)


# ---------------------------------------------------------------------------
# spec 3: L1 scaling target C honored
# ---------------------------------------------------------------------------
def test_scaling_target_default_keeps_m33_target() -> None:
    p = InputProjection(
        H * W,
        N_EXC + N_INH,
        n_excitatory=N_EXC,
        fanout=FANOUT,
        seed=SEED,
        plastic=True,
        synaptic_scaling=True,
        excitatory_only=True,
    )
    p.set_learning(True)
    p.synaptic_scale()
    sums = _per_neuron_sums(p)
    target = np.diff(p._in_offsets).astype(np.float64) * 0.30
    assert np.allclose(sums, target, atol=1e-6)
    assert p.scaling_target == 0.30


def test_scaling_target_C_configurable() -> None:
    means = {}
    for c in (0.30, 0.60, 1.00):
        p = InputProjection(
            H * W,
            N_EXC + N_INH,
            n_excitatory=N_EXC,
            fanout=FANOUT,
            seed=SEED,
            plastic=True,
            synaptic_scaling=True,
            scaling_target=c,
            excitatory_only=True,
        )
        p.set_learning(True)
        p.synaptic_scale()
        sums = _per_neuron_sums(p)
        per = np.diff(p._in_offsets).astype(np.float64)
        # weights are clipped to [0,1], so the renormalization can never exceed
        # the per-neuron target (hard clip may leave C=1.0 marginally below).
        assert np.all(sums <= per * c + 1e-9)
        if c == 0.30:
            assert np.allclose(sums, per * 0.30, atol=1e-6)
        # weights still bounded after the higher-C renormalization
        assert p._weights_flat.min() >= 0.0 and p._weights_flat.max() <= 1.0 + 1e-9
        means[c] = float(sums.mean())
    # higher C re-pins the pathway at higher drive (monotone on fresh weights)
    assert means[0.60] > means[0.30]
    assert means[1.00] > means[0.60]


# ---------------------------------------------------------------------------
# spec 4: determinism per lever config
# ---------------------------------------------------------------------------
def test_lever_config_deterministic_in_seed() -> None:
    def run() -> int:
        retina = Retina(seed=SEED, window_ms=80.0)
        proj = InputProjection(
            H * W,
            N_EXC + N_INH,
            n_excitatory=N_EXC,
            fanout=FANOUT,
            seed=SEED,
            plastic=True,
            synaptic_scaling=True,
            scaling_target=SCALE,
            excitatory_only=True,
        )
        stim = RetinaStimulus(
            retina, proj, [_bright_block()] * 4, gap_ms=30.0, ambient_drive=AMBIENT
        )
        pop = IzhikevichPopulation(
            n_excitatory=N_EXC, n_inhibitory=N_INH, seed=SEED, adaptive_thresholds=True
        )
        syn = SparseSynapses(
            n_excit=N_EXC, n_inhib=N_INH, out_degree=60, seed=SEED, gain=4.0
        )
        rec = simulate(
            pop,
            syn,
            T_ms=int(4 * stim.slot_ms),
            engine="sparse",
            seed=SEED,
            learning=True,
            stimulus_fn=stim,
            input_plastic_fn=proj.on_neurons_fired,
        )
        return int(rec.times_ms.size)

    assert run() == run()


# ---------------------------------------------------------------------------
# spec 5: integration health
# ---------------------------------------------------------------------------
def test_arm_c_short_integration_finite_and_deterministic() -> None:
    """Short ARM-C-style run: finite, fires, and reproduces spike count."""

    def run() -> tuple[int, bool]:
        retina = Retina(seed=SEED, window_ms=80.0)
        proj = InputProjection(
            H * W,
            N_EXC + N_INH,
            n_excitatory=N_EXC,
            fanout=FANOUT,
            seed=SEED,
            plastic=True,
            synaptic_scaling=True,
            scaling_target=SCALE,
            excitatory_only=True,
        )
        stim = RetinaStimulus(
            retina,
            proj,
            [_bright_block() for _ in range(8)],
            gap_ms=30.0,
            ambient_drive=AMBIENT,
        )
        pop = IzhikevichPopulation(
            n_excitatory=N_EXC, n_inhibitory=N_INH, seed=SEED, adaptive_thresholds=True
        )
        syn = SparseSynapses(
            n_excit=N_EXC, n_inhib=N_INH, out_degree=60, seed=SEED, gain=4.0
        )
        rec = simulate(
            pop,
            syn,
            T_ms=int(8 * stim.slot_ms),
            engine="sparse",
            seed=SEED,
            learning=True,
            stimulus_fn=stim,
            input_plastic_fn=proj.on_neurons_fired,
        )
        finite = bool(
            np.isfinite(rec.times_ms).all()
            and np.isfinite(rec.indices).all()
            and np.isfinite(syn.weights).all()
            and np.isfinite(proj._weights_flat).all()
        )
        return int(rec.times_ms.size), finite

    spikes1, finite1 = _arm_c_spikes()
    spikes2, finite2 = _arm_c_spikes()
    assert finite1 is True and finite2 is True
    assert spikes1 > 0
    assert spikes1 == spikes2


def _arm_c_spikes() -> tuple[int, bool]:
    retina = Retina(seed=SEED, window_ms=80.0)
    proj = InputProjection(
        H * W,
        N_EXC + N_INH,
        n_excitatory=N_EXC,
        fanout=FANOUT,
        seed=SEED,
        plastic=True,
        synaptic_scaling=True,
        scaling_target=SCALE,
        excitatory_only=True,
    )
    stim = RetinaStimulus(
        retina,
        proj,
        [_bright_block() for _ in range(8)],
        gap_ms=30.0,
        ambient_drive=AMBIENT,
    )
    pop = IzhikevichPopulation(
        n_excitatory=N_EXC, n_inhibitory=N_INH, seed=SEED, adaptive_thresholds=True
    )
    syn = SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=60, seed=SEED, gain=4.0
    )
    rec = simulate(
        pop,
        syn,
        T_ms=int(8 * stim.slot_ms),
        engine="sparse",
        seed=SEED,
        learning=True,
        stimulus_fn=stim,
        input_plastic_fn=proj.on_neurons_fired,
    )
    finite = bool(
        np.isfinite(rec.times_ms).all()
        and np.isfinite(rec.indices).all()
        and np.isfinite(syn.weights).all()
        and np.isfinite(proj._weights_flat).all()
    )
    return int(rec.times_ms.size), finite
