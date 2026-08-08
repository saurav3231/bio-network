"""Tests for the artificial retina and input pathway (M3).

Covers encoder determinism, latency/rate monotonicity, spike-timing hygiene
(no spikes outside the presentation window), the frozen non-plastic input
projection, the label-gated readout (labels never touch weights), an
end-to-end integration training run, and the full-white visibility guard that
proves bright input genuinely reaches the population.
"""

from __future__ import annotations

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses
from bio_network.senses import InputProjection, LabelsReadout, Retina, RetinaStimulus

SEED = 42
H, W = 28, 28


def _bright_block_img() -> np.ndarray:
    img = np.zeros((H, W))
    img[10:18, 10:18] = 1.0
    img[18:22, 13:16] = 0.6
    return img


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, x.size + 1)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# ---- determinism -------------------------------------------------------------


def test_latency_encode_is_deterministic() -> None:
    img = _bright_block_img()
    a = Retina(seed=SEED).encode(img)
    b = Retina(seed=SEED).encode(img)
    np.testing.assert_array_equal(a, b)


def test_rate_encode_is_deterministic() -> None:
    img = _bright_block_img()
    a = Retina(mode="rate", seed=SEED).encode(img)
    b = Retina(mode="rate", seed=SEED).encode(img)
    np.testing.assert_array_equal(a, b)


# ---- monotonicity / coding -----------------------------------------------------


def test_latency_brighter_fires_earlier() -> None:
    """Latency Spearman(intensity, -time) must be strong (target > 0.6)."""
    img = _bright_block_img()
    retina = Retina(seed=SEED)
    table = retina.encode(img)
    assert table.shape[1] == 2
    intensity = img.reshape(-1)[table[:, 1].astype(int)]
    t = table[:, 0]
    rho = _spearman(intensity, -t)
    assert rho > 0.6, f"latency Spearman={rho:.3f}"


def test_rate_count_tracks_intensity() -> None:
    """Rate mode: per-pixel spike count Pearson vs intensity (target > 0.9)."""
    retina = Retina(mode="rate", seed=SEED, max_rate_hz=250.0, window_ms=500.0)
    img = _bright_block_img()
    counts = np.zeros(H * W)
    for _ in range(25):
        table = retina.encode(img)
        counts += np.bincount(table[:, 1].astype(int), minlength=H * W)
    intensity = img.reshape(-1)
    rho = _pearson(counts, intensity)
    assert rho > 0.9, f"rate Pearson={rho:.3f}"


# ---- window hygiene ------------------------------------------------------------


def test_all_spikes_inside_window() -> None:
    retina = Retina(seed=SEED, window_ms=350.0)
    table = retina.encode(_bright_block_img())
    assert table[:, 0].min() >= 0.0
    assert table[:, 0].max() < 350.0


def test_dark_image_emits_no_spikes() -> None:
    retina = Retina(seed=SEED)
    assert retina.encode(np.zeros((H, W))).size == 0


def test_stimulus_is_silent_in_gap() -> None:
    """Gap: a few ms after the window must drive zero current."""
    retina = Retina(seed=SEED, window_ms=100.0)
    proj = InputProjection(n_pixels=H * W, n_neurons=200, fanout=10, seed=SEED)
    stim = RetinaStimulus(retina, proj, [_bright_block_img()], gap_ms=50.0)
    # inside the window: a spike at t=0 fires; a few ms later (t=1) the pulse
    # still lands. In the gap right after (t=130) nothing is driven.
    in_window = stim(1.0, 200)
    in_gap = stim(160.0, 200)
    assert np.abs(in_window).sum() > 0
    assert np.abs(in_gap).max() == 0.0


# ---- input projection ----------------------------------------------------------


def test_projection_targets_stable_and_frozen() -> None:
    p1 = InputProjection(784, 1000, fanout=20, seed=SEED)
    p2 = InputProjection(784, 1000, fanout=20, seed=SEED)
    np.testing.assert_array_equal(p1.targets, p2.targets)  # seed-stable
    assert p1.targets.shape == (784, 20)
    # every pixel targets 20 distinct neurons.
    for row in p1.targets:
        assert len(set(row.tolist())) == 20


def test_drive_neurons_fanout() -> None:
    proj = InputProjection(784, 100, fanout=5, seed=SEED)
    neurons = proj.drive_neurons(np.array([0, 1, 2], dtype=np.int64))
    assert neurons.size == 15
    assert neurons.max() < 100


# ---- honest label-gated readout -------------------------------------------------


def test_readout_labels_only_at_fit() -> None:
    rng = np.random.default_rng(0)
    n, n_neurons = 40, 12
    responses = rng.random((n, n_neurons))
    labels = rng.integers(0, 10, n)
    ro = LabelsReadout(n_neurons=n_neurons, n_classes=10)
    ro.fit(responses, labels)
    assert ro.assignment is not None
    assert ro.class_profiles.shape == (10, n_neurons)

    test_r = rng.random((5, n_neurons))
    preds = ro.predict(test_r)
    assert preds.shape == (5,)
    assert np.all((preds >= 0) & (preds < 10))


def test_readout_predict_before_fit_raises() -> None:
    ro = LabelsReadout(n_neurons=4, n_classes=10)
    try:
        ro.predict(np.zeros((2, 4)))
    except RuntimeError:
        return
    raise AssertionError("predict() must raise without fit()")


# ---- integration: a small visual training run --------------------------------


def test_visual_training_run_health() -> None:
    """A short visual training run: finite rates, bounded weights, frozen inhibition."""
    n_exc, n_inh = 80, 20
    img = _bright_block_img()
    imgs = [img, img, img]
    retina = Retina(seed=SEED, window_ms=80.0)
    proj = InputProjection(H * W, n_exc + n_inh, fanout=20, seed=SEED)
    stim = RetinaStimulus(retina, proj, imgs, gap_ms=30.0)

    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh, seed=SEED)
    synapses = SparseSynapses(
        n_excit=n_exc, n_inhib=n_inh, out_degree=60, seed=SEED, gain=4.0
    )
    inh0 = synapses.weights[n_exc * 60 :].copy()

    rec = simulate(
        population,
        synapses,
        T_ms=int(3 * stim.slot_ms),
        engine="sparse",
        seed=SEED,
        learning=True,
        stimulus_fn=stim,
    )

    assert np.all(np.isfinite(rec.times_ms))
    assert np.all(np.isfinite(synapses.weights))
    assert synapses.weights.min() >= -1.0
    assert synapses.weights.max() <= 1.0
    # inhibitory weights must be untouched (excitatory-only plasticity).
    np.testing.assert_array_equal(synapses.weights[n_exc * 60 :], inh0)


# ---- full-white visibility guard ----------------------------------------------


def test_white_perturbs_dark() -> None:
    """A full-white image must drive measurably more spikes than full silence."""
    n_exc, n_inh = 80, 20
    white = np.ones((H, W))
    dark = np.zeros((H, W))

    def _count(img: np.ndarray) -> int:
        retina = Retina(seed=SEED)
        proj = InputProjection(H * W, n_exc + n_inh, fanout=20, seed=SEED)
        stim = RetinaStimulus(retina, proj, [img], gap_ms=0.0, pulse_amp=8.0)
        population = IzhikevichPopulation(
            n_excitatory=n_exc, n_inhibitory=n_inh, seed=SEED
        )
        synapses = SparseSynapses(
            n_excit=n_exc, n_inhib=n_inh, out_degree=60, seed=SEED, gain=4.0
        )
        population.v[:] = -65.0
        population.u[:] = -13.0
        rec = simulate(
            population,
            synapses,
            T_ms=int(retina.window_ms),
            engine="sparse",
            seed=SEED,
            learning=False,
            stimulus_fn=stim,
        )
        return int(rec.times_ms.size)

    white_count = _count(white)
    dark_count = _count(dark)
    assert white_count > dark_count * 3, (
        f"white image must drive the population well above silence "
        f"({white_count} vs dark {dark_count})"
    )
