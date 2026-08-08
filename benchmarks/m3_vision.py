"""M3 experiments: artificial retina + unsupervised visual feature emergence.

Runs the four experiments specified for Milestone 3 on the sparse, event-driven
engine with STDP (M2) and the new sensory front-end (``bio_network/senses/``):

    E3a  Encoder fidelity. Latency coding must be monotonically related to
         pixel intensity (Spearman(intensity, -latency) > 0.6); rate coding
         must track intensity (Pearson(spikes, intensity) > 0.9).

    E3b  Unsupervised feature emergence. After presenting MNIST digits through
         a fixed, non-plastic input projection while STDP sculpts the
         recurrent weights, we measure "receptive-field image averaging"
         (RIA): for each neuron, the response-weighted mean of the images that
         drive it. We compare RIA selectivity before vs after training. A
         neuron with a dominant class should emerge that was not there at
         initialization.

    E3c  Zero-shot digit classification (the headline). The network is trained
         with STDP and NEVER sees a label. We then freeze plasticity, present
         digits, and attach a labels-only readout (``LabelsReadout``) whose
         assignment uses *training* responses only. We report held-out test
         accuracy and a confusion matrix against baselines: chance (10 %) and
         a numpy kNN (k=3) on raw pixels (~97 %).

    E3d  Stability guards. Population rates within [0.5, 100] Hz, plastic
         weights within [0, 1], inhibitory weights frozen, no NaN/Inf.

Run with:  python benchmarks/m3_vision.py [--train N] [--test N] [--probe N]
Results are written to docs/M3_RESULTS.md and figures to notebooks/output/.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses
from bio_network.senses import InputProjection, LabelsReadout, Retina, RetinaStimulus
from bio_network.senses.mnist import load_mnist, subsample_mnist

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
RESULTS = ROOT / "docs" / "M3_RESULTS.md"

N_EXC, N_INH = 800, 200
N = N_EXC + N_INH
GAIN = 8.0
SEED = 42
OUT_DEGREE = 100
WINDOW_MS = 350.0
GAP_MS = 150.0
FANOUT = 20
N_CLASSES = 10


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def make_network() -> tuple[IzhikevichPopulation, SparseSynapses]:
    population = IzhikevichPopulation(seed=SEED)
    synapses = SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=OUT_DEGREE, seed=SEED, gain=GAIN
    )
    return population, synapses


def normalize(x: np.ndarray) -> np.ndarray:
    """uint8 MNIST -> float [0, 1]."""
    return np.asarray(x, dtype=float) / 255.0


# ---- helpers ----------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation implemented in numpy (no scipy dependency)."""
    order = np.argsort(a, kind="mergesort")
    ranks_a = np.empty_like(order, dtype=float)
    ranks_a[order] = np.arange(1, a.size + 1)
    order = np.argsort(b, kind="mergesort")
    ranks_b = np.empty_like(order, dtype=float)
    ranks_b[order] = np.arange(1, b.size + 1)
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def per_image_response(stim, rec, n_neurons: int) -> np.ndarray:
    """Per-image per-neuron spike counts for a recording over ``stim``.

    Returns ``(n_images, n_neurons)``: each image's contribution is the number
    of spikes its neurons fired inside that image's presentation window.
    """
    n_img = len(stim)
    resp = np.zeros((n_img, n_neurons), dtype=np.int64)
    for i in range(n_img):
        t0, t1 = stim.slot_boundaries(i)
        in_w = (rec.times_ms >= t0) & (rec.times_ms < t1)
        if not in_w.any():
            continue
        resp[i] = np.bincount(rec.indices[in_w], minlength=n_neurons).astype(np.int64)
    return resp


def run_images(
    population,
    synapses,
    stim,
    *,
    learning: bool,
    seed: int = SEED,
) -> tuple[object, np.ndarray]:
    """Simulate ``stim`` and return ``(recording, response_matrix)``."""
    rec = simulate(
        population,
        synapses,
        T_ms=int(len(stim) * stim.slot_ms),
        engine="sparse",
        seed=seed,
        learning=learning,
        stimulus_fn=stim,
    )
    return rec, per_image_response(stim, rec, N)


# ---- Experiment 3a: encoder fidelity ----------------------------------------


def experiment_3a() -> dict:
    """Monotonicity of latency and rate coding on a synthetic gradient."""
    gradient = np.zeros((28, 28), dtype=float)
    for i, frac in enumerate(np.linspace(0.0, 1.0, 28)):
        gradient[i, :] = frac

    # latency
    lat = Retina(mode="latency", seed=SEED, window_ms=WINDOW_MS)
    table = lat.encode(gradient)
    intensity = gradient.reshape(-1)[table[:, 1].astype(int)]
    t = table[:, 0]
    spearman_lat = _spearman(intensity, -t)

    # rate (aggregated over repeats for a stable estimate)
    rate = Retina(mode="rate", seed=SEED, max_rate_hz=250.0, window_ms=500.0)
    counts = np.zeros(28 * 28)
    for _ in range(30):
        tab = rate.encode(gradient)
        counts += np.bincount(tab[:, 1].astype(int), minlength=28 * 28)
    px = np.flatnonzero(gradient.reshape(-1) >= rate.threshold)
    pearson_rate = _pearson(counts[px], gradient.reshape(-1)[px])

    return {
        "latency_spearman": float(spearman_lat),
        "rate_pearson": float(pearson_rate),
        "latency_n_spikes": int(table.shape[0]),
    }


# ---- Experiment 3b: unsupervised feature emergence ---------------------------


def _ria_tiles(imgs: list[np.ndarray], resp: np.ndarray) -> np.ndarray:
    """Response-weighted mean image (RIA) per neuron.

    ``RIA[n] = sum_i (resp[i,n] * img_i) / sum_i resp[i,n]`` (28x28). Neurons
    that never fire are kept as zeros and excluded downstream.
    """
    imgs_arr = np.stack([np.asarray(i) for i in imgs])  # (P, H, W)
    resp = np.asarray(resp, dtype=float)
    denom = resp.sum(axis=0)  # (n_neurons,)
    tiles = np.zeros((resp.shape[1], *imgs_arr.shape[1:]))
    for n in range(resp.shape[1]):
        if denom[n] > 0:
            tiles[n] = np.tensordot(resp[:, n], imgs_arr, axes=(0, 0)) / denom[n]
    return tiles


def _contrast(tiles: np.ndarray) -> np.ndarray:
    """Per-neuron RIA contrast ``(max-min)/(max+min)`` (0 = uniform).

    Note: with a fixed random fan-out every responsive neuron averages a small
    set of images, so this value saturates near 1 for any active neuron; it is
    reported for completeness but is not the discriminator between emergence.
    """
    flat = tiles.reshape(tiles.shape[0], -1)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    active = (mx + mn) > 1e-9
    out = np.zeros(tiles.shape[0])
    out[active] = (mx[active] - mn[active]) / (mx[active] + mn[active])
    return out


def _ria_spread(tiles: np.ndarray, frac: float = 0.5) -> float:
    """Mean fraction of pixels a neuron's RIA concentrates on (tuning width).

    A neuron that has learned a *localized* feature concentrates its RIA on a
    small pixel fraction; a neuron dominated by the global random fan-out has a
    widely spread RIA. Lower mean spread after training = sharper features.
    """
    flat = tiles.reshape(tiles.shape[0], -1)
    counts = 0.0
    n = 0
    for row in flat:
        mx = row.max()
        if mx <= 1e-9:
            continue
        n += 1
        counts += float((row >= frac * mx).mean())
    return counts / n if n else 0.0


def _selective_neurons(
    resp: np.ndarray, labels: np.ndarray, threshold: float = 2.0
) -> int:
    """Neurons whose best-class mean response exceeds the mean by ``threshold``x."""
    resp = np.asarray(resp, dtype=float)
    class_mean = np.zeros((N_CLASSES, resp.shape[1]))
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.any():
            class_mean[c] = resp[mask].mean(axis=0)
    total_mean = resp.mean(axis=0)
    best = class_mean.max(axis=0)
    sel = (best > threshold * np.maximum(total_mean, 1e-9)) & (best > 0)
    return int(sel.sum())


def experiment_3b(
    train_imgs: list[np.ndarray],
    train_labels: np.ndarray,
    probe_imgs: list[np.ndarray],
    probe_labels: np.ndarray,
) -> dict:
    """Measure RIA selectivity before vs after unsupervised STDP training."""
    retina = Retina(seed=SEED, window_ms=WINDOW_MS)
    projection = InputProjection(28 * 28, N, fanout=FANOUT, seed=SEED)

    probe_stim = RetinaStimulus(retina, projection, probe_imgs, gap_ms=GAP_MS)

    population, synapses = make_network()

    # BEFORE: fresh network, frozen responses to the probe set.
    _, resp_before = run_images(population, synapses, probe_stim, learning=False)
    tiles_before = _ria_tiles(probe_imgs, resp_before)
    contrast_before = _contrast(tiles_before)
    spread_before = _ria_spread(tiles_before)
    selective_before = _selective_neurons(resp_before, probe_labels)

    # TRAIN: present the training digits with STDP on (labels untouched).
    train_stim = RetinaStimulus(retina, projection, train_imgs, gap_ms=GAP_MS)
    rec_train, _ = run_images(population, synapses, train_stim, learning=True)

    # AFTER: same probe set, frozen.
    _, resp_after = run_images(population, synapses, probe_stim, learning=False)
    tiles_after = _ria_tiles(probe_imgs, resp_after)
    contrast_after = _contrast(tiles_after)
    spread_after = _ria_spread(tiles_after)
    selective_after = _selective_neurons(resp_after, probe_labels)

    n_active = int((resp_after.sum(axis=0) > 0).sum())

    # Guardrail from E3d: rate and weight health.
    rates = rec_train.mean_rates_hz()
    weight_min = float(synapses.weights.min())
    weight_max = float(synapses.weights.max())

    return {
        "contrast_before": float(contrast_before.mean()),
        "contrast_after": float(contrast_after.mean()),
        "spread_before": spread_before,
        "spread_after": spread_after,
        "selective_before": selective_before,
        "selective_after": selective_after,
        "n_active_after": n_active,
        "train_spikes": int(rec_train.times_ms.size),
        "mean_rate_hz": float(rates.mean()),
        "max_rate_hz": float(rates.max()),
        "weight_min": weight_min,
        "weight_max": weight_max,
        "is_finite": bool(np.isfinite(synapses.weights).all()),
        "tiles_after": tiles_after,
        "resp_after": resp_after,
    }


# ---- Experiment 3c: zero-shot digit classification ---------------------------


def _knn_accuracy(
    train_imgs: np.ndarray,
    train_labels: np.ndarray,
    test_imgs: np.ndarray,
    test_labels: np.ndarray,
    k: int = 3,
) -> float:
    """numpy kNN (Euclidean) on flattened 28x28 raw pixels."""
    train_flat = np.asarray(train_imgs).reshape(len(train_imgs), -1).astype(float)
    test_flat = np.asarray(test_imgs).reshape(len(test_imgs), -1).astype(float)
    correct = 0
    for row, true in zip(test_flat, test_labels):
        dist = ((train_flat - row) ** 2).sum(axis=1)
        nn = np.argsort(dist)[:k]
        votes = train_labels[nn]
        pred = int(np.bincount(votes, minlength=10).argmax())
        correct += pred == int(true)
    return correct / len(test_labels)


def experiment_3c(
    train_imgs: list[np.ndarray],
    train_labels: np.ndarray,
    test_imgs: list[np.ndarray],
    test_labels: np.ndarray,
) -> dict:
    """Frozen readout accuracy, confusion matrix, and baselines."""
    retina = Retina(seed=SEED, window_ms=WINDOW_MS)
    projection = InputProjection(28 * 28, N, fanout=FANOUT, seed=SEED)
    population, synapses = make_network()

    # Training pass: STDP, labels never seen.
    train_stim = RetinaStimulus(retina, projection, train_imgs, gap_ms=GAP_MS)
    run_images(population, synapses, train_stim, learning=True)

    # Assignment pass: frozen responses on the *training* digits.
    _, resp_train = run_images(population, synapses, train_stim, learning=False)

    # Held-out pass: frozen responses on the *test* digits.
    test_stim = RetinaStimulus(retina, projection, test_imgs, gap_ms=GAP_MS)
    _, resp_test = run_images(population, synapses, test_stim, learning=False)

    readout = LabelsReadout(n_neurons=N, n_classes=N_CLASSES)
    readout.fit(resp_train, train_labels)  # labels only used here
    pred = readout.predict(resp_test)

    accuracy = float(np.mean(pred == test_labels))
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for true, predi in zip(test_labels, pred):
        confusion[int(true), int(predi)] += 1

    chance = 1.0 / N_CLASSES
    knn = _knn_accuracy(train_imgs, train_labels, test_imgs, test_labels, k=3)

    # Which neurons does the readout actually rely on?
    used = readout.assignment
    class_counts = np.bincount(used, minlength=N_CLASSES) if used is not None else None

    return {
        "accuracy": accuracy,
        "chance": chance,
        "knn_accuracy": float(knn),
        "confusion": confusion,
        "n_neurons_used": int(class_counts.sum()) if class_counts is not None else 0,
        "class_counts": class_counts.tolist() if class_counts is not None else [],
    }


# ---- Experiment 3d: stability guards -----------------------------------------


def experiment_3d() -> dict:
    """Population-rate bounds, weight bounds, and the frozen-inhibition guard."""
    retina = Retina(seed=SEED, window_ms=WINDOW_MS)
    projection = InputProjection(28 * 28, N, fanout=FANOUT, seed=SEED)
    population, synapses = make_network()
    inh0 = synapses.weights[N_EXC * OUT_DEGREE :].copy()

    rng = _rng()
    imgs = []
    for _ in range(30):
        img = np.zeros((28, 28))
        y0, x0 = rng.integers(2, 22), rng.integers(2, 22)
        img[y0 : y0 + 5, x0 : x0 + 5] = 1.0
        imgs.append(img)
    stim = RetinaStimulus(retina, projection, imgs, gap_ms=GAP_MS)

    rec = simulate(
        population,
        synapses,
        T_ms=int(len(imgs) * stim.slot_ms),
        engine="sparse",
        seed=SEED,
        learning=True,
        stimulus_fn=stim,
    )

    rates = rec.mean_rates_hz()
    inh_frozen = bool(np.array_equal(synapses.weights[N_EXC * OUT_DEGREE :], inh0))
    return {
        "min_rate_hz": float(rates.min()),
        "max_rate_hz": float(rates.max()),
        "weight_min": float(synapses.weights.min()),
        "weight_max": float(synapses.weights.max()),
        "is_finite": bool(np.isfinite(synapses.weights).all()),
        "inh_frozen": inh_frozen,
        "n_spikes": int(rec.times_ms.size),
    }


# ---- figures and report ------------------------------------------------------


def _plot_confusion(conf: np.ndarray, acc: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"M3 E3c: unsupervised digit readout (accuracy {acc:.3f})")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = OUT_DIR / "m3_confusion.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def _plot_ria(
    tiles: np.ndarray,
    resp: np.ndarray,
    labels: np.ndarray,
    label: str,
    fname: str,
) -> None:
    """Grid of the neurons with the strongest emerged class selectivity.

    Ordering uses class-selectivity strength (best-class mean response relative
    to a neuron's overall mean), which is the honest `emerged feature` signal,
    rather than raw pixel contrast (saturated for every active neuron with a
    fixed random fan-out).
    """
    resp = np.asarray(resp, dtype=float)
    class_mean = np.zeros((N_CLASSES, tiles.shape[0]))
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.any():
            class_mean[c] = resp[mask].mean(axis=0)
    total_mean = resp.mean(axis=0)
    best = class_mean.max(axis=0)
    specificity = np.divide(
        best,
        np.maximum(total_mean, 1e-9),
        out=np.zeros_like(best),
        where=total_mean > 0,
    )

    grid = 20
    order = np.argsort(specificity)[::-1][: grid * grid]
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    for k, n in enumerate(order):
        ax = axes[k // grid, k % grid]
        ax.imshow(tiles[n], cmap="gray_r")
        ax.axis("off")
    fig.suptitle(
        f"M3 E3b: receptive-field images, {label}\n"
        f"(top-{grid * grid} neurons by class-selectivity)",
        y=0.92,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT_DIR / fname
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"[saved] {path}")


def _write_results(e3a: dict, e3b: dict, e3c: dict, e3d: dict) -> None:
    lines = [
        "# M3 Results -- artificial retina and unsupervised visual features",
        "",
        "Observed values from `benchmarks/m3_vision.py` on the sparse",
        "event-driven engine (gain=8, out_degree=100, N=1000, seed 42) with a",
        "28x28 artificial retina (latency coding, window 350 ms, gap 150 ms, 20",
        "non-plastic fan-out edges per pixel).",
        "",
        "## Experiment 3a -- encoder fidelity",
        f"- latency coding: Spearman(intensity, -latency) = {e3a['latency_spearman']:.3f} (target > 0.6)",
        f"- rate coding:    Pearson(spike count, intensity) = {e3a['rate_pearson']:.3f} (target > 0.9)",
        f"- latency spikes per gradient image: {e3a['latency_n_spikes']}",
        "",
        "## Experiment 3b -- unsupervised feature emergence",
        f"- mean RIA contrast before/after: {e3b['contrast_before']:.3f} / {e3b['contrast_after']:.3f}",
        f"- mean RIA pixel spread before/after (lower = sharper): {e3b['spread_before']:.3f} / {e3b['spread_after']:.3f}",
        f"- class-selective neurons before/after: {e3b['selective_before']} / {e3b['selective_after']}",
        f"- active neurons after training: {e3b['n_active_after']} / {N}",
        f"- training pass: {e3b['train_spikes']} spikes, mean rate {e3b['mean_rate_hz']:.2f} Hz, peak {e3b['max_rate_hz']:.2f} Hz",
        "",
        "## Experiment 3c -- zero-shot digit classification",
        f"- held-out accuracy (frozen soft readout): {e3c['accuracy']:.3f}",
        f"- chance baseline: {e3c['chance']:.2f}",
        f"- numpy kNN (k=3, raw pixels) baseline: {e3c['knn_accuracy']:.3f}",
        f"- neurons used by the readout: {e3c['n_neurons_used']} / {N}",
        "- confusion matrix (rows=true, cols=pred):",
        *(f"  {'  '.join(f'{v:3d}' for v in row)}" for row in e3c["confusion"]),
        "",
        "## Experiment 3d -- stability guards",
        f"- firing rates: [{e3d['min_rate_hz']:.2f}, {e3d['max_rate_hz']:.2f}] Hz (guardrail [0.5, 100])",
        f"- weights: [{e3d['weight_min']:.3f}, {e3d['weight_max']:.3f}] (bounds [0, 1] for excitatory)",
        f"- all weights finite: {e3d['is_finite']}",
        f"- inhibitory weights frozen: {e3d['inh_frozen']}",
        f"- E3d spikes: {e3d['n_spikes']}",
        "",
        "Caveat: feature emergence is measured at the population level (RIA",
        "contrast and class selectivity), not as per-pixel tuning, because the",
        "input pathway is a fixed random projection (no input plasticity). A",
        "learned input weight matrix (Diehl & Cook 2015 style) is the planned",
        "v2 upgrade and would let single neurons develop localized receptive",
        "fields.",
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 retina experiments")
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--probe", type=int, default=30)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist()
    x_train, y_train = subsample_mnist(x_train, y_train, per_class=args.train // 10)
    x_test, y_test = subsample_mnist(x_test, y_test, per_class=args.test // 10)

    train_imgs = [normalize(img) for img in x_train]
    test_imgs = [normalize(img) for img in x_test]

    # A balanced probe set for RIA tiles (per_class images of each digit).
    probe_x, probe_y = subsample_mnist(x_train, y_train, per_class=args.probe // 10)
    probe_imgs = [normalize(img) for img in probe_x]
    probe_labels = probe_y.astype(np.int64)

    t0 = time.time()
    e3a = experiment_3a()
    print(f"[3a] latency {e3a['latency_spearman']:.3f}, rate {e3a['rate_pearson']:.3f}")

    e3b = experiment_3b(train_imgs, y_train.astype(np.int64), probe_imgs, probe_labels)
    print(
        f"[3b] contrast {e3b['contrast_before']:.3f} -> {e3b['contrast_after']:.3f}; "
        f"spread {e3b['spread_before']:.3f} -> {e3b['spread_after']:.3f}; "
        f"selective {e3b['selective_before']} -> {e3b['selective_after']}"
    )
    _plot_ria(
        e3b["tiles_after"],
        e3b["resp_after"],
        probe_labels,
        "after unsupervised STDP",
        "m3_emergence_tiles.png",
    )

    e3c = experiment_3c(
        train_imgs, y_train.astype(np.int64), test_imgs, y_test.astype(np.int64)
    )
    print(
        f"[3c] accuracy {e3c['accuracy']:.3f} (chance {e3c['chance']:.2f}, kNN {e3c['knn_accuracy']:.3f})"
    )
    _plot_confusion(e3c["confusion"], e3c["accuracy"])

    e3d = experiment_3d()
    print(
        f"[3d] rates [{e3d['min_rate_hz']:.2f}, {e3d['max_rate_hz']:.2f}], frozen inh {e3d['inh_frozen']}"
    )

    _write_results(e3a, e3b, e3c, e3d)
    print(f"\nwall time: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
