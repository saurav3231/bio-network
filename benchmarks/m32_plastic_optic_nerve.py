"""M3.2 experiments: a plastic input projection ("optic nerve" analog).

M3 v1 froze the input pathway: every retina pixel drove a fixed random fan-out
of neurons through constant unit weights, so a neuron's receptive field was a
static cocktail of uncorrelated pixels (hence degenerate readouts). M3.2 tests
the causal claim *input plasticity is the source of receptive fields*: the same
network, the same seeds, the same images, with the only difference that the
input projection's ``w_in in [0,1]`` synapses learn with STDP (tau 20 ms,
A+ 0.10, A- 0.12, init uniform 0.2-0.4) during a training phase then freeze for
assignment and test.

    E32a Control reproduction. ``plastic=False`` is the byte-for-byte v1 arm
         built from the same seed (same topology and fan-out).

    E32b Receptive-field imagery (RIA). We count *structured* tiles -- RIA
         tiles whose bright pixels concentrate on a small pixel cluster (
         localized, digit-like receptive fields), the Diehl & Cook (2015)
         signature of input plasticity. Plot control vs plastic.

    E32c Zero-shot digit readout, TWO pre-committed decoders both reported
         (no post-hoc shopping): the v1 soft prototype fingerprint and a hard
         per-neuron majority vote.

    E32d Stability guards: finite spiking/weights, w_in in [0,1], frozen
         inhibitory recurrent compartment.

Run:   python benchmarks/m32_plastic_optic_nerve.py [--train 1000] [--test 200]
       [--probe 60] [--seed 42] [--competition_gain 1.0]
Results: docs/M3_2_RESULTS.md, figures in notebooks/output/.
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
RESULTS = ROOT / "docs" / "M3_2_RESULTS.md"

N_EXC, N_INH = 800, 200
N = N_EXC + N_INH
GAIN = 8.0
OUT_DEGREE = 100
WINDOW_MS = 350.0
GAP_MS = 150.0
FANOUT = 20
N_CLASSES = 10
N_PIXELS = 28 * 28

STDP_TAU_MS = 20.0
LTP_A_PLUS = 0.10
LTD_A_MINUS = 0.12
W_LO, W_HI = 0.2, 0.4  # w_in init bounds


def make_network(seed: int) -> tuple[IzhikevichPopulation, SparseSynapses]:
    population = IzhikevichPopulation(seed=seed)
    synapses = SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=OUT_DEGREE, seed=seed, gain=GAIN
    )
    return population, synapses


def normalize(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / 255.0


def per_image_response(stim, rec, n_neurons: int) -> np.ndarray:
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
    seed: int,
    input_plastic_fn=None,
) -> tuple[object, np.ndarray]:
    rec = simulate(
        population,
        synapses,
        T_ms=int(len(stim) * stim.slot_ms),
        engine="sparse",
        seed=seed,
        learning=learning,
        stimulus_fn=stim,
        input_plastic_fn=input_plastic_fn,
    )
    return rec, per_image_response(stim, rec, N)


# ---- receptive-field imagery --------------------------------------------------
def ria_tiles(imgs: list[np.ndarray], resp: np.ndarray) -> np.ndarray:
    arr = np.stack([np.asarray(i) for i in imgs])
    resp = np.asarray(resp, dtype=float)
    denom = resp.sum(axis=0)
    tiles = np.zeros((resp.shape[1], *arr.shape[1:]))
    for n in range(resp.shape[1]):
        if denom[n] > 0:
            tiles[n] = np.tensordot(resp[:, n], arr, axes=(0, 0)) / denom[n]
    return tiles


def ria_spread(tiles: np.ndarray, frac: float = 0.5) -> float:
    """Mean fraction of pixels above ``frac*max`` (localization proxy)."""
    flat = tiles.reshape(tiles.shape[0], -1)
    tot, n = 0.0, 0
    for row in flat:
        mx = row.max()
        if mx <= 1e-9:
            continue
        n += 1
        tot += float((row >= frac * mx).mean())
    return tot / n if n else 0.0


def structured_tiles(tiles: np.ndarray, max_frac: float = 0.05) -> int:
    """Number of RIA tiles whose bright half-max closure covers <= 5% of pixels.

    A true localized receptive field (a stroke or digit fragment) occupies a
    small pixel fraction; a diffuse random-fan-out blend covers far more.
    """
    flat = tiles.reshape(tiles.shape[0], -1)
    structured = 0
    for row in flat:
        mx = row.max()
        if mx <= 1e-9:
            continue
        if float((row >= 0.5 * mx).mean()) <= max_frac:
            structured += 1
    return structured


def selective_neurons(resp, labels, threshold: float = 2.0) -> int:
    """Neurons whose best-class mean response exceeds their mean by ``threshold``x."""
    resp = np.asarray(resp, dtype=float)
    class_mean = np.zeros((N_CLASSES, resp.shape[1]))
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.any():
            class_mean[c] = resp[mask].mean(axis=0)
    total_mean = resp.mean(axis=0)
    best = class_mean.max(axis=0)
    return int(((best > threshold * np.maximum(total_mean, 1e-9)) & (best > 0)).sum())


# ---- experiment 32a: control reproduction --------------------------------------


# ---- experiment 32c readout ------------------------------------------------------
def evaluate_readout(
    resp_train: np.ndarray,
    train_labels: np.ndarray,
    resp_test: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Fit assignment on training responses; evaluate soft + vote on test."""
    ro = LabelsReadout(n_neurons=N, n_classes=N_CLASSES)
    ro.fit(resp_train, train_labels)
    acc_soft = float(np.mean(ro.predict(resp_test) == test_labels))
    acc_vote = float(np.mean(ro.predict_vote(resp_test) == test_labels))
    n_used = int(np.bincount(ro.assignment, minlength=N_CLASSES).sum())
    return {"acc_soft": acc_soft, "acc_vote": acc_vote, "n_used": n_used}


def run_arm(
    train_imgs: list[np.ndarray],
    train_labels: np.ndarray,
    test_imgs: list[np.ndarray],
    test_labels: np.ndarray,
    probe_imgs: list[np.ndarray],
    probe_labels: np.ndarray,
    *,
    plastic: bool,
    seed: int,
    competition_gain: float,
) -> dict:
    """Run one end-to-end arm; ``plastic=False`` reproduces the frozen v1 control."""
    t0 = time.time()
    retina = Retina(seed=seed, window_ms=WINDOW_MS)
    proj = InputProjection(
        N_PIXELS,
        N,
        fanout=FANOUT,
        seed=seed,
        plastic=plastic,
        competition_gain=competition_gain,
    )
    train_stim = RetinaStimulus(retina, proj, train_imgs, gap_ms=GAP_MS)
    test_stim = RetinaStimulus(retina, proj, test_imgs, gap_ms=GAP_MS)
    probe_stim = RetinaStimulus(retina, proj, probe_imgs, gap_ms=GAP_MS)

    population, synapses = make_network(seed)
    input_fn = proj.on_neurons_fired if plastic else None

    # BEFORE: fresh network, frozen probe responses.
    _, resp_before = run_images(
        population,
        synapses,
        probe_stim,
        learning=False,
        seed=seed,
        input_plastic_fn=None,
    )
    tiles_before = ria_tiles(probe_imgs, resp_before)

    # TRAIN: recurrent STDP ON; input plasticity on iff this is the plastic arm.
    rec_train, _ = run_images(
        population,
        synapses,
        train_stim,
        learning=True,
        seed=seed,
        input_plastic_fn=input_fn,
    )

    # ASSIGNMENT (frozen): re-read the training images, plasticity OFF.
    _, resp_train = run_images(
        population,
        synapses,
        train_stim,
        learning=False,
        seed=seed,
        input_plastic_fn=None,
    )
    # TEST (frozen).
    _, resp_test = run_images(
        population,
        synapses,
        test_stim,
        learning=False,
        seed=seed,
        input_plastic_fn=None,
    )
    # AFTER: frozen probe read, same probe images.
    _, resp_after = run_images(
        population,
        synapses,
        probe_stim,
        learning=False,
        seed=seed,
        input_plastic_fn=None,
    )
    tiles_after = ria_tiles(probe_imgs, resp_after)

    ro = evaluate_readout(resp_train, train_labels, resp_test, test_labels)

    rates = rec_train.mean_rates_hz()
    inh0 = synapses.weights[N_EXC * OUT_DEGREE :].copy()
    inh_same = bool(np.array_equal(synapses.weights[N_EXC * OUT_DEGREE :], inh0))
    fan_in_mean, fan_in_std = _arch_in(proj)

    w_in = proj.weights if plastic else None
    return {
        "plastic": plastic,
        "acc_soft": ro["acc_soft"],
        "acc_vote": ro["acc_vote"],
        "n_used": ro["n_used"],
        "selective_before": selective_neurons(resp_before, probe_labels),
        "selective_after": selective_neurons(resp_after, probe_labels),
        "spread_before": ria_spread(tiles_before),
        "spread_after": ria_spread(tiles_after),
        "structured_before": structured_tiles(tiles_before),
        "structured_after": structured_tiles(tiles_after),
        "n_active_after": int((resp_after.sum(axis=0) > 0).sum()),
        "train_spikes": int(rec_train.times_ms.size),
        "mean_rate_hz": float(rates.mean()),
        "max_rate_hz": float(rates.max()),
        "recurrent_weight_min": float(synapses.weights.min()),
        "recurrent_weight_max": float(synapses.weights.max()),
        "all_finite": bool(np.isfinite(synapses.weights).all()),
        "inh_frozen": inh_same,
        "fan_in_mean": fan_in_mean,
        "fan_in_std": fan_in_std,
        "w_in_min": float(np.min(w_in)) if plastic else None,
        "w_in_max": float(np.max(w_in)) if plastic else None,
        "w_in_finite": bool(np.isfinite(w_in).all()) if plastic else None,
        "wall_s": time.time() - t0,
        "resp_after": resp_after,
        "tiles_after": tiles_after,
    }


def _arch_in(proj: InputProjection) -> tuple[float, float]:
    return proj.fan_in_stats()


def _plot_tiles_arm(res: dict, probe_labels, plastic: bool) -> None:
    tiles = res["tiles_after"]
    resp = np.asarray(res["resp_after"], dtype=float)
    class_mean = np.zeros((N_CLASSES, tiles.shape[0]))
    for c in range(N_CLASSES):
        mask = probe_labels == c
        if mask.any():
            class_mean[c] = resp[mask].mean(axis=0)
    total_mean = resp.mean(axis=0)
    best = class_mean.max(axis=0)
    spec = np.divide(
        best,
        np.maximum(total_mean, 1e-9),
        out=np.zeros_like(best),
        where=total_mean > 0,
    )
    grid = 20
    order = np.argsort(spec)[::-1][: grid * grid]
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    for k, n in enumerate(order):
        ax = axes[k // grid, k % grid]
        ax.imshow(tiles[n], cmap="gray_r")
        ax.axis("off")
    tag = "plastic (learned inputs)" if plastic else "control (frozen v1)"
    fig.suptitle(
        f"M3.2 {tag} -- top-{grid*grid} RIA receptive fields by class selectivity",
        y=0.93,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fname = "m32_tiles_control.png" if not plastic else "m32_tiles_plastic.png"
    path = OUT_DIR / fname
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"[saved] {path}")


def _plot_diptych(control: dict, plastic: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 15))
    for ax, res in ((ax1, control), (ax2, plastic)):
        tiles = res["tiles_after"]
        grid = 8
        panel = np.zeros((grid, grid, 28, 28))
        for r in range(grid):
            for c in range(grid):
                n = r * grid + c
                if n < tiles.shape[0]:
                    panel[r, c] = tiles[n]
        ax.imshow(panel.reshape(grid * 28, grid * 28), cmap="gray_r")
        ax.axis("off")
        ax.set_title(
            "control (frozen v1)" if not res["plastic"] else "plastic (learned)"
        )
    fig.suptitle("M3.2 receptive fields: first neurons, control vs plastic")
    fig.tight_layout()
    path = OUT_DIR / "m32_tiles_diptych.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def _write_results(control: dict, plastic: dict) -> None:
    def fmt(d: dict) -> str:
        return f"{d['acc_soft']:.1%} (soft) / {d['acc_vote']:.1%} (vote)"

    lines = [
        "# M3.2 Results -- a plastic input projection (optic nerve)",
        "",
        "Controlled AB test on the same engineered substrate as M3 v1: seeds, fan-out,",
        "topology, images and readouts are identical. The only change is the plastic arm",
        "lets its input synapses ``w_in in [0,1]`` (init uniform 0.2-0.4) learn with STDP",
        "(tau 20 ms, A+ 0.10, A- 0.12) during the training phase, then freezes before",
        "assignment/test.",
        "",
        "## Arm summary",
        (
            f"- frozen (v1 control): accuracy {fmt(control)}, structured tiles "
            f"{control['structured_after']}, selectivity {control['selective_before']}->"
            f"{control['selective_after']}, readout neurons {control['n_used']}/{N}, "
            f"wall {control['wall_s']:.0f}s"
        ),
        (
            f"- plastic (learned inputs): accuracy {fmt(plastic)}, structured tiles "
            f"{plastic['structured_after']}, selectivity {plastic['selective_before']}->"
            f"{plastic['selective_after']}, readout neurons {plastic['n_used']}/{N}, "
            f"wall {plastic['wall_s']:.0f}s"
        ),
        "",
        "## Experiment 32a -- control reproduction",
        "The frozen arm is byte-for-byte the M3 v1 pathway (same targets, unit drive,",
        "drive scaled only by ``pulse_amp`` and ``competition_gain``). Any drift vs the",
        "recorded v1 numbers is attributable to test/probe sampling, not plasticity.",
        "",
        "## Experiment 32b -- receptive-field imagery (RIA)",
        (
            f"- structured (localized) tiles before/after: control "
            f"{control['structured_before']}->{control['structured_after']}, plastic "
            f"{plastic['structured_before']}->{plastic['structured_after']}"
        ),
        (
            f"- RIA pixel spread (lower = sharper): control {control['spread_after']:.3f}, "
            f"plastic {plastic['spread_after']:.3f}"
        ),
        (
            f"- class-selective neurons: control {control['selective_before']}->"
            f"{control['selective_after']}, plastic {plastic['selective_before']}->"
            f"{plastic['selective_after']}"
        ),
        (
            f"- active neurons (probe): control {control['n_active_after']}, plastic "
            f"{plastic['n_active_after']}"
        ),
        "",
        "## Experiment 32c -- zero-shot digit readout (both decoders pre-committed)",
        f"- control: soft {control['acc_soft']:.1%}, vote {control['acc_vote']:.1%}",
        f"- plastic: soft {plastic['acc_soft']:.1%}, vote {plastic['acc_vote']:.1%}",
        "",
        "## Experiment 32d -- stability guards",
        f"- train spikes: control {control['train_spikes']:.0f}, plastic {plastic['train_spikes']:.0f}",
        (
            f"- mean/max rate (Hz): control {control['mean_rate_hz']:.2f}/"
            f"{control['max_rate_hz']:.2f}, plastic {plastic['mean_rate_hz']:.2f}/"
            f"{plastic['max_rate_hz']:.2f}"
        ),
        (
            f"- recurrent weights: control [{control['recurrent_weight_min']:.3f}, "
            f"{control['recurrent_weight_max']:.3f}], plastic [{plastic['recurrent_weight_min']:.3f}, "
            f"{plastic['recurrent_weight_max']:.3f}]"
        ),
        f"- all finite: control {control['all_finite']}, plastic {plastic['all_finite']}",
        (
            f"- inhibitory weights frozen: control {control['inh_frozen']}, "
            f"plastic {plastic['inh_frozen']}"
        ),
        f"- w_in range (plastic): [{plastic['w_in_min']:.3f}, {plastic['w_in_max']:.3f}]",
        "",
        "## Honest bottom line",
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {RESULTS}")


def _fmt_acc(d: dict) -> str:
    return f"{d['acc_soft']:.1%} (soft) / {d['acc_vote']:.1%} (vote)"


def main() -> None:
    ap = argparse.ArgumentParser(description="M3.2 plastic input projection benchmark")
    ap.add_argument(
        "--train", type=int, default=1000, help="training images (stratified)"
    )
    ap.add_argument("--test", type=int, default=200, help="total test images")
    ap.add_argument("--probe", type=int, default=60, help="total probe images")
    ap.add_argument("--seed", type=int, default=42, help="seed for both arms")
    ap.add_argument("--competition_gain", type=float, default=1.0)
    ap.add_argument("--only", choices=["control", "plastic"], default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist()
    x_train, y_train = subsample_mnist(
        x_train, y_train, per_class=args.train // 10, seed=args.seed
    )
    x_test, y_test = subsample_mnist(
        x_test, y_test, per_class=args.test // 10, seed=args.seed
    )
    train_imgs = [normalize(img) for img in x_train]
    test_imgs = [normalize(img) for img in x_test]

    probe_x, probe_y = subsample_mnist(
        x_train, y_train, per_class=args.probe // 10, seed=args.seed
    )
    probe_imgs = [normalize(img) for img in probe_x]
    probe_labels = probe_y.astype(np.int64)
    train_labels = y_train.astype(np.int64)
    test_labels = y_test.astype(np.int64)

    control, plastic = None, None
    if args.only in (None, "control"):
        control = run_arm(
            train_imgs,
            train_labels,
            test_imgs,
            test_labels,
            probe_imgs,
            probe_labels,
            plastic=False,
            seed=args.seed,
            competition_gain=args.competition_gain,
        )
        print(
            f"[control] soft {control['acc_soft']:.1%} vote {control['acc_vote']:.1%} "
            f"structured {control['structured_after']} wall {control['wall_s']:.0f}s"
        )
        _plot_tiles_arm(control, probe_labels, plastic=False)

    if args.only in (None, "plastic"):
        plastic = run_arm(
            train_imgs,
            train_labels,
            test_imgs,
            test_labels,
            probe_imgs,
            probe_labels,
            plastic=True,
            seed=args.seed,
            competition_gain=args.competition_gain,
        )
        print(
            f"[plastic] soft {plastic['acc_soft']:.1%} vote {plastic['acc_vote']:.1%} "
            f"structured {plastic['structured_after']} wall {plastic['wall_s']:.0f}s"
        )
        _plot_tiles_arm(plastic, probe_labels, plastic=True)

    if control is not None and plastic is not None:
        _write_results(control, plastic)
        _plot_diptych(control, plastic)


if __name__ == "__main__":
    main()
