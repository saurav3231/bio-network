"""M3.3 experiments: homeostatic regulators in the plastic input pathway.

M3.2 (commit 035a9b7) established the diagnosis: pure STDP on the input cable
starves the network. LTD (A- 0.12, applied on every arrival) outruns LTP
(A+ 0.10, which needs an actual postsynaptic spike), ``w_in`` drains toward
zero, only ~76/1000 neurons stay alive, and the survivors converge onto the
"7" and "9" archetype ghosts. M3.3 adds the two classic biological regulators:

    E33a Health.  Compare mean/max firing rates, probe-phase active-neuron
              counts and train-spike totals between the M3.2 baseline
              (ARM A) and the homeostatic arm (ARM B). Dream number:
              >= 2 Hz mean and >= 400/1000 active; any honest number counts.

    E33b Emergence. Receptive-field (RIA) tile grids. The structured-tile
              proxy is recalibrated (see calibration note in docs) so the
              M3.2 7/9 stroke-like archetype tiles count as structured.

    E33c Recognition. The identical readout suite as M3 (soft prototype
              + per-neuron plurality vote), plus prediction histograms,
              confusion matrices and the human-style top confusions.

    E33d Stability. Bounds, inhibitory-frozen recurrent weights, all finite,
              theta within [1, 30].

Run:   python benchmarks/m33_homeostasis.py [--train 1000] [--test 200]
       [--probe 60] [--seed 42]
Results: docs/M3_3_RESULTS.md, figures in notebooks/output/.
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
RESULTS = ROOT / "docs" / "M3_3_RESULTS.md"

N_EXC, N_INH = 800, 200
N = N_EXC + N_INH
GAIN = 8.0
OUT_DEGREE = 100
WINDOW_MS = 350.0
GAP_MS = 150.0
FANOUT = 20
N_CLASSES = 10
N_PIXELS = 28 * 28

# Synaptic-scaling constant (Turrigiano et al. 1998-style per-neuron target).
SCALING_TARGET = 0.30

# Adaptive-threshold constants (M3.3).
TARGET_RATE_HZ = 5.0
RATE_TAU_MS = 2000.0
THETA_MIN, THETA_MAX = 1.0, 30.0

# Structured-tile calibration: max_frac 0.12 counts the M3.2 7/9 archetype
# (stroke-like, ~5-12% of the image) as structured while a uniform blend is not.
STRUCTURED_MAX_FRAC = 0.12


def make_network(
    seed: int, adapt_thresholds: bool
) -> tuple[IzhikevichPopulation, SparseSynapses]:
    population = IzhikevichPopulation(
        seed=seed,
        adaptive_thresholds=adapt_thresholds,
        target_rate_hz=TARGET_RATE_HZ,
        rate_tau_ms=RATE_TAU_MS,
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
    )
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
        if in_w.any():
            resp[i] = np.bincount(rec.indices[in_w], minlength=n_neurons).astype(
                np.int64
            )
    return resp


def slot_spike_buckets(rec, slot_ms: float, n_slots: int) -> np.ndarray:
    """Spike count per presentation slot (feed the health-rate timeline)."""
    ids = np.floor(np.asarray(rec.times_ms, dtype=float) / slot_ms).astype(np.int64)
    ids = np.clip(ids, 0, n_slots - 1)
    return np.bincount(ids, minlength=n_slots)


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


# ---- receptive-field imagery ---------------------------------------------
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


def structured_tiles(tiles: np.ndarray, max_frac: float = STRUCTURED_MAX_FRAC) -> int:
    """Number of RIA tiles whose bright half-max closure covers <= max_frac.

    Recalibrated for M3.3 (max_frac 0.12 = ~94 px of 784): a stroke / digit
    fragment tile counts; a uniform random-fan blend does not.
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
    """Neurons whose best-class mean response exceeds overall mean by threshx."""
    resp = np.asarray(resp, dtype=float)
    class_mean = np.zeros((N_CLASSES, resp.shape[1]))
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.any():
            class_mean[c] = resp[mask].mean(axis=0)
    total_mean = resp.mean(axis=0)
    best = class_mean.max(axis=0)
    return int(((best > threshold * np.maximum(total_mean, 1e-9)) & (best > 0)).sum())


# ---- readout (identical suite to M3 / M3.2) -------------------------------
def evaluate_readout(
    resp_train: np.ndarray,
    train_labels: np.ndarray,
    resp_test: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    ro = LabelsReadout(n_neurons=N, n_classes=N_CLASSES)
    ro.fit(resp_train, train_labels)
    pred_soft = ro.predict(resp_test)
    pred_vote = ro.predict_vote(resp_test)
    acc_soft = float(np.mean(pred_soft == test_labels))
    acc_vote = float(np.mean(pred_vote == test_labels))
    n_used = int(np.bincount(ro.assignment, minlength=N_CLASSES).sum())
    hist = np.bincount(pred_soft, minlength=N_CLASSES).astype(np.int64)
    return {
        "acc_soft": acc_soft,
        "acc_vote": acc_vote,
        "n_used": n_used,
        "pred_soft": pred_soft,
        "hist": hist,
        "n_classes_used": int((hist > 0).sum()),
    }


def top_confusions(pred, truth, k: int = 5) -> list[tuple[int, int, int]]:
    """Return [(true, predicted, count)] sorted descending, ignoring diagonal."""
    pred = np.asarray(pred, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    pairs: dict[tuple[int, int], int] = {}
    for tr, pr in zip(truth, pred):
        if tr != pr:
            pairs[(int(tr), int(pr))] = pairs.get((int(tr), int(pr)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [(t, p, c) for (t, p), c in ranked]


# ---- one arm ----------------------------------------------------------------
def run_arm(
    name: str,
    train_imgs,
    train_labels,
    test_imgs,
    test_labels,
    probe_imgs,
    probe_labels,
    *,
    adapt_thresholds: bool,
    synaptic_scaling: bool,
    excitatory_only: bool,
    seed: int,
) -> dict:
    t0 = time.time()
    retina = Retina(seed=seed, window_ms=WINDOW_MS)
    proj = InputProjection(
        N_PIXELS,
        N,
        n_excitatory=N_EXC,
        fanout=FANOUT,
        seed=seed,
        plastic=True,
        synaptic_scaling=synaptic_scaling,
        excitatory_only=excitatory_only,
    )
    train_stim = RetinaStimulus(retina, proj, train_imgs, gap_ms=GAP_MS)
    test_stim = RetinaStimulus(retina, proj, test_imgs, gap_ms=GAP_MS)
    probe_stim = RetinaStimulus(retina, proj, probe_imgs, gap_ms=GAP_MS)

    population, synapses = make_network(seed, adapt_thresholds)
    input_fn = proj.on_neurons_fired

    # fresh probe (BEFORE)
    _, resp_before = run_images(
        population, synapses, probe_stim, learning=False, seed=seed
    )
    tiles_before = ria_tiles(probe_imgs, resp_before)

    # TRAIN
    rec_train, _ = run_images(
        population,
        synapses,
        train_stim,
        learning=True,
        seed=seed,
        input_plastic_fn=input_fn,
    )
    train_bucket = slot_spike_buckets(rec_train, train_stim.slot_ms, len(train_stim))

    # ASSIGNMENT + TEST + AFTER (frozen)
    _, resp_train = run_images(
        population, synapses, train_stim, learning=False, seed=seed
    )
    _, resp_test = run_images(
        population, synapses, test_stim, learning=False, seed=seed
    )
    _, resp_after = run_images(
        population, synapses, probe_stim, learning=False, seed=seed
    )
    tiles_after = ria_tiles(probe_imgs, resp_after)

    ro = evaluate_readout(resp_train, train_labels, resp_test, test_labels)

    rates = rec_train.mean_rates_hz()
    inh0 = synapses.weights[N_EXC * OUT_DEGREE :].copy()
    inh_same = bool(np.array_equal(synapses.weights[N_EXC * OUT_DEGREE :], inh0))

    w_in = proj._weights_flat if proj.plastic else None
    return {
        "name": name,
        "adapt_thresholds": adapt_thresholds,
        "synaptic_scaling": synaptic_scaling,
        "acc_soft": ro["acc_soft"],
        "acc_vote": ro["acc_vote"],
        "n_used": ro["n_used"],
        "pred_soft": ro["pred_soft"],
        "hist": ro["hist"],
        "n_classes_used": ro["n_classes_used"],
        "selective_before": selective_neurons(resp_before, probe_labels),
        "selective_after": selective_neurons(resp_after, probe_labels),
        "spread_before": ria_spread(tiles_before),
        "spread_after": ria_spread(tiles_after),
        "structured_before": structured_tiles(tiles_before),
        "structured_after": structured_tiles(tiles_after),
        "n_active_before": int((resp_before.sum(axis=0) > 0).sum()),
        "n_active_after": int((resp_after.sum(axis=0) > 0).sum()),
        "train_spikes": int(rec_train.times_ms.size),
        "mean_rate_hz": float(rates.mean()),
        "max_rate_hz": float(rates.max()),
        "train_bucket": np.asarray(train_bucket, dtype=np.int64),
        "recurrent_weight_min": float(synapses.weights.min()),
        "recurrent_weight_max": float(synapses.weights.max()),
        "all_finite": bool(np.isfinite(synapses.weights).all()),
        "inh_frozen": inh_same,
        "w_in_min": float(w_in.min()) if w_in is not None else None,
        "w_in_max": float(w_in.max()) if w_in is not None else None,
        "w_in_finite": bool(np.isfinite(w_in).all()) if w_in is not None else None,
        "theta_min": float(population.theta.min()) if adapt_thresholds else None,
        "theta_max": float(population.theta.max()) if adapt_thresholds else None,
        "theta_in_bounds": (
            bool(
                np.all(
                    (population.theta >= THETA_MIN) & (population.theta <= THETA_MAX)
                )
            )
            if adapt_thresholds
            else None
        ),
        "wall_s": time.time() - t0,
        "resp_after": resp_after,
        "tiles_after": tiles_after,
    }


# ---- plots -----------------------------------------------------------------
def _plot_tiles_arm(res: dict, fname: str, title: str) -> None:
    tiles = res["tiles_after"]
    grid = 8
    order = np.argsort(tiles.reshape(tiles.shape[0], -1).max(axis=1))[::-1][
        : grid * grid
    ]
    fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
    for k, n in enumerate(order):
        ax = axes[k // grid, k % grid]
        ax.imshow(tiles[n], cmap="gray_r")
        ax.axis("off")
    fig.suptitle(title, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / fname
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"[saved] {path}")


def _plot_health(arm_a: dict, arm_b: dict) -> None:
    nb = min(len(arm_a["train_bucket"]), len(arm_b["train_bucket"]))
    x = np.arange(nb)
    sprites = WINDOW_MS / 1000.0
    rates_a = arm_a["train_bucket"][:nb] / sprites / N
    rates_b = arm_b["train_bucket"][:nb] / sprites / N

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(x, rates_a, alpha=0.6, label="ARM A (M3.2 baseline)")
    ax.plot(x, rates_b, label="ARM B (M3.3 homeostatic)")
    ax.set_title("population mean rate per training slot (Hz)")
    ax.set_xlabel("training slot #")
    ax.set_ylabel("Hz")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.bar(["ARM A", "ARM B"], [arm_a["n_active_after"], arm_b["n_active_after"]])
    ax.axhline(400, color="gray", ls="--", label="target 400")
    ax.axhline(200, color="gray", ls=":", label="stretch 200")
    ax.set_title("active neurons in probe phase")
    ax.set_ylabel("count / 1000")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "m33_health.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[saved] {path}")


def _plot_accuracy_ladder(arm_a: dict, arm_b: dict) -> None:
    rows = [
        ("M3 v1 (frozen)", 0.11, 0.10),
        ("M3.2 plastic", arm_a["acc_soft"], arm_a["acc_vote"]),
        ("M3.3 homeostatic", arm_b["acc_soft"], arm_b["acc_vote"]),
    ]
    labels = [r[0] for r in rows]
    soft = [r[1] for r in rows]
    vote = [r[2] for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(x - 0.15, soft, 0.3, label="soft prototype")
    ax.bar(x + 0.15, vote, 0.3, label="per-neuron vote")
    ax.axhline(0.10, color="gray", ls="--", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, max(1.0, max(max(soft), max(vote)) * 1.25))
    ax.set_title("held-out digit accuracy")
    ax.legend()

    ax = axes[1]
    ax.bar(np.arange(10), arm_a["hist"], alpha=0.5, label="M3.2 baseline")
    ax.bar(np.arange(10), arm_b["hist"], alpha=0.7, label="M3.3 homeostatic")
    ax.set_xticks(np.arange(10))
    ax.set_title("prediction histograms")
    ax.set_xlabel("predicted class")
    ax.legend()
    fig.tight_layout()
    path = OUT_DIR / "m33_accuracy_ladder.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[saved] {path}")


def _write_results(arm_a: dict, arm_b: dict, top_conf: dict) -> None:
    def acc(d):
        return f"{d['acc_soft']:.1%} (soft) / {d['acc_vote']:.1%} (vote)"

    lines = [
        "# M3.3 Results -- homeostatic regulators in the plastic optic nerve",
        "",
        "Controlled follow-up to M3.2 (commit 035a9b7). The two arms share all of",
        "the substrate: seeds, fan-out, images, recurrent weight engine and",
        "readouts. The only change between them is the M3.3 physiology: ARM B",
        "switches on 1) **synaptic scaling** (Turrigiano et al. 1998): after every",
        "training window each neuron's incoming weight sum is renormalized to",
        f"`sum(w_in) == n_in_per_neuron * {SCALING_TARGET}`; and 2) **adaptive",
        "spike thresholds** (Diehl & Cook 2015 intrinsic plasticity): excitatory",
        "neurons track a slow-rate estimate and drift the firing threshold theta",
        f"toward {TARGET_RATE_HZ} Hz target within [{THETA_MIN},{THETA_MAX}] so over-active",
        "neurons back off and silent ones get recruited.",
        "",
        "## Arm summary",
        (
            f"- M3.2 baseline (ARM A): {acc(arm_a)}, structured tiles "
            f"{arm_a['structured_after']}, rate {arm_a['mean_rate_hz']:.2f} Hz, "
            f"active {arm_a['n_active_after']}/1000"
        ),
        (
            f"- M3.3 homeostatic (ARM B): {acc(arm_b)}, structured tiles "
            f"{arm_b['structured_after']}, rate {arm_b['mean_rate_hz']:.2f} Hz, "
            f"active {arm_b['n_active_after']}/1000"
        ),
        "",
        "## Economics story",
        "",
        "M3.2 diagnosis: LTD (A- 0.12) lands on every input arrival while LTP",
        "(A+ 0.10) only lands when a neuron actually fires; at the observed",
        "firing rate the one-sided depression starves the pathway. Synaptic",
        "scaling re-pins each neuron's total input power so the losing race can't",
        "run away -- and adaptive thresholds make under-driven neurons cheaper to",
        "excite, so pattern claimants diversify instead of collapsing to the",
        "7/9 ghosts.",
        "",
        "## E33a -- health (did the brain wake up?)",
        "",
        "- mean/max rate (Hz): ARM A {:.2f}/{:.2f}, ARM B {:.2f}/{:.2f}".format(
            arm_a["mean_rate_hz"],
            arm_a["max_rate_hz"],
            arm_b["mean_rate_hz"],
            arm_b["max_rate_hz"],
        ),
        "- active neurons (probe): ARM A {}, ARM B {}".format(
            arm_a["n_active_after"], arm_b["n_active_after"]
        ),
        "- train spikes: ARM A {:.0f}, ARM B {:.0f}".format(
            arm_a["train_spikes"], arm_b["train_spikes"]
        ),
        "",
        "## E 33b -- emergence (recalibrated proxy)",
        "",
        (
            "The structured-tile proxy was recalibrated (max covering fraction "
            f"{STRUCTURED_MAX_FRAC}) so the M3.2 7/9-prototype stroke-like tiles count as"
        ),
        "structured: a stroke occupying ~5-12% of the image qualifies, a uniform",
        "random blend doesn't. See calibration note.",
        "- structured tiles before/after: ARM A {}->{}, ARM B {}->{}".format(
            arm_a["structured_before"],
            arm_a["structured_after"],
            arm_b["structured_before"],
            arm_b["structured_after"],
        ),
        "- RIA pixel spread (lower = sharper): ARM A {:.3f}, ARM B {:.3f}".format(
            arm_a["spread_after"], arm_b["spread_after"]
        ),
        "- class-selective neurons: ARM A {}->{}, ARM B {}->{}".format(
            arm_a["selective_before"],
            arm_a["selective_after"],
            arm_b["selective_before"],
            arm_b["selective_after"],
        ),
        "",
        "## E 33c -- zero-shot readout (pre-committed decoders)",
        "",
        "- ARM A: soft {:.1%}, vote {:.1%}; {} classes predicted".format(
            arm_a["acc_soft"], arm_a["acc_vote"], arm_a["n_classes_used"]
        ),
        "- ARM B: soft {:.1%}, vote {:.1%}; {} classes predicted".format(
            arm_b["acc_soft"], arm_b["acc_vote"], arm_b["n_classes_used"]
        ),
        "",
        "## Top confusions (human-style)",
        "",
    ]
    for t, p, c in top_conf["arm_b"]:
        lines.append(f"- true {t} confused with predicted {p} ({c} cases)")
    lines.append(
        "- ARM A top: {}".format(
            ", ".join(f"{t}->{p} ({c})" for t, p, c in top_conf["arm_a"])
        )
    )
    lines += [
        "",
        "## E 33d -- stability guards",
        "",
        "- recurrent weights: ARM A [{:.3f}, {:.3f}], ARM B [{:.3f}, {:.3f}]".format(
            arm_a["recurrent_weight_min"],
            arm_a["recurrent_weight_max"],
            arm_b["recurrent_weight_min"],
            arm_b["recurrent_weight_max"],
        ),
        "- all finite: ARM A {}, ARM B {}".format(
            arm_a["all_finite"], arm_b["all_finite"]
        ),
        "- inhibitory frozen: ARM A {}, ARM B {}".format(
            arm_a["inh_frozen"], arm_b["inh_frozen"]
        ),
        "- w_in range: ARM A [{:.4f}, {:.4f}], ARM B [{:.4f}, {:.4f}]".format(
            arm_a["w_in_min"],
            arm_a["w_in_max"],
            arm_b["w_in_min"],
            arm_b["w_in_max"],
        ),
        "- w_all finite: ARM A {}, ARM B {}".format(
            arm_a["w_in_finite"], arm_b["w_in_finite"]
        ),
        "- theta bounds OK: ARM B {} (range [{:.3f}, {:.3f}])".format(
            arm_b["theta_in_bounds"], arm_b["theta_min"], arm_b["theta_max"]
        ),
        "- wall: ARM A {:.0f}s, ARM B {:.0f}s".format(arm_a["wall_s"], arm_b["wall_s"]),
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {RESULTS}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M3.3 homeostatic regulator benchmark")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--probe", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--only",
        choices=["a", "b", "ab"],
        default="ab",
        help="which arm(s) to run (others loaded from cache)",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def cache_path(name: str) -> pathlib.Path:
        key = f"t{args.train}_e{args.test}_p{args.probe}_s{args.seed}"
        return OUT_DIR / f"m33_{key}_{name.lower().replace(' ', '_')}.npz"

    def save_cache(arm: dict) -> None:
        np.savez(
            cache_path(arm["name"]), **{k: v for k, v in arm.items() if v is not None}
        )

    def load_cache(name: str) -> dict | None:
        path = cache_path(name)
        if not path.exists():
            return None
        raw = np.load(path, allow_pickle=True)
        out = {}
        for k in raw.files:
            v = raw[k]
            out[k] = v.item() if v.ndim == 0 else v
        out["name"] = name
        return out

    def get_arm(flag: str, cfg: dict) -> dict:
        cached = load_cache(cfg["name"])
        if cached is not None:
            print(f"[{cfg['name']}] loaded from cache")
            return cached
        arm = run_arm(**cfg)
        save_cache(arm)
        return arm

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
    train_labels = y_train.astype(np.int64)
    test_labels = y_test.astype(np.int64)
    probe_labels = probe_y.astype(np.int64)

    print(
        f"[M3.3] train {len(train_imgs)} test {len(test_imgs)} probe {len(probe_imgs)}"
    )

    cfg_a = {
        "name": "ARM A",
        "train_imgs": train_imgs,
        "train_labels": train_labels,
        "test_imgs": test_imgs,
        "test_labels": test_labels,
        "probe_imgs": probe_imgs,
        "probe_labels": probe_labels,
        "adapt_thresholds": False,
        "synaptic_scaling": False,
        "excitatory_only": False,
        "seed": args.seed,
    }
    cfg_b = {
        "name": "ARM B",
        "train_imgs": train_imgs,
        "train_labels": train_labels,
        "test_imgs": test_imgs,
        "test_labels": test_labels,
        "probe_imgs": probe_imgs,
        "probe_labels": probe_labels,
        "adapt_thresholds": True,
        "synaptic_scaling": True,
        "excitatory_only": True,
        "seed": args.seed,
    }
    arm_a = get_arm("a", cfg_a) if args.only in ("a", "ab") else load_cache("ARM A")
    arm_b = get_arm("b", cfg_b) if args.only in ("b", "ab") else load_cache("ARM B")
    if arm_a is not None:
        print(
            f"[{arm_a['name']}] soft {arm_a['acc_soft']:.1%} vote {arm_a['acc_vote']:.1%} "
            f"structured {arm_a['structured_after']} rate {arm_a['mean_rate_hz']:.2f} "
            f"active {arm_a['n_active_after']} wall {arm_a['wall_s']:.0f}s"
        )
        _plot_tiles_arm(
            arm_a, "m33_tiles_armA.png", "M3.3 ARM A -- M3.2 baseline (frozen modes)"
        )

    if arm_b is not None:
        print(
            f"[{arm_b['name']}] soft {arm_b['acc_soft']:.1%} vote {arm_b['acc_vote']:.1%} "
            f"structured {arm_b['structured_after']} rate {arm_b['mean_rate_hz']:.2f} "
            f"active {arm_b['n_active_after']} wall {arm_b['wall_s']:.0f}s"
        )
        _plot_tiles_arm(
            arm_b, "m33_tiles_armB.png", "M3.3 ARM B -- homeostatic plastic"
        )

    if arm_a is not None and arm_b is not None:
        top_conf = {
            "arm_a": top_confusions(arm_a["pred_soft"], test_labels),
            "arm_b": top_confusions(arm_b["pred_soft"], test_labels),
        }
        _plot_health(arm_a, arm_b)
        _plot_accuracy_ladder(arm_a, arm_b)
        _write_results(arm_a, arm_b, top_conf)


if __name__ == "__main__":
    main()
