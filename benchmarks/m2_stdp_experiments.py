"""M2 STDP experiments: hallmark, fire-together, and cue/pattern-completion.

Runs the three experiments specified for Milestone 2 on the sparse,
event-driven engine with STDP enabled (gain = 8 for every training run so LTP
headroom exists below the ~160 Hz avalanche regime):

    E1  Hallmark stability: N=1000 network driven with noise, learning on,
        T=30 s. STDP must keep the network stable with a bimodal excitatory
        weight distribution, no NaN, and mean rate in the cortical range.
    E2  Fire-together / wire-together: three groups A (0..49+1), B (50..99+1) and
        C (100..149). A and B are co-driven with a pulse every 200 ms through a
        training phase (T=20 s); C only sees noise. In a frozen test phase
        (T=2 s) we compare the mean strength of A->B vs A->C synapses and
        expect the co-driven pair to be stronger (ratio > 1.5).
    E3  Cue / pattern completion (stretch): after E2-style training, drive A
        alone and check B's spike count in the 50 ms after an A pulse rises
        above a pre-training reference. Reported honestly.

Run with:  python benchmarks/m2_stdp_experiments.py
Results are written to docs/M2_RESULTS.md and figures to notebooks/output/.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses_sparse import SparseSynapses

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
RESULTS = ROOT / "docs" / "M2_RESULTS.md"

N_EXC, N_INH = 800, 200
N = N_EXC + N_INH
GAIN = 8.0
SEED = 42
OUT_DEGREE = 100
PULSE_PERIOD = 200  # ms
PULSE_AMP = 20.0
PULSE_WIDTH_A = 3  # ms, brief causal poke on A
LEAD_AB = 25  # ms, A leads B by roughly the axonal-delay window

GROUP_A = (0, 50)
GROUP_B = (50, 100)
GROUP_C = (100, 150)


def make_network() -> tuple[IzhikevichPopulation, SparseSynapses]:
    population = IzhikevichPopulation(seed=SEED)
    synapses = SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=OUT_DEGREE, seed=SEED, gain=GAIN
    )
    return population, synapses


def exc_mean(
    synapses: SparseSynapses, pre: tuple[int, int], post: tuple[int, int]
) -> float:
    """Mean excitatory weight of pre->post synapses between two neuron ranges."""
    collected = []
    for pre_i in range(pre[0], pre[1]):
        start, end = int(synapses.offsets[pre_i]), int(synapses.offsets[pre_i + 1])
        tg = synapses.targets[start:end]
        sel = (tg >= post[0]) & (tg < post[1])
        if sel.any():
            collected.append(synapses.weights[start:end][sel])
    if not collected:
        return 0.0
    return float(np.concatenate(collected).mean())


def pair_pulse_stim(a: tuple[int, int], b: tuple[int, int]):
    """M2 pairing stimulus: A fires first, B ~20 ms later, every 200 ms.

    Driving A *before* B makes the A->B synapse causal (pre before post within
    the axonal-delay window), which is exactly the 'fire together, wire
    together' association STDP is meant to capture. Driving them synchronously
    would instead depress A->B (post-before-pre), so the lead is essential.
    """
    _rng = np.random.default_rng(SEED)
    _scale = np.ones(N)
    _scale[:N_EXC] = 5.0
    _scale[N_EXC:] = 2.0
    _lead = LEAD_AB  # ms: A's spike should reach B just before B fires

    def _stim(t: float, n_neurons: int):
        base = _scale * _rng.standard_normal(N)
        phase = int(t) % PULSE_PERIOD
        if phase < PULSE_WIDTH_A:
            base[a[0] : a[1]] += PULSE_AMP
        if _lead <= phase < _lead + PULSE_WIDTH_A:
            base[b[0] : b[1]] += PULSE_AMP
        return base

    return _stim


def cue_stim(a: tuple[int, int]):
    """M2 utility: drive A alone, pure noise elsewhere."""
    _rng = np.random.default_rng(SEED)
    _scale = np.ones(N)
    _scale[:N_EXC] = 5.0
    _scale[N_EXC:] = 2.0

    def _stim(t: float, n_neurons: int):
        base = _scale * _rng.standard_normal(N)
        if int(t) % PULSE_PERIOD < PULSE_WIDTH_A:
            base[a[0] : a[1]] += PULSE_AMP
        return base

    return _stim


# ---- Experiment 1 ----------------------------------------------------------
def experiment_1(T_ms: int = 30000) -> dict:
    population, synapses = make_network()
    rec = simulate(population, synapses, T_ms=T_ms, engine="sparse", learning=True)
    rates = rec.mean_rates_hz()
    exc = synapses.weights[: synapses.n_excit * synapses.out_degree]

    seg = 1000
    counts, _ = np.histogram(rec.times_ms, bins=int(T_ms / seg), range=(0, T_ms))
    seg_rate = counts / 1.0 / N

    bins = np.arange(0.0, 1.01, 0.02)
    hist, _ = np.histogram(exc, bins=bins)

    return {
        "T_ms": T_ms,
        "spikes": int(rec.times_ms.size),
        "mean_exc_hz": float(rates[:N_EXC].mean()),
        "mean_inh_hz": float(rates[N_EXC:].mean()),
        "seg_min_hz": float(seg_rate.min()),
        "seg_max_hz": float(seg_rate.max()),
        "w_mean": float(exc.mean()),
        "w_min": float(exc.min()),
        "w_max": float(exc.max()),
        "w_frac_zero": float((exc < 1e-6).mean()),
        "w_frac_max": float((exc > 1 - 1e-6).mean()),
        "hist": hist.tolist(),
        "bins": bins.tolist(),
        "finite": bool(
            np.all(np.isfinite(population.v)) and np.all(np.isfinite(population.u))
        ),
    }


# ---- Experiment 2 ----------------------------------------------------------
def experiment_2(T_train_ms: int = 20000, T_test_ms: int = 2000) -> dict:
    population, synapses = make_network()
    train_stim = pair_pulse_stim(GROUP_A, GROUP_B)  # A+B co-driven, learning on
    simulate(
        population,
        synapses,
        T_ms=T_train_ms,
        engine="sparse",
        learning=True,
        stimulus_fn=train_stim,
    )
    ab_train = exc_mean(synapses, GROUP_A, GROUP_B)
    ac_train = exc_mean(synapses, GROUP_A, GROUP_C)

    # Frozen test phase: pulses keep co-driving A+B but plasticity is off.
    test_stim = pair_pulse_stim(GROUP_A, GROUP_B)
    rec = simulate(
        population,
        synapses,
        T_ms=T_test_ms,
        engine="sparse",
        learning=True,
        stimulus_fn=test_stim,
        freeze_at_ms=0,
    )
    ab_test = exc_mean(synapses, GROUP_A, GROUP_B)
    ac_test = exc_mean(synapses, GROUP_A, GROUP_C)
    ratio = ab_test / ac_test if ac_test > 0 else float("inf")
    return {
        "ab_train": ab_train,
        "ac_train": ac_train,
        "ab_test": ab_test,
        "ac_test": ac_test,
        "ratio": float(ratio),
        "test_spikes": int(rec.times_ms.size),
    }


# ---- Experiment 3 ----------------------------------------------------------
def _cue_b_count(population, synapses, T_cue_ms: int, a: tuple[int, int]) -> int:
    """Spikes from B neurons inside the 50 ms after each A pulse, learning frozen."""
    cue_st = cue_stim(a)
    rec = simulate(
        population,
        synapses,
        T_ms=T_cue_ms,
        engine="sparse",
        learning=True,
        stimulus_fn=cue_st,
        freeze_at_ms=0,
    )
    b_sel = (rec.indices >= GROUP_B[0]) & (rec.indices < GROUP_B[1])
    n_windows = int(T_cue_ms // PULSE_PERIOD)
    total = 0
    for k in range(n_windows):
        t0 = k * PULSE_PERIOD
        total += int(((rec.times_ms >= t0) & (rec.times_ms < t0 + 50) & b_sel).sum())
    return total


def experiment_3(T_train_ms: int = 20000, T_cue_ms: int = 2000) -> dict:
    # Baseline: a freshly built, never-trained network responding to an A cue.
    pop_base, syn_base = make_network()
    baseline = _cue_b_count(pop_base, syn_base, T_cue_ms, GROUP_A)

    # Train A+B together, then probe B's response to an A-only cue (frozen).
    population, synapses = make_network()
    simulate(
        population,
        synapses,
        T_ms=T_train_ms,
        engine="sparse",
        learning=True,
        stimulus_fn=pair_pulse_stim(GROUP_A, GROUP_B),
        freeze_at_ms=0,
    )
    trained = _cue_b_count(population, synapses, T_cue_ms, GROUP_A)

    n_windows = int(T_cue_ms // PULSE_PERIOD)
    return {
        "baseline_b_in_50ms": baseline,
        "trained_b_in_50ms": trained,
        "n_pulses": n_windows,
        "fold": float(trained) / baseline if baseline > 0 else float("inf"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e1 = experiment_1()
    e2 = experiment_2()
    e3 = experiment_3()
    _save_fig(e1)
    _write_results(e1, e2, e3)
    _print_all(e1, e2, e3)


def _save_fig(e1: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.array(e1["bins"])
    hist = np.array(e1["hist"])
    ax.bar(bins[:-1], hist, width=(bins[1] - bins[0]))
    ax.set_xlabel("excitatory weight")
    ax.set_ylabel("count")
    ax.set_title("M2 E1: STDP hallmark weight distribution (N=1000, T=30 s)")
    path = OUT_DIR / "m2_hallmark_weights.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def _write_results(e1: dict, e2: dict, e3: dict) -> None:
    lines = [
        "# M2 Results -- self-organizing STDP",
        "",
        "Observed values from `benchmarks/m2_stdp_experiments.py` on the",
        "event-driven sparse engine (gain=8, out_degree=100, N=1000, seed 42).",
        "",
        "## Experiment 1 -- hallmark stability (T = 30 s)",
        f"- spikes: {e1['spikes']}",
        f"- mean excitatory rate: {e1['mean_exc_hz']:.2f} Hz",
        f"- mean inhibitory rate: {e1['mean_inh_hz']:.2f} Hz",
        f"- per-second mean-rate range: [{e1['seg_min_hz']:.2f}, {e1['seg_max_hz']:.2f}] Hz",
        (
            f"- excitatory weights: min {e1['w_min']:.3f}, max {e1['w_max']:.3f}, "
            f"mean {e1['w_mean']:.3f}"
        ),
        f"- weight fraction pinned at 0: {e1['w_frac_zero']:.3f}",
        f"- weight fraction pinned at 1: {e1['w_frac_max']:.3f}",
        f"- finite (no NaN/Inf): {e1['finite']}",
        "Weight histogram image: `notebooks/output/m2_hallmark_weights.png`.",
        "",
        "## Experiment 2 -- fire-together / wire-together (train 20 s, test 2 s)",
        f"- A->B weight after training: {e2['ab_train']:.4f}",
        f"- A->C weight after training: {e2['ac_train']:.4f}",
        f"- A->B weight (frozen test): {e2['ab_test']:.4f}",
        f"- A->C weight (frozen test): {e2['ac_test']:.4f}",
        f"- ratio A->B / A->C: {e2['ratio']:.3f} (expect > 1.5)",
        "",
        "## Experiment 3 -- cue / pattern completion (stretch)",
        (
            f"- B spikes within 50 ms of an A cue over {e3['n_pulses']} pulses: "
            f"{e3['trained_b_in_50ms']} after training vs {e3['baseline_b_in_50ms']} "
            f"before training ({e3['fold']:.2f}x)."
        ),
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {RESULTS}")


def _print_all(e1: dict, e2: dict, e3: dict) -> None:
    print("\n=== E1 hallmark ===")
    for k, v in e1.items():
        if k in ("hist", "bins"):
            continue
        print(f"  {k}: {v}")
    print("=== E2 fire-together ===")
    for k, v in e2.items():
        print(f"  {k}: {v}")
    print("=== E3 cue ===")
    for k, v in e3.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
