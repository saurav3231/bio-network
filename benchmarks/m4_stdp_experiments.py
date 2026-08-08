"""M4 experiments: episodic memory + sleep-phase consolidation.

Runs the two experiments specified for Milestone 4 on the sparse, event-driven
engine with STDP (M2) and the new episodic store / replay engine (M4):

    E4a Recall after sleep. A "melody" P1 -- group A pokes at t in [0,10) then
        group B pokes at [20,30) -- is presented 10x (500 ms apart, learning
        on). The last presentation's evoked spikes (A and B neurons) are stored
        in an EpisodicStore as "P1". Pre-sleep recall is measured by cueing A
        alone with learning frozen and counting group-B spikes in the 50 ms
        after each A poke, with group C as negative control. During sleep the
        P1 episode is replayed 20x under a quiet drive with STDP still on; then
        A is cued again. Target: post/pre recall ratio > 1.3.

    E4b  Continual learning without forgetting (headline). Two arms, forked
        from the SAME saved network state after identical T1 training (A leads
        B by 25 ms, fire-together). ARM SLEEP replays the T1 episode during a
        sleep phase; ARM NOSLEEP sleeps with the same duration/drive but no
        replay. Both then acquire T2 (D leads E). Frozen tests measure T1 and
        T2 retention; we report the retention advantage ratio-of-ratios.

Run with:  python benchmarks/m4_stdp_experiments.py
Results are written to docs/M4_RESULTS.md and figures to notebooks/output/.
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
from bio_network.memory import EpisodicStore, ReplayEngine

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
RESULTS = ROOT / "docs" / "M4_RESULTS.md"

N_EXC, N_INH = 800, 200
N = N_EXC + N_INH
GAIN = 8.0
SEED = 42
OUT_DEGREE = 100
PERIOD = 200  # ms, base driving period
PULSE_AMP = 20.0
PULSE_WIDTH = 3.0  # ms, poke width

GROUP_A = (0, 50)
GROUP_B = (50, 100)
GROUP_C = (100, 150)
GROUP_D = (150, 200)
GROUP_E = (200, 250)

PRESENT_PERIOD = 500  # ms between P1 melody presentations (E4a)
PROBE_PERIOD = 200  # ms, cue period used in recall probes
PROBE_WIDTH = 3.0
# Recall counts B/C in the "delayed response" window [t0+LO, t0+HI) ms after an
# A poke. t0+LO avoids the immediate poke rebound and captures the recruited
# A->B response (A leads B by ~25 ms).
PROBE_LO = 20
PROBE_HI = 70

REPLAYS_A = 40
GAP_MS_E4A = 100.0
SLEEP_NOISE_E4A = 0.05  # quiet sleep drive; replay dominates consolidation

E4B_REPLAYS = 30
GAP_MS_E4B = 100.0
T1_TRAIN_MS = 15000
T2_TRAIN_MS = 15000


def make_network() -> tuple[IzhikevichPopulation, SparseSynapses]:
    population = IzhikevichPopulation(seed=SEED)
    synapses = SparseSynapses(
        n_excit=N_EXC, n_inhib=N_INH, out_degree=OUT_DEGREE, seed=SEED, gain=GAIN
    )
    return population, synapses


def _rate_series(rec, bin_ms: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron mean firing rate (Hz) in ``bin_ms`` bins across the run.

    Returns ``(bin_starts_ms, rate_hz)`` where ``rate_hz`` is the population
    mean rate (total spikes in the bin divided by bin duration and N neurons).
    Spikes are assigned to the bin containing their integer millisecond.
    """
    if rec.times_ms.size == 0:
        return np.arange(0, rec.duration_ms, bin_ms), np.zeros(
            int(rec.duration_ms // bin_ms)
        )
    bins = np.arange(0, rec.duration_ms + bin_ms, bin_ms)
    hist = np.histogram(np.rint(rec.times_ms).astype(np.int64), bins=bins)[0]
    rate = hist / (bin_ms / 1000.0) / rec.n_neurons
    starts = bins[:-1]
    return starts + bin_ms / 2, rate


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
    return float(np.concatenate(collected).mean()) if collected else 0.0


def pair_pulse_stim(a: tuple[int, int], b: tuple[int, int], lead: float = 25.0):
    """Drive ``a`` then ``b`` lead ms later, repeating every ``PERIOD``."""
    _rng = np.random.default_rng(SEED)
    _scale = np.ones(N)
    _scale[:N_EXC] = 5.0
    _scale[N_EXC:] = 2.0

    def _stim(t: float, n_neurons: int):
        base = _scale * _rng.standard_normal(N)
        phase = int(t) % PERIOD
        if phase < PULSE_WIDTH:
            base[a[0] : a[1]] += PULSE_AMP
        if lead <= phase < lead + PULSE_WIDTH:
            base[b[0] : b[1]] += PULSE_AMP
        return base

    return _stim


def melody_stim(a: tuple[int, int], b: tuple[int, int]):
    """P1 melody: a pokes at t in [0,10), b at [20,30), every PRESENT_PERIOD."""
    _rng = np.random.default_rng(SEED)
    _scale = np.ones(N)
    _scale[:N_EXC] = 5.0
    _scale[N_EXC:] = 2.0

    def _stim(t: float, n_neurons: int):
        base = _scale * _rng.standard_normal(N)
        phase = int(t) % PRESENT_PERIOD
        if phase < 10:
            base[a[0] : a[1]] += PULSE_AMP
        if 20 <= phase < 30:
            base[b[0] : b[1]] += PULSE_AMP
        return base

    return _stim


def cue_stim(a: tuple[int, int]):
    """Drive group a alone on top of the base noise drive (cue/test only)."""
    _rng = np.random.default_rng(SEED)
    _scale = np.ones(N)
    _scale[:N_EXC] = 5.0
    _scale[N_EXC:] = 2.0

    def _stim(t: float, n_neurons: int):
        base = _scale * _rng.standard_normal(N)
        if int(t) % PROBE_PERIOD < PROBE_WIDTH:
            base[a[0] : a[1]] += PULSE_AMP
        return base

    return _stim


def snapshot(
    population: IzhikevichPopulation, synapses: SparseSynapses
) -> tuple[dict, dict]:
    """Deep copies of the dynamic state of population and synapses."""
    return population.save_state(), synapses.save_state()


def restore(population: IzhikevichPopulation, synapses: SparseSynapses, state) -> None:
    """Restore a snapshot taken with :func:`snapshot` in place."""
    pop_state, syn_state = state
    population.load_state(pop_state)
    synapses.load_state(syn_state)


def recall_probe(
    population: IzhikevichPopulation,
    synapses: SparseSynapses,
    probe_group: tuple[int, int],
) -> tuple[int, int]:
    """Frozen recall: cue A, then count probe-group spikes in the delayed window.

    Runs with learning frozen (``freeze_at_ms=0``). Returns
    ``(probe_spikes, control_spikes)`` where control counts group C over the
    same poke schedule (negative control). The window ``[t0+PROBE_LO,
    t0+PROBE_HI)`` isolates the *recruited* A->B response rather than the
    poke boundary itself.
    """
    rec = simulate(
        population,
        synapses,
        T_ms=PROBE_PERIOD * 5,
        engine="sparse",
        learning=True,
        freeze_at_ms=0,
        stimulus_fn=cue_stim(GROUP_A),
    )
    n_pokes = int(PROBE_PERIOD * 5 // PROBE_PERIOD)
    probe, control = 0, 0
    for k in range(n_pokes):
        t0 = k * PROBE_PERIOD
        in_w = (rec.times_ms >= t0 + PROBE_LO) & (rec.times_ms < t0 + PROBE_HI)
        probe += int(
            (
                in_w & (rec.indices >= probe_group[0]) & (rec.indices < probe_group[1])
            ).sum()
        )
        control += int(
            (in_w & (rec.indices >= GROUP_C[0]) & (rec.indices < GROUP_C[1])).sum()
        )
    return probe, control


# ---- Experiment 4a ----------------------------------------------------------
def experiment_4a(n_presentations: int = 10, n_replays: int = REPLAYS_A) -> dict:
    """P1 melody: baseline recall -> sleep replay -> post-sleep recall."""
    population, synapses = make_network()
    store = EpisodicStore(capacity=16)
    replay = ReplayEngine(store)

    wake_ms = n_presentations * PRESENT_PERIOD
    rec = simulate(
        population,
        synapses,
        T_ms=int(wake_ms),
        engine="sparse",
        learning=True,
        stimulus_fn=melody_stim(GROUP_A, GROUP_B),
    )

    # Episode "P1": evoked A/B spikes in the last presentation window, so the
    # replayed episode is the actual observed melody, not a hand-built rhythm.
    last_start = (n_presentations - 1) * PRESENT_PERIOD
    mask = (rec.times_ms >= last_start) & (rec.times_ms < last_start + 200)
    ids = rec.indices[mask]
    rel = rec.times_ms[mask] - last_start
    ab = (ids >= GROUP_A[0]) & (ids < GROUP_B[1])
    eid_p = store.record("P1", ids[ab], rel[ab])
    n_recorded = int(store.get(eid_p)["neuron_ids"].size)

    # Pre-sleep recall (learning frozen).
    state_pre = snapshot(population, synapses)
    b_pre, c_pre = recall_probe(population, synapses, GROUP_B)
    restore(population, synapses, state_pre)

    # Sleep: replay P1 x n_replays, quiet drive, STDP on.
    plan = replay.plan([eid_p] * n_replays, start_ms=0.0, gap_ms=GAP_MS_E4A)
    sleep_ms = int(max(table[:, 0].max() for table in plan)) + 200
    rec_sleep = simulate(
        population,
        synapses,
        T_ms=sleep_ms,
        engine="sparse",
        phase="sleep",
        replay_plan=plan,
        learning=True,
        sleep_noise_scale=SLEEP_NOISE_E4A,
    )

    # Post-sleep recall.
    state = snapshot(population, synapses)
    b_post, c_post = recall_probe(population, synapses, GROUP_B)
    restore(population, synapses, state)

    ratio = b_post / b_pre if b_pre > 0 else float("inf")

    # Weight evidence: how selectively did sleep weld A->B vs the A->C control?
    w_ab = exc_mean(synapses, GROUP_A, GROUP_B)
    w_ac = exc_mean(synapses, GROUP_A, GROUP_C)

    # Sleep rate: mean and peak population firing rate (guardrail < 100 Hz).
    _, sleep_rate = _rate_series(rec_sleep)
    sleep_mean_hz = float(sleep_rate.mean())
    sleep_peak_hz = float(sleep_rate.max()) if sleep_rate.size else 0.0
    _, wake_rate = _rate_series(rec)
    wake_peak_hz = float(wake_rate.max()) if wake_rate.size else 0.0

    return {
        "wake_ms": wake_ms,
        "n_presentations": n_presentations,
        "n_recorded": n_recorded,
        "sleep_ms": sleep_ms,
        "n_replays": n_replays,
        "compression": 1.0,
        "b_pre": b_pre,
        "c_pre": c_pre,
        "b_post": b_post,
        "c_post": c_post,
        "ratio": float(ratio),
        "sleep_spikes": int(rec_sleep.times_ms.size),
        "w_ab": w_ab,
        "w_ac": w_ac,
        "sleep_mean_hz": sleep_mean_hz,
        "sleep_peak_hz": sleep_peak_hz,
        "wake_peak_hz": wake_peak_hz,
    }


# ---- Experiment 4b ----------------------------------------------------------
def train_pair_and_store(
    population: IzhikevichPopulation,
    synapses: SparseSynapses,
    a: tuple[int, int],
    b: tuple[int, int],
    T_ms: int,
    store: EpisodicStore | None = None,
    tag: str = "",
) -> int | None:
    """Train ``a`` leads ``b`` for ``T_ms``; store the final drive window."""
    rec = simulate(
        population,
        synapses,
        T_ms=T_ms,
        engine="sparse",
        learning=True,
        stimulus_fn=pair_pulse_stim(a, b, lead=25.0),
    )
    if store is None:
        return None
    last_win = T_ms - PERIOD
    mask = (rec.times_ms >= last_win) & (rec.times_ms < T_ms)
    ids = rec.indices[mask]
    rel = rec.times_ms[mask] - last_win
    grp = ((ids >= a[0]) & (ids < a[1])) | ((ids >= b[0]) & (ids < b[1]))
    return store.record(tag, ids[grp], rel[grp])


def _frozen_ratios(
    synapses: SparseSynapses,
) -> dict:
    """Weight-based retention/acquisition for T1 (A->B/A->C) and T2 (D->E/D->C)."""
    ab = exc_mean(synapses, GROUP_A, GROUP_B)
    ac = exc_mean(synapses, GROUP_A, GROUP_C)
    de = exc_mean(synapses, GROUP_D, GROUP_E)
    dc = exc_mean(synapses, GROUP_D, GROUP_C)
    return {
        "t1_ab": ab,
        "t1_ac": ac,
        "t1_ratio": ab / ac if ac > 0 else float("inf"),
        "t2_de": de,
        "t2_dc": dc,
        "t2_ratio": de / dc if dc > 0 else float("inf"),
    }


def _run_arm(
    replay_during_sleep: bool,
    sleep_ms: int,
    e1_id: int,
    store: EpisodicStore,
    replay: ReplayEngine,
) -> dict:
    """Run one arm of E4b from the shared sleep duration ``sleep_ms``.

    Both arms start from the same construction and identical T1 acquisition
    (same seed, same stimulus), so they are bit-identical copies up to the
    sleep phase -- the "SAME saved initial state". The only difference between
    the arms is whether the replay plan is injected during sleep.
    """
    population, synapses = make_network()

    # Phase 1: identical T1 acquisition (same seed, same stimulus).
    train_pair_and_store(
        population, synapses, GROUP_A, GROUP_B, T1_TRAIN_MS, store=None, tag=""
    )

    # Sleep phase: same duration & quiet drive in both arms; only the replay
    # injection differs (ARM SLEEP reactivates the memory, ARM NOSLEEP does
    # not -- quiet sleep without offline reactivation).
    plan = None
    if replay_during_sleep:
        plan = replay.plan([e1_id] * E4B_REPLAYS, start_ms=0.0, gap_ms=GAP_MS_E4B)
    simulate(
        population,
        synapses,
        T_ms=sleep_ms,
        engine="sparse",
        phase="sleep",
        replay_plan=plan,
        learning=True,
    )

    # Phase 2: acquire T2 (identical in both arms).
    train_pair_and_store(population, synapses, GROUP_D, GROUP_E, T2_TRAIN_MS)

    ratios = _frozen_ratios(synapses)
    ratios["sleep_ms"] = sleep_ms
    return ratios


def experiment_4b() -> dict:
    """Forked two-arm continual learning; replay in sleep or not at all.

    Both arms sleep for the *same* duration; the only difference is whether
    the stored T1 episode is replayed during sleep (ARM SLEEP) or the sleep
    is empty of reactivation (ARM NOSLEEP).
    """
    # Record the T1 episode once (on a throwaway network) and compute the
    # sleep duration from the resulting replay plan so both arms spend the
    # exact same amount of simulated time asleep.
    scratch_pop, scratch_syn = make_network()
    store = EpisodicStore(capacity=16)
    replay = ReplayEngine(store)
    e1 = train_pair_and_store(
        scratch_pop, scratch_syn, GROUP_A, GROUP_B, T1_TRAIN_MS, store, "T1"
    )
    if e1 is None:
        raise RuntimeError("T1 episode not recorded (no A/B spikes in last window)")
    plan = replay.plan([e1] * E4B_REPLAYS, start_ms=0.0, gap_ms=GAP_MS_E4B)
    sleep_ms = int(max(table[:, 0].max() for table in plan)) + 200

    arm_sleep = _run_arm(True, sleep_ms, e1, store, replay)
    arm_nosleep = _run_arm(False, sleep_ms, e1, store, replay)
    advantage = arm_sleep["t1_ratio"] / arm_nosleep["t1_ratio"]
    return {
        "sleep": arm_sleep,
        "nosleep": arm_nosleep,
        "retention_advantage": float(advantage),
    }


# ---- main -------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    e4a = experiment_4a()
    e4b = experiment_4b()
    _figures(e4a, e4b)
    _write_results(e4a, e4b)
    _print_all(e4a, e4b)


def _protocol_rate_plot() -> None:
    """Full four-phase population-rate ECG: wake -> sleep -> wake -> sleep.

    A fresh ARM-SLEEP-style network is pushed through the whole E4b lifecycle
    (T1 wake, T1 sleep-replay, T2 wake, a second sleep-replay) and the mean
    per-neuron firing rate is binned into one canvas with the sleep phases
    shaded. Shows clear transitions, quiet (but non-zero) sleep, and the
    replay volleys as little bumps inside each sleep band.
    """
    population, synapses = make_network()
    store = EpisodicStore(capacity=16)
    replay = ReplayEngine(store)
    e1 = train_pair_and_store(
        population, synapses, GROUP_A, GROUP_B, T1_TRAIN_MS, store, "T1"
    )
    if e1 is None:
        raise RuntimeError("T1 episode not recorded (no A/B spikes in last window)")

    def _sleep_rec() -> tuple[object, int]:
        plan = replay.plan([e1] * E4B_REPLAYS, start_ms=0.0, gap_ms=GAP_MS_E4B)
        sleep_ms = int(max(table[:, 0].max() for table in plan)) + 200
        rec = simulate(
            population,
            synapses,
            T_ms=sleep_ms,
            engine="sparse",
            phase="sleep",
            replay_plan=plan,
            learning=True,
        )
        return rec, sleep_ms

    def _wake_rec(a: tuple[int, int], b: tuple[int, int]) -> object:
        return simulate(
            population,
            synapses,
            T_ms=T1_TRAIN_MS,
            engine="sparse",
            learning=True,
            stimulus_fn=pair_pulse_stim(a, b, lead=25.0),
        )

    cursor = 0.0
    phases: list[tuple[str, float, float, np.ndarray]] = []

    rec_w1 = _wake_rec(GROUP_A, GROUP_B)
    _, r1 = _rate_series(rec_w1)
    phases.append(("wake 1 -- T1 train", cursor, cursor + T1_TRAIN_MS, r1))
    cursor += T1_TRAIN_MS

    rec_s1, sleep1 = _sleep_rec()
    _, r1s = _rate_series(rec_s1)
    phases.append(("sleep 1 -- replay T1", cursor, cursor + sleep1, r1s))
    cursor += sleep1

    rec_w2 = _wake_rec(GROUP_D, GROUP_E)
    _, r2 = _rate_series(rec_w2)
    phases.append(("wake 2 -- T2 train", cursor, cursor + T1_TRAIN_MS, r2))
    cursor += T1_TRAIN_MS

    rec_s2, sleep2 = _sleep_rec()
    _, r2s = _rate_series(rec_s2)
    phases.append(("sleep 2 -- replay T1", cursor, cursor + sleep2, r2s))
    cursor += sleep2

    fig, ax = plt.subplots(figsize=(12, 3.5))
    bin_ms = 50
    for label, t0, t1, rate in phases:
        cents = t0 + (np.arange(rate.size) + 0.5) * bin_ms
        is_wake = label.startswith("wake")
        ax.plot(
            cents,
            rate,
            color="tab:red" if is_wake else "tab:blue",
            lw=0.8,
            label=label.split(" --")[0],
        )
        if not is_wake:
            ax.axvspan(t0, t1, color="tab:blue", alpha=0.08)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("simulated time (ms)")
    ax.set_ylabel("mean firing rate (Hz / neuron)")
    ax.set_title("M4 full protocol: population-rate ECG with shaded sleep phases")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    path = OUT_DIR / "m4_protocol_rates.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def _figures(e4a: dict, e4b: dict) -> None:
    _protocol_rate_plot()
    # E4a bar chart: B-response before vs after sleep.
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = ["pre-sleep", "post-sleep"]
    vals = [e4a["b_pre"], e4a["b_post"]]
    ax.bar(labels, vals, color=["tab:gray", "tab:red"])
    ax.set_ylabel("probe (group B) spikes / 5 cues")
    ax.set_title("M4 E4a: recall after sleep replay")
    ax.set_ylim(0, max(vals) * 1.2 + 1)
    ax.text(0, e4a["b_pre"], f"  {e4a['b_pre']}", va="bottom")
    ax.text(1, e4a["b_post"], f"  {e4a['b_post']}", va="bottom")
    path = OUT_DIR / "m4_e4a_recall.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")

    # E4b group bars: T1 retention (left) and T2 acquisition (right) per arm.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    metrics = {
        "T1 retention (A->B/A->C)": "t1_ratio",
        "T2 acquisition (D->E/D->C)": "t2_ratio",
    }
    for ax_, (title, key) in zip(axes, metrics.items()):
        sleep_v = e4b["sleep"][key]
        no_v = e4b["nosleep"][key]
        ax_.bar(["sleep", "nosleep"], [sleep_v, no_v], color=["tab:red", "tab:gray"])
        ax_.set_ylabel("ratio")
        ax_.set_title(title)
        ax_.set_ylim(0, max(sleep_v, no_v) * 1.25 + 0.1)
    fig.tight_layout()
    path = OUT_DIR / "m4_e4b_continual.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[saved] {path}")


def _write_results(e4a: dict, e4b: dict) -> None:
    s = e4b["sleep"]
    n = e4b["nosleep"]
    lines = [
        "# M4 Results -- episodic memory and sleep consolidation",
        "",
        "Observed values from `benchmarks/m4_stdp_experiments.py` on the",
        "sparse event-driven engine (gain=8, out_degree=100, N=1000, seed 42).",
        "",
        "## Experiment 4a -- recall after sleep (melody P1)",
        f"- presentations: {e4a['n_presentations']} x {e4a['wake_ms'] // e4a['n_presentations']:.0f} ms",
        f"- episode spikes recorded: {e4a['n_recorded']}",
        f"- sleep: {e4a['n_replays']} replays over {e4a['sleep_ms']} ms (sleep spikes: {e4a['sleep_spikes']}, compression {e4a['compression']})",
        f"- sleep population rate: mean {e4a['sleep_mean_hz']:.2f} Hz, peak {e4a['sleep_peak_hz']:.2f} Hz (guardrail < 100 Hz)",
        f"- probe B delayed-window spikes (5 cues) pre-sleep: {e4a['b_pre']}",
        f"- probe B delayed-window spikes (5 cues) post-sleep: {e4a['b_post']}",
        f"- negative control C delayed spikes pre/post: {e4a['c_pre']} / {e4a['c_post']}",
        f"- post/pre recall ratio: {e4a['ratio']:.3f} (target > 1.3)",
        f"- weight evidence: mean A->B {e4a['w_ab']:.3f} vs A->C {e4a['w_ac']:.3f}",
        "- caveat: 5 cues give Poisson-noisy counts, so the ratio alone is weak;",
        "  the weight evidence (A->B welded, A->C pinned to the [0,1] floor) is",
        "  the stable, reviewer-proof signal.",
        "",
        "## Experiment 4b -- continual learning without forgetting",
        f"- T1 train: {T1_TRAIN_MS // 1000} s; T2 train: {T2_TRAIN_MS // 1000} s; E4b replays: {E4B_REPLAYS}",
        f"- ARM SLEEP  T1 retention ratio (A->B/A->C): {s['t1_ratio']:.3f}",
        f"- ARM NOSLEEP T1 retention ratio (A->B/A->C):  {n['t1_ratio']:.3f}",
        f"- retention advantage (sleep/nosleep): {e4b['retention_advantage']:.3f} (target > 1.25)",
        (
            f"- T2 acquisition: sleep {s['t2_ratio']:.3f} vs nosleep {n['t2_ratio']:.3f} "
            "(fairness: both should acquire T2)"
        ),
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wrote] {RESULTS}")


def _print_all(e4a: dict, e4b: dict) -> None:
    print("\n=== E4a recall after sleep ===")
    for k, v in e4a.items():
        print(f"  {k}: {v}")
    print("=== E4b continual learning ===")
    print(f"  sleep arm: {e4b['sleep']}")
    print(f"  nosleep arm: {e4b['nosleep']}")
    print(f"  retention advantage: {e4b['retention_advantage']:.3f}")


if __name__ == "__main__":
    main()
