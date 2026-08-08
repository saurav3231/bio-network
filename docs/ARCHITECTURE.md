# Architecture

This document maps each brain mechanism from the README to a software module.
Milestones M1 through M4 are built: the dense and sparse engines, STDP
learning, and the episodic/sleep-replay memory loop all exist in `bio_network/`.
Sensory encoding (M3) now ships as `bio_network/senses/`; neuromodulation (M5)
remains planned. The layout is deliberately small so each mechanism can be
studied, benchmarked, and swapped independently.

```
+----------+     +---------+     +------------------+     +-----------+
|  senses  | --> | encoder | --> | spiking network  | --> | episodic  |
| (input)  |     |         |     |  (engine/)       |     | memory    |
+----------+     +---------+     +------------------+     | (memory/) |
                                   |    ^      ^          +-----------+
                                   |    |      |                |
                                   v    |      |                v
                              +---------+   +----------------+  |
                              | learning|   | replay loop    |--+
                              | (STDP)  |   | (sleep phase)  |
                              +---------+   +----------------+
                                   ^
                                   |
                             +------------+
                             | modulation |
                             | (reward)   |
                             +------------+

         structural plasticity grows/prunes connections over time
         neuromodulation gates how strongly learning applies
         sleep phase re-runs episodes to consolidate memory
```

## Modules

### `bio_network/engine/` -- spiking neurons and event scheduler

Event-driven simulation core. Implements the Izhikevich neuron model
(`neurons.py`) so a single neuron supports many biologically plausible spiking
regimes, plus a synapse container (`synapses.py`) that holds connection weights
and conduction delays. The scheduler (`scheduler.py`) advances simulation time
by processing spike events only when they occur, rather than stepping the whole
network at every tick. This is the performance-sensitive part and the natural
candidate for a later Rust core.

### Event-driven synaptic delivery

The dense engine in `synapses.py` computes every synapse every millisecond --
including the 95 %+ that carry no spike. Real brains are sparse and
event-driven: a synapse only matters when a spike arrives, and spikes take
1-20 ms to travel along an axon. The alternative engine
(`synapses_sparse.py`) implements this:

- Each neuron projects to `out_degree` distinct random targets, stored in
  CSR-style flat arrays (`targets`, `weights`, `delays`, per-neuron
  `offsets`). No `N x N` matrix is kept anywhere, so memory scales as
  `O(N * out_degree)` instead of `O(N^2)`.
- Dale's principle is unchanged: excitatory outgoing weights uniform in
  `[0, 0.5]`, inhibitory outgoing weights uniform in `[-1, 0]`.
- Delays follow Izhikevich (2006): excitatory synapses draw a uniform integer
  delay in 1-20 ms; inhibitory synapses are fixed at 1 ms.
- Delivery is event-driven through a preallocated ring buffer of shape
  `(max_delay, n_neurons)`: firing a neuron touches only its `out_degree`
  targets, so the cost per step is `O(fired x out_degree)`, never `O(N^2)`.

The scheduler exposes both engines through `simulate(..., engine="dense" |
"sparse")`. Trajectories are chaotic, so the two engines diverge after ~100 ms
and agree only statistically (rates, ISI distributions, rhythmicity), not
spike-for-spike.

**Fan-in controls the mean rate (balanced-network theory).** The dense engine
gives every neuron a fan-in of `~N` (it sums all fired columns); the sparse
engine gives each neuron a fan-in of only `out_degree`. In a recurrent network
the mean firing rate tracks the balance of excitation and inhibition, and that
balance must be re-normalized as the fan-in changes. In balanced network theory
(van Vreeswijk & Sompolinsky, 1998) neurons sit in a fluctuating, near-threshold
regime only when the total recurrent input grows linearly with the fan-in;
shrinking the fan-in without rescaling the weights pulls the network into a
lower-gain, quieter regime (the sparse engine idles near ~5 Hz while dense
sits near ~10 Hz). The excitatory `gain` parameter of `SparseSynapses`
re-establishes that drive: at `out_degree=100` a gain of `10.0` reproduces the
dense baseline mean rate (calibration in `docs/M1_5_RESULTS.md`).

References:

- Izhikevich, E. M. (2003). Simple model of spiking neurons.
  *IEEE Transactions on Neural Networks*, 14(6), 1569--1572.
- Izhikevich, E. M. (2006). Polychronization: computation with spikes.
  *Neural Computation*, 18(2), 245--282.
- van Vreeswijk, C., & Sompolinsky, H. (1998). Chaotic balanced state in a model
  of cortical circuits. *Neural Computation*, 10(6), 1321--1371.

### `bio_network/learning/` -- synaptic plasticity

Local learning rules. The first rule is spike-timing-dependent plasticity
(`stdp.py`): a synapse strengthens when a pre-synaptic spike precedes a
post-synaptic spike and weakens in the reverse order, following the qualitative
window of Bi & Poo 1998. Rules are local -- each synapse only needs the timing
of its own pre- and post-synaptic spikes.

### STDP (M2) -- spike-timing-dependent plasticity

STDP is implemented directly inside the sparse engine
(`SparseSynapses.enable_learning`, `synapses_sparse.py`) so the plasticity
updates run event-driven alongside spike delivery; there is no separate
`stdp.py` module, because the rule needs the engine's arrival ledger and ring
buffer to be computed on the exact spike timing.

Rule details:

- **Excitatory-only, with hard bounds.** Only outgoing excitatory synapses are
  plastic. Weights are clamped to `[0, 1]` (normalized bounds of Song, Miller
  & Abbott 2000) before the `gain` scale is applied downstream. Inhibitory
  weights are frozen: E/I balance is what keeps the recurrent network stable,
  and letting inhibition grow is the classic road to runaway activity.
- **Asymmetric amplitudes.** `A_plus = 0.10` (LTP) and `A_minus = 0.12`
  (LTD). Depression slightly stronger than potentiation is the Song-Miller-
  Abbott stability mechanism: in a steady state the potentiation events
  slightly outnumber the depression events (pairing-rate asymmetry), and the
  `A_minus > A_plus` tilt keeps the mean weight bounded away from the ceiling.
- **Exact lazy traces.** Every plastic synapse keeps a pre-trace
  (`syn_trace`) and every neuron a post-trace (`post_trace`), each with a
  "last updated" timestamp. Traces decay exponentially with time constants
  `tau_plus = tau_minus = 20 ms` but the exponential is only evaluated at the
  moment the trace is *used*, never every millisecond (event-driven).
- **Causality is measured at arrival time.** A synapse is potentiated or
  depressed based on the **arrival** of the pre-synaptic spike at the
  post-synaptic neuron (after the axonal delay), not the emission time. This
  matters: an axon's conduction delay is 1-20 ms, which is of the same order
  as the STDP window, so emission-time causality would systematically mislabel
  many pre/post orderings. The engine books every fired excitatory synapse
  into an arrival ledger at delivery time and applies LTD + the pre-trace
  increment at the exact arrival millisecond; LTP and the post-trace refresh
  run when the post-synaptic neuron fires.
- **Learning toggle and freeze.** `simulate(..., learning=True)` enables STDP;
  `freeze_at_ms` stops weight updates after a given simulated time while
  spikes keep propagating, which provides the train/test split used by the M2
  experiments (training phase, then a frozen test phase).
- **Gain = 8 for all M2 training.** The calibrated sparse engine runs near
  ~10 Hz at `gain = 10`; learning uses `gain = 8` so LTP has headroom below
  the ~160 Hz avalanche regime (see `docs/M1_5_RESULTS.md` for the fan-in /
  gain discussion).

Experiment results are recorded in `docs/M2_RESULTS.md`; the running notebook
is `notebooks/m2_fire_together.ipynb`.

References:

- Bi, G., & Poo, M.-m. (1998). Synaptic modifications in cultured hippocampal
  neurons: dependence on spike timing, synaptic strength, and postsynaptic
  cell type. *Journal of Neuroscience*, 18(24), 10464--10472.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning
  through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*,
  3(9), 919--926.
- van Vreeswijk, C., & Sompolinsky, H. (1998). Chaotic balanced state in a
  model of cortical circuits. *Neural Computation*, 10(6), 1321--1371.

### `bio_network/memory/` -- episodic store and replay

Fast one-shot memory, realized in M4. The episodic store (`episodic.py`)
records spike events and network states during "wake" phases. The replay
module (`replay.py`) plays stored episodes back during a simulated "sleep"
phase, giving the network a chance to consolidate memories offline.

### Episodic store and replay (M4)

Two modules, both deliberately small and honest about what they model:

- **`EpisodicStore`** (`episodic.py`) is a bounded, FIFO one-shot memory. An
  episode is a verbatim spike pattern -- `(tag, neuron_ids, rel_times_ms)` --
  with no learned compression. `record()` accepts a spike burst; `get()` returns
  it unchanged (round trip is exact); `all()` lists stored episodes. Sorted by
  time on replay. Capacity is a constructor parameter with FIFO eviction of the
  oldest episode (an unreplayed memory's natural fate).
- **`ReplayEngine`** (`replay.py`) turns an episode into a `(time_ms,
  neuron_id)` replay timetable. `schedule(episode_id, start_ms,
  compression=1.0)` shifts the episode in time and optionally time-compresses it;
  `plan(episode_ids, start_ms, gap_ms, compression)` loops many episode copies
  into a single timetable -- the sleep-phase "replay sequence".

The scheduler gained a real **sleep phase** (`simulate(..., phase="sleep",
replay_plan=..., sleep_noise_scale=0.25)`): external background drive is
attenuated by `sleep_noise_scale`, replay pulses are injected as current onto
the implicated neurons, and STDP stays ON so the replayed pattern is
consolidated into the *same* recurrent weights used during wake. The dense
engine does not implement sleep and rejects `phase="sleep"` with a ValueError;
only the sparse event-driven engine (the one with STDP) supports it.

**Compressed replay is the biological anchor.** Hippocampal replay in slow-wave
sleep runs 5-20x faster than the original experience (Wilson & McNaughton 1994;
Diba & Buzsaki 2007). The `compression` parameter is the faithful knob: M4
runs at `compression=1.0` (untimed replay) to keep the consolidation mechanism
itself the experimental variable, but the engine and the replay timetable both
support compressed replay.

**Why replay stays in the same weights.** M4 does not implement a separate
hippocampus-to-cortex weight copy (two-pathway consolidation such as Rasch &
Born 2013). Instead, sleep replay drives the *same* plastic excitatory synapses
that learned during wake. This is the simplest mechanism that turns the wake
episode into relaunch quantum: replay re-runs the pattern under the same STDP
rule, and the replayed spike timing structurally strengthen the same
association. The `episodic.py`/`replay.py` split is kept so a separate
consolidation path can be added later without touching the engine.

**`simulate` also exposes `save_state()` / `load_state()`** on
`IzhikevichPopulation` (v, u) and `SparseSynapses` (weights, spike queue,
learning ledger, sorted spike times, traces, and arrival ledger), in part for
the M4 continual-learning experiment, which needs to start two arms from the
identical post-training state.

Experiment results are recorded in `docs/M4_RESULTS.md`; the running notebook
is `notebooks/m4_sleep_consolidation.ipynb`.

References:

- Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal
  ensemble memories during sleep. *Science*, 265(5172), 676--679.
- Diba, K., & Buzsaki, G. (2007). Forward and reverse hippocampal place-cell
  sequences during ripples. *Nature Neuroscience*, 10(10), 1241--1242.
- Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiological
  Reviews*, 93(2), 681--766.

### `bio_network/modulation/` -- neuromodulation

Global, dopamine-like reward signals. A reward prediction error signal (Schultz
1997) gates how strongly the local learning rules update weights. Modulation is
a scalar signal delivered to the whole network rather than a per-synapse rule.

### `bio_network/viz/` -- plotting and dashboard

Visualization of dynamics. The raster module (`raster.py`) plots spike times
across neurons. Later milestones add firing-rate heatmaps and a live web
dashboard (M6) that shows network activity and structural growth.

### `bio_network/senses/` -- sensory encoding (M3)

The artificial retina and input pathway that turn images into the spike
timetables the sparse engine consumes. Four small modules:

- **`retina.py` -- `Retina`**: converts a 28x28 intensity image (values in
  [0, 1]) into a sorted `(t_ms, pixel_index)` spike timetable. Two deterministic
  coding modes: **latency** (one spike per bright pixel at
  `t = (1 - intensity) * window_ms`, so brighter pixels fire earlier) and
  **rate** (a Poisson train at `intensity * max_rate_hz`). Determinism comes
  from an explicit `np.random.default_rng(seed)`; `encode()` is the only entry
  point.
- **`projections.py` -- `InputProjection`**: by default a **frozen** random
  fan-out from pixels to neurons: each pixel connects to `fanout` distinct
  target neurons and the `targets` array is generated once and never updated
  (`drive_neurons(pixels)` maps a pixel list to the neurons a stimulus should
  inject; `fan_in_stats()` reports the per-neuron input fan-in). Since M3.2 the
  projection may instead be **plastic** (`plastic=True`): it carries per-edge
  input synapses `w_in in [0, 1]` (init uniform 0.2-0.4) that learn with the
  same arrival-time STDP as the recurrent engine (tau 20 ms, A+ 0.10,
A- 0.12; Song-Miller-Abbott bounds) -- the firing-side LTP runs through an
  incoming-edge reverse adjacency built once at construction, and a per-neuron
  homeostatic target keeps each neuron's total input power equal to its fan-in
  so the plastic arm's drive matches the frozen arm's. `set_learning()` gates
  plasticity to the training phase; `drive_weights()` is a vector of ones when
  frozen and the (homeostatically balanced) `w_in` when plastic.
- **Homeostatic regulation (M3.3)**. Two biological regulators keep the plastic
  pathway from starving (the M3.2 diagnosis: LTD on every arrival outpaces LTP
  that needs a real spike, so `w_in` drains toward zero):
  - *Synaptic scaling* (`synaptic_scaling=True`, Turrigiano et al. 1998). After
    every training image window ``RetinaStimulus`` calls
    ``InputProjection.synaptic_scale()``, a fixed-point loop that renormalizes
    each neuron's incoming weights so ``sum(w_in) == n_in_per_neuron * 0.30``
    (the STDP init mean, so day-one total power is preserved and Scaling only
    reallocates correlation structure within a constant budget). The loop keeps
    every weight within ``[0, 1]`` by scale-then-clamp-then-rescale until the
    target is hit to 1e-9 (clamping to the writer-pinned bound). Scaling is
    gated to the training phase only (`_learning`), exactly like STDP.
  - *Adaptive spike thresholds* (`adaptive_thresholds=True` on
    ``IzhikevichPopulation``, Diehl & Cook 2015 intrinsic plasticity). Each
    excitatory neuron tracks a per-ms exponential rate low-pass
    (``rate_tau_ms=2000``, ``target_rate_hz=5``) and drifts its firing
    threshold ``theta`` by ``theta_gain * (ema - target_ema)`` clipped to
    ``[theta_min, theta_max] = [1, 30]`` mV -- over-active neurons self-limit,
    silent ones are recruited. Inhibitory neurons keep the canonical threshold.
    ``save_state/load_state`` preserve ``theta`` and the rate estimate; when
    disabled ``theta == 30 == _THRESHOLD_MV`` so M1-M3.2 behaviour is
    bit-identical.
  - *Structural constraint* (`excitatory_only=True`): the fan-out draws input
    targets exclusively from the excitatory population so no input synapse
    lands on an inhibitory interneuron.
- **`stimulus.py` -- `RetinaStimulus`**: a drop-in `stimulus_fn` for
  `simulate(...)`. Each image owns a time slot of `window_ms + gap_ms`; within
  the window it emits current pulses to the pixels' target neurons (scaled by
  `drive_weights` when the projection is plastic), and it is silent during the
  gap. `slot_boundaries(slot)` returns the slot's [t0, t1)
  and `__len__` the number of images, so post-hoc per-image spike counts are
  exact (this is how response matrices are built). `set_learning()` propagates
  the learning toggle so the arrival side of input STDP fires exactly on each
  pixel spike's arrival millisecond. At the start of each new slot it applies
  per-window synaptic scaling when training (M3.3).
- **`readout.py` -- `LabelsReadout`**: the label-scoped decoder. `fit()` builds
  a per-class mean response fingerprint from the responses of the *training*
  split only; `predict()` scores a trial with the frozen fingerprints. Labels
  never predict the weights: the readout reads *responses*, and fitting happens
  after learning is frozen. `predict_vote()` is the second pre-committed
  decoder (hard per-neuron plurality vote over the frozen assignment), reported
  alongside `predict()` in the M3.2 and M3.3 arms.

Two honesty rules are structural: plasticity is **phase-gated** (input and
recurrent STDP are on only during training; both parents freeze before
assignment and test) and **labels never touch the recurrent weights** -- M3 is
"unsupervised feature emergence," not supervised tuning. Results are recorded
in `docs/M3_RESULTS.md`, `docs/M3_2_RESULTS.md` and `docs/M3_3_RESULTS.md`; the
running notebooks are `notebooks/m3_first_light.ipynb`,
`notebooks/m32_plastic_optic_nerve.ipynb` and
`notebooks/m33_critical_period.ipynb`.

References:

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition
  using spike-timing-dependent plasticity. *Frontiers in Computational
  Neuroscience*, 9, 99.
- Masquelier, T., & Thorpe, S. J. (2007). Unsupervised learning of visual
  features through spike timing dependent plasticity. *PLoS Computational
  Biology*, 3(2), e31.

### `bio_network/viz/` -- plotting and dashboard

Visualization of dynamics. The raster module (`raster.py`) plots spike times
across neurons. Later milestones add firing-rate heatmaps and a live web
dashboard (M6) that shows network activity and structural growth.

## Data flow (one cycle)

1. Input arrives from `senses` (e.g. a pixel array).
2. The `encoder` (the `Retina` in `senses/`) converts it to spike trains.
3. Spikes propagate through the `engine`; the scheduler dispatches events.
4. `learning` updates weights using local STDP rules.
5. `memory` records the episode; during a `sleep` phase, `replay` re-runs it.
6. `modulation` scales plasticity based on a reward prediction error signal.
7. `structural plasticity` (engine) grows or prunes connections over time.
8. `viz` renders rasters and network state for inspection.
