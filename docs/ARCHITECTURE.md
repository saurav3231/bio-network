# Architecture

This document maps each brain mechanism from the README to a planned software
module. Everything here is a plan; Milestone 1 is the only code that exists so
far. The layout is deliberately small so each mechanism can be studied,
benchmarked, and swapped independently.

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

Fast one-shot memory. The episodic store (`episodic.py`) records spike events
and network states during "wake" phases. The replay module (`replay.py`) plays
stored episodes back during a simulated "sleep" phase, giving the network a
chance to consolidate memories offline, in line with Rasch & Born 2013.

### `bio_network/modulation/` -- neuromodulation

Global, dopamine-like reward signals. A reward prediction error signal (Schultz
1997) gates how strongly the local learning rules update weights. Modulation is
a scalar signal delivered to the whole network rather than a per-synapse rule.

### `bio_network/viz/` -- plotting and dashboard

Visualization of dynamics. The raster module (`raster.py`) plots spike times
across neurons. Later milestones add firing-rate heatmaps and a live web
dashboard (M6) that shows network activity and structural growth.

### `bio_network/encoding/` (planned, M3)

Sensory encoding: converts images (or other high-dimensional inputs) into spike
trains that the spiking network consumes.

## Data flow (one cycle)

1. Input arrives from `senses` (e.g. a pixel array).
2. The `encoder` converts it to spike trains.
3. Spikes propagate through the `engine`; the scheduler dispatches events.
4. `learning` updates weights using local STDP rules.
5. `memory` records the episode; during a `sleep` phase, `replay` re-runs it.
6. `modulation` scales plasticity based on a reward prediction error signal.
7. `structural plasticity` (engine) grows or prunes connections over time.
8. `viz` renders rasters and network state for inspection.
