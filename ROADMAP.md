# Roadmap

Milestone-based development plan for Bio Network. Each milestone is independently
useful and verifiable. Statuses: `Current` (in active development), `Planned`
(not started), `Stretch` (nice to have, may not ship).

## M1 -- Spiking Neuron Engine (Complete)

A minimal but solid spiking neural network core.

- Izhikevich neuron model (`bio_network/engine/neurons.py`).
- Event-driven scheduler that advances the network only when spikes occur
  (`bio_network/engine/scheduler.py`).
- Synapse container and spike-delay handling (`bio_network/engine/synapses.py`).
- Spike raster visualization (`bio_network/viz/raster.py`).
- Validation against published Izhikevich dynamics (regular spiking, fast
  spiking, chattering).

**Exit criteria:** run a small recurrent network, render a raster plot, and
reproduce known spiking regimes from Izhikevich 2003.

**Status:** Complete. Spike conduction delays remain a planned follow-up.

## M2 -- Self-Organizing Learning (Complete)

- Spike-timing-dependent plasticity (STDP) local learning rule, event-driven on
  the sparse engine (`SparseSynapses.enable_learning`; see `docs/ARCHITECTURE.md`).
- Unsupervised structure emergence: inputs self-organize into stable, selective
  response patterns without labels.

**Exit criteria:** run a recurrent network, render a raster plot, and
reproduce known spiking regimes from Izhikevich 2003.

**Status:** Complete. STDP produces a stable, bimodal weight distribution
(E1) and a "fire-together, wire-together" association (E2, mean A->B vs A->C
ratio 1.72 > 1.5) in `docs/M2_RESULTS.md`. Cue/pattern completion (E3) was
explored and honestly reported as absent under the current gain/layout.

## M4 -- Memory + Sleep Phase (Complete)

- One-shot episodic store (`bio_network/memory/episodic.py`).
- Offline replay of stored episodes (`bio_network/memory/replay.py`).
- Sleep/consolidation loop that replays and strengthens recent memories while
  learning new ones.

**Exit criteria:** on the sparse engine with STDP, replaying a stored wake
episode during sleep measurably strengthens the learned association, and
learning a second association afterward does not erase the first.

**Status:** Complete. 48 tests pass. Sleep replay of the P1 melody lifts
recall after a cue (E4a, post/pre 1.407 > 1.3) in `docs/M4_RESULTS.md`, and
the replay arm retains T1 after learning T2 (E4b, retention advantage 2.49x >
1.25) while both arms acquire T2 equally. See `notebooks/m4_sleep_consolidation.ipynb`.

## M3 -- Sensory Encoding (Complete)

- Convert images to spike trains (`bio_network/senses/`: `Retina` encoder,
  `InputProjection` fan-out, `RetinaStimulus` presenter).
- Unsupervised pattern recognition: the network clusters or discriminates input
  classes from spike statistics alone (`LabelsReadout` on frozen responses).

**Status:** Complete. The retina encoder is exact (E3a, latency Spearman 1.000,
rate Pearson 0.999); unsupervised STDP on a frozen input projection creates
class-selective feature neurons (E3b, selective 0 -> 44) and a frozen
label-scoped readout decodes held-out digits at 0.20 vs chance 0.10 and kNN
0.60 (E3c) in `docs/M3_RESULTS.md`. 61 tests pass. The input projection is
intentionally non-plastic; a learned input weight matrix (Diehl & Cook 2015;
Masquelier & Thorpe 2007) is the planned v2 upgrade. See
`notebooks/m3_first_light.ipynb`.

### M3.2 -- Plastic Input Projection ("optic nerve") (Complete, honest negative)

Make the 784 -> neuron input cable learnable (`w_in` STDP, tau 20 ms,
A+ 0.10, A- 0.12, init uniform 0.2-0.4, plastic path only during training,
frozen at assignment/test) and test causally whether input plasticity creates
receptive fields and better recognition.

**Status:** Complete as a *controlled marginal/negative result*: plastic drive
moves held-out accuracy 11% -> 15% soft / 10% -> 16% vote against the identical
frozen control, but contextualized honestly -- with A- 0.12 applied on every
arrival vs A+ 0.10 requiring an actual spike, the plastic arm starves the
population (45k vs 2.6 M train spikes, 0.09 vs 5.24 Hz) and forms no localized
tiles. See `docs/M3_2_RESULTS.md` and `notebooks/m32_plastic_optic_nerve.ipynb`.
A balanced homeostatic update rule is the documented follow-up.

### M3.3 -- Homeostatic Regulators ("critical period") (Complete, partial positive)

Fix the M3.2 starvation by adding the two classic biological regulators to the
plastic optic nerve during training: **synaptic scaling** (Turrigiano et al.
1998: after every training window each neuron's incoming `w_in` is re-pinned to
`sum(w_in) == n_in * 0.30`, its day-one power) and **adaptive spike thresholds**
(Diehl & Cook 2015 intrinsic plasticity: excitatory neurons drift their firing
threshold within [1, 30] mV toward a 5 Hz target so over-active neurons back
off and silent ones are recruited), plus the excitatory-only structural
constraint.

**Status:** Complete as an honest *partial positive*. ARM B beats the
byte-identical M3.2 baseline on every axis: held-out accuracy 15% -> 39% soft /
16% -> 34% vote, active neurons 76 -> 342/1000, mean probe rate 0.09 -> 0.17 Hz,
structured RIA tiles 3 -> 110, with the stability box held (w_in in [0, 1],
theta in [1, 30], finite recurrent weights, inhibitory frozen). The brain wakes
up but is not fully awake -- rate stays below the 2 Hz aspiration and only
~1/3 of the population is recruited. See `docs/M3_3_RESULTS.md` and
`notebooks/m33_critical_period.ipynb`. A rate-dense / higher-drive variant
(greater scaling budget or faster threshold recruitment) is the documented
follow-up.

## M5 -- Neuromodulation

- Dopamine-like reward prediction error signal (`bio_network/modulation/`).
- Reward-gated plasticity: learning strength scales with the modulatory signal.

## M6 -- Live Dashboard

- Web UI (`viz/`) with live spike raster, firing-rate heatmaps, and network
  growth/pruning plots.

## Stretch -- C. elegans Connectome

- Run the engine on the openly published 302-neuron connectome of *C. elegans*
  (WormAtlas / White et al. 1986 connectivity data).
- Show that structural plasticity and modulation produce qualitatively
  interesting dynamics on a real biological graph.
