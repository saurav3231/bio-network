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

## M3 -- Sensory Encoding

- Convert images to spike trains (`bio_network/encoding/`).
- Unsupervised pattern recognition: the network clusters or discriminates input
  classes from spike statistics alone.

## M4 -- Memory + Sleep Phase

- One-shot episodic store (`bio_network/memory/episodic.py`).
- Offline replay of stored episodes (`bio_network/memory/replay.py`).
- Sleep/consolidation loop that replays and strengthens recent memories while
  protecting older ones.

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
