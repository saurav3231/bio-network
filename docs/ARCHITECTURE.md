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

### `bio_network/learning/` -- synaptic plasticity

Local learning rules. The first rule is spike-timing-dependent plasticity
(`stdp.py`): a synapse strengthens when a pre-synaptic spike precedes a
post-synaptic spike and weakens in the reverse order, following the qualitative
window of Bi & Poo 1998. Rules are local -- each synapse only needs the timing
of its own pre- and post-synaptic spikes.

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
