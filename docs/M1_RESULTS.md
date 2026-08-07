# M1 Results: Verified Baseline

Milestone 1 shipped a working spiking neuron engine based on the Izhikevich
(2003) model, vectorized with NumPy, plus spike raster visualization and a
demonstration notebook. This file records the verified baseline that all
future engines must reproduce.

## Golden reference

The **dense engine** (`bio_network.engine.synapses.RandomSynapses` +
`scheduler.simulate(..., engine="dense")`) is the golden reference. It is
deliberately left untouched; every alternative synaptic engine is expected to
match it **statistically** (rates, ISI distributions, rhythmicity), never
spike-for-spike, because the network dynamics are chaotic and summation order
differs between implementations.

- Test suite: 16/16 passing (neurons, network integration, visualization).
- Seed: 42 (default) for the whole network and stimulus.

## Measured behavior (N = 1000, T = 1000 ms, seed = 42)

| Metric | Value |
|--------|-------|
| Total spikes | 9737 |
| Mean excitatory rate | 9.76 Hz |
| Mean inhibitory rate | 9.64 Hz |
| Maximum rate | 14 Hz |
| Population rhythm | ~3 Hz, clear UP/DOWN alternation (bursts separated by near-silent epochs) |

The population-rate series (10 ms bins) shows strong rhythmic bursting: bins
alternate between ~0 Hz and ~85-95 Hz, with a standard deviation far above the
noise floor.

## Acceptance criteria for future engines

An alternative engine passes the wind tunnel if, for N = 1000, T = 1000 ms,
seed = 42:

1. Mean excitatory rate within 25 % of the dense baseline (9.76 Hz), and both
   rates in the active regime 0.5..60 Hz.
2. Both population-rate series are non-flat (std > 5 Hz), i.e. the network is
   neither silent nor saturated.
3. No NaN/Inf in the neuron state (`v`, `u`).

The event-driven sparse engine (`SparseSynapses`, `engine="sparse"`) satisfies
these criteria when given an equal synaptic budget (see the statistical
equivalence test in `tests/test_sparse.py`). Note that the default sparse
connectivity (`out_degree=100`) is deliberately cheaper and quieter; that
sparseness is a feature of the engine, not a regression of the baseline.
