# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial repository scaffolding: README, ROADMAP, ARCHITECTURE, community
  files (CONTRIBUTING, CODE_OF_CONDUCT, issue/PR templates).
- Python package skeleton (`bio_network`) with empty module stubs marked
  `TODO(M1)`.
- `pyproject.toml` with NumPy and matplotlib runtime dependencies and
  pytest/black/ruff dev dependencies.
- Smoke test verifying the package imports and exposes a version string.

### Added (M1: spiking neuron engine)

- `IzhikevichPopulation` implementing the Izhikevich (2003) spiking neuron
  model, vectorized with NumPy and integrated with two 0.5 ms Euler half-steps
  (`bio_network/engine/neurons.py`).
- `RandomSynapses`: a dense, Dale-principle weight matrix with excitatory
  columns in `[0, 0.5]` and inhibitory columns in `[-1, 0]`
  (`bio_network/engine/synapses.py`).
- `scheduler.simulate` and the `SpikeRecording` container with per-neuron
  mean-rate computation, plus a default thalamic-noise stimulus
  (`bio_network/engine/scheduler.py`).
- Raster, population-rate, and voltage-trace plotting for the Agg backend
  (`bio_network/viz/raster.py`).
- `notebooks/m1_first_spikes.ipynb` demonstration notebook with rendered
  outputs and saved figures in `notebooks/output/`.
- Unit, integration, and visualization tests (`tests/test_neurons.py`,
  `tests/test_network.py`, `tests/test_viz.py`).

### Added (event-driven sparse synapses)

- `SparseSynapses`: a sparse, event-driven synaptic engine with axonal
  transmission delays (`bio_network/engine/synapses_sparse.py`). CSR-style
  flat arrays, a `(max_delay, n_neurons)` ring buffer, `O(fired x out_degree)`
  delivery, and delays per Izhikevich (2006).
- `scheduler.simulate(..., engine="sparse")`: a drop-in alternative to the
  dense reference with statistically equivalent behavior.
- `benchmarks/m1_dense_vs_sparse.py`: wall-time, peak-RAM, and firing-rate
  comparison across N = 1k/10k/50k.
- Sparse engine tests (`tests/test_sparse.py`): delay correctness,
  determinism, Dale's principle, sparsity, statistical equivalence with dense,
  a 50k-neuron memory ceiling, and NaN/Inf checks.
- `docs/M1_RESULTS.md` recording the verified M1 baseline.

### Fixed (sparse gain calibration)

- Added the `gain` parameter to `SparseSynapses`, scaling excitatory weights
  at construction. Calibrated `gain = 10.0` at `out_degree=100` reproduces the
  dense baseline mean rate (9.70 Hz vs 9.76 Hz).
- Documented the fan-in dependence of mean firing rate (balanced-network
  theory, van Vreeswijk & Sompolinsky 1998) in `docs/ARCHITECTURE.md`.
- Added `docs/M1_5_RESULTS.md` recording the gain sweep and the dense N=10k
  saturation event ("dense fan-in scaling trap", first recorded pathology).

### Added (M2: self-organizing STDP learning)

- STDP on the sparse event-driven engine (`SparseSynapses.enable_learning`):
  arrival-time causality (the booked arrival ledger applies LTD and the
  pre-trace at the exact arrival millisecond), exact lazy exponential traces
  (`tau_plus = tau_minus = 20 ms`), asymmetric amplitudes (`A_plus = 0.10`,
  `A_minus = 0.12`), hard `[0, 1]` bounds, and excitatory-only plasticity with
  frozen inhibitory weights (Song-Miller-Abbott stability).
- `simulate(..., learning=True, freeze_at_ms=None)`: STDP toggle and train/test
  freeze (`bio_network/engine/scheduler.py`).
- Micro- and mesoscopic unit tests (`tests/test_stdp.py`): causal/non-causal
  windows, gate dependence, bounds, bit-identical no-learning control,
  reproducibility, stability, and frozen-inhibitory guarantee.
- Experiments (`benchmarks/m2_stdp_experiments.py`): hallmark stability
  (E1), fire-together / wire-together (E2, ratio 1.72 > 1.5), and cue /
  pattern-completion (E3, honestly negative: ~1.02x).
- `docs/M2_RESULTS.md` and `notebooks/m2_fire_together.ipynb` (with executed
  outputs) and `notebooks/output/m2_hallmark.png`. STDP documented in
  `docs/ARCHITECTURE.md`; M2 marked Complete in `ROADMAP.md`.

### Added (M4: episodic memory + sleep replay)

- `EpisodicStore` (`bio_network/memory/episodic.py`): bounded, FIFO one-shot
  verbatim spike-pattern memory with exact round trip, time-sorted listing,
  and eviction.
- `ReplayEngine` (`bio_network/memory/replay.py`): turns episodes into
  `(time_ms, neuron_id)` replay timetables with optional time compression and
  multi-copy sleep-phase plans.
- Sleep phase in `scheduler.simulate`: `phase="sleep"` attenuates background
  drive (`sleep_noise_scale`), injects replay pulses, leaves STDP ON, and
  rejects the dense engine (sparse-only).
- `SparseSynapses.save_state/load_state` and `IzhikevichPopulation.save_state/
  load_state` for bit-identical checkpoint / restore (two-armed experiments,
  snapshot-after-training).
- M4 experiments (`benchmarks/m4_stdp_experiments.py`): E4a recall after sleep
  (post/pre 1.407 > 1.3), E4b continual learning (SLEEP vs NOSLEEP arm,
  retention advantage 2.45x > 1.25).
- `docs/M4_RESULTS.md`, `docs/ARCHITECTURE.md` (episodic/sleep-replay section
  with compressed-replay biology citations), ROADMAP M4 Complete,
  `notebooks/m4_sleep_consolidation.ipynb` (executed, with
  `notebooks/output/m4_e4a_recall.png`, `m4_e4b_continual.png`).
