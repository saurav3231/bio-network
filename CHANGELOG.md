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
