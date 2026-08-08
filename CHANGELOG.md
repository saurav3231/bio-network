# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-08

### Added (M3.2: plastic input projection / optic-nerve STDP)

- `InputProjection` learns (`plastic=True`): per-channel input synapses
  `w_in in [0, 1]` initialized uniform 0.2-0.4 (`bio_network/senses/projections.py`)
  with M2-style arrival-time STDP (tau 20 ms, A+ 0.10, A- 0.12), arrival-side
  LTD `on_input_arrival`, firing-side LTP `on_neurons_fired`, reverse
  (incoming-edge) adjacency for O(fired x fan-in) LTP, `set_learning` gates for
  the train/freeze boundary, and a per-neuron homeostatic power target that
  matches the frozen arm's drive. `plastic=False` is byte-for-byte the frozen
  v1 pathway (unit drive, motherboard `drive_weights` = ones).
- `RetinaStimulus` scales pulses by `w_in` when plastic and triggers
  arrival-side STDP at the exact ms of each pixel spike; `set_learning`
  propagates the training-phase toggle
  (`bio_network/senses/stimulus.py`).
- `scheduler.simulate` gains an optional `input_plastic_fn(fired, t, learn_now)`
  hook applied post-firing in the sparse engine (default None: M1-M4 behavior
  bit-identical) and toggles `stimulus.set_learning` in lock-step
  (`bio_network/engine/scheduler.py`).
- `LabelsReadout.predict_vote` (`bio_network/senses/readout.py`): a frozen hard
  plurality-vote decoder, pre-committed alongside the soft prototype readout --
  both reported on both arms, no retroactive selection.
- `tests/test_projection_plastic.py`: 9 tests (frozen-arm identity, single-pair
  causal LTP / non-causal LTD, [0,1] bounds, structural edge integrity, frozen
  phase no-op, same-seed trajectory determinism, end-to-end training health).
- `benchmarks/m32_plastic_optic_nerve.py`: two-arm AB (frozen v1 control vs
  plastic) with pre-committed readouts, structured-tile and selectivity
  metrics, stability guards; saves `m32_tiles_control.png`,
  `m32_tiles_plastic.png`, `m32_tiles_diptych.png`.
- `docs/M3_2_RESULTS.md`: **honest negative / marginal result** -- input
  plasticity alone (A+ < A- at the observed firing rate) starves the recurrent
  engine (plastic 44,950 vs control 2.61 M train spikes; rate 0.09 vs 5.24 Hz),
  `w_in` stays in [0, 0.4], no structured tiles formed; readout moves 11% to
  15% soft / 10% to 16% vote. Follow-up (balanced updating / Euclidean
  homeostasis) is required before a "learned receptive field" claim.

### Added (M3.3: homeostatic regulators -- the "critical period")

- `InputProjection` gains **synaptic scaling** (`synaptic_scaling=True`,
  Turrigiano et al. 1998) in `bio_network/senses/projections.py`:
  `synaptic_scale()` renormalizes every neuron's incoming `w_in` after each
  training window to `sum(w_in) == n_in_per_neuron * 0.30` (the STDP init
  mean, so day-one power is preserved and regulation only reallocates
  structure) via a fixed-point scale -> clamp -> rescale loop (up to 64
  iterations, target to 1e-9) that keeps all weights inside `[0, 1]`.
- `InputProjection` gains the **excitatory-only structural constraint**
  (`excitatory_only=True`, `n_excitatory=`): the pixel->neuron fan-out draws
  only excitatory targets, so no input synapse lands on an inhibitory
  interneuron; the historical all-neuron wiring stays the default for ARM A.
- `RetinaStimulus` triggers per-window synaptic scaling at the slot boundary
  during training (`bio_network/senses/stimulus.py`), frozen in test/assign.
- `IzhikevichPopulation` gains **adaptive spike thresholds**
  (`adaptive_thresholds=True`, Diehl & Cook 2015 intrinsic plasticity) in
  `bio_network/engine/neurons.py`: a per-ms exponential rate low-pass
  (`rate_tau_ms`, `target_rate_hz` defaults 2000 ms / 5 Hz) drifts each
  excitatory neuron's threshold `theta` by `theta_gain * (ema - target)` within
  `[1, 30]` mV; inhibitory neurons stay at the canonical 30. `save_state` /
  `load_state` now carry `theta` + rate EMA. When disabled the population is
  bit-canonical to M1-M3.2 (`theta == _THRESHOLD_MV`).
- `tests/test_homeostasis.py`: 10 tests (scaling reaches target and keeps
  bounds, scaling-off no-op, adaptive-threshold rig with bounds + determinism,
  thresholds-off bit-canonical, freeze boundaries, excitatory-only wiring
  invariant, and a reproducibility + animation-health ARM-style run).
- `benchmarks/m33_homeostasis.py`: controlled two-arm AB (ARM A = the exact
  M3.2 plastic baseline, ARM B = scaling + thresholds + excitatory-only) with
  per-arm `.npz` caches (`--only a|b|ab`) so arms re-run independently,
  tile / selectivity / readout / stability metrics, health + accuracy-ladder
  figures, and `docs/M3_3_RESULTS.md`. Protocol: train 1000 / test 200 /
  probe 60 / seed 42.
- `docs/M3_3_RESULTS.md` and the executed `notebooks/m33_critical_period.ipynb`.

**Result:** ARM B (scaling + adaptive thresholds) rescues the starved pathway.
Against the byte-identical ARM A baseline it raises held-out accuracy
15% -> 39% soft / 16% -> 34% vote, active neurons 76 -> 342/1000, probe-time
mean rate 0.09 -> 0.17 Hz and structured RIA tiles 3 -> 110, with the stability
box held (w_in in [0, 1], theta in [1, 30], recurrent weights finite,
inhibitory frozen). The brain wakes up -- but is not yet awake: rate stays below
the 2 Hz target and only ~1/3 of the population is recruited, so M3.3 is a
partial positive after M3.2's controlled negative; rate homeostasis alone does
not yet float the pathway into a dense, high-rate regime.

### Added (M3: senses -- retina encoder and visual feature emergence)

- `Retina` (`bio_network/senses/retina.py`): deterministic image -> spike
  timetable encoder in two modes -- latency (one spike per bright pixel at
  `t = (1 - intensity) * window_ms`) and rate (Poisson at
  `intensity * max_rate_hz`) -- with validation, explicit seeding, and a
  stable `encode()` API.
- `InputProjection` (`bio_network/senses/projections.py`): frozen, non-plastic
  random pixel-to-neuron fan-out (`fanout` targets per pixel), with
  `drive_neurons()` and `fan_in_stats()`.
- `RetinaStimulus` (`bio_network/senses/stimulus.py`): drop-in `stimulus_fn`
  presenting images as windowed current pulses with silent gaps, slot
  boundaries, and `__len__` for exact per-image response slicing.
- `LabelsReadout` (`bio_network/senses/readout.py`): label-scoped decoder that
  fits per-class response fingerprints on the training split only and predicts
  with frozen fingerprints (labels never touch weights).
- `mnist.load_mnist()` / `mnist.subsample_mnist()` (`bio_network/senses/mnist.py`):
  Keras -> scikit-learn -> raw IDX download fallback loader with a local
  `notebooks/data/` cache.
- `tests/test_retina.py`: 13 tests (encoder determinism, latency Spearman /
  rate Pearson fidelity, window and gap silence, dark-image silence, frozen
  projections, readout label-gating and pre-fit guard, training-run health,
  white-over-dark visibility).
- `benchmarks/m3_vision.py` (E3a encoder fidelity, E3b RIA feature emergence,
  E3c frozen-readout vs chance and kNN, E3d stability guards) and
  `benchmarks/build_m3_notebook.py`; executed `notebooks/m3_first_light.ipynb`
  with `notebooks/output/m3_emergence_tiles.png` and `m3_confusion.png`.
- `docs/M3_RESULTS.md`, ARCHITECTURE senses section, ROADMAP M3 Complete.

### Fixed (sparse-delivery learning bug)

- `SparseSynapses.deliver` was booking arrival-time LTD events for *all* fired
  neurons, including inhibitory ones, clipping their negative weights toward 0
  even though inhibitory weights must stay frozen. Delivery now books learning
  events only for excitatory neurons (`i < n_excit`), keeping inhibitory
  weights bit-frozen during STDP. Verified by the M3 training run
  (16 inhibitory spikes exercised, weights unchanged).

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
  retention advantage 2.485x > 1.25).
- 15 M4 tests (`tests/test_memory.py`): store/replay round trips, sleep quiet
  drive, replay-pulse injection, causal consolidation, and the sleep fork
  guarantees -- `save_state/load_state` lockstep determinism, noise-gating
  (scale 0.25 quieter), and `learning=False` during sleep leaving weights
  bit-identical.
- `docs/M4_RESULTS.md`, `docs/ARCHITECTURE.md` (episodic/sleep-replay section
  with compressed-replay biology citations), ROADMAP M4 Complete (before M3),
  `notebooks/m4_sleep_consolidation.ipynb` (executed, with
  `notebooks/output/m4_e4a_recall.png`, `m4_e4b_continual.png`,
  `m4_protocol_rates.png`).

[Unreleased]: https://github.com/saurav3231/bio-network/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/saurav3231/bio-network/releases/tag/v0.1.0
