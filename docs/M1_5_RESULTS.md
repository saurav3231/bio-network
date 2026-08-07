# M1.5 Results: Sparse Engine Calibration and First Pathology

Results for the event-driven sparse engine (`SparseSynapses`,
`engine="sparse"`) added after the M1 dense baseline (`docs/M1_RESULTS.md`).

## Gain calibration

The sparse engine with default connectivity (`out_degree=100`) sits in a
lower-gain regime than the dense reference because its per-neuron fan-in is
~100 instead of ~N. The `gain` parameter rescales excitatory weights at
construction to re-establish the recurrent drive. Calibration below:
N = 1000, T = 1000 ms, seed = 42, 80/20 split, out_degree = 100.

| gain | mean exc (Hz) | mean inh (Hz) | notes |
|------|---------------|---------------|-------|
| 1.0  | 4.88          | 0.00          | low-gain regime, inhibition silent |
| 1.5  | 5.01          | 0.00          | |
| 2.0  | 5.13          | 0.00          | |
| 3.0  | 5.47          | 0.00          | |
| 5.0  | 6.38          | 0.06          | inhibition barely active |
| 8.0  | 8.03          | 7.29          | near dense baseline |
| **10.0** | **9.70**  | 13.19         | **matches dense (9.76 Hz)** |
| 12.0 | 14.53         | 24.05         | above target |
| 20.0 | 160.61        | 211.28        | runaway (saturated) |
| 40.0 | 979.78        | 979.95        | pathological saturation |

**Calibrated default: `gain = 10.0`** at `out_degree=100`, reproducing the
dense baseline mean excitatory rate (9.70 Hz vs 9.76 Hz). The neutral
constructor default remains `gain = 1.0` (identity); use `gain = 10.0` for
parity with the dense engine at `out_degree=100`.

The response is sharply nonlinear: below gain ~8 the network is subcritical and
inhibition stays silent; around gain 8-12 it enters the balanced regime; beyond
~12 it runs away. This is expected in balanced-network theory (van Vreeswijk &
Sompolinsky, 1998): the mean rate is a sensitive function of the
excitation/inhibition balance set by the fan-in.

## First recorded network pathology: the dense fan-in scaling trap

During the N-scaling benchmark (`benchmarks/m1_dense_vs_sparse.py`), the dense
engine at N = 10,000 saturated: mean excitatory rate ~961 Hz (measured at
T = 200 ms, seed = 42). The dense matrix gives every neuron a fan-in of ~N, so
as N grows the recurrent excitation in each neuron grows linearly with N while
the network's ability to clamp it does not keep pace -- the all-to-all
positive feedback runs away.

This is the first recorded pathology of the simulator and it is specific to the
dense engine's O(N^2) fan-in scaling. The sparse engine does not exhibit it: at
N = 10,000 it maintains ~5 Hz with a fixed fan-in of 100. Dense, "golden
reference" results are therefore only meaningful at N = 1000; larger dense runs
should be treated with caution.

Recorded at N = 10,000, T = 200 ms, seed = 42 (see benchmark table):

| engine | N      | time (s) | peak RAM | mean exc (Hz) |
|--------|--------|----------|----------|---------------|
| dense  | 10,000 | 53.06    | ~493 MB  | 961.36 (saturated) |
| sparse | 10,000 | 0.58     | ~0.7 MB  | 5.14 |
