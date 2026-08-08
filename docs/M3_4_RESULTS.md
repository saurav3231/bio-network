# M3.4 Results -- morning coffee (bounded parameter tuning)

## Step 1 -- diagnosis (instrumented ARM B replay, end of training)

- theta histogram (bins over [1,30]): [800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- thetas at floor (1.0 mV): 100.0%; ceiling 0.0%; mean 1.00
- per-neuron w_in sum: mean 5.870, zero-incoming neurons 0.0% (excitatory block only)
- mean window drive (mV-equivalent over the whole window): 117.39
- verdict: **DRIVE-bound (per-ms strength)** -- thresholds bottomed out at the 1.0 floor (100% of excitatory neurons) so excitation cost is already minimal, while w_in is still supplied (0% zero, scaling held it at the C=0.30 line); the sleep is therefore a per-ms current-strength problem -- the pulsed drive is too weak to push membranes to threshold inside each window. L2 ambient drive & L1 both raise exactly that per-ms drive.

## Step 2 -- 3x2 pilot sweep (200-train pilot, seed 42)

| scaling C | ambient | rate (Hz) | active | acc soft |
|---|---|---|---|---|
| 0.30 | 0.0 | 0.54 | 623 | 37.5% |
| 0.30 | 1.0 | 0.58 | 619 | 36.0% |
| 0.30 | 2.0 | 0.63 | 617 | 37.5% |
| 0.60 | 0.0 | 1.47 | 936 | 10.5% |
| 0.60 | 1.0 | 1.55 | 942 | 10.5% |
| 0.60 | 2.0 | 1.69 | 946 | 10.5% |
| 1.00 | 0.0 | 5.15 | 1000 | 18.0% |
| 1.00 | 1.0 | 5.23 | 1000 | 18.0% |
| 1.00 | 2.0 | 5.30 | 1000 | 18.0% |
- **picked:** C = 0.60, ambient drive = 2.0 -- rate 1.69 Hz (the only lever setting hitting the [1,4] band) with 946 active, but accuracy 10.5% is 27.0% below the best pilot 37.5% -- acceptance will document the tradeoff

## Step 3 -- full-scale ARM C (train 1000, C = 0.60, ambient = 2.0)

| arm | acc soft/vote | active | rate (Hz) | structured |
|---|---|---|---|---|
| ARM A (M3.2 baseline) | 15.0% soft / 16.0% vote | 76 | 0.09 | 3 |
| ARM B (M3.3 homeostatic) | 39.0% soft / 34.0% vote | 342 | 0.17 | 110 |
| ARM C (M3.4 morning) | 13.5% soft / 13.0% vote | 805 | 1.31 | 112 |

## Acceptance

targets: rate >= 1.0 Hz, active >= 400, soft acc >= 30%.
- ARM C: rate 1.31 Hz, active 805/1000, soft acc 13.5%
- HEALTH target met: rate 1.31 Hz >= 1.0 Hz (+) and active 805 >= 400 (+), but soft acc 13.5% < 30% (--): the morning coffee wakes the network but over-drives it past the class-selective operating point.
- Per M3.4 acceptance policy (health passed, accuracy failed): keep the ARM C run, document the tradeoff, and STOP -- no architecture redesign within this bounded iteration.