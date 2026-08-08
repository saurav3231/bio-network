# M3.3 Results -- homeostatic regulators in the plastic optic nerve

Controlled follow-up to M3.2 (commit 035a9b7). The two arms share all of
the substrate: seeds, fan-out, images, recurrent weight engine and
readouts. The only change between them is the M3.3 physiology: ARM B
switches on 1) **synaptic scaling** (Turrigiano et al. 1998): after every
training window each neuron's incoming weight sum is renormalized to
`sum(w_in) == n_in_per_neuron * 0.3`; and 2) **adaptive
spike thresholds** (Diehl & Cook 2015 intrinsic plasticity): excitatory
neurons track a slow-rate estimate and drift the firing threshold theta
toward 5.0 Hz target within [1.0,30.0] so over-active
neurons back off and silent ones get recruited.

## Arm summary
- M3.2 baseline (ARM A): 15.0% (soft) / 16.0% (vote), structured tiles 3, rate 0.09 Hz, active 76/1000
- M3.3 homeostatic (ARM B): 39.0% (soft) / 34.0% (vote), structured tiles 110, rate 0.17 Hz, active 342/1000

## Economics story

M3.2 diagnosis: LTD (A- 0.12) lands on every input arrival while LTP
(A+ 0.10) only lands when a neuron actually fires; at the observed
firing rate the one-sided depression starves the pathway. Synaptic
scaling re-pins each neuron's total input power so the losing race can't
run away -- and adaptive thresholds make under-driven neurons cheaper to
excite, so pattern claimants diversify instead of collapsing to the
7/9 ghosts.

## E33a -- health (did the brain wake up?)

- mean/max rate (Hz): ARM A 0.09/0.24, ARM B 0.17/1.41
- active neurons (probe): ARM A 76, ARM B 342
- train spikes: ARM A 44950, ARM B 82717

## E 33b -- emergence (recalibrated proxy)

The structured-tile proxy was recalibrated (max covering fraction 0.12) so the M3.2 7/9-prototype stroke-like tiles count as
structured: a stroke occupying ~5-12% of the image qualifies, a uniform
random blend doesn't. See calibration note.
- structured tiles before/after: ARM A 0->3, ARM B 0->110
- RIA pixel spread (lower = sharper): ARM A 0.139, ARM B 0.128
- class-selective neurons: ARM A 0->76, ARM B 0->339

## E 33c -- zero-shot readout (pre-committed decoders)

- ARM A: soft 15.0%, vote 16.0%; 8 classes predicted
- ARM B: soft 39.0%, vote 34.0%; 9 classes predicted

## Top confusions (human-style)

- true 1 confused with predicted 0 (11 cases)
- true 3 confused with predicted 2 (10 cases)
- true 0 confused with predicted 2 (10 cases)
- true 6 confused with predicted 2 (9 cases)
- true 9 confused with predicted 7 (9 cases)
- ARM A top: 1->0 (20), 4->0 (19), 8->0 (18), 7->0 (18), 3->0 (18)

## E 33d -- stability guards

- recurrent weights: ARM A [-1.000, 1.000], ARM B [-1.000, 1.000]
- all finite: ARM A True, ARM B True
- inhibitory frozen: ARM A True, ARM B True
- w_in range: ARM A [0.0000, 0.4000], ARM B [0.0000, 1.0000]
- w_all finite: ARM A True, ARM B True
- theta bounds OK: ARM B True (range [1.000, 30.000])
- wall: ARM A 304s, ARM B 453s
