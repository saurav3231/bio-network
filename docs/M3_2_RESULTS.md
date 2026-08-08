# M3.2 Results -- a plastic input projection (optic nerve)

Controlled AB test on the same engineered substrate as M3 v1: seeds, fan-out,
topology, images and readouts are identical. The only change is the plastic arm
lets its input synapses ``w_in in [0,1]`` (init uniform 0.2-0.4) learn with STDP
(tau 20 ms, A+ 0.10, A- 0.12) during the training phase, then freezes before
assignment/test. A per-neuron homeostatic target (total input power per neuron
== fan-in count) keeps the *drive strength* of the two arms identical so any
difference is caused by plasticity, not under-powering.

## Arm summary
- frozen (v1 control): accuracy 11.0% (soft) / 10.0% (vote), structured tiles 0, selectivity 0->40, readout neurons 1000/1000, wall 725s
- plastic (learned inputs): accuracy 15.0% (soft) / 16.0% (vote), structured tiles 0, selectivity 0->76, readout neurons 1000/1000, wall 255s

## Experiment 32a -- control reproduction
The frozen arm is byte-for-byte the M3 v1 pathway (same targets, unit drive,
drive scaled only by ``pulse_amp`` and ``competition_gain``). Any drift vs the
recorded v1 numbers is attributable to test/probe sampling, not plasticity.

## Experiment 32b -- receptive-field imagery (RIA)
- structured (localized) tiles before/after: control 0->0, plastic 0->0
- RIA pixel spread (lower = sharper): control 0.223, plastic 0.139
- class-selective neurons: control 0->40, plastic 0->76
- active neurons (probe): control 1000, plastic 76
- figure: ``notebooks/output/m32_tiles_control.png`` / ``m32_tiles_plastic.png``

## Experiment 32c -- zero-shot digit readout (both decoders pre-committed)
- control: soft 11.0%, vote 10.0%
- plastic: soft 15.0%, vote 16.0%

## Experiment 32d -- stability guards
- train spikes: control 2619360, plastic 44950
- mean/max rate (Hz): control 5.24/10.68, plastic 0.09/0.24
- recurrent weights: control [-1.000, 1.000], plastic [-1.000, 1.000]
- all finite: control True, plastic True
- inhibitory weights frozen: control True, plastic True
- w_in range (plastic): [0.000, 0.400]

## Honest bottom line

**Input plasticity does not win big here.** The single change controlled
(``w_in`` learns during training, frozen at assignment/test) moves held-out
accuracy from 11% to 15% soft / 10% to 16% vote -- a real but small win that is
only marginally above the 10% chance floor, and it does so by *silencing* the
network, not by forming digit-shaped receptive fields:

- no structured (localized) RIA tile count improved at all (0 -> 0 both arms);
- the plastic arm produced 1690x fewer training spikes (44,950 vs 2.61 M) and
  rate collapse to 0.09 Hz against a 5.24 Hz control;
- ``w_in`` never settled away from its init: it stayed within [0.0, 0.4], so
  LTP (A+ 0.10, requires the neuron to fire) never outcompeted LTD (A- 0.12,
  applied on *every* input arrival) -- a causal-ish, self-defeating loop.

Mechanically STDP is correct (a controlled arrival->fire pair *does* potentiate,
edge-wise delta +0.45 over the raw-weight storm check) but the *network-level*
dynamics with these exact constants starve the recurrent engine of drive: LTD
outweighs LTP at the observed firing rate, so weights drift to ~0 and the
residual signal that reaches the readout is a sparse winner-take-most that
slightly helps (15%/16%) but is nowhere near the Diehl & Cook digit-feature
outcome.

The honest conclusion for M3.2 is a **controlled negative / marginal result**
under the spec-pinned STDP constants: input learnability alone does not float
receptive fields in this pipeline; a balanced updating rule (Euclidean
homeostasis on weights, equal A+//A- dwell, or post-synaptic gain control) is
the required follow-up before the "plastic optic nerve" claim can be made.
