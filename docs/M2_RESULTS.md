# M2 Results -- self-organizing STDP

Observed values from `benchmarks/m2_stdp_experiments.py` on the
event-driven sparse engine (gain=8, out_degree=100, N=1000, seed 42).

## Experiment 1 -- hallmark stability (T = 30 s)
- spikes: 126911
- mean excitatory rate: 5.28 Hz
- mean inhibitory rate: 0.03 Hz
- per-second mean-rate range: [4.07, 4.87] Hz
- excitatory weights: min 0.000, max 1.000, mean 0.583
- weight fraction pinned at 0: 0.017
- weight fraction pinned at 1: 0.035
- finite (no NaN/Inf): True
Weight histogram image: `notebooks/output/m2_hallmark_weights.png`.

## Experiment 2 -- fire-together / wire-together (train 20 s, test 2 s)
- A->B weight after training: 0.9826
- A->C weight after training: 0.5700
- A->B weight (frozen test): 0.9826
- A->C weight (frozen test): 0.5700
- ratio A->B / A->C: 1.724 (expect > 1.5)

## Experiment 3 -- cue / pattern completion (stretch)
- B spikes within 50 ms of an A cue over 10 pulses: 231 after training vs 226
  before training (1.02x). Pattern completion is effectively **negative** in
  this configuration: the baseline is already saturated because the recurrent
  network (gain = 8) recruits group B strongly from an A cue whether or not A
  and B were co-trained. Reported honestly as a negative/stretch result, not a
  win; this is a known limitation of the current gain and single-layer design.
