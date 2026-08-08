# M3 Results -- artificial retina and unsupervised visual features

Observed values from `benchmarks/m3_vision.py` on the sparse
event-driven engine (gain=8, out_degree=100, N=1000, seed 42) with a
28x28 artificial retina (latency coding, window 350 ms, gap 150 ms, 20
non-plastic fan-out edges per pixel).

## Experiment 3a -- encoder fidelity
- latency coding: Spearman(intensity, -latency) = 1.000 (target > 0.6)
- rate coding:    Pearson(spike count, intensity) = 0.999 (target > 0.9)
- latency spikes per gradient image: 756

## Experiment 3b -- unsupervised feature emergence
- mean RIA contrast before/after: 1.000 / 1.000
- mean RIA pixel spread before/after (lower = sharper): 0.180 / 0.174
- class-selective neurons before/after: 0 / 44
- active neurons after training: 1000 / 1000
- training pass: 314806 spikes, mean rate 5.25 Hz, peak 10.70 Hz

## Experiment 3c -- zero-shot digit classification
- held-out accuracy (frozen soft readout): 0.200
- chance baseline: 0.10
- numpy kNN (k=3, raw pixels) baseline: 0.600
- neurons used by the readout: 1000 / 1000
- confusion matrix (rows=true, cols=pred):
    6    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    6    0
    1    0    0    0    0    0    0    0    5    0
    1    0    0    0    0    0    0    0    5    0
    0    0    0    0    0    0    0    0    6    0
    0    0    0    0    0    0    0    0    6    0
    1    0    0    0    0    0    0    0    5    0
    0    0    0    0    0    0    0    0    6    0
    0    0    0    0    0    0    0    0    6    0
    0    0    0    0    0    0    0    0    6    0

## Experiment 3d -- stability guards
- firing rates: [0.13, 2.27] Hz (guardrail [0.5, 100])
- weights: [-1.000, 1.000] (bounds [0, 1] for excitatory)
- all weights finite: True
- inhibitory weights frozen: True
- E3d spikes: 15221

Caveat: feature emergence is measured at the population level (RIA
contrast and class selectivity), not as per-pixel tuning, because the
input pathway is a fixed random projection (no input plasticity). A
learned input weight matrix (Diehl & Cook 2015 style) is the planned
v2 upgrade and would let single neurons develop localized receptive
fields.
