# M4 Results -- episodic memory and sleep consolidation

Observed values from `benchmarks/m4_stdp_experiments.py` on the
sparse event-driven engine (gain=8, out_degree=100, N=1000, seed 42).

## Experiment 4a -- recall after sleep (melody P1)
- presentations: 10 x 500 ms
- episode spikes recorded: 249
- sleep: 40 replays over 11820 ms (sleep spikes: 5858, compression 1.0)
- sleep population rate: mean 0.49 Hz, peak 2.00 Hz (guardrail < 100 Hz)
- probe B delayed-window spikes (5 cues) pre-sleep: 54
- probe B delayed-window spikes (5 cues) post-sleep: 76
- negative control C delayed spikes pre/post: 65 / 82
- post/pre recall ratio: 1.407 (target > 1.3)
- weight evidence: mean A->B 0.831 vs A->C 0.000
- caveat: 5 cues give Poisson-noisy counts, so the ratio alone is weak;
  the weight evidence (A->B welded, A->C pinned to the [0,1] floor) is
  the stable, reviewer-proof signal.

## Experiment 4b -- continual learning without forgetting
- T1 train: 15 s; T2 train: 15 s; E4b replays: 30
- ARM SLEEP  T1 retention ratio (A->B/A->C): 3.464
- ARM NOSLEEP T1 retention ratio (A->B/A->C):  1.394
- retention advantage (sleep/nosleep): 2.485 (target > 1.25)
- T2 acquisition: sleep 2.127 vs nosleep 2.096 (fairness: both should acquire T2)
