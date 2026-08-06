# Bio Network

Brain-inspired computation for research and education: a biologically plausible neural simulation engine.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)
![Status](https://img.shields.io/badge/Status-Early%20Research-orange.svg)

## About

Bio Network is an open-source research project exploring brain-inspired computation. It aims to build a biologically plausible neural simulation engine that implements core brain mechanisms that mainstream deep learning ignores: event-driven spiking neurons, local synaptic learning rules, predictive processing, fast episodic memory, sleep-phase consolidation, neuromodulation, and structural plasticity. The prototype is written in Python with NumPy; a performance core in Rust is planned later.

This is an honest research and educational project. It is not an AGI project and it is not medical software.

## Motivation

The human brain runs on roughly 20 watts, learns continuously from a single stream of experience without catastrophic forgetting, and requires no error backpropagation across a frozen network graph. Modern neural networks exhibit none of these properties: they need massive GPU energy, they forget old tasks when learning new ones, and they depend on global, non-local training signals. We treat these differences as engineering inspiration -- as constraints and design patterns worth studying -- not as claims that we are recreating consciousness or the human brain.

## Core Principles

1. **Spiking neurons** -- Computation only happens when a neuron fires an event, not on a fixed clock cycle.
2. **Synaptic plasticity** -- Connections change through local, biologically inspired rules (Hebbian / STDP), not backpropagation.
3. **Predictive processing** -- The network predicts its own inputs and learns from prediction error.
4. **Fast episodic memory** -- A hippocampus-like module stores experiences in one shot for later use.
5. **Sleep-phase consolidation** -- Periodic "replay" phases stabilize and consolidate memories.
6. **Neuromodulation** -- Dopamine-like global signals gate when and how strongly learning occurs.
7. **Structural plasticity** -- The network grows and prunes its own connections over time.

## Goals / Non-Goals

**Goals**

- A research-grade, inspectable simulation of spiking neural networks with biologically plausible learning rules.
- A clear, well-documented codebase that serves as an educational reference for computational neuroscience.
- Honest, reproducible results backed by citations to the primary literature.

**Non-Goals**

- Recreating the human brain or consciousness.
- Building an AGI system or an AGI-adjacent product.
- Medical or clinical claims of any kind.
- Outperforming mainstream deep learning on benchmark tasks.

## Roadmap

| Milestone | Focus | Status |
|-----------|-------|--------|
| M1 | Spiking neuron engine (Izhikevich model, event scheduler, spike raster) | Current |
| M2 | Self-organizing learning (STDP, label-free emergence) | Planned |
| M3 | Sensory encoding (images to spike trains) | Planned |
| M4 | Memory + sleep-phase consolidation | Planned |
| M5 | Neuromodulation (reward-gated plasticity) | Planned |
| M6 | Live web dashboard | Planned |
| Stretch | Run the engine on the published C. elegans connectome | Stretch |

See [ROADMAP.md](ROADMAP.md) for the full milestone plan.

## Getting Started

Milestone 1 (spiking neuron engine) is under active development. See [ROADMAP.md](ROADMAP.md) for status and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the planned module layout. Installation and usage instructions will be added once M1 lands.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

## License

Released under the [MIT License](LICENSE).

## References

- Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.
- Bi, G.-q., & Poo, M.-m. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *The Journal of Neuroscience*, 18(24), 10464--10472.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569--1572.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79--87.
- Rasch, B., & Born, J. (2013). About sleep's role in memory. *Physiological Reviews*, 93(2), 681--766.
- Schultz, W., Dayan, P., & Montague, P. R. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593--1599.

## Disclaimer

This is a research and educational project. It makes no medical or biological claims. The simulations are abstract models of certain neural mechanisms, not representations of real nervous systems, and they are not intended to diagnose, treat, or inform any medical decision.
