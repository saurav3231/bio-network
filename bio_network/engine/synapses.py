"""Synapse container and conduction delay handling.

Provides a dense random synaptic weight matrix with sign-constrained weights
reflecting Dale's principle. Conduction delays are a planned M1 extension.
"""

from __future__ import annotations

import numpy as np


class RandomSynapses:
    """A dense random synaptic weight matrix following Dale's principle.

    Rows are post-synaptic neurons and columns are pre-synaptic neurons, so
    ``S[i, j]`` is the weight from neuron ``j`` to neuron ``i``. Columns
    corresponding to excitatory pre-synaptic neurons carry weights uniformly
    in ``[0, 0.5]``; columns corresponding to inhibitory pre-synaptic neurons
    carry weights uniformly in ``[-1, 0]``.
    """

    def __init__(
        self,
        n_pre_excit: int = 800,
        n_pre_inhib: int = 200,
        seed: int = 42,
    ) -> None:
        """Initialize the synaptic weight matrix.

        Args:
            n_pre_excit: number of excitatory pre-synaptic neurons.
            n_pre_inhib: number of inhibitory pre-synaptic neurons.
            seed: random seed for reproducible weights.
        """
        self.n_pre_excit = n_pre_excit
        self.n_pre_inhib = n_pre_inhib

        rng = np.random.default_rng(seed)
        n_neurons = n_pre_excit + n_pre_inhib
        self.S = np.empty((n_neurons, n_neurons))
        self.S[:, :n_pre_excit] = rng.uniform(0.0, 0.5, size=(n_neurons, n_pre_excit))
        self.S[:, n_pre_excit:] = rng.uniform(-1.0, 0.0, size=(n_neurons, n_pre_inhib))

    def deliver(self, fired: np.ndarray) -> np.ndarray:
        """Compute post-synaptic input current from fired pre-synaptic neurons.

        Args:
            fired: indices of pre-synaptic neurons that spiked this step.

        Returns:
            The summed post-synaptic current for every neuron,
            ``sum(S[:, fired], axis=1)``.
        """
        if fired.size == 0:
            return np.zeros(self.S.shape[0])
        return self.S[:, fired].sum(axis=1)
