"""Izhikevich spiking neuron model.

Implements the Izhikevich (2003) simple model of spiking neurons, vectorized
over a population with NumPy.

Reference:
    Izhikevich, E. M. (2003). Simple model of spiking neurons.
    IEEE Transactions on Neural Networks, 14(6), 1569--1572.
"""

from __future__ import annotations

import numpy as np

_THRESHOLD_MV = 30.0


class IzhikevichPopulation:
    """A population of Izhikevich (2003) neurons with cortex-like parameters.

    The membrane potential of neuron ``i`` follows::

        dv/dt = 0.04 * v^2 + 5 * v + 140 - u + I
        du/dt = a * (b * v - u)
        if v >= 30 mV: v = c, u = u + d

    Parameters follow the cortex-like scheme of Izhikevich (2003): 800
    excitatory regular-spiking neurons (a=0.02, b=0.2, c=-65, d=8) and 200
    inhibitory fast-spiking neurons (a=0.1, b=0.2, c=-65, d=2), obeying Dale's
    principle at the population level.
    """

    def __init__(
        self,
        n_excitatory: int = 800,
        n_inhibitory: int = 200,
        seed: int = 42,
    ) -> None:
        """Initialize the neuron population.

        Args:
            n_excitatory: number of excitatory (regular spiking) neurons.
            n_inhibitory: number of inhibitory (fast spiking) neurons.
            seed: random seed reserved for future stochastic population
                initialization; keeps experiments reproducible.
        """
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        n_neurons = n_excitatory + n_inhibitory
        self.is_excitatory = np.zeros(n_neurons, dtype=bool)
        self.is_excitatory[:n_excitatory] = True

        self.a = np.concatenate(
            (0.02 * np.ones(n_excitatory), 0.1 * np.ones(n_inhibitory))
        )
        self.b = 0.2 * np.ones(n_neurons)
        self.c = -65.0 * np.ones(n_neurons)
        self.d = np.concatenate(
            (8.0 * np.ones(n_excitatory), 2.0 * np.ones(n_inhibitory))
        )

        # Canonical initial conditions from Izhikevich (2003).
        self.v = -65.0 * np.ones(n_neurons)
        self.u = 0.2 * self.v

    def save_state(self) -> dict:
        """Deep copy of the dynamic state (``v``, ``u``) needed to resume."""
        return {
            "v": self.v.copy(),
            "u": self.u.copy(),
        }

    def load_state(self, state: dict) -> None:
        """Restore the state captured by :meth:`save_state` in place."""
        self.v = state["v"].copy()
        self.u = state["u"].copy()

    def step(self, I: np.ndarray) -> np.ndarray:
        """Advance the population by 1 ms and return the neurons that spiked.

        Integrates the membrane potential with two 0.5 ms explicit Euler
        half-steps (the canonical reference implementation from Izhikevich
        2003, required for numerical stability) and one 1 ms update of the
        recovery variable, then applies the spike-and-reset rule: neurons with
        ``v >= 30`` mV are reset to ``v = c`` and ``u = u + d``.

        Args:
            I: input current (mV) for every neuron this millisecond.

        Returns:
            Indices of the neurons that spiked during this step.
        """
        v = self.v
        u = self.u

        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + I)
        u += self.a * (self.b * v - u)

        fired = v >= _THRESHOLD_MV
        v[fired] = self.c[fired]
        u[fired] += self.d[fired]

        self.v = v
        self.u = u
        return np.flatnonzero(fired)
