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

    M3.3 adds an optional **intrinsic plasticity** regulator (Diehl & Cook 2015
    adaptive firing threshold; Desai, Rutherford & Turrigiano 1999 intrinsic
    excitability). When ``adaptive_thresholds=True`` every *excitatory* neuron
    tracks a slow low-pass estimate of its own firing rate (timescale tau, on
    the order of a second). A per-neuron firing threshold ``theta`` -- the
    molar analogue of the fixed ``_THRESHOLD_MV = 30`` line -- drifts in small
    steps toward a target rate: neurons firing faster than ``target_rate_hz``
    raise their threshold (become harder to excite), neurons firing slower
    lower it (become easier to recruit). ``theta`` is clipped to
    ``[theta_min, theta_max]`` = [1.0, 30.0] mV. This is the classic
    rate-homeostatic thermostat that keeps the population from either starving
    (the M3.2 pathology) or seizing, and it lets under-active neurons
    claim inputs during the critical period. With the flag off (default), the
    population is bit-identical to the canonical implementation.
    """

    def __init__(
        self,
        n_excitatory: int = 800,
        n_inhibitory: int = 200,
        seed: int = 42,
        adaptive_thresholds: bool = False,
        target_rate_hz: float = 5.0,
        rate_tau_ms: float = 2000.0,
        theta_min: float = 1.0,
        theta_max: float = 30.0,
        theta_gain: float = 1e-4,
    ) -> None:
        """Initialize the neuron population.

        Args:
            n_excitatory: number of excitatory (regular spiking) neurons.
            n_inhibitory: number of inhibitory (fast spiking) neurons.
            seed: random seed reserved for future stochastic population
                initialization; keeps experiments reproducible.
            adaptive_thresholds: enable intrinsic-plasticity firing thresholds
                (M3.3). Default False reproduces the canonical run bit-for-bit.
            target_rate_hz: homeostatic target firing rate (default 5 Hz).
            rate_tau_ms: low-pass timescale of the per-neuron rate estimate.
            theta_min / theta_max: bounds on the adaptive threshold (mV).
            theta_gain: per-Hz per-ms step of the threshold update.
        """
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.seed = seed
        self.adaptive_thresholds = bool(adaptive_thresholds)
        self.target_rate_hz = float(target_rate_hz)
        self.rate_tau_ms = float(rate_tau_ms)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)
        self.theta_gain = float(theta_gain)
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

        self.v = -65.0 * np.ones(n_neurons)
        self.u = 0.2 * self.v

        # Intrinsic-plasticity state (M3.3); theta == 30 == canonical when off.
        # _rate_ema is a per-neuron exponential low-pass of the spike train; at
        # steady state a neuron firing at ``f`` Hz converges to
        # ema = 1 / (1 - exp(-1000 / (f * rate_tau_ms))).
        self.theta = self.theta_max * np.ones(n_neurons, dtype=np.float64)
        self._rate_ema = np.zeros(n_neurons, dtype=np.float64)
        self._rate_decay = float(np.exp(-1.0 / self.rate_tau_ms))

    def save_state(self) -> dict:
        """Deep copy of the dynamic state needed to resume."""
        return {
            "v": self.v.copy(),
            "u": self.u.copy(),
            "theta": self.theta.copy(),
            "rate_ema": self._rate_ema.copy(),
        }

    def load_state(self, state: dict) -> None:
        """Restore the state captured by :meth:`save_state` in place."""
        self.v = state["v"].copy()
        self.u = state["u"].copy()
        if "theta" in state:
            self.theta = state["theta"].copy()
        if "rate_ema" in state:
            self._rate_ema = state["rate_ema"].copy()

    def _target_ema(self) -> float:
        """Steady-state rate-EMA a neuron firing at ``target_rate_hz`` reaches."""
        step = 1000.0 / (self.target_rate_hz * self.rate_tau_ms)
        return 1.0 / (1.0 - np.exp(-step))

    def _update_thresholds(self) -> None:
        """Intrinsic plasticity: drift firing thresholds toward homeostasis.

        Every excitatory neuron with a spike-filtered rate estimate above the
        target raises its threshold (harder to fire; self-limiting), and every
        neuron below the target lowers it (cheap to recruit). The drift step is
        ``theta_gain * (rate_ema - target_ema)`` per millisecond, so the
        timescale is set by ``target_rate_hz`` and ``rate_tau_ms`` together.
        Inhibitory neurons keep the canonical ``theta_max``: their gain is
        already set by the fixed recurrent inhibitory wiring.
        """
        exc = self.is_excitatory
        err = self._rate_ema[exc] - self._target_ema()
        self.theta[exc] = np.clip(
            self.theta[exc] + self.theta_gain * err,
            self.theta_min,
            self.theta_max,
        )

    def step(self, I: np.ndarray) -> np.ndarray:
        """Advance the population by 1 ms and return the neurons that spiked.

        Integrates the membrane potential with two 0.5 ms explicit Euler
        half-steps (the canonical reference implementation from Izhikevich
        2003, required for numerical stability) and one 1 ms update of the
        recovery variable, then applies the spike-and-reset rule: neurons with
        ``v >= threshold`` are reset to ``v = c`` and ``u = u + d``.

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

        theta = self.theta if self.adaptive_thresholds else _THRESHOLD_MV
        fired = v >= theta
        v[fired] = self.c[fired]
        u[fired] += self.d[fired]

        self.v = v
        self.u = u

        if self.adaptive_thresholds:
            self._rate_ema = self._rate_decay * self._rate_ema
            self._rate_ema[fired] += 1.0
            self._update_thresholds()

        return np.flatnonzero(fired)
