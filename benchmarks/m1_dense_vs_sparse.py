"""Benchmark: dense vs event-driven sparse synaptic engine.

Compares wall time, peak RAM, and mean excitatory firing rate for the dense
M1 reference engine and the sparse, delay-based engine across population sizes
(N = 1k, 10k, 50k, keeping the 80/20 excitatory/inhibitory split).

Run from the repository root:

    python benchmarks/m1_dense_vs_sparse.py

Expected: the sparse engine uses tens of MB and completes at 50k neurons, while
the dense engine (a float32 ``N x N`` matrix) becomes infeasible well before
50k for lack of RAM.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass

import numpy as np

from bio_network.engine.neurons import IzhikevichPopulation
from bio_network.engine.scheduler import simulate
from bio_network.engine.synapses import RandomSynapses
from bio_network.engine.synapses_sparse import SparseSynapses

N_EXC_FRAC = 0.8
T_MS = 1000
SEED = 42
OUT_DEGREE = 100


@dataclass
class Result:
    """Outcome of a single benchmark run."""

    n_neurons: int
    engine: str
    wall_s: float
    peak_mb: float
    exc_rate_hz: float
    notes: str = ""


def _make_dense_float32(n_exc: int, n_inh: int, seed: int) -> RandomSynapses:
    """Dense engine with a float32 weight matrix (M1 connectivity)."""
    synapses = RandomSynapses(n_pre_excit=n_exc, n_pre_inhib=n_inh, seed=seed)
    synapses.S = synapses.S.astype(np.float32)
    return synapses


def _make_sparse(n_exc: int, n_inh: int, seed: int) -> SparseSynapses:
    return SparseSynapses(
        n_excit=n_exc, n_inhib=n_inh, out_degree=OUT_DEGREE, seed=seed
    )


def _run(engine: str, n_exc: int, n_inh: int) -> Result:
    n_neurons = n_exc + n_inh
    note = ""
    if engine == "dense":
        try:
            synapses = _make_dense_float32(n_exc, n_inh, SEED)
        except MemoryError:
            return Result(
                n_neurons, engine, 0.0, 0.0, float("nan"), "infeasible (>RAM)"
            )
    else:
        synapses = _make_sparse(n_exc, n_inh, SEED)

    population = IzhikevichPopulation(n_excitatory=n_exc, n_inhibitory=n_inh)

    tracemalloc.start()
    start = time.perf_counter()
    try:
        recording = simulate(population, synapses, T_ms=T_MS, seed=SEED, engine=engine)
    except MemoryError:
        note = "infeasible (>RAM)"
        wall_s = time.perf_counter() - start
        peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        return Result(n_neurons, engine, wall_s, peak_mb, float("nan"), note)
    wall_s = time.perf_counter() - start
    peak_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    if recording.times_ms.size:
        exc_rate = float(recording.mean_rates_hz()[:n_exc].mean())
    else:
        exc_rate = 0.0
    return Result(n_neurons, engine, wall_s, peak_mb, exc_rate)


def _header() -> str:
    return (
        f"| {'N':>7} | {'engine':<6} | {'time (s)':>9} | {'peak RAM (MB)':>14} "
        f"| {'mean exc (Hz)':>13} | note |"
    )


def _row(r: Result) -> str:
    n = f"{r.n_neurons:7d}"
    eng = f"{r.engine:<6}"
    if np.isnan(r.exc_rate_hz):
        t = f"{r.wall_s:9.2f}"
        ram = f"{r.peak_mb:14.1f}"
        rate = f"{'--':>13}"
    else:
        t = f"{r.wall_s:9.2f}"
        ram = f"{r.peak_mb:14.1f}"
        rate = f"{r.exc_rate_hz:13.2f}"
    note = r.notes
    return f"| {n} | {eng} | {t} | {ram} | {rate} | {note} |"


def main() -> None:
    sizes = [1000, 10000, 50000]
    print(_header())
    print(f"|{'-'*9}|{'-'*8}|{'-'*11}|{'-'*16}|{'-'*15}|{'-'*6}|")
    for n in sizes:
        n_exc = int(n * N_EXC_FRAC)
        n_inh = n - n_exc
        for engine in ("dense", "sparse"):
            result = _run(engine, n_exc, n_inh)
            print(_row(result))


if __name__ == "__main__":
    main()
