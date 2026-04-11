#!/usr/bin/env python
"""
Minibatch Size Benchmark for Half-Sarcomere Simulation

Finds the optimal minibatch_size for _run_sim_kernel by timing each candidate
chunk size. Splitting large batches into fixed-size minibatches bounds peak
GPU RAM to minibatch_size × T × n_metrics and can improve L2 cache utilisation.

Usage:
    python examples/benchmarks/benchmark_minibatch.py [--n N] [--duration_ms D] [--total_runs R]

Defaults:
    --n            2        SarcTopology nrows=ncols=N
    --duration_ms  100      Simulation duration per run
    --total_runs   256      Total batch size to benchmark against

Output:
    Timing table (seconds per simulation) for each candidate minibatch size,
    followed by the recommended optimal value.
"""

import sys
import time
import argparse

import jax
import jax.numpy as jnp

from multifil_jax.simulation import run, get_skeletal_params, BATCH_BUCKETS
from multifil_jax.core.sarc_geometry import SarcTopology


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=2, help='SarcTopology nrows=ncols (default: 2)')
    parser.add_argument('--duration_ms', type=float, default=100.0, help='Simulation duration ms (default: 100)')
    parser.add_argument('--total_runs', type=int, default=256, help='Total batch size (default: 256)')
    parser.add_argument('--min_m', type=int, default=1, help='Minimum minibatch size to test (default: 1)')
    return parser.parse_args()


def main():
    args = parse_args()
    n = args.n
    duration_ms = args.duration_ms
    total_runs = args.total_runs
    min_m = args.min_m

    print("Minibatch Size Benchmark")
    print("=" * 60)
    print(f"Devices: {jax.devices()}")
    print(f"SarcTopology: nrows={n}, ncols={n}")
    print(f"Duration: {duration_ms} ms per simulation")
    print(f"Total runs: {total_runs}")
    print()

    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(nrows=n, ncols=n, static_params=static, dynamic_params=dynamic)

    # Candidate chunk sizes: powers of 2 up to total_runs (inclusive)
    # M == total_runs uses the single-call path (no minibatching) as baseline
    candidate_sizes = [M for M in BATCH_BUCKETS if min_m <= M <= total_runs]
    if not candidate_sizes:
        candidate_sizes = [total_runs]
    # Ensure total_runs itself is included as the no-minibatch baseline
    if total_runs not in candidate_sizes:
        candidate_sizes.append(total_runs)

    print(f"Candidate minibatch sizes: {candidate_sizes}")
    print()

    results = {}

    for M in candidate_sizes:
        label = f"M={M}" if M < total_runs else f"M={M} (no minibatch)"
        minibatch_arg = M if M < total_runs else None

        # Warmup: pays JIT compile for this chunk size once
        print(f"  Warming up {label}...", end='', flush=True)
        r = run(topo, pCa=4.5, z_line=1100.0, duration_ms=duration_ms,
                replicates=total_runs, minibatch_size=minibatch_arg)
        r.axial_force.block_until_ready()
        print(" done")

        # Three timed runs
        times = []
        for seed in range(1, 4):
            t0 = time.perf_counter()
            r = run(topo, pCa=4.5, z_line=1100.0, duration_ms=duration_ms,
                    replicates=total_runs, minibatch_size=minibatch_arg, rng_seed=seed)
            r.axial_force.block_until_ready()
            times.append(time.perf_counter() - t0)

        best_s = min(times)
        per_sim_ms = best_s / total_runs * 1000.0
        results[M] = best_s / total_runs  # seconds per simulation

        print(f"    {label}: best {best_s:.3f}s total, {per_sim_ms:.3f} ms/sim")

    print()
    print("=" * 60)
    print("Results summary (ms per simulation, best of 3):")
    print(f"  {'M':>8}  {'ms/sim':>10}  {'vs no-minibatch':>16}")

    baseline_s = results.get(total_runs, min(results.values()))
    for M in candidate_sizes:
        s = results[M]
        pct = (s / baseline_s - 1.0) * 100.0
        sign = "+" if pct >= 0 else ""
        label = "(baseline)" if M == total_runs else ""
        print(f"  {M:>8}  {s*1000:>10.3f}  {sign}{pct:>+14.1f}%  {label}")

    optimal_M = min(results, key=results.get)
    print()
    if optimal_M == total_runs:
        print(f"Optimal: no minibatching (single kernel call) — M={total_runs}")
    else:
        print(f"Optimal minibatch_size={optimal_M}  ({results[optimal_M]*1000:.3f} ms/sim)")
        print(f"  Usage: run(..., minibatch_size={optimal_M})")


if __name__ == '__main__':
    main()
