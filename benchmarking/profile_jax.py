#!/usr/bin/env python
"""
JAX Profiler Script for Half-Sarcomere Simulation

Generates Perfetto traces for GPU performance analysis.

Usage:
    python benchmarking/profile_jax.py

Output:
    Trace files in /tmp/jax_trace/plugins/profile/
    Open https://ui.perfetto.dev/ and drag the .json.gz file to view.
    Use WASD keys to navigate the timeline.
"""

import jax
import jax.numpy as jnp
import time

from multifil_jax.simulation import run, get_default_params
from multifil_jax.core.sarc_geometry import SarcTopology

# Configuration
TRACE_DIR = "/tmp/jax_trace"
DURATION_MS = 500
DT = 1.0
N_STEPS = int(DURATION_MS / DT)


def main():
    print("JAX Half-Sarcomere Profiler")
    print("=" * 60)
    print(f"Devices: {jax.devices()}")
    print(f"Configuration:")
    print(f"  Duration: {DURATION_MS} ms ({N_STEPS} steps)")
    print(f"  dt: {DT} ms")
    print()

    # Create topology
    static, dynamic = get_default_params()
    topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
    topo = jax.device_put(topo)

    # Warmup run (triggers JIT compilation)
    print("Warmup run (JIT compilation)...")
    t0 = time.time()
    warmup_result = run(topo, pCa=4.5, z_line=900.0, duration_ms=DURATION_MS, dt=DT)
    warmup_result.axial_force.block_until_ready()
    warmup_time = time.time() - t0
    print(f"  Warmup (compile + run): {warmup_time:.2f}s")
    print(f"  Mean force: {float(warmup_result.axial_force.mean()):.2f} pN")
    print()

    # Second run (execution only, no compile)
    print("Timed run (execution only)...")
    t0 = time.time()
    timed_result = run(topo, pCa=4.5, z_line=900.0, duration_ms=DURATION_MS, dt=DT, rng_seed=1)
    timed_result.axial_force.block_until_ready()
    exec_time = time.time() - t0
    print(f"  Execution time: {exec_time:.3f}s")
    print(f"  Time per step:  {exec_time / N_STEPS * 1000:.3f} ms")
    print(f"  Simulation speed: {DURATION_MS / exec_time:.1f}x realtime")
    print()

    # Profiled run
    print(f"Profiled run (saving trace to {TRACE_DIR})...")
    t0 = time.time()
    with jax.profiler.trace(TRACE_DIR):
        profile_result = run(
            topo, pCa=4.5, z_line=900.0, duration_ms=DURATION_MS, dt=DT, rng_seed=2
        )
        profile_result.axial_force.block_until_ready()
    profiled_time = time.time() - t0
    print(f"  Profiled run: {profiled_time:.3f}s")
    print()

    # Summary
    print("Performance Summary")
    print("-" * 60)
    print(f"  JIT compilation: {warmup_time - exec_time:.2f}s (approx)")
    print(f"  Execution time:  {exec_time:.3f}s")
    print(f"  Time per step:   {exec_time / N_STEPS * 1000:.3f} ms")
    print(f"  Speed:           {DURATION_MS / exec_time:.1f}x realtime")
    print()
    print("=" * 60)
    print("To view the trace:")
    print("  1. Open Chrome/Edge and visit https://ui.perfetto.dev/")
    print(f"  2. Drag the .json.gz file from {TRACE_DIR}/plugins/profile/")
    print("     into the browser window")
    print("  3. Use WASD keys to navigate the timeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
