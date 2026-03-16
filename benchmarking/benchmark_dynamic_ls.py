#!/usr/bin/env python
"""
Dynamic Lattice Spacing Benchmark

Two sections:
  A. PERFORMANCE: compile time and ms/step for fixed LS vs dynamic LS (2x2 lattice).
     Run on a COLD XLA cache for valid compile times:
       rm -rf ~/.cache/multifil_jax/xla/ && python benchmarking/benchmark_dynamic_ls.py

  B. LATTICE SCALING: verify that the K_lat per-filament scaling fix works.
     d_deviation from d_ref should be roughly constant across lattice sizes.
     If it scales with n_thick, the fix is broken.

Usage:
    python benchmarking/benchmark_dynamic_ls.py --section A
    python benchmarking/benchmark_dynamic_ls.py --section B
    python benchmarking/benchmark_dynamic_ls.py --section AB  (both)
"""

import sys
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, '.')
from multifil_jax.simulation import run
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import get_default_params


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--section', default='AB', choices=['A', 'B', 'AB'])
    p.add_argument('--duration_ms', type=float, default=100.0)
    p.add_argument('--replicates', type=int, default=8,
                   help='Replicates for section A force comparison (default: 8)')
    p.add_argument('--rng_seed', type=int, default=42)
    return p.parse_args()


def block(result):
    jax.block_until_ready(result.metrics['axial_force'])
    return result


def run_section_a(dur, reps, seed):
    """Performance: fixed LS vs dynamic LS at 2x2."""
    print("=" * 70)
    print("SECTION A — Performance: fixed LS vs dynamic LS (2x2 lattice)")
    print("=" * 70)
    print("NOTE: compile times are only meaningful on a cold XLA cache.")
    print("      To clear: rm -rf ~/.cache/multifil_jax/xla/")
    print()

    static, dynamic = get_default_params()
    topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
    n_steps = int(dur)

    common = dict(duration_ms=dur, dt=1.0, pCa=4.5, z_line=900.0,
                  lattice_spacing=14.0, replicates=reps, rng_seed=seed)

    # Two kernels: fixed LS and dynamic LS (K_lat is traced so rigid/soft share one kernel)
    configs = [
        ("Fixed LS",               lambda: run(topo, **common)),
        ("Dynamic LS (K_lat=500)", lambda: run(topo, K_lat=500., nu=0.5, **common)),
        ("Dynamic LS (K_lat=5.0)", lambda: run(topo, K_lat=5.0, nu=0.5, **common)),
    ]

    print(f"{'Config':<28} {'1st call (s)':>12} {'ms/step':>8} {'Resid_max':>10} {'Force mean':>11}")
    print("-" * 72)

    stored = {}
    for label, fn in configs:
        # First call: compile + execute
        t0 = time.perf_counter()
        r = block(fn())
        first_call = time.perf_counter() - t0

        # Second call: execute only (kernel cached in-process)
        t0 = time.perf_counter()
        r = block(fn())
        exec_s = time.perf_counter() - t0

        ms_per_step = exec_s / n_steps * 1000.0
        resid_max = float(jnp.max(r.metrics['solver_residual']))
        force_mean = float(jnp.mean(r.metrics['axial_force']))
        stored[label] = r
        print(f"{label:<28} {first_call:>12.1f} {ms_per_step:>8.2f} {resid_max:>10.3f} {force_mean:>11.1f}")

    print()
    print("K_lat=500 and K_lat=5.0 share one compiled kernel (K_lat is traced).")
    print("Dynamic LS (K_lat=5.0) first-call ≈ 0 s because K_lat=500 already compiled it.")

    # Force comparison: fixed LS vs rigid dynamic LS
    # With enough replicates, means should converge. Not a precise test with reps=8.
    f_fixed = float(jnp.mean(stored["Fixed LS"].metrics['axial_force']))
    f_rigid = float(jnp.mean(stored["Dynamic LS (K_lat=500)"].metrics['axial_force']))
    f_soft  = float(jnp.mean(stored["Dynamic LS (K_lat=5.0)"].metrics['axial_force']))
    print()
    print(f"Mean force comparison (pCa=4.5, isometric):")
    print(f"  Fixed LS:           {f_fixed:.1f} pN")
    print(f"  Dynamic K_lat=500:  {f_rigid:.1f} pN  (should be close to fixed LS)")
    print(f"  Dynamic K_lat=5.0:  {f_soft:.1f} pN  (may differ — d compresses lattice)")
    print(f"  Note: with only {reps} replicates stochastic noise >> systematic diff.")
    print(f"        Use --replicates 64+ for a meaningful force comparison.")

    d_rigid = np.array(stored["Dynamic LS (K_lat=500)"].metrics['lattice_spacing'])
    d_soft  = np.array(stored["Dynamic LS (K_lat=5.0)"].metrics['lattice_spacing'])
    print()
    print(f"Emergent d (n_thick=4):")
    print(f"  K_lat=500 (K_lat_eff=2000 pN/nm): mean={np.mean(d_rigid):.3f} nm  deviation={np.mean(d_rigid) - 14.0:+.3f} nm")
    print(f"  K_lat=5.0 (K_lat_eff=20   pN/nm): mean={np.mean(d_soft):.3f} nm  deviation={np.mean(d_soft)  - 14.0:+.3f} nm")


def run_section_b(dur, seed):
    """Lattice scaling: performance and d_deviation across nxn lattice sizes."""
    print("=" * 70)
    print("SECTION B — Lattice scaling (dynamic LS, K_lat=5.0 per filament)")
    print("=" * 70)
    print("Reference (fixed LS, sessions 1-2): ms/step ~1-2 ms, Newton iters ~1")
    print("K_lat per-filament fix: delta_d should be roughly constant across sizes.")
    print()

    static, dynamic = get_default_params()
    K_LAT = 5.0
    NU = 0.5
    n_steps = int(dur)

    cols = f"{'Lattice':<8} {'n_thick':>7} {'compile(s)':>11} {'ms/step':>8} {'iters(mean/max)':>16} {'Resid_max':>10} {'d_mean':>7} {'delta_d':>8}"
    print(cols)
    print("-" * len(cols))

    for nrows in [2, 4, 6, 8, 10]:
        topo = SarcTopology.create(nrows=nrows, ncols=nrows,
                                   static_params=static, dynamic_params=dynamic)

        common = dict(
            duration_ms=dur, dt=1.0,
            pCa=4.5, z_line=900.0, lattice_spacing=14.0,
            K_lat=K_LAT, nu=NU,
            replicates=1, rng_seed=seed,
        )

        # First call: compile + execute
        t0 = time.perf_counter()
        result = run(topo, **common)
        jax.block_until_ready(result.metrics['axial_force'])
        compile_s = time.perf_counter() - t0

        # Second call: execute only
        t0 = time.perf_counter()
        result = run(topo, **common)
        jax.block_until_ready(result.metrics['axial_force'])
        exec_s = time.perf_counter() - t0

        ms_per_step  = exec_s / n_steps * 1000.0
        d            = np.array(result.metrics['lattice_spacing'])
        resid_max    = float(jnp.max(result.metrics['solver_residual']))
        iters_mean   = float(jnp.mean(result.metrics['newton_iters']))
        iters_max    = int(jnp.max(result.metrics['newton_iters']))
        d_mean       = float(np.mean(d))
        delta        = d_mean - 14.0

        print(f"{nrows}x{nrows:<4}   {topo.n_thick:>7}   {compile_s:>11.1f}   {ms_per_step:>8.2f} "
              f"   {iters_mean:>5.2f} / {iters_max:<2}       {resid_max:>10.3f}   {d_mean:>7.3f}   {delta:>+8.3f}")

    print()
    print("PASS criteria:")
    print("  ms/step:      ~1-5 ms, roughly flat across lattice sizes (GPU parallelism)")
    print("  iters mean:   ~1-2 (while_loop exits early; augmented system not much harder than fixed LS)")
    print("  delta_d:      roughly constant across rows (K_lat per-filament scaling correct)")


def main():
    args = parse_args()
    print(f"Devices: {jax.devices()}")
    print()

    if 'A' in args.section:
        run_section_a(args.duration_ms, args.replicates, args.rng_seed)
        print()

    if 'B' in args.section:
        run_section_b(args.duration_ms, args.rng_seed)


if __name__ == '__main__':
    main()
