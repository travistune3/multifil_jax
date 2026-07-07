#!/usr/bin/env python
"""
Subpopulation: modeling a mixed population of motors

Demonstrates the Subpopulation feature: run() can simulate a lattice where a
fraction of XB motors (or TM regulatory units) have different kinetics than
the rest -- e.g. a heterozygous mutation, or a cMyBP-C-restrained C-zone.

Three modes, all shown below:
    mean_field  -- deterministic, blends the population rate matrices
    random      -- stochastic per-XB/TM assignment (needs replicates)
    c_zone      -- deterministic spatial band (e.g. cMyBP-C C-zone)

Run this script:
    python examples/subpopulation.py
"""
import jax
import numpy as np

from multifil_jax.simulation import run, get_skeletal_params
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax import Subpopulation

print("=" * 70)
print("Subpopulation Feature")
print("=" * 70)

static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=4, ncols=4, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

RUN_KWARGS = dict(pCa=4.5, z_line=1100.0, duration_ms=200, dt=1.0)

# ===========================================================================
# 1. MEAN-FIELD: fraction sweep of a "weak motor" mutant
# ===========================================================================
# xb_g_k_strong scaled by 0.3 models motors with a much softer power stroke.
# A list of Subpopulation objects becomes a sweep axis, just like a pCa list.
print("\n1. Mean-field fraction sweep (0%, 25%, 50%, 75%, 100% weak motors)")
print("-" * 50)

fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
subpops = [Subpopulation.mean_field(f, xb_g_k_strong=0.3) for f in fractions]

result = run(topo, subpopulation=subpops, replicates=1, **RUN_KWARGS)
force = np.asarray(result.axial_force[:, 0, -50:].mean(axis=-1))  # steady-state, per fraction

for f, force_i in zip(fractions, force):
    print(f"  mutant fraction={f:.2f}  ->  steady-state force = {force_i:.1f} pN")
print("  (fraction=0.0 is bit-exact WT; fraction=1.0 is bit-exact fully-mutant)")

# ===========================================================================
# 2. RANDOM: stochastic per-XB assignment, compared to mean-field
# ===========================================================================
# Same scale and fraction as above, but each XB independently draws WT/mutant.
# Use several replicates so the per-XB noise averages out.
print("\n2. Random (stochastic) assignment vs. mean-field, fraction=0.5")
print("-" * 50)

subpop_mf = Subpopulation.mean_field(0.5, xb_g_k_strong=0.3)
subpop_rand = Subpopulation.random(0.5, seed=0, xb_g_k_strong=0.3)

result_mf = run(topo, subpopulation=subpop_mf, replicates=1, **RUN_KWARGS)
result_rand = run(topo, subpopulation=subpop_rand, replicates=8, **RUN_KWARGS)

force_mf = float(result_mf.axial_force[0, -50:].mean())
force_rand = float(result_rand.axial_force[:, -50:].mean())

print(f"  mean_field force:        {force_mf:.1f} pN")
print(f"  random force (8 reps):   {force_rand:.1f} pN")
print("  (should agree within noise for a uniformly-mixed population)")

# ===========================================================================
# 3. C_ZONE: spatially localized mutation (e.g. cMyBP-C C-zone)
# ===========================================================================
# Only crowns whose axial offset falls in [c_zone_min_nm, c_zone_max_nm] get
# the modified kinetics; everywhere else stays WT.
print("\n3. C-zone: mutation confined to a band of the thick filament")
print("-" * 50)

subpop_czone = Subpopulation.c_zone(topo, c_zone_min_nm=350.0, c_zone_max_nm=650.0,
                                     xb_g_k_strong=0.3)
print(f"  empirical mutant fraction (crowns in band): {float(subpop_czone.fractions[1]):.2f}")

result_czone = run(topo, subpopulation=subpop_czone, replicates=4, **RUN_KWARGS)
force_czone = float(result_czone.axial_force[:, -50:].mean())
print(f"  c_zone force (4 reps):   {force_czone:.1f} pN")

# ===========================================================================
# 4. SWEPT RANDOM: severity x fraction 2D sweep in one run() call
# ===========================================================================
# A list of `random` Subpopulations (sharing one seed) is a genuine sweep
# axis, same as a pCa list -- no outer Python loop or manual result stacking.
print("\n4. Swept random: severity x fraction grid, single run() call")
print("-" * 50)

severities = [1.0, 0.5, 0.2]   # xb_g_k_strong scale
sweep_fractions = [0.25, 0.5, 0.75]
subpops_swept = [Subpopulation.random(f, seed=0, xb_g_k_strong=sev)
                  for sev in severities for f in sweep_fractions]

result_sweep = run(topo, subpopulation=subpops_swept, replicates=4, **RUN_KWARGS)
force_sweep = np.asarray(result_sweep.axial_force[:, :, -50:].mean(axis=(1, 2)))
grid = force_sweep.reshape(len(severities), len(sweep_fractions))

print("  steady-state force (pN), rows=severity, cols=fraction:")
print("           " + "".join(f"frac={f:<8.2f}" for f in sweep_fractions))
for sev, row in zip(severities, grid):
    print(f"  sev={sev:.2f}  " + "".join(f"{v:<13.1f}" for v in row))

print("\n" + "=" * 70)
print("Subpopulation demo complete.")
print("=" * 70)
