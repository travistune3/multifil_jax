#!/usr/bin/env python
"""
Quickstart: the run() patterns.

    python examples/quickstart.py
"""
#%%
import time

import jax.numpy as jnp

from multifil_jax import SarcTopology, run, get_params, draw, SimulationResult

print("=" * 70)
print("JAX Muscle Simulation - Quickstart")
print("=" * 70)

#%%
# ===========================================================================
# SETUP: one muscle class -> its structure, physics and operating point.
# Build the topology once and reuse it for every run.
# ===========================================================================
static, dynamic, z0, d0 = get_params('skeletal')
topo = SarcTopology.create(2, 2, static)

print(f" Topology: {topo.n_thick} thick, {topo.n_thin} thin filaments")
print(f" Crowns: {topo.n_crowns}/thick, binding candidates: {topo.n_cand}/thin, "
      f"thin nodes: {topo.n_nodes}")
print(f" Total XBs: {topo.total_xbs}")

DRV = dict(z_line=1100.0, lattice_spacing=d0, dt=1.0)

#%%
# ===========================================================================
# 1. SIMPLE ISOMETRIC SIMULATION
# ===========================================================================
print("\n1. Simple Isometric Simulation")
print("-" * 50)

start = time.time()
result = run(topo, dynamic, pCa=4.5, duration_ms=100, verbose=True, **DRV)
elapsed = time.time() - start

print(f"Elapsed: {elapsed:.2f}s")
print(f"Result shape: {result.axial_force.shape}")  # (1, 100) = (replicates, time)
print(f"Mean force: {float(result.axial_force.mean()):.2f} pN")
print(f"Available metrics: {list(result.metrics.keys())[:5]} ...")
print(result.summary())

#%%
# ===========================================================================
# 2. CALCIUM TRANSIENT (array pCa = a time course)
# ===========================================================================
print("\n2. Calcium Transient (Array pCa)")
print("-" * 50)

n_steps = 200
time_ms = jnp.arange(n_steps)
pCa_trace = 9.0 - 5.0 * jnp.exp(-0.5 * ((time_ms - 50) / 20) ** 2)

result_twitch = run(topo, dynamic, pCa=pCa_trace, duration_ms=200, **DRV)

print(f"Peak force: {float(jnp.max(result_twitch.axial_force)):.2f} pN")
print(f"pCa trace shape: {pCa_trace.shape} -> force shape: {result_twitch.axial_force.shape}")

#%%
# ===========================================================================
# 3. pCa SWEEP (list = sweep axis)
# ===========================================================================
print("\n3. pCa Sweep (list -> sweep axis)")
print("-" * 50)

result_sweep = run(topo, dynamic, pCa=[9.0, 6.0, 5.0, 4.5], duration_ms=100,
                   replicates=3, verbose=True, **DRV)

print(f"Shape: {result_sweep.axial_force.shape}")  # (4, 3, 100) = (pCa, reps, time)
print(f"Axis names: {result_sweep._axis_names}")
print(f"pCa coords: {result_sweep.coords['pCa']}")

mean_result = result_sweep.mean()
std_result = result_sweep.std()
print(f"After .mean(): {mean_result.axial_force.shape}")   # (4, 100)
print(f"After .std():  {std_result.axial_force.shape}")    # (4, 100)

#%%
# ===========================================================================
# 4. 2D GRID SWEEP (z_line x pCa) — every list is crossed with every other
# ===========================================================================
print("\n4. 2D Grid Sweep (z_line x pCa)")
print("-" * 50)

result_2d = run(topo, dynamic, z_line=[1000.0, 1100.0, 1200.0], lattice_spacing=d0,
                pCa=[6.0, 5.0, 4.5], replicates=2, dt=1.0, duration_ms=100, verbose=True)

print(f"Shape: {result_2d.axial_force.shape}")  # (3, 3, 2, 100)
print(f"Axes: {result_2d._axis_names}")

sliced_z = result_2d.sel(z_line=1100.0)
print(f"result_2d.sel(z_line=1100.0) shape: {sliced_z.axial_force.shape}")  # (3, 2, 100)

#%%
# ===========================================================================
# 5. PARAMETER SWEEP — any DynamicParams field as a keyword:
#    scalar = override, list = sweep axis
# ===========================================================================
print("\n5. Parameter sweep (thick_k list, titin_rest override)")
print("-" * 50)

result_param = run(topo, dynamic, pCa=4.5, duration_ms=100,
                   thick_k=[5000.0, 7500.0, 10000.0], titin_rest=200.0, verbose=True, **DRV)

print(f"Shape: {result_param.axial_force.shape}")  # (3, 1, 100)
mean_forces = [float(result_param.axial_force[i].mean()) for i in range(3)]
print(f"Mean forces by thick_k: {[f'{f:.1f}' for f in mean_forces]} pN")

# A random grid inside the field's literature range for this muscle class:
r34 = draw(dynamic, 'xb_r34_coeff', 4, seed=0)
result_draw = run(topo, dynamic, pCa=4.5, duration_ms=100, xb_r34_coeff=r34, **DRV)
print(f"Drawn xb_r34_coeff {[f'{v:.3f}' for v in r34]} -> shape {result_draw.axial_force.shape}")

#%%
# ===========================================================================
# 6. REPLICATES
# ===========================================================================
print("\n6. Replicates (rng_seed variation)")
print("-" * 50)

result_reps = run(topo, dynamic, pCa=4.5, duration_ms=100, replicates=5, **DRV)
print(f"Shape: {result_reps.axial_force.shape}")  # (5, 100)
print(f"Replicate forces (mean over time): {[float(result_reps.axial_force[i].mean()) for i in range(5)]}")

#%%
# ===========================================================================
# 7. STRUCTURAL SWEEP (stack results from different topologies)
# ===========================================================================
print("\n7. Structural Sweep (SimulationResult.stack)")
print("-" * 50)

results_list = []
for nrows in [2, 3]:
    topo_n = SarcTopology.create(nrows, nrows, static)
    results_list.append(run(topo_n, dynamic, pCa=4.5, duration_ms=100, **DRV))

stacked = SimulationResult.stack(results_list, axis_name='nrows')
print(f"Stacked shape: {stacked.axial_force.shape}")  # (2, 1, 100)
print(f"Axes: {stacked._axis_names}")

#%%
print("\n" + "=" * 70)
print("Quickstart Complete!")
print("=" * 70)
print("""
Cheatsheet:

    from multifil_jax import SarcTopology, run, get_params, draw

    static, dynamic, z0, d0 = get_params('skeletal')   # or 'cardiac_beta',
    topo = SarcTopology.create(2, 2, static)           # 'cardiac_alpha', 'insect'

    # drivers: scalar = constant, list = sweep axis, array = time course
    result = run(topo, dynamic, pCa=4.5, z_line=z0, lattice_spacing=d0,
                 dt=1.0, duration_ms=1000)

    # sweeps: every list is an axis, all axes crossed
    result = run(topo, dynamic, pCa=[9.0, 6.0, 4.5], z_line=z0, lattice_spacing=d0,
                 dt=1.0, duration_ms=1000, thick_k=[5000, 7500], replicates=5)
    result = run(topo, [dynamic, dynamic.copy(xb_r12_coeff=0.3)], ...)  # candidates

    print(result.summary())
    mean = result.mean()            # collapse replicates axis
    sliced = result.sel(pCa=4.5)    # select by coordinate value
""")
