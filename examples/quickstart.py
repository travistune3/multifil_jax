#!/usr/bin/env python
"""
Quickstart: JAX Muscle Simulation (v3.0 API)

Demonstrates the canonical v3.0 patterns using run() + SarcTopology.

Run this script:
    python examples/quickstart.py
"""
#%%
import jax
import jax.numpy as jnp
import numpy as np
import time

from multifil_jax.simulation import run, get_skeletal_params, SimulationResult
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams

print("=" * 70)
print("JAX Muscle Simulation - Quickstart (v3.0 API)")
print("=" * 70)

#%%
# ===========================================================================
# SETUP: Create topology once, reuse across all runs
# ===========================================================================
static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

print(f" Topology: {topo.n_thick} thick, {topo.n_thin} thin filaments")
print(f" Crowns: {topo.n_crowns}/thick, Sites: {topo.n_sites}/thin")
print(f" Total XBs: {topo.total_xbs}")

#%%
# ===========================================================================
# 1. SIMPLE ISOMETRIC SIMULATION
# ===========================================================================
print("\n1. Simple Isometric Simulation")
print("-" * 50)

start = time.time()
result = run(topo, pCa=4.5, z_line=1100.0, duration_ms=100, dt=1.0, verbose=True)
elapsed = time.time() - start

print(f"Elapsed: {elapsed:.2f}s")
print(f"Result shape: {result.axial_force.shape}")  # (1, 100) = (replicates, time)
print(f"Mean force: {float(result.axial_force.mean()):.2f} pN")
print(f"Available metrics: {list(result.metrics.keys())[:5]} ...")
print(result.summary())

#%%
# ===========================================================================
# 2. CALCIUM TRANSIENT (Array pCa Input)
# ===========================================================================
print("\n2. Calcium Transient (Array pCa)")
print("-" * 50)

n_steps = 200
time_ms = jnp.arange(n_steps)
pCa_trace = 9.0 - 5.0 * jnp.exp(-0.5 * ((time_ms - 50) / 20) ** 2)

result_twitch = run(topo, pCa=pCa_trace, z_line=1100.0, duration_ms=200, dt=1.0)

print(f"Peak force: {float(jnp.max(result_twitch.axial_force)):.2f} pN")
print(f"pCa trace shape: {pCa_trace.shape} -> force shape: {result_twitch.axial_force.shape}")

#%%
# ===========================================================================
# 3. pCa SWEEP (List -> Sweep Axis)
# ===========================================================================
print("\n3. pCa Sweep (list -> sweep axis)")
print("-" * 50)

result_sweep = run(
    topo,
    pCa=[9.0, 6.0, 5.0, 4.5],   # List = sweep axis
    z_line=1100.0,
    duration_ms=100,
    replicates=3,
    verbose=True,
)

print(f"Shape: {result_sweep.axial_force.shape}")  # (4, 3, 100) = (pCa, reps, time)
print(f"Axis names: {result_sweep._axis_names}")
print(f"pCa coords: {result_sweep.coords['pCa']}")

# Collapse replicate axis
mean_result = result_sweep.mean()
std_result = result_sweep.std()
print(f"After .mean(): {mean_result.axial_force.shape}")   # (4, 100)
print(f"After .std():  {std_result.axial_force.shape}")    # (4, 100)

#%%
# ===========================================================================
# 4. 2D GRID SWEEP (z_line x pCa)
# ===========================================================================
print("\n4. 2D Grid Sweep (z_line x pCa)")
print("-" * 50)

result_2d = run(
    topo,
    z_line=[1000.0, 1100.0, 1200.0],  # axis 0
    pCa=[6.0, 5.0, 4.5],           # axis 1
    replicates=2,
    duration_ms=100,
    verbose=True,
)

print(f"Shape: {result_2d.axial_force.shape}")  # (3, 3, 2, 100)
print(f"Axes: {result_2d._axis_names}")

# Slice by coordinate
sliced_z = result_2d.sel(z_line=1100.0)
print(f"result_2d.sel(z_line=1100.0) shape: {sliced_z.axial_force.shape}")  # (3, 2, 100)

#%%
# ===========================================================================
# 5. DynamicParams DICT SWEEP (thick_k values)
# ===========================================================================
print("\n5. DynamicParams Sweep (thick_k)")
print("-" * 50)

result_param = run(
    topo,
    pCa=4.5,
    z_line=1100.0,
    duration_ms=100,
    dynamic_params={'thick_k': [1000.0, 2020.0, 4000.0]},
    verbose=True,
)

print(f"Shape: {result_param.axial_force.shape}")  # (3, 1, 100)
mean_forces = [float(result_param.axial_force[i].mean()) for i in range(3)]
print(f"Mean forces by thick_k: {[f'{f:.1f}' for f in mean_forces]} pN")

#%%
# ===========================================================================
# 6. REPLICATES
# ===========================================================================
print("\n6. Replicates (rng_seed variation)")
print("-" * 50)

result_reps = run(topo, pCa=4.5, z_line=1100.0, duration_ms=100, replicates=5)
print(f"Shape: {result_reps.axial_force.shape}")  # (5, 100)
print(f"Replicate forces (mean over time): {[float(result_reps.axial_force[i].mean()) for i in range(5)]}")

#%%
# ===========================================================================
# 7. INVERTEBRATE GEOMETRY
# ===========================================================================
print("\n7. Invertebrate Geometry")
print("-" * 50)

static_invert = StaticParams(actin_geometry='invertebrate')
topo_invert = SarcTopology.create(nrows=2, ncols=2, static_params=static_invert, dynamic_params=dynamic)
topo_invert = jax.device_put(topo_invert)

result_invert = run(topo_invert, pCa=4.5, z_line=1100.0, duration_ms=100)
print(f"Invertebrate mean force: {float(result_invert.axial_force.mean()):.2f} pN")

#%%
# ===========================================================================
# 8. STRUCTURAL SWEEP (stack results from different topologies)
# ===========================================================================
print("\n8. Structural Sweep (SimulationResult.stack)")
print("-" * 50)

results_list = []
for nrows in [2, 3]:
    topo_n = SarcTopology.create(nrows=nrows, ncols=nrows, static_params=static, dynamic_params=dynamic)
    topo_n = jax.device_put(topo_n)
    r = run(topo_n, pCa=4.5, z_line=1100.0, duration_ms=100)
    results_list.append(r)

stacked = SimulationResult.stack(results_list, axis_name='nrows')
print(f"Stacked shape: {stacked.axial_force.shape}")  # (2, 1, 100)
print(f"Axes: {stacked._axis_names}")

#%%
# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "=" * 70)
print("Quickstart Complete!")
print("=" * 70)
print("""
v3.0 API Cheatsheet:

    from multifil_jax.simulation import run, get_skeletal_params, SimulationResult
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.params import StaticParams

    # Create topology once (reuse across runs)
    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

    # Run simulation (pCa/z_line: scalar=constant, list=sweep, array=trace)
    result = run(topo, pCa=4.5, z_line=1100.0, duration_ms=1000)

    # Parameter sweeps
    result = run(topo, pCa=[9.0, 6.0, 4.5], replicates=5)
    result = run(topo, pCa=4.5, dynamic_params={'thick_k': [1000, 2000, 3000]})

    # Analyze results
    print(result.summary())
    mean = result.mean()        # collapse replicates axis
    std  = result.std()         # standard deviation over replicates
    sliced = result.sel(pCa=4.5)  # select by coordinate value

    # Invertebrate geometry
    static = StaticParams(actin_geometry='invertebrate')
""")
