#!/usr/bin/env python
# NOTE: Designed for interactive Jupyter/IPython use (VS Code or jupyter notebook).
# Plain `python examples/test.py` will fail due to %magic syntax.


#%%


# %load_ext autoreload
# %autoreload 2


#%%
import numpy as np
import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from multifil_jax.core.params import get_skeletal_params, StaticParams, DynamicParams
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.simulation import run, SimulationResult


#%% define corner plot

def plot_metric_corner(results, metric_name='axial_force', last_n=20, cmap='viridis'):
    """Generate corner plot for parameter sweep results.

    Args:
        results: SimulationResult from run()
        metric_name: 'axial_force' or a key from results.metrics
        last_n: Number of timesteps to average for steady-state
        cmap: Colormap for heatmaps
    """
    # 1. Extract data: (P1, P2, ..., Replicates, Time)
    data = results.metrics[metric_name]

    # 2. Collapse Time (always the last axis)
    grid_data = np.mean(data[..., -last_n:], axis=-1)

    # 3. Collapse Replicates to sync array rank with parameter count
    if results.replicate_axis is not None:
        grid_data = np.mean(grid_data, axis=results.replicate_axis)

    # 4. Get only the physical parameter names (excludes 'replicates' and 'time')
    param_names = [n for n in results._axis_names if n not in ('replicates', 'time')]
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params),
                             constrained_layout=True, squeeze=False)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')
                continue

            # Calculate marginal: grid_data rank is now exactly n_params
            axes_to_avg = tuple(k for k in range(n_params) if k not in (i, j))
            marginal = np.mean(grid_data, axis=axes_to_avg)

            if i == j:
                # Diagonal: 1D Plot
                x_vals = results.coords[param_names[i]]
                ax.plot(x_vals, marginal, 'o-', color='#2c3e50', lw=2)
                ax.set_title(param_names[i])
            else:
                # Off-diagonal: 2D Heatmap
                ax.imshow(marginal.T if i > j else marginal, origin='lower', aspect='auto', cmap=cmap)
                if j == 0: ax.set_ylabel(param_names[i])
                if i == n_params - 1: ax.set_xlabel(param_names[j])

    plt.suptitle(f"Corner Plot: {metric_name}")
    plt.show()


#%%
# =============================================================================
# STIFFNESS PARAMETER SWEEP
# =============================================================================

print("Creating topology and running stiffness parameter sweep...")

static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, 
                           ncols=2, 
                           static_params=static, 
                           dynamic_params=dynamic)
topo = jax.device_put(topo)

# Stiffnesses
# thick_sweep = [float(dynamic.thick_k) * (i * 0.5) for i in range(1, 10)]
# thin_sweep = [float(dynamic.thin_k) * (i * 0.5) for i in range(1, 10)]
# titin_sweep = [float(dynamic.titin_b) * (i * 0.5) for i in range(1, 10)]
# xb_c_k_strong_sweep = [float(dynamic.xb_c_k_strong) * (i * 0.5) for i in range(1, 10)]
# xb_g_k_strong_sweep = [float(dynamic.xb_g_k_strong) * (i * 0.5) for i in range(1, 10)]

# srx -> drx
# xb_srx_b_sweep = [float(dynamic.xb_srx_b) * (i * 0.5) for i in range(1, 10)]
xb_srx_kmax_sweep = [float(dynamic.xb_srx_kmax) * (i * 0.5) for i in range(1, 30)]
# xb_srx_ca50_sweep = [float(dynamic.xb_srx_ca50) * (i * 0.5) for i in range(1, 10)]

xb_r45_coeff_sweep = [float(dynamic.xb_r45_coeff) * (i * 0.5) for i in range(1, 30)]


print("Compiling + executing sweep...")
start = time.time()
results = run(
    topo,
    pCa=[4, 4.5, 5, 5.5, 6, 6.5, 7],
    z_line=1100,
    lattice_spacing=[14.,15.,16.,17.,18.],
    duration_ms=1000,
    dt=1.0,
    dynamic_params={
        # 'xb_srx_b': xb_srx_b_sweep,
        # 'xb_srx_kmax': xb_srx_kmax_sweep,
        # 'xb_r45_coeff': xb_r45_coeff_sweep,
        # 'xb_r51': xb_r51_sweep,
        # 'xb_srx_ca50': xb_srx_b_sweep,
        # 'thick_k': thick_sweep,
        # 'thin_k': thin_sweep,
        # 'titin_b': titin_sweep,
        # 'xb_c_k_strong': xb_c_k_strong_sweep,
        # 'xb_g_k_strong': xb_g_k_strong_sweep,
    },
    replicates=10,
)

results.axial_force.block_until_ready()
end = time.time()
print(f"Sweep completed in {end-start:.2f}s")
print(results.summary())


#%%
# Generate corner plot

# Derived metric
# results.metrics['energy_per_atp'] = results.metrics['thick_energy_first_delta_avg'] / results.metrics['atp_expected_q']
# results.metrics['solver_residual'] = results.solver_residual

plot_metric_corner(results, metric_name='axial_force', last_n=900)


#%%
# =============================================================================
# SINGLE SIMULATION BENCHMARK
# =============================================================================
print("\n" + "=" * 60)
print("Single Simulation Benchmark")
print("=" * 60)

# First run (includes JIT compilation)
start = time.time()
result = run(topo, 
             pCa=4.0, 
             z_line=1100.0, 
             duration_ms=10, 
             dt=1.0)
result.axial_force.block_until_ready()
print(f"First run (with JIT): {time.time() - start:.2f}s")

# Second run (cached)
start = time.time()
result = run(topo, 
             pCa=4.0, 
             z_line=1100.0, 
             duration_ms=10, 
             dt=1.0, 
             rng_seed=1)
result.axial_force.block_until_ready()
print(f"Second run (cached): {time.time() - start:.2f}s")

print(result.summary())


#%%
# =============================================================================
# PROFILING (Optional)
# =============================================================================

print("\n" + "=" * 60)
print("Profiling")
print("=" * 60)

print('Executing with profiler...')
start = time.time()
# jax.profiler.start_trace("/tmp/jax-trace")
result = run(topo, 
             pCa=4.0, 
             z_line=1100.0, 
             duration_ms=25, 
             dt=1.0)
result.axial_force.block_until_ready()
# jax.profiler.stop_trace()
end = time.time()
print(f'Execution Time: {end-start:.2f}s')
# print("Trace saved to /tmp/jax-trace")
# print("Open https://ui.perfetto.dev/ and drag the trace file to view")


#%%
# =============================================================================
# LARGER GRID EXAMPLE (15x15 sweep)
# =============================================================================

static, dynamic = get_skeletal_params()
topo_4x4 = SarcTopology.create(nrows=4, 
                               ncols=4, 
                               static_params=static, 
                               dynamic_params=dynamic)
topo_4x4 = jax.device_put(topo_4x4)

thick_sweep = [float(dynamic.thick_k) * (i * 0.25) for i in range(1, 16)]
thin_sweep = [float(dynamic.thin_k) * (i * 0.25) for i in range(1, 16)]

print(f"15x15 sweep = {len(thick_sweep) * len(thin_sweep)} runs")
# result = run(topo_4x4, pCa=4, z_line=1100, lattice_spacing=14,
#              duration_ms=1000, dynamic_params={'thick_k': thick_sweep, 'thin_k': thin_sweep})


#%%
# =============================================================================
# STRUCTURAL SWEEP EXAMPLE (SimulationResult.stack)
# =============================================================================

print("\n" + "=" * 60)
print("Structural Sweep (different topologies)")
print("=" * 60)

results_by_size = []
sizes = [2, 3]
for n in sizes:
    topo_n = SarcTopology.create(nrows=n, 
                                 ncols=n, 
                                 static_params=static, 
                                 dynamic_params=dynamic)
    topo_n = jax.device_put(topo_n)
    r = run(topo_n, 
            pCa=4.0, 
            z_line=1100.0, 
            duration_ms=10, 
            dt=1.0)
    results_by_size.append(r)
    print(f"  {n}x{n}: force shape = {r.axial_force.shape}")

stacked = SimulationResult.stack(results_by_size, axis_name='lattice_size', axis_values=sizes)
print(f"Stacked shape: {stacked.axial_force.shape}")
print(f"Stacked axes: {stacked._axis_names}")








