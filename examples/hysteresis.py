#!/usr/bin/env python


#%%


%load_ext autoreload
%autoreload 2


#%%
import numpy as np
import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from multifil_jax.core.params import get_default_params, StaticParams, DynamicParams
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.simulation import run, SimulationResult


#%%

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

static, dynamic = get_default_params()
topo_vert = SarcTopology.create(nrows=2, 
                                  ncols=2, 
                                  static_params=static, 
                                  dynamic_params=dynamic)
topo_vert = jax.device_put(topo_vert)

# Create parameter sweep values
thick_sweep = [float(dynamic.thick_k) * (i * 0.5) for i in range(1, 20)]
thin_sweep = [float(dynamic.thin_k) * (i * 0.5) for i in range(1, 20)]
titin_sweep = [float(dynamic.titin_b) * (i * 0.5) for i in range(1, 20)]

print("Compiling + executing sweep...")
start = time.time()
results_vertebrate = run(
    topo_vert,
    pCa=4,
    z_line=900,
    lattice_spacing=14,
    duration_ms=1000,
    dt=1.0,
    dynamic_params={
        'thick_k': thick_sweep,
        'thin_k': thin_sweep,
        'titin_b': titin_sweep,
    },
    replicates=1,
)
end = time.time()
print(f"Sweep completed in {end-start:.2f}s")
print(results_vertebrate.summary())


#%%
# 
# Derived metric

pseudo_m6 = results.metrics['thick_displace_mean']

force = results.axial_force


# hystereisis = ...


# plot_metric_corner(hystereisis, metric_name='thick_energy_first_avg', last_n=5)

#%%



static, dynamic = get_default_params()
topo_invert = SarcTopology.create(nrows=2, 
                            ncols=2, 
                            static_params=static.replace(actin_geometry = 'invertebrate'), 
                            dynamic_params=dynamic)
topo_invert = jax.device_put(topo_invert)



print("Compiling + executing sweep...")
start = time.time()
results_invertebrate = run(
    topo_invert,
    pCa=4,
    z_line=900,
    lattice_spacing=14,
    duration_ms=1000,
    dt=1.0,
    dynamic_params={
        'thick_k': thick_sweep,
        'thin_k': thin_sweep,
        'titin_b': titin_sweep,
    },
    replicates=1,
)
end = time.time()
print(f"Invertebrate sweep completed in {end-start:.2f}s")


#%%
# =============================================================================
# ANALYSIS
# =============================================================================

import numpy as np

# Steady-state force: mean of last 100 timesteps, collapsed across sweep axes
last_n = 100

vert_ss_force = np.array(results_vertebrate.axial_force[..., -last_n:].mean(axis=(-2, -1)))
invert_ss_force = np.array(results_invertebrate.axial_force[..., -last_n:].mean(axis=(-2, -1)))

print("\nSteady-state force comparison (vertebrate vs invertebrate):")
print(f"  Vertebrate:   mean={vert_ss_force.mean():.1f} pN, max={vert_ss_force.max():.1f} pN")
print(f"  Invertebrate: mean={invert_ss_force.mean():.1f} pN, max={invert_ss_force.max():.1f} pN")

# Thick filament displacement metric
vert_disp = np.array(results_vertebrate.metrics['thick_displace_mean'][..., -last_n:].mean(axis=(-2, -1)))
invert_disp = np.array(results_invertebrate.metrics['thick_displace_mean'][..., -last_n:].mean(axis=(-2, -1)))

print("\nThick filament displacement comparison:")
print(f"  Vertebrate:   mean={vert_disp.mean():.3f} nm, max={vert_disp.max():.3f} nm")
print(f"  Invertebrate: mean={invert_disp.mean():.3f} nm, max={invert_disp.max():.3f} nm")

# Optional: matplotlib plots
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Force vs time (diagonal of sweep: equal stiffness scaling)
    diag_n = min(len(thick_sweep), 5)
    ax = axes[0]
    for i in range(diag_n):
        t = np.arange(results_vertebrate.axial_force.shape[-1])
        ax.plot(t, np.array(results_vertebrate.axial_force[i, i, i, 0, :]),
                label=f'vert k={thick_sweep[i]:.0f}', alpha=0.7)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Force (pN)')
    ax.set_title('Vertebrate Force Traces (diagonal sweep)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Force vs stiffness (diagonal)
    ax = axes[1]
    ks = thick_sweep[:diag_n]
    ax.plot(ks, vert_ss_force.diagonal()[:diag_n], 'o-', label='Vertebrate')
    ax.plot(ks, invert_ss_force.diagonal()[:diag_n], 's-', label='Invertebrate')
    ax.set_xlabel('thick_k (pN/nm)')
    ax.set_ylabel('Steady-state Force (pN)')
    ax.set_title('Force vs Stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hysteresis_analysis.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to hysteresis_analysis.png")
    plt.show()
except ImportError:
    print("\n(matplotlib not available, skipping plots)")










