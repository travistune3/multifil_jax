#!/usr/bin/env python
"""Three-way stiffness sweep (thick_k x thin_EA x titin_b), skeletal preset.

19 values per axis, crossed: 6,859 simulations in one run() call. The result is
shown as a corner plot of steady-state force — each diagonal panel is one
stiffness averaged over the other two, each off-diagonal panel a pair.

The %% markers are cells for VS Code / Jupyter interactive mode.
"""

#%%
import time

import numpy as np
import matplotlib.pyplot as plt

from multifil_jax.core.params import get_params
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.simulation import run


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

    fig.suptitle(f"Corner plot: {metric_name}")
    return fig


#%%
print("Creating topology and running stiffness parameter sweep...")

static, dynamic, z0, d0 = get_params('skeletal')
topo_vert = SarcTopology.create(2, 2, static)

# Create parameter sweep values
thick_sweep = [float(dynamic.thick_k) * (i * 0.5) for i in range(1, 20)]
thin_sweep = [float(dynamic.thin_EA) * (i * 0.5) for i in range(1, 20)]
titin_sweep = [float(dynamic.titin_b) * (i * 0.5) for i in range(1, 20)]

print("Compiling + executing sweep...")
start = time.time()
results_vertebrate = run(
    topo_vert, dynamic,
    pCa=4,
    z_line=1100,
    lattice_spacing=d0,
    duration_ms=1000,
    dt=1.0,
    thick_k=thick_sweep,
    thin_EA=thin_sweep,
    titin_b=titin_sweep,
    replicates=1,
)
end = time.time()
print(f"Sweep completed in {end-start:.2f}s")
print(results_vertebrate.summary())


#%%
force = np.array(results_vertebrate.axial_force[..., -100:].mean(axis=(-2, -1)))
print(f"\nSteady-state force: mean={force.mean():.1f} pN, max={force.max():.1f} pN")

fig = plot_metric_corner(results_vertebrate, 'axial_force', last_n=100)
out = __file__.replace('.py', '.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Plot saved to {out}")
