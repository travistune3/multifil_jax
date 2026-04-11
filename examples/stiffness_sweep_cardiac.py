"""Stiffness sweep with cardiac params, plotting thick_energy_first_delta_avg."""

import time
import numpy as np
import matplotlib.pyplot as plt
import jax

from multifil_jax.core.params import get_cardiac_params
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.simulation import run

# ── Topology ─────────────────────────────────────────────────────────────────
static, dynamic = get_cardiac_params()
topo = SarcTopology.create(nrows=4, ncols=4, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

# ── Sweep definition ──────────────────────────────────────────────────────────
thick_sweep = [float(dynamic.thick_k) * f for f in np.linspace(0.25, 3.0, 12)]
thin_sweep  = [float(dynamic.thin_k)  * f for f in np.linspace(0.25, 3.0, 12)]

print(f"Sweep: {len(thick_sweep)} thick × {len(thin_sweep)} thin = {len(thick_sweep)*len(thin_sweep)} conditions")

# ── Run ───────────────────────────────────────────────────────────────────────
print("Compiling + running...")
t0 = time.time()
results = run(
    topo,
    pCa=4.5,
    z_line=1000.0,
    duration_ms=1000,
    dt=1.0,
    replicates=3,
    dynamic_params={
        'thick_k': thick_sweep,
        'thin_k':  thin_sweep,
    },
)
results.metrics['axial_force'].block_until_ready()
print(f"Done in {time.time()-t0:.1f}s")

# ── Extract metric ─────────────────────────────────────────────────────────────
# Shape: (thick_k, thin_k, replicates, time)
data = np.array(results.metrics['thick_energy_first_delta_avg'])

# Steady-state mean: last 100 timesteps, averaged over replicates and time
steady = data[..., -200:].mean(axis=(-1, -2))   # (thick_k, thin_k)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(
    steady.T,
    origin='lower',
    aspect='auto',
    extent=[thick_sweep[0], thick_sweep[-1], thin_sweep[0], thin_sweep[-1]],
    cmap='RdBu_r',
)
fig.colorbar(im, ax=ax, label='thick_energy_first_delta_avg (pN·nm)')

ax.set_xlabel('thick_k (pN/nm)')
ax.set_ylabel('thin_k (pN/nm)')
ax.set_title('Stiffness sweep — cardiac params\nSteady-state Δ thick-filament tip energy')

plt.tight_layout()
out = 'examples/stiffness_sweep_cardiac.png'
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
