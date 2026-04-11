"""Dynamic Lattice Spacing Example

Demonstrates run(..., K_lat=..., nu=...) — lattice spacing (d) as an emergent DOF
solved from radial force balance rather than prescribed as a driver.

Four panels:
    1. Isometric activation — emergent d vs time
    2. Force comparison — fixed LS vs dynamic LS
    3. K_lat sensitivity — d deviation vs stiffness
    4. Length ramp — Poisson prediction vs emergent d

Usage:
    python examples/dynamic_lattice_spacing.py

Requirements:
    matplotlib, numpy, jax, multifil_jax
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from multifil_jax.simulation import run
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams, DynamicParams, get_skeletal_params

# ---------------------------------------------------------------------------
# Topology (small 2x2 for fast examples)
# ---------------------------------------------------------------------------
static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

D0 = 14.0    # nm — reference lattice spacing
Z0 = 1100.0  # nm — reference z-line (skeletal SL ~2.2 µm)
NU = 0.5     # Poisson exponent
DT = 1.0     # ms timestep

# K_lat is per-filament stiffness (pN/nm per thick filament).
# run() multiplies by n_thick internally so d_deviation is
# lattice-size-independent. For a 2x2 topology (n_thick=4), K_lat_eff = K_lat * 4.

print("Topology created. Running four experiments...")
print("(First run will trigger JIT compilation — subsequent runs are fast.)\n")


# ===========================================================================
# Panel 1: Isometric activation — emergent d vs time
# ===========================================================================
print("Panel 1: Isometric activation...")

dur1 = 500.0   # ms
n1 = int(dur1 / DT)

# pCa trace: rest → activation → rest
pCa_trace1 = np.full(n1, 9.0)
pCa_trace1[100:300] = 4.5

result_dyn1 = run(
    topo,
    duration_ms=dur1,
    dt=DT,
    pCa=jnp.array(pCa_trace1),
    z_line=Z0,
    lattice_spacing=D0,
    K_lat=5.0,
    nu=NU,
    replicates=1,
    rng_seed=42,
    verbose=True,
)

# Extract results (shape: replicates=1, time=n1)
time1 = np.arange(n1) * DT
d_trace = np.array(result_dyn1.metrics['lattice_spacing'][0])  # (time,)
force1 = np.array(result_dyn1.metrics['axial_force'][0])

# Poisson reference at constant z_line (constant)
d_ref_isometric = D0  # z_line is constant, so d_ref = D0

print(f"  Mean d during rest:   {np.mean(d_trace[:100]):.4f} nm")
print(f"  Mean d during active: {np.mean(d_trace[100:300]):.4f} nm")
print(f"  Poisson reference:    {d_ref_isometric:.4f} nm")
print(f"  d shift during activation: {np.mean(d_trace[100:300]) - d_ref_isometric:+.4f} nm\n")


# ===========================================================================
# Panel 2: Force comparison — fixed LS vs dynamic LS
# ===========================================================================
print("Panel 2: Force comparison (fixed vs dynamic LS)...")

# Reuse same pCa trace, same conditions
result_fixed1 = run(
    topo,
    duration_ms=dur1,
    dt=DT,
    pCa=jnp.array(pCa_trace1),
    z_line=Z0,
    lattice_spacing=D0,
    replicates=1,
    rng_seed=42,
    verbose=False,
)

force_fixed = np.array(result_fixed1.metrics['axial_force'][0])
force_dynamic = force1
force_diff = force_dynamic - force_fixed

print(f"  Peak force (fixed LS):   {np.max(np.abs(force_fixed)):.1f} pN")
print(f"  Peak force (dynamic LS): {np.max(np.abs(force_dynamic)):.1f} pN")
print(f"  Max absolute difference: {np.max(np.abs(force_diff)):.2f} pN")
print(f"  Max relative difference: {np.max(np.abs(force_diff)) / max(np.max(np.abs(force_fixed)), 1e-10) * 100:.3f}%\n")


# ===========================================================================
# Panel 3: K_lat sensitivity sweep
# ===========================================================================
print("Panel 3: K_lat sensitivity sweep...")

K_lat_values = [1.0, 2.0, 5.0, 10.0, 50.0]
dur3 = 200.0
n3 = int(dur3 / DT)

# Steady-state activation: pCa=4.5 throughout (after 50ms warmup)
pCa_trace3 = np.full(n3, 9.0)
pCa_trace3[50:] = 4.5

d_deviations = []
for k in K_lat_values:
    res = run(
        topo,
        duration_ms=dur3,
        dt=DT,
        pCa=jnp.array(pCa_trace3),
        z_line=Z0,
        lattice_spacing=D0,
        K_lat=k,
        nu=NU,
        replicates=1,
        rng_seed=42,
    )
    d_t = np.array(res.metrics['lattice_spacing'][0])
    # Mean deviation from Poisson reference during steady-state activation
    d_dev = np.mean(d_t[150:]) - D0  # D0 = d_ref at constant z_line
    d_deviations.append(d_dev)
    print(f"  K_lat={k:5.1f} pN/nm -> d deviation = {d_dev:+.4f} nm")

d_deviations = np.array(d_deviations)
K_lat_arr = np.array(K_lat_values)

# Expected: d_deviation ~ 1/K_lat (linear in soft limit)
# Fit for annotation
print()


# ===========================================================================
# Panel 4: Length ramp — Poisson prediction vs emergent d
# ===========================================================================
print("Panel 4: Length ramp...")

dur4 = 400.0
n4 = int(dur4 / DT)

# Ramp z_line from 1100 to 900 nm over 200 ms, then hold (shortening by 200 nm = SL 2.2→1.8 µm)
z_trace4 = np.linspace(Z0, 900.0, n4)
z_trace4[200:] = 900.0

# pCa = 4.5 throughout (activated)
pCa4 = 4.5

result_ramp = run(
    topo,
    duration_ms=dur4,
    dt=DT,
    pCa=pCa4,
    z_line=jnp.array(z_trace4),
    lattice_spacing=D0,
    K_lat=5.0,
    nu=NU,
    replicates=1,
    rng_seed=42,
    verbose=True,
)

time4 = np.arange(n4) * DT
d_emergent = np.array(result_ramp.metrics['lattice_spacing'][0])
d_poisson = D0 * (Z0 / z_trace4) ** NU  # Poisson prediction

d_xb_perturbation = d_emergent - d_poisson

print(f"  Max Poisson d (at shortest z): {np.max(d_poisson):.3f} nm")
print(f"  Max emergent d (at shortest z): {np.max(d_emergent):.3f} nm")
print(f"  XB perturbation (mean during ramp): {np.mean(d_xb_perturbation[50:200]):+.4f} nm\n")


# ===========================================================================
# PLOT
# ===========================================================================
print("Plotting...")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Dynamic Lattice Spacing — Emergent d from Radial Force Balance", fontsize=13)

# --- Panel 1: Emergent d vs time ---
ax = axes[0, 0]
ax.axhline(d_ref_isometric, color='gray', linestyle='--', lw=1.5, label=f'Poisson ref = {d_ref_isometric:.1f} nm')
ax.plot(time1, d_trace, color='royalblue', lw=1.5, label='Emergent d')
ax.axvspan(100, 300, alpha=0.12, color='orange', label='pCa=4.5')
ax2 = ax.twinx()
ax2.plot(time1, force1, color='tomato', lw=1, alpha=0.6, linestyle=':')
ax2.set_ylabel('Axial force (pN)', color='tomato', fontsize=9)
ax2.tick_params(axis='y', labelcolor='tomato')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Lattice spacing d (nm)')
ax.set_title('Panel 1: Isometric Activation\n(K_lat=5 pN/nm, nu=0.5)')
ax.legend(fontsize=8, loc='upper right')

# --- Panel 2: Force comparison ---
ax = axes[0, 1]
ax.plot(time1, force_fixed, color='steelblue', lw=1.5, label='Fixed LS')
ax.plot(time1, force_dynamic, color='tomato', lw=1.5, label='Dynamic LS', linestyle='--')
ax.axvspan(100, 300, alpha=0.1, color='orange')
ax3 = ax.twinx()
ax3.plot(time1, force_diff, color='purple', lw=1, alpha=0.7)
ax3.set_ylabel('Force difference (pN)', color='purple', fontsize=9)
ax3.tick_params(axis='y', labelcolor='purple')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Axial force (pN)')
ax.set_title('Panel 2: Force Comparison\nFixed LS vs Dynamic LS')
ax.legend(fontsize=8)

# --- Panel 3: K_lat sensitivity ---
ax = axes[1, 0]
ax.semilogx(K_lat_values, d_deviations, 'o-', color='darkgreen', lw=2, ms=7)
ax.axhline(0, color='gray', linestyle='--', lw=1)
ax.set_xlabel('K_lat (pN/nm)')
ax.set_ylabel('d deviation from Poisson (nm)')
ax.set_title('Panel 3: K_lat Sensitivity\n(pCa=4.5 steady state)')
ax.grid(True, which='both', alpha=0.3)

# Annotate expected 1/K scaling
if len(d_deviations) > 1 and d_deviations[0] != 0:
    ax.annotate('Slope ~ -1/K_lat', xy=(2.0, d_deviations[1]),
                fontsize=8, color='darkgreen',
                xytext=(10, d_deviations[1] * 0.6),
                arrowprops=dict(arrowstyle='->', color='darkgreen'))

# --- Panel 4: Length ramp ---
ax = axes[1, 1]
ax.plot(time4, d_poisson, color='gray', lw=2, linestyle='--', label=f'Poisson (nu={NU})')
ax.plot(time4, d_emergent, color='royalblue', lw=1.5, label='Emergent d')
ax4 = ax.twinx()
ax4.fill_between(time4, d_xb_perturbation, 0, alpha=0.25, color='orange', label='XB perturbation')
ax4.plot(time4, d_xb_perturbation, color='orange', lw=1, alpha=0.8)
ax4.set_ylabel('XB perturbation (nm)', color='darkorange', fontsize=9)
ax4.tick_params(axis='y', labelcolor='darkorange')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Lattice spacing d (nm)')
ax.set_title('Panel 4: Length Ramp (900→800 nm)\n(pCa=4.5, K_lat=5 pN/nm)')
ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()

outfile = 'dynamic_lattice_spacing.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {outfile}")
plt.show()
