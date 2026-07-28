"""
Vertebrate vs insect flight muscle: what does the GEOMETRY alone change?

Vertebrate striated muscle and insect indirect flight muscle (IFM) are built
from the same proteins but packed very differently. This example holds the
chemistry fixed and varies only the structure, to ask a narrow question:

    with identical crossbridge and tropomyosin kinetics, how much does the
    lattice geometry alone change force production?

WHAT DIFFERS BETWEEN THE TWO PRESETS

                              vertebrate        Lethocerus (IFM)
    thick : thin ratio        1:2               1:3
    thin filament sits at     interstices       hexagon edge midpoints
    faces per thin filament   3                 2
    myosin heads per crown    3                 4
    azimuthal step per crown  60 deg            33.75 deg
    crowns per half filament  52                100
    thick filament tip at     787 nm            1494 nm
    actin helix               26 monomers/12 turns   28/13
    actin half pitch          36.0 nm           38.7 nm

Because 33.75 deg rotation is not a multiple of the
60 deg separating hexagonal neighbours many heads end up pointing at empty space. Those heads can never bind.
The model flags them in `topology.xb_valid`, and at these settings they are the
MAJORITY of crossbridge slots — see the structural summary the script prints.


1. THE KINETICS ARE NOT INSECT KINETICS. Both presets use identical, unmodified
   skeletal rate constants. The only difference here is xb and filament geometry.

2. Titin differs in the two sets, so passive force
   is not comparable between the two, and every force below is
   PASSIVE-SUBTRACTED: active = F(pCa 4.5) - F(pCa 9) at the same z_line.

3. Z-LINE IS NOT "MATCHED", because there is no principled way to match it.
   Each species is run across its own anatomical range. A sweep rather than a
   single point is used deliberately, so you can see whether a difference is
   robust or an artifact of where the operating point was placed.

4. FORCE PER FILAMENT CONFLATES LENGTH WITH GEOMETRY. The IFM thick filament
   carries roughly twice as many crowns, so more force per filament may say only
   that the filament is longer. Force per BOUND crossbridge is the more
   geometry-specific readout, and both are reported.

6. The thin filament's per-segment stiffness is a leaky invariant across these
   presets: the wider IFM acceptance window creates more, shorter spring
   segments, leaving the IFM thin filament ~19% more compliant per unit length
   at equal `thin_k`. See the `thin_k` notes in core/params.py.

Run:
    python examples/vertebrate_vs_invertebrate_geometry.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from multifil_jax import SarcTopology, run
from multifil_jax.core.params import get_skeletal_params, get_lethocerus_params

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Invertebrate geometry requires an even nrows: under periodic boundaries an
# odd row count leaves thick filament faces with no thin filament neighbour, and
# SarcTopology.create() raises rather than building a malformed lattice.
NROWS, NCOLS = 4, 4

PCA_RELAXED = 9.0     # passive reference
PCA_ACTIVE = 4.5      # near-saturating activation
DURATION_MS = 400.0
STEADY_MS = 100       # average over the final this-many ms
REPLICATES = 3

# Each species is swept across its own anatomical range. Vertebrate titin sits
# at rest near z = 1002 nm; the Lethocerus connecting filament near z = 1544 nm.
Z_LINES = {
    'vertebrate': [1000.0, 1050.0, 1100.0, 1150.0],   # sarcomere 2.0-2.3 um
    'lethocerus': [1500.0, 1525.0, 1550.0, 1575.0],   # sarcomere 3.0-3.2 um
}

PRESETS = {
    'vertebrate': get_skeletal_params,
    'lethocerus': get_lethocerus_params,
}

# Solver settings are held IDENTICAL across species, so that any difference in
# the results is geometry and not solver tuning.
#
# Both are raised from their defaults for the IFM case. The default Newton cap
# of 4 leaves residuals near 17 pN at some IFM lengths; raising it to 16 brings
# them to ~1.8 pN, and beyond that more iterations change nothing — that is a
# float32 precision floor, not a failure to converge. The floor is higher here
# than for vertebrate because IFM nodes sit at ~1500 nm rather than ~1100 nm and
# there are more of them, and float32 spacing grows with magnitude. Forces at
# 4, 16 and 32 Newton steps agree to the printed precision.
N_NEWTON_STEPS = 16
SOLVER_RESIDUAL_TOL = 2.5   # pN — above the achievable floor for both geometries

# ---------------------------------------------------------------------------
# CAVEAT 1, ENFORCED: confirm the two presets really do share their kinetics.
# ---------------------------------------------------------------------------
_KINETIC_FIELDS = [f for f in vars(type(get_skeletal_params()[1])).get('__slots__', ())
                   if f.startswith(('xb_', 'tm_'))]

_skeletal_dynamic = get_skeletal_params()[1]
_ifm_dynamic = get_lethocerus_params()[1]
_differing = [f for f in _KINETIC_FIELDS
              if float(getattr(_skeletal_dynamic, f)) != float(getattr(_ifm_dynamic, f))]
assert not _differing, (
    f"Presets differ in kinetics {_differing}; this comparison assumes they do not. "
    "Either the IFM preset has been given real insect kinetics (in which case this "
    "script is measuring geometry AND chemistry together and its conclusions are "
    "invalid), or something has drifted."
)
print("Kinetics identity check: all xb_*/tm_* parameters match between presets.")


# ---------------------------------------------------------------------------
# STRUCTURE
# ---------------------------------------------------------------------------
def describe(name, topo, dynamic):
    """Print the structural facts that shape how the results must be read."""
    valid_idx, _, _ = topo.valid_xb_targets()
    n_valid = int(valid_idx.size)
    tip = float(topo.crown_offsets.max())
    print(f"\n{name}")
    print(f"  thick filaments      {topo.n_thick}")
    print(f"  thin filaments       {topo.n_thin}   ({topo.n_faces_per_thin} faces each)")
    print(f"  crowns per filament  {topo.n_crowns}   x {topo.n_xb_per_crown} heads")
    print(f"  crossbridge slots    {topo.total_xbs}")
    print(f"    with a real target {n_valid}  ({100.0 * n_valid / topo.total_xbs:.1f}%)")
    print(f"  binding sites/thin   {topo.n_sites}")
    print(f"  thick filament tip   {tip:.0f} nm from M-line")
    print(f"  titin at rest near   z = {tip + float(dynamic.titin_rest):.0f} nm")
    return n_valid


topologies, n_valid_xbs, dynamics = {}, {}, {}
print("\n" + "=" * 72)
print("STRUCTURE")
print("=" * 72)
for name, factory in PRESETS.items():
    static, dynamic = factory()
    static = static.replace(n_newton_steps=N_NEWTON_STEPS,
                            solver_residual_tol=SOLVER_RESIDUAL_TOL)
    topo = SarcTopology.create(nrows=NROWS, ncols=NCOLS,
                               static_params=static, dynamic_params=dynamic)
    topologies[name] = (jax.device_put(topo), static)
    dynamics[name] = dynamic
    n_valid_xbs[name] = describe(name, topo, dynamic)

print("\nNote the crossbridge-slot line. Slots without a real geometric target")
print("are padding required by fixed-shape arrays, not heads that merely happen")
print("to be unbound. They can never bind, so any 'fraction bound' computed")
print("against the total slot count is diluted by them. This script divides by")
print("the valid count instead.")


# ---------------------------------------------------------------------------
# SIMULATE
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SIMULATING")
print("=" * 72)

results = {}
for name, (topo, static) in topologies.items():
    z_list = Z_LINES[name]
    print(f"  {name}: {len(z_list)} z_line x 2 pCa x {REPLICATES} replicates ...",
          end="", flush=True)
    t0 = time.time()
    res = run(
        topo,
        pCa=[PCA_RELAXED, PCA_ACTIVE],
        z_line=z_list,
        duration_ms=DURATION_MS,
        dt=1.0,
        replicates=REPLICATES,
        dynamic_params=dynamics[name],
        static_params=static,
    )
    res.axial_force.block_until_ready()
    print(f" {time.time() - t0:.1f}s")
    results[name] = res


# ---------------------------------------------------------------------------
# ANALYSE
# ---------------------------------------------------------------------------
def steady(metric):
    """Mean over the final STEADY_MS, then over replicates.

    Applied to a result already sliced to one pCa, so the remaining axes are
    (z_line, replicates, time) and this returns one value per z_line.
    """
    return np.asarray(metric[..., -STEADY_MS:].mean(axis=(-2, -1)))


print("\n" + "=" * 72)
print("ACTIVE FORCE (passive-subtracted: pCa 4.5 minus pCa 9 at matched z_line)")
print("=" * 72)

summary = {}
for name, res in results.items():
    # Select by coordinate VALUE rather than by axis position. run() orders
    # sweep axes internally (z_line before pCa, whatever order they were passed
    # in), so indexing by position is a good way to silently transpose a result.
    rel = res.sel(pCa=PCA_RELAXED)
    act = res.sel(pCa=PCA_ACTIVE)

    force_passive = steady(rel.axial_force)
    force_total = steady(act.axial_force)
    bound = steady(act.metrics['n_bound'])
    tm_open = steady(act.metrics['frac_tm_state_3_overlap'])
    max_resid = float(np.asarray(res.metrics['solver_residual']).max())

    active = force_total - force_passive
    force = np.stack([force_passive, force_total])
    bound = np.stack([np.zeros_like(bound), bound])
    tm_open = np.stack([np.zeros_like(tm_open), tm_open])
    n_thick = topologies[name][0].n_thick

    summary[name] = dict(
        z=np.asarray(Z_LINES[name]),
        active_per_thick=active / n_thick,
        bound_active=bound[1],
        force_per_bound=np.where(bound[1] > 0, active / np.maximum(bound[1], 1e-9), np.nan),
        frac_valid_bound=bound[1] / n_valid_xbs[name],
        tm_open=tm_open[1],
        passive_per_thick=force[0] / n_thick,
    )

    print(f"\n{name}   (max solver residual {max_resid:.2f} pN)")
    print("   z_line   active/thick   bound XBs   active/bound   bound/valid   TM open")
    print("     (nm)          (pN)                       (pN)           (%)       (%)")
    s = summary[name]
    for i, z in enumerate(s['z']):
        print(f"   {z:6.0f}   {s['active_per_thick'][i]:12.1f}   "
              f"{s['bound_active'][i]:9.1f}   {s['force_per_bound'][i]:12.2f}   "
              f"{100 * s['frac_valid_bound'][i]:11.1f}   {100 * s['tm_open'][i]:7.1f}")

print("\n  active/thick   active force per thick filament (pN)")
print("  bound XBs      crossbridges bound at pCa 4.5, lattice total")
print("  active/bound   active force divided by bound crossbridge count (pN)")
print("  bound/valid    bound as a fraction of crossbridges that CAN bind")
print("  TM open        tropomyosin in the open state, overlap zone only")


# ---------------------------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("COMPARISON (across each species' own z_line range)")
print("=" * 72)

v, l = summary['vertebrate'], summary['lethocerus']


def band(x):
    return f"{np.nanmin(x):.2f} to {np.nanmax(x):.2f}"


print(f"\n  active force per thick filament (pN)")
print(f"    vertebrate   {band(v['active_per_thick'])}")
print(f"    lethocerus   {band(l['active_per_thick'])}")
print(f"\n  active force per bound crossbridge (pN)")
print(f"    vertebrate   {band(v['force_per_bound'])}")
print(f"    lethocerus   {band(l['force_per_bound'])}")
print(f"\n  bound fraction of BINDING-CAPABLE crossbridges (%)")
print(f"    vertebrate   {band(100 * v['frac_valid_bound'])}")
print(f"    lethocerus   {band(100 * l['frac_valid_bound'])}")
print(f"\n  passive force per thick filament (pN) — NOT comparable, differing titin")
print(f"    vertebrate   {band(v['passive_per_thick'])}")
print(f"    lethocerus   {band(l['passive_per_thick'])}")

print("""
HOW TO READ THIS
  Force per filament mixes geometry with filament length: the IFM thick
  filament carries ~2x the crowns, so a difference there may only reflect size.
  Force per bound crossbridge is closer to a per-motor quantity and is the
  fairer geometric comparison. If the two point in opposite directions, the
  effect is about how many heads engage, not how hard each one pulls.

  Ranges are given rather than single numbers because each species is at its own
  operating point. A difference that holds across the whole z_line range is more
  believable than one that appears at a single length.

  None of this addresses stretch activation, which is what actually makes insect
  flight muscle work and which this model does not implement.
""")

# ---------------------------------------------------------------------------
# OPTIONAL PLOT
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, style in (('vertebrate', 'o-'), ('lethocerus', 's-')):
        s = summary[name]
        axes[0].plot(s['z'], s['active_per_thick'], style, label=name)
        axes[1].plot(s['z'], s['force_per_bound'], style, label=name)
        axes[2].plot(s['z'], 100 * s['frac_valid_bound'], style, label=name)

    for ax, ylab, title in zip(
            axes,
            ['pN per thick filament', 'pN per bound crossbridge', '% of valid crossbridges'],
            ['Active force per filament', 'Active force per bound XB', 'Bound fraction']):
        ax.set_xlabel('z-line (nm)')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Geometry-only comparison — identical kinetics, no stretch activation',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig('vertebrate_vs_invertebrate_geometry.png', dpi=150, bbox_inches='tight')
    print("Plot saved to vertebrate_vs_invertebrate_geometry.png")
except ImportError:
    print("(matplotlib not available, skipping plot)")
