# LOGIC_FLOW: Architectural Guide to the JAX Half-Sarcomere Model (v3.0)

This document traces the complete execution logic of the JAX-based half-sarcomere
simulation. Each section cross-references the relevant file and approximate line
numbers to help navigate the codebase.

---

## 1. Architecture Overview

### Tiered Design

The v3.0 architecture separates concerns into four tiers:

| Tier | Name | Type | Description |
|------|------|------|-------------|
| 0 | **State** | `State` NamedTuple | Pure simulation state (no params/geometry) |
| 1 | **Topology** | `SarcTopology` | Structural index maps. Changing requires recompile. |
| 2 | **Constants** | `DynamicParams` | Physics values. Sweepable without recompile. Alias: `Constants` |
| 3 | **Drivers** | `Drivers` | Per-timestep values (pCa, z_line, lattice_spacing) |

**Orchestrator signature:** `kinetics_step(state, constants, drivers, topology, rng_key, *, dt)`

### Explicit driver threading

The drivers are **not** in `DynamicParams`, and there is no fallback value
behind them. `run()` requires all three (keyword-only, no defaults) and rejects
non-finite input. Each preset returns its natural operating point alongside the
params: `static, dynamic, z0, d0 = get_skeletal_params()`.

`Drivers(pCa, z_line, lattice_spacing)` is a plain NamedTuple used only at the
orchestration layer — `kinetics_step`, `timestep`, `compute_all_metrics` and
`KineticsTrace`. Below that, each kernel takes the scalars it actually uses as
required positional arguments right after `topology`, so its signature states
its driver dependence:

| Kernel | Driver arguments |
|--------|------------------|
| `update_nearest_neighbors(state, topology, z_line, lattice_spacing)` | both geometry drivers |
| `thin_transitions(state, constants, topology, pCa, rng_key, dt, ...)` | `pCa` |
| `xb_binned_generator(state, constants, topology, pCa, lattice_spacing, dt, ...)` | `pCa`, spacing |
| `solve_equilibrium(state, constants, topology, z_line, lattice_spacing, ...)` | both (spacing = initial d in dynamic LS) |
| `compute_forces_from_state_vectorized(state, constants, topology, z_line, lattice_spacing)` | both |
| `axial_force_at_mline(state, constants, topology)` | none |

### vmap-outside-scan Architecture

```
run() → vmap(run_single_sim) → lax.scan(scan_fn)
         ↑ batch dim                 ↑ time dim
```

All batch elements run in a single fused XLA kernel. XLA fuses vmap+scan
into one GPU kernel — maximum parallelism with minimum kernel launch overhead.

---

## 2. Primary API: `run()`

**File:** `multifil_jax/simulation.py`

```python
result = run(
    topology,            # SarcTopology (Tier 1); everything after is keyword-only
    *,
    pCa,                 # REQUIRED  float | list[float] | array(n_steps)
    z_line,              # REQUIRED  float | list[float] | array(n_steps)
    lattice_spacing,     # REQUIRED  float | list[float] | array(n_steps)
    duration_ms=1000.0,
    dt=1.0,
    K_lat=None,          # float | list[float] | None  — lattice stiffness
    nu=0.0,              # float | list[float]          — Poisson exponent
    dynamic_params=None, # DynamicParams | dict[str, float|list] | list[DynamicParams]
    static_params=None,  # StaticParams — solver knobs + residual warning threshold
    replicates=1,
    rng_seed=0,
    unroll=1,
    minibatch_size="auto",  # "auto" | int | None
    verbose=False,
    subpopulation=None,  # Subpopulation | list[Subpopulation]
)
```

**Input semantics:**
- `float` → broadcast to all timesteps (constant)
- `list[float]` → Cartesian product sweep axis
- `array(n_steps)` → time-varying trace

**`static_params`**: `run()` reads `n_cg_steps` and `n_newton_steps` (both
JIT-static) from it. Defaults to
`StaticParams()` if omitted — pass the *same* StaticParams used to build the
topology whenever it was customized.

**`dynamic_params` forms:**
| Form | Meaning |
|------|---------|
| `DynamicParams` | base constants for every sim |
| `dict[str, float]` | scalar overrides |
| `dict[str, list]` | one sweep axis per key |
| `list[DynamicParams]` | a `'candidates'` sweep axis (one element per entry) |

⚠️ The **dict** form builds on a fresh `DynamicParams()` (skeletal defaults), *not*
on any preset — every non-skeletal field must be restated. To sweep a cardiac/IFM
preset, use the candidate-list form: `[dynamic.copy(field=v) for v in values]`.

**Result shape convention:** `(Sweep_1, ..., Sweep_N, Replicates, Time)`

**Batch padding:** sweep sizes rounded up to `BATCH_BUCKETS = (1, 2, 4, ..., 16384)`
for JIT cache reuse. A 225-run and 256-run sweep share the same compiled kernel.

**Minibatch:** `minibatch_size="auto"` chunks large batches (≥16384) into 4096-size
pieces for L2 cache efficiency and VRAM bounding. An explicit int overrides the
heuristic; `None` disables chunking.

### Three Lattice Spacing Modes

| Mode | Parameters | Behavior |
|------|-----------|----------|
| **Fixed** | `K_lat=None, nu=0` | Lattice spacing held constant (default) |
| **Poisson** | `K_lat=None, nu>0` | `ls = d0 * (z0/z)^nu` pre-computed as time-series |
| **Dynamic** | `K_lat>0` | `d` solved as DOF from radial force balance each timestep |

`K_lat` is per-filament stiffness (pN/nm per thick filament); `run()` internally
scales by `n_thick` so that `d_deviation` is lattice-size-independent.

### Typical usage

```python
from multifil_jax.simulation import run
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams, get_skeletal_params

static, dynamic, z0, d0 = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
drv = dict(z_line=z0, lattice_spacing=d0)

# Simple isometric
result = run(topo, pCa=4.5, **drv, duration_ms=1000, static_params=static)

# pCa sweep
result = run(topo, pCa=[9.0, 6.0, 4.5], **drv, replicates=5)

# DynamicParams sweep (dict form — skeletal base, see warning above)
result = run(topo, pCa=4.5, **drv, dynamic_params={'thick_k': [5000, 7500, 10000]})

# Preset-preserving sweep (candidate-list form)
result = run(topo, pCa=4.5, **drv,
             dynamic_params=[dynamic.copy(thick_k=k) for k in (5000, 7500, 10000)])

# Dynamic lattice spacing (lattice_spacing is the reference d0 / initial guess)
result = run(topo, pCa=4.5, **drv, K_lat=5.0, nu=0.5, duration_ms=500)
```

### Species presets

`core/params.py` ships four `(StaticParams, DynamicParams, z0, d0)` factories
(z0 = natural z_line, d0 = lattice spacing, both nm; all four currently 900 / 14):

| Factory | Geometry | Notes |
|---------|----------|-------|
| `get_skeletal_params()` | vertebrate | fast-twitch skeletal, ~26 °C — the defaults |
| `get_cardiac_params()` | vertebrate | cardiac rate/titin overrides |
| `get_lethocerus_params()` | invertebrate | IFM: 1:3 lattice, 4 XB/crown, `n_crowns=100` |
| `get_drosophila_params()` | invertebrate | Lethocerus + `n_superlattice_classes=3` |

Only the first two are re-exported from the package root; import the IFM presets
from `multifil_jax.core.params`. `get_drosophila_params()` requires `nrows` even
and `ncols` a multiple of 3 (the 3-coloring must close under periodic boundaries);
`SarcTopology.create()` raises otherwise. IFM `xb_*`/`tm_*` kinetics are
skeletal-inherited placeholders, not fitted values.

---

## 3. SimulationResult

**File:** `multifil_jax/simulation.py`

```
result.axial_force      # property → metrics['axial_force'] (pN)
result.metrics          # MetricsDict of 43 metric arrays, same shape
result.z_line           # z_line trace used
result.pCa              # pCa trace used
result.metrics['solver_residual']  # Newton solver residual at each step
result.metrics['newton_iters']     # Newton iterations per step
result.metrics['lattice_spacing']  # actual LS each step (emergent if dynamic)
result.dt               # timestep (ms)
result.coords           # {'pCa': [...], 'z_line': [...], ...}
result._axis_names      # ['pCa', 'replicates', 'time']
result.topology_config  # dict: n_thick, n_thin, n_crowns, n_sites, n_titin,
                        #       n_faces_per_thin, total_xbs
result.metadata         # {'master_seed': int} — everything else is first-class
```

`MetricsDict` (defined in `core/state.py`) is a dict subclass with attribute access:
`result.metrics.axial_force == result.metrics['axial_force']`.

**Methods:**
- `.mean()` — collapse replicates axis (returns SimulationResult)
- `.std()` — standard deviation over replicates axis
- `.sel(**kwargs)` — slice by coordinate value, e.g. `.sel(pCa=4.5)`
- `SimulationResult.stack(results, axis_name, axis_values=None)` — classmethod;
  stacks independent runs (different topologies allowed) into a new outer sweep axis
- `.summary()` — human-readable text summary

`.mean()`/`.std()` both route through `_reduce_replicates(reduce_fn, suffix)`;
drivers (`z_line`/`pCa`/`lattice_spacing`) are always *averaged* over replicates
regardless of which reduction was requested.

---

## 4. JIT-compiled Kernel: `_run_sim_kernel`

**File:** `multifil_jax/simulation.py`

This is the vmapped+scanned simulation kernel:

```python
@partial(jax.jit, static_argnames=[
    'dt', 'unroll', 'is_dynamic_ls', 'n_cg_steps', 'n_newton_steps',
    'is_subpop_active', 'is_mean_field', 'subpop_has_xb', 'subpop_has_tm',
    'scaled_field_names', 'n_pops'])
def _run_sim_kernel(
    topology,           # passed via closure (vmap in_axes=None)
    batched_params,     # vmap in_axes=0
    z_batched,          # (batch, n_steps)
    pCa_batched,        # (batch, n_steps)
    ls_batched,         # (batch, n_steps)
    rng_keys,           # (batch,)
    dt, unroll,
    is_dynamic_ls=False,  # static — controls fixed vs dynamic LS code path
    K_lat_batched=None,   # (batch,) lattice stiffness
    nu_batched=None,      # (batch,) Poisson exponent
    n_cg_steps=6,         # static — from StaticParams
    n_newton_steps=16,    # static — from StaticParams (run() passes 4)
    subpop_arrays=None,   # dict of (batch, ...) arrays, or None
    is_subpop_active=False, is_mean_field=False,   # static subpop flags
    subpop_has_xb=False, subpop_has_tm=False,
    scaled_field_names=(), n_pops=1,
) -> MetricsDict:         # shape (batch, n_steps) for each key
```

`is_dynamic_ls` is a JIT static arg — fixed LS and dynamic LS compile to separate
kernels. `K_lat` and `nu` are traced (not static), so different stiffness values
share the same compiled kernel. The solver knobs and every subpopulation *shape*
flag are static too: changing any of them recompiles,
while the subpopulation *scale values* and masks ride the batch axis and do not.

Scan carry is `(state, rng_key, current_ls)` — the third element tracks the
emergent lattice spacing (identity passthrough for fixed LS).

When a subpopulation is active, `subpop_arrays` joins the vmap as one extra
(dict) axis; when inactive the vmap signature is unchanged, so the WT trace is
byte-identical to a build without the feature.

All 43 metrics are always computed. No `metrics`/`manifest` in JIT
`static_argnames` — changing metric selection never triggers recompilation.

---

## 5. Single Timestep

**File:** `multifil_jax/timestep.py`

Two public functions:

### `kinetics_step()` — stochastic phase (steps 0–5)

```python
state, rng_key, trace = kinetics_step(
    state, constants, drivers, topology, rng_key, dt=dt,
    xb_subpop=None, tm_subpop=None,
)
```

Performs driver resolution, nearest neighbors, and stochastic TM/XB transitions.
Returns a **`KineticsTrace`** (`core/state.py`) carrying everything the metrics
layer needs to describe the step that actually happened:

| field | what it is |
|---|---|
| `state` | the **mid** state — after `thin_transitions`, before `thick_transitions` |
| `constants` | driver-resolved `DynamicParams`, at the **pre-solve** lattice spacing |
| `xb_subpop` | the already-resolved `(mode, constants_k, extra)` tuple, or `None` |
| `torn` | per-head bool mask of heads tropomyosin tore off this step |

`state` is the mid one because that is what `thick_transitions` builds its
generator from; a metric that rebuilds that generator off the pre-step state
describes a step that never happened (measured bias 0.06–0.46 % on
`atp_expected`). `torn` is carried rather than re-derived because a torn head
lands where an ordinary one does. The trace **holds a whole `State`**, so it is a
within-step value only — never a scan carry or a scan output.

Separated from the mechanical solve to support future FE coupling: run kinetics
across all coupled sarcomeres, then perform a coupled equilibration.

### `timestep()` — full step (kinetics + equilibrium)

```python
(new_state, new_key, residual, new_ls, n_iters, trace,
 residual_norm, tolerance) = timestep(
    state, constants, drivers, topology, rng_key, dt=dt,
    K_lat=None, d_ref=None,
    n_cg_steps=6, n_newton_steps=16,
    precond_params=None, prefactored_precond=None,
    xb_subpop=None, tm_subpop=None,
)
```

Returns an 8-tuple. `residual_norm` is `max(|F| / tol_vec)` — dimensionless,
`<= 1` means converged — and `tolerance` is the axial tolerance in pN. The
`KineticsTrace` is the 6th element — feed it
straight to `compute_all_metrics()`. `K_lat is None` selects fixed LS mode (resolved at trace time,
no runtime branch). When `K_lat` is not None, passes `K_lat` and `d_ref` to
`solve_equilibrium()` which handles the augmented (n+1)-DOF dynamic LS solve.

**Workflow:**

1. **update_nearest_neighbors** — per-XB geometry (axial/radial distance to nearest site)
2. **thin_transitions** — TM 4-state Markov transitions, 54 unique Q matrices
3. **thick_transitions** — XB 6-state Markov transitions (binned Q → gather)
4. **solve_equilibrium** — Newton-CG solver (unified fixed/dynamic LS)

Steps 1–3 are `kinetics_step()`. Step 4 is the mechanical solve. Each kernel
gets its drivers as explicit scalars from the `Drivers` bundle (see §1).

In the kinetics phase `z_line` is read only to place the thin sites for the
nearest-site search — cooperativity is derived from neighbouring TM states, not
from filament tension.
It is deprecated and slated for removal.

`xb_subpop`/`tm_subpop` are `(mode, constants_k, extra)` tuples or `None`;
`None` reproduces the single-population path verbatim. See §15.

---

## 6. State Hierarchy

**File:** `multifil_jax/core/state.py`

```python
State(
    thick = ThickState(
        axial,          # (n_thick, n_crowns) crown positions (nm)
        xb_states,      # (n_thick, n_crowns, n_xb_per_crown) XB states (0-5), int8
        xb_bound_to,    # (n_thick, n_crowns, n_xb_per_crown) bound site index (-1=unbound)
        xb_nearest_bs,  # (n_thick, n_crowns, n_xb_per_crown) nearest BS index
        xb_distances,   # (n_thick, n_crowns, n_xb_per_crown, 2) — (axial, radial) to that site
    ),
    thin = ThinState(
        axial,           # (n_thin, n_sites) site positions (nm)
        tm_states,       # (n_thin, n_sites) TM states (0-3), int8
        bound_to,        # (n_thin, n_sites) XB address (-1=unbound)
        # rests: moved to SarcTopology (Tier 1)
        # permissiveness: derived inline as (tm_states == 3).astype(float32)
    ),
)
```

**XB state indices are 0-based** (they were 1-based in pre-3.0 notes):

| Index | Name | Meaning |
|-------|------|---------|
| 0 | DRX | disordered relaxed, detached |
| 1 | Loose | weakly bound |
| 2 | Tight_1 | strongly bound, post-stroke, phosphate released |
| 3 | Tight_2 | strongly bound, ADP-release-competent |
| 4 | Free_2 | just detached, post-ATP |
| 5 | SRX | super-relaxed / parked |

Bound heads are states 1–3. The rate-coefficient names follow the same
convention (`xb_r01_coeff`, `xb_r12_coeff`, `xb_r23_coeff`, `xb_r34_coeff`,
`xb_r40`, `xb_r04`, `xb_r05`).

**MetricsDict** — scan output. A dict subclass with attribute access, registered
as a JAX PyTree. Contains all 43 metric scalars per timestep (including
`axial_force`, `solver_residual`, `newton_iters`).

**Immutable updates** via `._replace()`:
```python
new_state = state._replace(thick=state.thick._replace(axial=new_axial))
```

**State creation:**
```python
state = realize_state(topology, constants, z_line, pCa, lattice_spacing)
```

**Drivers** — per-step values, always finite (no fallback):
```python
Drivers(pCa=4.5, z_line=900.0, lattice_spacing=14.0)
```

**KineticsTrace** carries `drivers` — the PRE-solve drivers the kinetics ran at.
The scan body builds a second, POST-solve `Drivers(pCa, z_line, new_ls)` for the
mechanics metrics. Both are passed to `compute_all_metrics` with one `constants`.

---

## 7. Topology: `SarcTopology`

**File:** `multifil_jax/core/sarc_geometry.py`

Registered as JAX PyTree. Pre-computes all structural index maps for GPU efficiency:

```python
topo = SarcTopology.create(
    nrows=2, ncols=2,
    static_params=StaticParams(),
    dynamic_params=DynamicParams(),
    periodic=True,
    lattice_spacing=14.0,
    thin_starts=None,     # None = deterministic unbiased spread (the default)
    thick_starts=None,    # None = deterministic unbiased spread (the default)
)
```

**Key fields:**
```
n_thick, n_crowns, n_thin, n_sites          # Dimensions (int, aux_data)
n_titin, total_xbs, n_faces_per_thin, n_xb_per_crown, max_sites_per_face

crown_offsets      # (n_thick, n_crowns) crown rest positions from M-line
crown_rests        # (n_thick, n_crowns) rest spacings between crowns
binding_offsets    # (n_thin, n_sites) site rest positions
binding_rests      # (n_thin, n_sites) rest spacings between sites
titin_connections  # (n_titin, 4) (thick_idx, thick_face, thin_idx, thin_face)

xb_to_thin_id      # (total_xbs,) XB → target thin filament
xb_to_thin_face    # (total_xbs,) XB → target face on that filament
xb_to_site_indices # (total_xbs, max_sites_per_face) fixed-width candidate sites
xb_valid           # (total_xbs,) bool — False where the XB has no real partner

tm_chains          # (n_thin, n_sites) TM chain assignment (0 or 1)
tm_prev_neighbor   # (n_thin, n_sites) same-chain predecessor site index
tm_next_neighbor   # (n_thin, n_sites) same-chain successor site index
face_to_sites      # (n_thin, n_faces, max_sites_per_face) site indices per face
n_sites_per_face   # (n_thin, n_faces) valid count per face
thick_to_thin      # (n_thick, 6, 2) hex-neighborhood map
thin_to_thick      # (n_thin, n_faces, 2)
thick_starts       # (n_thick,) crown level start offset
thin_starts        # (n_thin,) helical twist start offset

n_xb_bins, xb_bin_edges, xb_bin_centers     # XB axial-distance binning grid
eye_4, eye_6                                 # identities for the matrix exponentials
```

`crown_offsets`/`crown_rests` are **per-filament** `(n_thick, n_crowns)`. They are
identical across filaments unless `n_superlattice_classes > 1`, which offsets each
class by `thick_crown_spacing / n_superlattice_classes`.

⚠️ **`xb_valid` gate.** `xb_to_thin_id`/`xb_to_thin_face` are dense fixed-width
arrays, so XBs with no real geometric partner carry a **placeholder `(0, 0)`** that
is indistinguishable from a real "thin 0, face 0" target. The kernels already gate
on `xb_valid`; analysis code must too. Use `topo.valid_xb_targets()` →
`(xb_index, thin_id, thin_face)` for the masked view. This matters most for small
or asymmetric lattices — at IFM geometry ~56 % of XB slots were invalid in one
measured case.

**Filament starts.** `thin_starts`/`thick_starts` default to a deterministic
coprime-step spread (`_spread_starts`), which removes the single-topology phase
bias that random starts used to introduce. Invertebrate `thin_starts` instead
defaults to all-zero, because the Squire 3-fold registration class
(`_compute_thin_registration_classes`, derived from each thin filament's hex-edge
direction) already supplies the systematic phase variation. Passing an explicit
list overrides either default and is length-validated.

**Vertebrate** (default): 1 thick : 2 thin, 3 faces/thin, `n_xb_per_crown=3`
**Invertebrate:** `StaticParams(actin_geometry='invertebrate')` — 1:3, 2 faces;
IFM presets use `n_xb_per_crown=4`, `crown_rotation_deg=33.75`

**`create()` preconditions** (raised as `ValueError`): under `periodic=True` every
thick face must reach a real thin filament (invertebrate geometry requires an even
`nrows`), and `n_superlattice_classes=3` additionally requires `nrows` even and
`ncols` a multiple of 3.

---

## 8. Parameters

### StaticParams (frozen, recompile trigger)

**File:** `multifil_jax/core/params.py`, `StaticParams` dataclass

Structural config — 21 fields; changing any of them requires recompilation:
```python
static = StaticParams(
    # --- lattice / filament dimensions ---
    n_crowns=52,                     # crowns per thick filament (IFM: 100)
    n_polymers_per_thin=15,          # actin polymer repeats per thin (IFM: 20)
    actin_geometry='vertebrate',     # or 'invertebrate'
    thick_bare_zone=58.0,            # nm, M-line to first crown
    thick_crown_spacing=14.3,        # nm, inter-crown rest spacing (IFM: 14.5)

    # --- solver ---
    n_newton_steps=4,                # Newton while_loop cap (exits early at convergence)
    n_cg_steps=6,                    # CG iterations per Newton step (0=Richardson, diverges with bound XBs)
    solver_max_iter=50,

    # --- XB transition-matrix binning ---
    n_xb_bins=200,                   # bins per permissiveness level
    xb_bin_lo=-8.0, xb_bin_hi=35.0,  # nm, axial-distance bin range

    # --- thin-filament actin helix ---
    actin_half_pitch=36.0,           # nm, long-pitch half-repeat (IFM: 38.7)
    mono_per_poly=26,                # monomers per polymer repeat (IFM: 28)
    polymer_base_turns=12.0,         # turns per polymer repeat (IFM: 13)
    target_zone_wiggle=np.radians(15.0),  # rad, angular acceptance half-width (IFM: 26°)

    # --- crown-face geometry ---
    n_xb_per_crown=3,                # XBs per crown; sets total_xbs (IFM: 4)
    crown_rotation_deg=60.0,         # azimuthal step per crown (IFM: 33.75)
    crown_face_wiggle_deg=15.0,      # face-match half-angle (inert at n_xb_per_crown=3)
    legacy_crown_geometry=False,     # True = exact pre-generalization face table (n=3 only)

    # --- myosin superlattice ---
    n_superlattice_classes=1,        # 1 = none; 3 = Drosophila (Squire 2006). Only {1,3} valid.
)
```

`StaticParams.solver_residual_tol` is gone. It was a third copy of a float32
precision floor that no longer exists: node positions are stored as
displacements from rest, so the achievable residual does not scale with
stiffness. Convergence is judged on the dimensionless `solver_residual_norm`
metric, and the post-run warning fires when it exceeds 1.

`target_zone_wiggle` (which actin monomer falls in a target zone) and
`crown_face_wiggle_deg` (which of six hex neighbors a crown's arm points at) are
different physical questions — do not conflate them. The real default is the
float32 round-trip of `2π/24` (≈15°), chosen so vertebrate topology stays
byte-identical to the pre-generalization code; the `np.radians(15.0)` above is
illustrative.

### DynamicParams / Constants (JAX PyTree, sweepable)

All 46 physical parameters as JAX arrays. Sweepable without recompile. The
drivers are not fields; an unknown name raises (`TypeError` from `__init__`,
`ValueError` from `copy()` and from `run(dynamic_params={...})`):
```python
dynamic = DynamicParams(thick_k=7500.0, thin_k=5500.0, ...)
dynamic_modified = dynamic.copy(thick_k=9000.0)
```

Defaults live in the `_DYNAMIC_DEFAULTS` ordered dict at the top of `params.py`,
with literature citations inline. That dict is the single source of truth:
`__slots__`, `DYNAMIC_FIELDS`, and `__init__` all derive from it, so **adding a
parameter is one edit**. Insertion order is the `tree_flatten`/`tree_unflatten`
order; it is derived, never positional — nothing indexes a field by position.

Current defaults are literature-anchored, not the old pre-3.0 values:
`thick_k=7500` (whole-filament ≈ 144 pN/nm) and `thin_k=5500` (≈ 61 pN/nm), per
Brunello 2014 / Mijailovich 2021 — these replace the uncited 2020/1743.

---

## 9. Cooperativity

One TM cooperativity model: a symmetric Ising chain. A second, tension-dependent
span model was removed in v3.0, verified bit-exact against the Ising path.
Removed with it:

| removed | was |
|---|---|
| `kernels/cooperativity.py` | the whole legacy module |
| `legacy_coop=` kwarg on `run()` / `timestep()` (earlier spelled `ising_coop=`) | selected between the two models |
| `State.subject_to_coop` | `(n_thin, n_sites)` bool field |
| `tm_coop_magnitude`, `tm_span_base`, `tm_span_force50`, `tm_span_steep` | `DynamicParams` entries |
| `tm_Keq_30` | `DynamicParams` entry, already unused — the 3→0 step is one-way |

Pre-v3.0 scripts passing any of these will raise. Set cooperativity through
`tm_J_M` (and `tm_J_C`) instead.

**File:** `multifil_jax/kernels/transitions.py` — `thin_transitions()`,
`_compute_unique_tm_Q_matrices()`, `count_neighbor_states_split()`. Rate laws
themselves live in `kernels/rate_functions.py` (`tm_rate_01` … `tm_rate_30`) and
are called by the builder, not inlined.

Each site's field comes from its two **topological** same-chain neighbors
(`topology.tm_prev_neighbor` / `tm_next_neighbor` — one up, one down its own
tropomyosin chain; there is no distance threshold):

```
h(i) = J_C * n_2 + J_M * n_3 - 0.5*(J_C + J_M) * n_closed
```

`n_2`, `n_3`, `n_closed` count neighbor states (each capped at 2), so there are
`3³ = 27` unique Q matrices per step. Forward TM rates (0→1, 1→2, 2→3) are scaled
by `exp(+h/2)` and their reverses by `exp(-h/2)` — Glauber-symmetric, so detailed
balance is preserved. `k_30` stays at base (boosting it produces anti-cooperative
cascades).

- `tm_J_M` (XB-bound neighbor coupling) sets the emergent Hill steepness.
- `tm_J_C` is empirically inert and defaults to 0.
- The old `tm_span_ising_nm` window parameter was **retired** when coupling became
  topological — a literal nearest-neighbor chain is intrinsically sharper than the
  windowed version, so `tm_J_M` was re-fit downward.
- `tm_J_M` is **not currently calibrated**: the shipped 2.70 came from an
  unconfirmed cardiac force-pCa fit, and the correlation-length target that once
  bracketed it at 1.5–2.5 has been withdrawn (unit and observable mismatch — see
  the `tm_J_M` entry in `core/params.py`).

Cooperativity reaches the rate functions as a single `mod` multiplier: `exp(+h/2)`
to the forward rates, `exp(-h/2)` to their reverses. `tm_rate_30` takes no `mod`
argument, making the anti-cooperative mistake structurally unrepresentable.

---

## 10. Geometry Kernel

**File:** `multifil_jax/kernels/geometry.py`

`update_nearest_neighbors(state, topology, z_line, lattice_spacing)` → updated `State`

**Logic:**
- For each XB, finds the single nearest candidate binding site via a fixed-width
  gather over `topology.xb_to_site_indices` (constant width → full GPU parallelism)
- Site search uses the head position (crown base **+ 13 nm** reach); the stored
  distance is measured from the crown base, not the head
- Stores `xb_distances` as `(axial, radial)` — radial is the current lattice
  spacing, not a second site
- Sites at position ≤ 0 are masked invalid (infinite distance) so M-line-proximal
  XBs cannot bind behind the M-line
- Updates `xb_nearest_bs` and `xb_distances`

---

## 11. Transitions Kernel

**File:** `multifil_jax/kernels/transitions.py`

### `thin_transitions(state, constants, topology, rng_key, dt, tm_subpop=None)`
- 4-state TM chain Markov model with symmetric Ising coupling (§9)
- 54 unique Q matrices indexed by `(n_2, n_3, n_closed)` (each capped at 2) **× whether a crossbridge is bound**, as `(n_2*9 + n_3*3 + n_c) + 27*is_bound`
- One batched `expm_pade6_batch` call, then a per-site gather by config index
- The crossbridge lock is applied **to rates, inside Q**: the locked half has
  `k_30 /= (1+K2)`, `k_32 /= (1+K2)`, `k_33 = -(k_30+k_32)`. McKillop & Geeves'
  `K_T(1+K2)` is an *equilibrium ratio*, and this is the form that reproduces it
  at any `dt`. It was applied to probabilities after the exponential until S136,
  which equals this only to first order in `dt` — at `dt = 1 ms` a nominal
  `xb_tm_K2 = 79` ran as an effective ~174. `xb_tm_K2 = jnp.inf` still gives the
  hard `[0,0,0,1]` lock bit-exactly
- Neighbor counts come from `count_neighbor_states_split()` in this same module
- `tm_subpop` supports mean-field (`Q_eff = Σ f_k Q_k`, one exponential) and
  explicit per-site label select (`(K, 54, 4, 4)` → gather). Each population now uses its own `xb_tm_K2`, which the post-hoc form could not do

### `thick_transitions(state, constants, topology, rng_key, dt)`
- 6-state XB Markov model (states 0–5, see §6)
- Per-XB probabilities gathered from a **binned** matrix-exponential grid
- Probabilities from `xb_step_probabilities()` → `P_all`, `(n_xb_total, 6, 6)`
- Updates `xb_states` and `xb_bound_to` stochastically

### XB Q/P binning

Rate construction is shared; exponentiation is not, because the sampling path and
the metrics path need exponentials of *different* generators. One matrix
exponential per caller, never two.

- `_build_xb_Q_bins(state, constants, topology, pCa, z_line, lattice_spacing)` → `(Q_bins, key)`.
  Builds `(3 * n_xb_bins, 6, 6)` rate matrices — closed (permissiveness 0),
  open, and open-but-screened — each evaluated at the `n_xb_bins` axial bin
  centers. The screened block is for targets in the thin-thin double-overlap
  zone: between the M-line and the hiding line (how far the equal-length thin
  filaments pass the M-line, mirrored from the opposite half). Its `r01` is
  scaled by `1 - thin_thin_overlap_screening`; `r10` is not, so occupancy falls.
  Each XB's `key` is its axial bin — arithmetic on the uniform `xb_bin_edges`,
  clipped at both ends — plus its block. The key depends only on geometry,
  permissiveness and the hiding line, never on rates, so it is shared across
  subpopulations.
- `_xb_Q_resolved(...)` → `(Q_bins, key, labels)`. All subpopulation handling
  (`mean_field` blends generators, `explicit` keeps them stacked), no
  exponentials. This is the shared stage that guarantees sampling and metrics
  cannot disagree about the physics of a step.
- `xb_binned_generator(...)` → `XBBins(Q, P, G, key, labels)`. **One** expm per
  step, taken in `kinetics_step` and carried on `KineticsTrace`. `P = exp(Q·dt)`
  is what `thick_transitions()` samples from; `G = ∫₀^dt exp(Qt)dt` is the
  occupancy-time integral, which rides along for ~5 % more wall (measured, 400
  matrices) and is what the metrics read.
- `xb_expected_crossings(bins, states)` → `(n_pairs, n_xb)`. Exact expected
  crossing counts, `E[# i→j | start s] = q_ij · G[s,i]`, for the pairs the ATP
  metrics need. Metrics only.

Until 2026-09-11 there were **two** exponentials per step: the second was of a
generator with rows 4 and 0 zeroed (`xb_exit_probabilities`, now deleted), which
made Free_2 and DRX absorbing so that `P_abs[i,3,4]` read "*visited* Free_2".
That is `P(visit ≥ 1)`, a lower bound on `E(visits)`, and it was biased low by
−0.07 % (cardiac) and −1.2 % (skeletal) at dt = 1 ms. Counting the **edge** 3→4
is exact, needs no absorbing construction, and cannot confuse `r04` — the reverse
recovery stroke — with hydrolysis.

This replaces one expm per crossbridge with `2 × n_xb_bins` — ~6× fewer expm calls
at 4×4, where the step was ~71 % matrix exponential. Binning resolution is set by
`StaticParams.n_xb_bins` / `xb_bin_lo` / `xb_bin_hi` and is baked into the
topology as `xb_bin_edges` / `xb_bin_centers`.

XBs with `xb_valid == False` are forced to permissiveness 0, routing them into the
AP=0 block where `r01` (the only entry rate into a bound state) is exactly zero —
a hard gate, not a distance-decay approximation.

### `expm_pade6_batch` squaring

6th-order Padé with scaling-and-squaring; the squaring phase is a fixed-trip
`fori_loop` with a per-matrix predicate:

```python
jax.lax.fori_loop(0, 18, _square_step, result)   # _square_step squares only where i < s
```

18 squarings cover ‖A‖ up to 2¹⁸ = 262144. Rows are renormalized afterwards to
absorb float32 drift.

---

## 12. Forces Kernel

**File:** `multifil_jax/kernels/forces.py`

### Axial forces (for equilibrium solver and output)
- `axial_force_at_mline(state, constants)` — total M-line force (pN)
- `compute_forces_vectorized(...)` — per-node axial residual forces for solver
- `compute_forces_from_state_vectorized(state, constants, topology, z_line, lattice_spacing)` — convenience wrapper

Force contributions: thick spring chain, thin spring chain, XB (converter + globular
springs for states 2-4), titin (exponential model).

### Radial forces (for dynamic lattice spacing solver)
- `_xb_radial_force_total(...)` — total XB radial force Σ dV_XB/dd (pN). Differentiable w.r.t. lattice_spacing for the augmented Newton JVP.
- `_titin_radial_force_total(...)` — total titin radial force from all thick filaments

`_xb_radial_force_total` shares the two-spring primitives with the axial path
rather than replicating them — it is the *other component of the same rotation*,
`polar_to_filament(...)[1]` where the axial path takes `[0]`. Used by
`_radial_residual()` in `solver.py`, and differentiated there, so it must stay
`jax.grad`-able w.r.t. `lattice_spacing`.

---

## 13. Solver Kernel

**File:** `multifil_jax/kernels/solver.py`

### Unified solver (fixed and dynamic lattice spacing)

```python
solve_equilibrium(
    state, constants, topology, z_line, lattice_spacing,
    K_lat=None, d_ref=None, tolerance=None,
    n_newton_steps=16, n_cg_steps=6,
    precond_params=None, prefactored_precond=None,
) -> (State, residual, new_ls, n_iters)
```

Returns a 4-tuple: the equilibrated state, scalar max residual (pN), the
lattice spacing used (solved `d` in dynamic mode, the prescribed `lattice_spacing` in
fixed mode), and the number of Newton iterations used. `K_lat is None` selects
fixed LS mode at trace time (no runtime branch).

The iteration caps default to 16/6 here, but `run()` overrides them from
`StaticParams` (`n_newton_steps=4`, `n_cg_steps=6`) — the StaticParams values are
what production runs actually use.

### Newton-CG with while_loop

Outer loop: `jax.lax.while_loop` — body traced once, exits at convergence or cap:
```
while max|F(x)| > tol AND iter < n_newton_steps:
    dx = CG_solve(-J, F(x))   # n_cg_steps CG iterations
    x += dx
```

Inner CG: Python `for` loop (unrolled at trace time) — enables XLA fusion.

### Thomas Algorithm (pre-factored preconditioner)

Tridiagonal preconditioner factored once before the scan loop, reused across all timesteps:
```python
precond = build_prefactored_preconditioner(precond_params)
# precond passed into scan, applied every step
```

`thomas_factor`: Python `for` loop — called once before the scan loop, not in the hot path.
`thomas_solve`: `jax.lax.associative_scan` for both forward and back substitution — 5× fewer
jaxpr equations vs the previous for-loop approach; 20% faster.
**Note:** `fori_loop` was tried for `thomas_solve` and caused 20× runtime regression (XLA cannot
fuse across WhileOp boundaries). Do NOT revert to fori_loop.

### Convergence tolerance — scale-free and per-block

```python
tol_axial = solver_atol + solver_rtol * RMS(backbone spring force)
tol_rad   = solver_atol + solver_rtol * |K_lat| * d_ref     # dynamic LS only
converged when   max(|F| / tol_vec) <= 1
```

`_run_newton` takes a tolerance **vector**, one entry per residual row. Fixed LS
passes a uniform one; dynamic LS appends `tol_rad` for the radial row, which is
a whole-lattice aggregate and would otherwise share one `max` with per-node
axial forces. `scale_axial` is evaluated ONCE per `solve_equilibrium` call from
the incoming state, so the tolerance is fixed through the Newton loop.

There used to be a `max(tolerance, thick_k * 1e-4, MIN_FLOAT32_TOLERANCE)`
floor here. It existed because absolute node positions (~1000 nm) put a float32
quantum under every backbone force, and it scaled WITH the problem rather than
with the physics: at 100x stiffness the tolerance became 75 pN and Newton
exited on its first iteration. It also ignored `thin_k` entirely. Displacement
coordinates removed the quantum; see `core/state.py`.

### Dynamic lattice spacing solve path

When `K_lat is not None`, `solve_equilibrium()` appends lattice spacing `d` as an
extra DOF to the position vector, creating an augmented (n+1)-dim system:

```
augmented residual:  [f_axial(positions, d), f_radial(positions, d)]
augmented solution:  [positions..., d]
```

JAX's JVP on the augmented residual automatically captures all cross-coupling
terms (∂f_axial/∂d, ∂f_radial/∂positions).

Key functions:
- `_radial_residual(d, ...)` — radial force balance: `F_rad = -K_lat*(d-d_ref) - f_xb - f_titin = 0`
- `_augmented_residual_fn(pos_aug, ...)` — joint `[f_axial, f_radial]` residual
- `_newton_solve_dynamic_ls(...)` — while_loop Newton with `d > 1.0 nm` projection
- `_apply_augmented_preconditioner(...)` — block-diagonal: Thomas for axial, exact Jacobian diagonal inverse for d

The d-block preconditioner uses `jax.grad(_radial_residual)` to compute `J_dd`
(the exact Jacobian diagonal at d), giving `d_block_inv = -1/J_dd`. This replaces
the naive `1/K_lat` which was ill-conditioned when XB radial stiffness dominated.

---

## 14. Metrics

**File:** `multifil_jax/metrics_fn.py`

Single function:
```python
metrics = compute_all_metrics(
    old_state, new_state, constants, drivers, topology,
    force, solver_residual, newton_iters, dt, trace,
    solver_residual_norm, solver_tolerance,
)
```

`drivers` carries the **solved** lattice spacing (force and the
reported `lattice_spacing` are post-solve quantities); `trace.drivers` carries
the **pre-solve** one (the rates were evaluated there). Both are correct for
their own question and must not be unified.

Returns a `MetricsDict` with **43 keys** (same keys every call). Always computed —
no selection needed.

**Every key is independent.** Nothing here is computable from the other keys
plus `result.topology_config`. Twenty keys that were — the all-site occupancy
fractions, `n_bound`, `actin_permissiveness`, `frac_tm_available_overlap`,
`xb_detach_atp`, `sarcomere_work`, `thick_energy_first_delta_avg`,
`xb_work_per_atp`, and the `z_line`/`pCa` duplicates of the `SimulationResult`
fields — were deleted on 2026-09-19. See "Reconstructions" below.

**Metric groups (43 total):**
| Group | Keys | n |
|-------|------|---|
| Protocol | `axial_force`, `solver_residual`, `solver_residual_norm`, `solver_tolerance`, `lattice_spacing` | 5 |
| XB counts | `n_xb_drx`, `n_xb_loose`, `n_xb_tight_1`, `n_xb_tight_2`, `n_xb_free_2`, `n_xb_srx` | 6 |
| XB force by state | `force_xb_loose`, `force_xb_tight_1`, `force_xb_tight_2` | 3 |
| TM counts | `n_tm_state_0` … `n_tm_state_3` | 4 |
| TM overlap-zone | `frac_tm_state_2_overlap`, `frac_tm_state_3_overlap`, `n_overlap_sites` | 3 |
| Transitions | `atp_consumed`, `newly_bound`, `closure_detach_free`, `closure_detach_atp` | 4 |
| Displacement | `thick_displace_mean/max/min/std`, `thin_displace_mean/max/min/std` | 8 |
| Energy | `thick_energy_first_avg`, `titin_energy_avg`, `titin_energy_delta_avg` | 3 |
| Work | `xb_work_on_filaments` | 1 |
| ATP | `atp_expected` | 1 |
| Cycle fluxes | `xb_detach_free`, `xb_give_up`, `atp_net_pi_release`, `atp_net_hydrolysis` | 4 |
| Solver | `newton_iters` | 1 |

**Reconstructions**, for the keys that used to exist:
| Was | Is now |
|-----|--------|
| `z_line`, `pCa` | `result.z_line`, `result.pCa` — bit-identical, already stored |
| `frac_xb_*` | count / `result.topology_config['total_xbs']` |
| `frac_tm_state_*`, `actin_permissiveness` | count / (`n_thin` × `n_sites`) |
| `n_bound` | `n_xb_loose + n_xb_tight_1 + n_xb_tight_2` |
| `frac_tm_available_overlap` | `frac_tm_state_2_overlap + frac_tm_state_3_overlap` |
| `xb_detach_atp` | `atp_expected - closure_detach_atp` |
| `thick_energy_first_delta_avg` | `np.diff(thick_energy_first_avg, axis=-1)` |
| `sarcomere_work` | `-0.5*(F[..., :-1] + F[..., 1:]) * np.diff(z, axis=-1)` |
| `xb_work_per_atp` | **do not form it per step** — see below |

`titin_energy_delta_avg` is deliberately NOT on that list: both its endpoints
are evaluated at the current step's `z_line` and lattice spacing, so it is the
change from filament motion alone. `np.diff` of the level also picks up the
driver's contribution, and under a shortening ramp the two differ by more than
the delta itself.

The **overlap-zone** group (`compute_overlap_tm_fractions()`) restricts the TM
fractions to crossbridge-reachable sites: within
`[crown_offsets.min() - 13, crown_offsets.max() + 13]` (the same 13 nm head reach
used in `geometry.py`) **and** past the hiding line (absolute position > 0,
reconstructed with `thin_axial()`). The plain
all-site fractions (counts over a constant denominator) average over *every*
site, including the thick filament's bare zone and sites beyond its tip, so
they are diluted by permanently unreachable sites. That is why the `_overlap`
variants are exported and the all-site fractions are not — a
filament-length change once moved the all-site metric 13.5 %→17.3 % almost
entirely through that denominator while the true overlap value barely moved.

`atp_expected` is the **exact** expected number of ATP-consuming crossings:
`q34 · G[s,3]` summed over **every** head, from its own mid state and its own
generator, with no mask. `G` is the occupancy-time integral carried on
`trace.xb_bins`, so the metric reads the very generator `thick_transitions`
sampled from rather than rebuilding one.

Every head is read, including those starting in states 0, 4 and 5, and that is
correct — a head that runs `0→1→2→3→4` inside one step really does spend an ATP.
The old mask existed because "reached state 4" *from state 0* would have counted
`r04`, the non-ATP reverse recovery stroke; an edge count cannot make that
mistake. A head torn off a closing site sits in state 4 and contributes exactly
zero, structurally: `r43` is always zero, so it cannot reach Tight_2 within the
step at all.

`atp_expected` additionally carries the realised **strong closure tears**, as a
count rather than an expectation: that event is fully observed and its charge is
deterministic, so there is nothing to take an expectation over. The two terms are
disjoint by construction — a torn head is in state 0 or 4 in `trace.state`, so it
falls outside the 1–3 mask.

**The one bias that remains.** `P_abs[·,s,4]` is P(visit Free_2 at least once),
not E(visits). The exact quantity is
`q34 · ∫₀^dt [exp(Qt)]_{s,3} dt`, the top-right block of `expm([[Q, I],[0, 0]]·dt)`
— see `local_projects/tension_cost/atp_estimator_spy.py`, which validates it
against a 200,000-substep Riemann sum. P(at least one) is a *lower bound* on
E(visits), so the gap is always negative. Measured 2026-09-10 at 4×4, `dt = 1 ms`,
pCa 4.5 and 6.2, over isometric, lengthening (+0.05 nm/ms) and shortening
(−0.05 nm/ms):

| preset | gap |
|---|---|
| cardiac | −0.06 % to −0.08 % |
| skeletal | −1.17 % to −1.39 % |

**The split is by cycling rate, not by protocol.** Skeletal turns over ~2.6× faster
(31 vs 12 ATP/ms at pCa 4.5), so far more heads complete `3 → 4` twice inside one
step; imposing a ramp moves the gap by only ~0.1 points on top of that. So the
honest bound is ~0.1 % for cardiac and ~1.4 % for skeletal at `dt = 1 ms` — not
the "~1 %" an isometric-cardiac-only measurement would have suggested. It shrinks
with `dt`. Not fixed; not to be described as exact. A skeletal tension-cost study
should use `dt = 0.1 ms` or carry the correction.

**Work: two different quantities, one exported.** `xb_work_on_filaments` is
what the crossbridges did to the lattice — path-dependent and per-head, so it
cannot be reconstructed from the returned traces and must be computed in the
scan. The external work at the driven z-line is exactly reconstructible
afterwards (see the table above) and is therefore not exported; it is also
identically zero under an isometric hold, where crossbridge work is not.

**Never form work-per-ATP per timestep.** `mean(w/a)` is the unweighted mean of
per-step efficiencies; the efficiency over a window is `sum(w)/sum(a)`, the
ATP-weighted one. The bias is the squared CV of the denominator: +1.3 % at
pCa 4.5, **+15.7 %** at pCa 6.2, and `0/0` reported as `0.0000` at pCa 9. The
deleted `xb_work_per_atp` key did exactly this. Reduce numerator and
denominator over the window, then divide — and likewise across replicates.

These replace `work_thick`/`work_thick_mean`/`work_per_atp`,
which were M-line force times the mean displacement of *every* thick crown —
neither quantity, and dominated by internal backbone strain redistribution.

---

## 15. Subpopulations

**Files:** `multifil_jax/core/subpopulation.py` (config), plumbed through
`simulation.py` → `timestep.py` → `transitions.py` / `metrics_fn.py`.

Models a fraction of XB motors or TM units running modified kinetics — mutations,
cMyBP-C / C-zone effects, or mean-field-vs-stochastic validation.

A config is a list of **K populations** (index 0 = WT, all scales 1.0), each a
dict of **multiplicative** factors applied to whatever `DynamicParams` was passed
to `run()`. Constructors currently emit K=2; nothing in the data path hardcodes it.

```python
from multifil_jax import Subpopulation

sp = Subpopulation.mean_field(0.5, xb_srx_kmax=0.3, xb_r01_coeff=4.0)
sp = Subpopulation.random(0.5, seed=0, xb_srx_kmax=0.3)
sp = Subpopulation.c_zone(topo, 350.0, 650.0, xb_r01_coeff=2.0)

result = run(topo, pCa=4.5, z_line=z0, lattice_spacing=d0, subpopulation=sp)
result = run(topo, pCa=4.5, z_line=z0, lattice_spacing=d0,
             subpopulation=[sp_a, sp_b, sp_c])   # sweep axis
```

| Mode | Mechanism | Determinism |
|------|-----------|-------------|
| `mean_field` | generator blend `Q_eff = Σ_k f_k Q_k`, then **one** expm | deterministic |
| `random` | per-XB / per-site integer labels, Bernoulli(fraction) | masks redrawn per sim from `seed + sim_index` |
| `c_zone` | labels by crown axial band (nm from M-line), default 350–650 | deterministic, built at construction |

**Constraints (all enforced, not conventions):**
- Scale keys must be `xb_*` or `tm_*` fields — mechanics/forces always use the WT
  base, only transition rates scale. Anything else raises.
- A `subpopulation` list is a sweep axis, mutually exclusive with a
  `dynamic_params` candidate list. All entries must share one mode and one K; a
  swept `random` list must share one `seed` (fraction/severity may vary).
- `c_zone` membership is genuinely per-filament under a myosin superlattice, since
  filaments in different classes sit at different axial positions for the same
  crown index.

Edge cases are exact, not approximate: `fraction=0.0` ≡ WT and `fraction=1.0` ≡ a
global scale, bit-for-bit, in every mode.

---

## 16. Appendix: Key File Reference

| File | Purpose |
|------|---------|
| `multifil_jax/simulation.py` | `run()`, `SimulationResult`, `_run_sim_kernel`, `BATCH_BUCKETS` |
| `multifil_jax/timestep.py` | `kinetics_step()`, `timestep()` — single step orchestrator |
| `multifil_jax/metrics_fn.py` | `compute_all_metrics()` — 43-metric MetricsDict |
| `multifil_jax/core/state.py` | State hierarchy, `realize_state()`, `Drivers`, `KineticsTrace`, `MetricsDict`, `PreconditionerParams` |
| `multifil_jax/core/params.py` | `StaticParams`, `DynamicParams`/`Constants`, `_DYNAMIC_DEFAULTS`, the four species presets |
| `multifil_jax/core/sarc_geometry.py` | `SarcTopology` — PyTree topology, `create()`, `valid_xb_targets()` |
| `multifil_jax/core/subpopulation.py` | `Subpopulation` dataclass + mask generation |
| `multifil_jax/kernels/geometry.py` | `update_nearest_neighbors()` |
| `multifil_jax/kernels/transitions.py` | `thin_transitions()`, `thick_transitions()`, `count_neighbor_states_split()`, `xb_binned_generator()`, `xb_expected_crossings()`, `expm_pade6_batch()` |
| `multifil_jax/kernels/forces.py` | `xb_geometry()`, `xb_springs_for_state()`, `xb_elastic_energy()`, `xb_polar_forces()`, `polar_to_filament()` (the two-spring primitives — the rate path in `transitions.py` imports them too), `xb_axial_work()`, `axial_force_at_mline()`, `compute_forces_vectorized()`, `_xb_radial_force_total()`, `_titin_radial_force_total()` |
| `multifil_jax/kernels/solver.py` | `solve_equilibrium()` (unified fixed/dynamic LS), Thomas algorithm |
| `multifil_jax/kernels/rate_functions.py` | Rate functions (absolute values, no multipliers) |
| `multifil_jax/utils/hardware.py` | GPU detection, XLA persistent-cache configuration |
| `multifil_jax/helper.py` | `count_transitions()`, `validate_forces_numerical()`, `validate_equilibrium()` |
| `examples/quickstart.py` | Full API demo: isometric, sweeps, transients, structural stack |
| `examples/dynamic_lattice_spacing.py` | Dynamic LS demo: isometric, force comparison, K_lat sweep, length ramp |
| `examples/subpopulation.py` | Subpopulation demo (mean_field / random / c_zone) |
| `examples/hysteresis.py` | Length-ramp hysteresis / work-loop protocol |
| `examples/sinusoidal_analysis.py` | Kawai-style complex-modulus analysis |
| `examples/stiffness_sweep_cardiac.py` | Cardiac stiffness sweep |
| `examples/benchmarks/benchmark_minibatch.py` | Minibatch size benchmark CLI |
| `examples/benchmarks/benchmark_dynamic_ls.py` | Dynamic LS performance and lattice scaling benchmark |
| `examples/benchmarks/profile_jax.py` | JAX/XLA profiling harness |

---

## 17. Performance Notes

### GPU Kernel Fusion Optimizations (implemented)

| Tag | Description |
|-----|-------------|
| 1A | TM/XB diagonal rates: `-(a + b)` instead of `vmap(ordered_sum)` |
| 1B | `expm_pade6_batch` squaring: `fori_loop(0, 18, ...)` |
| 2A | Drivers threaded as explicit scalars — no per-step DynamicParams rebuild (`with_drivers()` deleted) |
| 2B | `precond_params` built once before scan, reused |
| 3A | Thomas algorithm replaces Lineax cusparse; pre-factored before scan |
| 3B | `thick_transitions` per-XB probability gather: `vmap(lambda P, s: P[s])(P_all, states)` |
| 3C | XB Q/P binning: `2 × n_xb_bins` matrix exponentials + per-XB gather (§11) |
| 3D | Thomas back-substitution via `associative_scan` (5× fewer jaxpr eqns, ~20 % faster) |

### Do NOT Re-attempt

- **Thomas fori_loop**: 20× runtime regression (XLA cannot fuse across WhileOp barriers)
- **GPU autotune level=2**: Suboptimal kernel selection
- **float16 sampling (cumsum+argmax)**: Systematic one-state-downward bias (~0.088% of samples). No throughput benefit.
- **`n_cg_steps=0` (Richardson) as the default**: diverges once XBs are attached (`M ≈ J` holds only in the elastic case)
- **`@jax.jit` inside `scan`/`vmap`**: creates pjit boundaries that block XLA fusion. Only `_run_sim_kernel` carries `@jax.jit`
- **Carrying Q/P matrices in the scan carry**: ~12 MB extra carry at batch=256, net slower
- **Hiding-line clamping**: clamping sites below the hiding line (instead of masking them invalid) makes M-line-proximal XBs bind behind the M-line
