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
| 3 | **Drivers** | `Drivers` | Per-timestep overrides (pCa, z_line, lattice_spacing) |

**Kernel signature:** `kernel(state, constants, drivers, topology, rng_key, *, dt)`

### `resolve_value()` Pattern

`resolve_value(driver_val, constant_val)` — selects Tier 3 if not NaN, else Tier 2:

```python
# In core/state.py
def resolve_value(driver_val, constant_val):
    return jnp.where(jnp.isnan(driver_val), constant_val, driver_val)
```

Used in every kernel to merge per-step overrides with defaults without branching.

### vmap-outside-scan Architecture

```
run() → vmap(_run_single_sim) → lax.scan(_run_sim_kernel)
         ↑ batch dim                 ↑ time dim
```

All batch elements run in a single fused XLA kernel. XLA fuses vmap+scan
into one GPU kernel — maximum parallelism with minimum kernel launch overhead.

---

## 2. Primary API: `run()`

**File:** `multifil_jax/simulation.py`, ~line 530

```python
result = run(
    topology,          # SarcTopology (Tier 1)
    pCa=4.5,           # float | list[float] | array(n_steps)
    z_line=900.0,      # float | list[float] | array(n_steps)
    lattice_spacing=14.0,
    dynamic_params=None,  # DynamicParams | dict[str, float|list]
    duration_ms=1000.0,
    dt=1.0,
    replicates=1,
    rng_seed=0,
    unroll=1,
    verbose=False,
)
```

**Input semantics:**
- `float` → broadcast to all timesteps (constant)
- `list[float]` → Cartesian product sweep axis
- `array(n_steps)` → time-varying trace

**Result shape convention:** `(Sweep_1, ..., Sweep_N, Replicates, Time)`

**Batch padding:** sweep sizes rounded up to `BATCH_BUCKETS = (1, 2, 4, ..., 16384)`
for JIT cache reuse. A 225-run and 256-run sweep share the same compiled kernel.

### Typical usage

```python
from simulation import run, get_default_params
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams

static, dynamic = get_default_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

# Simple isometric
result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000)

# pCa sweep
result = run(topo, pCa=[9.0, 6.0, 4.5], replicates=5)

# DynamicParams sweep
result = run(topo, pCa=4.5, dynamic_params={'thick_k': [1000, 2000, 3000]})
```

---

## 3. SimulationResult

**File:** `multifil_jax/simulation.py`, ~line 64

```
result.axial_force      # (... replicates, n_steps) force trace (pN)
result.metrics          # dict of ~43 metric arrays, same shape
result.z_line           # z_line trace used
result.pCa              # pCa trace used
result.metrics["solver_residual"]  # Newton solver residual at each step
result.dt               # timestep (ms)
result.coords           # {'pCa': [...], 'z_line': [...], ...}
result._axis_names      # ['pCa', 'replicates', 'time']
result.topology_config  # dict with n_thick, n_thin, n_sites, etc.
```

**Methods:**
- `.mean()` — collapse replicates axis (returns SimulationResult)
- `.std()` — standard deviation over replicates axis
- `.sel(**kwargs)` — slice by coordinate value, e.g. `.sel(pCa=4.5)`
- `.stack(results, axis_name)` — stack list of SimulationResult into sweep axis
- `.summary()` — human-readable text summary

---

## 4. JIT-compiled Kernel: `_run_sim_kernel`

**File:** `multifil_jax/simulation.py`, ~line 455

This is the vmapped+scanned simulation kernel:

```python
# Outer: vmap over batch (parameter sweep × replicates)
# Inner: lax.scan over time
batched_summaries, batched_metrics = _run_sim_kernel(
    topology=topology,          # passed via closure (vmap in_axes=None)
    batched_params=batched_params,  # vmap in_axes=0
    z_batched=z_batched,        # (batch, n_steps) vmap in_axes=0
    pCa_batched=pCa_batched,
    ls_batched=ls_batched,
    rng_keys=rng_keys,          # (batch,) vmap in_axes=0
    dt=dt,
)
```

All metrics are always computed (~43 metrics). No `metrics`/`manifest` in JIT
`static_argnames` — avoids recompilation when metrics selection changes.

---

## 5. Single Timestep: `timestep()`

**File:** `multifil_jax/timestep.py`

Always returns `(state, rng_key, solver_residual)`.

**7-step workflow:**

1. **resolve_value** — merge Drivers (Tier 3) with Constants (Tier 2)
2. **update_cooperativity** — TM chain cooperative activation (tension-dependent span)
3. **update_nearest_neighbors** — per-XB geometry (distances to binding sites)
4. **thin_transitions** — TM 4-state Markov transitions (matrix exponential)
5. **thick_transitions** — XB 6-state Markov transitions (per-XB matrix exponential)
6. **solve_equilibrium** — Newton-CG solver for force balance
7. **compute_all_metrics** → `SummaryState(force, solver_residual)` + metrics dict

---

## 6. State Hierarchy

**File:** `multifil_jax/core/state.py`

```python
State(
    thick = ThickState(
        axial,          # (n_thick, n_crowns) crown positions (nm)
        rests,          # (n_thick, n_crowns) rest spacings
        xb_states,      # (n_thick, n_crowns, 3) XB states (1-6)
        xb_bound_to,    # (n_thick, n_crowns, 3) bound site index (-1=unbound)
        xb_nearest_bs,  # (n_thick, n_crowns, 3) nearest BS index
        xb_distances,   # (n_thick, n_crowns, 3, 2) distances to nearest BS
    ),
    thin = ThinState(
        axial,           # (n_thin, n_sites) site positions (nm)
        rests,           # (n_thin, n_sites) rest spacings
        tm_states,       # (n_thin, n_sites) TM states (0-3)
        permissiveness,  # (n_thin, n_sites) float 0-1
        subject_to_coop, # (n_thin, n_sites) bool
        bound_to,        # (n_thin, n_sites) XB address (-1=unbound)
    ),
)
```

**SummaryState** — scan output (minimal, just force + residual):
```python
SummaryState(force, solver_residual)
```

**Immutable updates** via `._replace()`:
```python
new_state = state._replace(thick=state.thick._replace(axial=new_axial))
```

**State creation:**
```python
state = realize_state(topology, constants, z_line, pCa, lattice_spacing)
```

**Drivers** — per-step overrides (NaN = use constant):
```python
Drivers(pCa=jnp.nan, z_line=jnp.nan, lattice_spacing=jnp.nan)
```

---

## 7. Topology: `SarcTopology`

**File:** `multifil_jax/core/sarc_geometry.py`

Registered as JAX PyTree. Pre-computes all structural index maps for GPU efficiency:

```python
topo = SarcTopology.create(
    nrows=2, ncols=2,
    static_params=StaticParams(),
    dynamic_params=DynamicParams(),
)
```

**Key fields:**
```
n_thick, n_crowns, n_thin, n_sites   # Dimensions (int)
n_titin, total_xbs, n_faces_per_thin

crown_offsets      # (n_crowns,) crown rest positions relative to M-line
crown_rests        # (n_crowns,) rest spacings between crowns
binding_offsets    # (n_sites,) site rest positions from Z-line
binding_rests      # (n_sites,) rest spacings between sites
titin_connections  # (n_titin, 2) thick-thin pairs for titin

tm_chains          # TM chain structure
face_to_sites      # (n_thin, n_faces, n_per_face) site indices per face
thick_to_thin      # (n_thick, ...) neighborhood maps
```

**Vertebrate** (default): 1 thick : 2 thin, 3 XBs/crown, 3 faces/thin
**Invertebrate:** `StaticParams(actin_geometry='invertebrate')` — 1:3, 2 faces

---

## 8. Parameters

### StaticParams (frozen, recompile trigger)

**File:** `multifil_jax/core/params.py`, `StaticParams` dataclass

Structural config — changing any field requires recompilation:
```python
static = StaticParams(
    n_crowns=52,
    n_polymers_per_thin=15,
    actin_geometry='vertebrate',     # or 'invertebrate'
    n_newton_steps=4,                # Newton solver cap (static for while_loop)
    n_cg_steps=6,                    # CG iterations per Newton step
    solver_residual_tol=0.5,         # post-run warning threshold (pN)
)
```

### DynamicParams / Constants (JAX PyTree, sweepable)

All ~45 physical parameters as JAX arrays. Sweepable without recompile:
```python
dynamic = DynamicParams(thick_k=2020.0, thin_k=1743.0, pCa=4.5, ...)
dynamic_modified = dynamic.copy(thick_k=3000.0)
```

**Drivers fast path** — creates new DynamicParams with only the 3 driver fields updated:
```python
constants = base_constants.with_drivers(pCa, z_line, lattice_spacing)
```
Avoids 42 redundant identity-copy XLA ops per timestep.

---

## 9. Cooperativity Kernel

**File:** `multifil_jax/kernels/cooperativity.py`

`update_cooperativity(state, constants, topology)` → updated `ThinState`

**Logic:**
- Computes force on each TM site (via thin filament spring chain)
- Force-dependent cooperative span: `span = base + (force50 / (force50 + force))`
- Sites within span of a "coop-active" site inherit cooperative activation
- Updates `subject_to_coop` and `permissiveness` arrays

---

## 10. Geometry Kernel

**File:** `multifil_jax/kernels/geometry.py`

`update_nearest_neighbors(state, constants, topology)` → updated `ThickState`

**Logic:**
- For each XB, finds the 2 nearest binding sites (per-XB, exact distances)
- No binning — exact per-XB geometry computed every timestep
- Updates `xb_nearest_bs` and `xb_distances` arrays

---

## 11. Transitions Kernel

**File:** `multifil_jax/kernels/transitions.py`

### `thin_transitions(state, constants, topology, rng_key, dt)`
- 4-state TM chain Markov model
- Computes 2 unique Q matrices (cooperative / non-cooperative)
- Matrix exponential via `expm_pade6_batch` with 18-iteration squaring:
  `P = expm_pade6(Q * dt)` — covers ‖Q‖ up to 2¹⁸
- Updates `tm_states` stochastically

### `thick_transitions(state, constants, topology, rng_key, dt)`
- 6-state XB Markov model
- Per-XB transition matrices (no binning)
- Shared helper `compute_xb_transition_matrices()` — distance-dependent rates
- Gathered via `P_all[jnp.arange(n_xb_total), current_states, :]`
- Updates `xb_states` and `xb_bound_to` stochastically

### `expm_pade6_batch` squaring
```python
jax.lax.fori_loop(0, 18, lambda i, X: X @ X, A_scaled)
```
18 squarings cover ‖A‖ up to 2¹⁸ = 262144.

---

## 12. Forces Kernel

**File:** `multifil_jax/kernels/forces.py`

- `axial_force_at_mline(state, constants)` — total M-line force (pN)
- `compute_forces_vectorized(...)` — per-node residual forces for solver

Force contributions: thick spring chain, thin spring chain, XB (converter + globular
springs for states 2-4), titin (exp model).

---

## 13. Solver Kernel

**File:** `multifil_jax/kernels/solver.py`

`solve_equilibrium(state, constants, topology) → (State, residual_scalar)`

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

Python `for` loops for forward/back substitution — enables XLA fusion.
**Note:** `fori_loop` was tried and caused 20× runtime regression (XLA cannot fuse across
WhileOp boundaries). Do NOT revert to fori_loop.

### Tolerance floor

```python
tolerance = max(tolerance, thick_k * 1e-4, MIN_FLOAT32_TOLERANCE)
```

Prevents the while_loop from chasing an unreachable target at stiff parameter values.

---

## 14. Metrics

**File:** `multifil_jax/metrics_fn.py`

Single function: `compute_all_metrics(state, constants, topology) → dict`

Returns ~43 metrics as a fixed dict. Always computed — no selection needed.

**Metric groups:**
- Force: `axial_force`, `xb_force_*`, `titin_force`
- XB counts: `n_bound`, `n_srx`, `n_drx`, `n_weak_bound`, `n_strong_bound`
- XB positions: `xb_displace_mean`, `xb_displace_std`, `thick_displace_mean`
- TM states: `n_tm_state_0/1/2/3`, `mean_permissiveness`
- Energetics: `thick_energy_*`, `xb_energy_*`
- Solver: `solver_residual`

---

## 15. Appendix: Key File Reference

| File | Purpose |
|------|---------|
| `multifil_jax/simulation.py` | `run()`, `SimulationResult`, `_run_sim_kernel` |
| `multifil_jax/timestep.py` | `timestep()` — single step orchestrator |
| `multifil_jax/metrics_fn.py` | `compute_all_metrics()` — ~43-metric dict |
| `multifil_jax/core/state.py` | State hierarchy, `realize_state()`, `Drivers`, `resolve_value()` |
| `multifil_jax/core/params.py` | `StaticParams`, `DynamicParams`, `get_default_params()` |
| `multifil_jax/core/sarc_geometry.py` | `SarcTopology` — PyTree topology |
| `multifil_jax/kernels/cooperativity.py` | `update_cooperativity()` |
| `multifil_jax/kernels/geometry.py` | `update_nearest_neighbors()` |
| `multifil_jax/kernels/transitions.py` | `thin_transitions()`, `thick_transitions()`, `compute_xb_transition_matrices()` |
| `multifil_jax/kernels/forces.py` | `axial_force_at_mline()`, `compute_forces_vectorized()` |
| `multifil_jax/kernels/solver.py` | `solve_equilibrium()`, Thomas algorithm |
| `multifil_jax/kernels/rate_functions.py` | Rate functions (absolute values, no multipliers) |
| `multifil_jax/helper.py` | `count_transitions()` and other utilities |
| `jax/tests/` | Test suite |

---

## 16. Performance Notes

### GPU Kernel Fusion Optimizations (implemented)

| Tag | Description |
|-----|-------------|
| 1A | TM/XB diagonal rates: `-(a + b)` instead of `vmap(ordered_sum)` |
| 1B | `expm_pade6_batch` squaring: `fori_loop(0, 18, ...)` |
| 2A | `DynamicParams.with_drivers()` fast path for scan body |
| 2B | `precond_params` built once before scan, reused |
| 3A | Thomas algorithm replaces Lineax cusparse; pre-factored before scan |
| 3B | `thick_transitions` per-XB gather: `P_all[arange(n_xb_total), states, :]` |

### Do NOT Re-attempt

- **Thomas fori_loop**: 20× runtime regression (XLA cannot fuse across WhileOp barriers)
- **GPU autotune level=2**: Suboptimal kernel selection
