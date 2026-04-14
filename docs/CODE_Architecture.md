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
    topology,            # SarcTopology (Tier 1)
    pCa=4.5,             # float | list[float] | array(n_steps)
    z_line=900.0,        # float | list[float] | array(n_steps)
    lattice_spacing=14.0,
    K_lat=None,          # float | list[float] | None  — lattice stiffness
    nu=0.0,              # float | list[float]          — Poisson exponent
    dynamic_params=None, # DynamicParams | dict[str, float|list]
    duration_ms=1000.0,
    dt=1.0,
    replicates=1,
    rng_seed=0,
    unroll=1,
    minibatch_size="auto",  # "auto" | int | None
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

static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

# Simple isometric
result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000)

# pCa sweep
result = run(topo, pCa=[9.0, 6.0, 4.5], replicates=5)

# DynamicParams sweep
result = run(topo, pCa=4.5, dynamic_params={'thick_k': [1000, 2000, 3000]})

# Dynamic lattice spacing
result = run(topo, pCa=4.5, K_lat=5.0, nu=0.5, duration_ms=500)
```

---

## 3. SimulationResult

**File:** `multifil_jax/simulation.py`

```
result.axial_force      # property → metrics['axial_force'] (pN)
result.metrics          # MetricsDict of ~46 metric arrays, same shape
result.z_line           # z_line trace used
result.pCa              # pCa trace used
result.metrics['solver_residual']  # Newton solver residual at each step
result.metrics['newton_iters']     # Newton iterations per step
result.metrics['lattice_spacing']  # actual LS each step (emergent if dynamic)
result.dt               # timestep (ms)
result.coords           # {'pCa': [...], 'z_line': [...], ...}
result._axis_names      # ['pCa', 'replicates', 'time']
result.topology_config  # dict with n_thick, n_thin, n_sites, etc.
```

`MetricsDict` (defined in `core/state.py`) is a dict subclass with attribute access:
`result.metrics.axial_force == result.metrics['axial_force']`.

**Methods:**
- `.mean()` — collapse replicates axis (returns SimulationResult)
- `.std()` — standard deviation over replicates axis
- `.sel(**kwargs)` — slice by coordinate value, e.g. `.sel(pCa=4.5)`
- `.stack(results, axis_name)` — stack list of SimulationResult into sweep axis
- `.summary()` — human-readable text summary

---

## 4. JIT-compiled Kernel: `_run_sim_kernel`

**File:** `multifil_jax/simulation.py`

This is the vmapped+scanned simulation kernel:

```python
@partial(jax.jit, static_argnames=['dt', 'unroll', 'is_dynamic_ls'])
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
) -> MetricsDict:         # shape (batch, n_steps) for each key
```

`is_dynamic_ls` is a JIT static arg — fixed LS and dynamic LS compile to separate
kernels. `K_lat` and `nu` are traced (not static), so different stiffness values
share the same compiled kernel.

Scan carry is `(state, rng_key, current_ls)` — the third element tracks the
emergent lattice spacing (identity passthrough for fixed LS).

All ~46 metrics are always computed. No `metrics`/`manifest` in JIT
`static_argnames` — changing metric selection never triggers recompilation.

---

## 5. Single Timestep

**File:** `multifil_jax/timestep.py`

Two public functions:

### `kinetics_step()` — stochastic phase (steps 0–5)

```python
state, rng_key, resolved_constants = kinetics_step(
    state, constants, drivers, topology, rng_key, dt=dt
)
```

Performs driver resolution, cooperativity, nearest neighbors, and stochastic
TM/XB transitions. Returns `resolved_constants` with driver values baked in.

Separated from the mechanical solve to support future FE coupling: run kinetics
across all coupled sarcomeres, then perform a coupled equilibration.

### `timestep()` — full step (kinetics + equilibrium)

```python
new_state, new_key, residual, new_ls, n_iters = timestep(
    state, constants, drivers, topology, rng_key, dt=dt,
    K_lat=None, d_ref=None,
    precond_params=None, prefactored_precond=None,
)
```

Returns a 5-tuple. `K_lat is None` selects fixed LS mode (resolved at trace time,
no runtime branch). When `K_lat` is not None, passes `K_lat` and `d_ref` to
`solve_equilibrium()` which handles the augmented (n+1)-DOF dynamic LS solve.

**7-step workflow:**

1. **resolve_value** — merge Drivers (Tier 3) with Constants (Tier 2)
2. **calculate_thin_forces_for_cooperativity** — internal thin filament spring forces
3. **update_cooperativity** — TM chain cooperative activation (tension-dependent span)
4. **update_nearest_neighbors** — per-XB geometry (distances to binding sites)
5. **thin_transitions** — TM 4-state Markov transitions (matrix exponential)
6. **thick_transitions** — XB 6-state Markov transitions (per-XB matrix exponential)
7. **solve_equilibrium** — Newton-CG solver (unified fixed/dynamic LS)

Steps 0–6 are `kinetics_step()`. Step 7 is the mechanical solve.

---

## 6. State Hierarchy

**File:** `multifil_jax/core/state.py`

```python
State(
    thick = ThickState(
        axial,          # (n_thick, n_crowns) crown positions (nm)
        xb_states,      # (n_thick, n_crowns, 3) XB states (1-6), int8
        xb_bound_to,    # (n_thick, n_crowns, 3) bound site index (-1=unbound)
        xb_nearest_bs,  # (n_thick, n_crowns, 3) nearest BS index
        xb_distances,   # (n_thick, n_crowns, 3, 2) distances to nearest BS
    ),
    thin = ThinState(
        axial,           # (n_thin, n_sites) site positions (nm)
        tm_states,       # (n_thin, n_sites) TM states (0-3), int8
        subject_to_coop, # (n_thin, n_sites) bool
        bound_to,        # (n_thin, n_sites) XB address (-1=unbound)
        # rests: moved to SarcTopology (Tier 1)
        # permissiveness: derived inline as (tm_states == 3).astype(float32)
    ),
)
```

**MetricsDict** — scan output. A dict subclass with attribute access, registered
as a JAX PyTree. Contains all ~46 metric scalars per timestep (including
`axial_force`, `solver_residual`, `newton_iters`).

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
    n_cg_steps=2,                    # CG iterations per Newton step (2 recommended; 0=Richardson, diverges with bound XBs)
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

### Axial forces (for equilibrium solver and output)
- `axial_force_at_mline(state, constants)` — total M-line force (pN)
- `compute_forces_vectorized(...)` — per-node axial residual forces for solver
- `compute_forces_from_state_vectorized(state, constants, topology)` — convenience wrapper

Force contributions: thick spring chain, thin spring chain, XB (converter + globular
springs for states 2-4), titin (exponential model).

### Radial forces (for dynamic lattice spacing solver)
- `_xb_radial_force_total(...)` — total XB radial force Σ dV_XB/dd (pN). Differentiable w.r.t. lattice_spacing for the augmented Newton JVP.
- `_titin_radial_force_total(...)` — total titin radial force from all thick filaments

Both functions replicate the geometry from their axial counterparts but accumulate
the radial component instead. Used by `_radial_residual()` in `solver.py`.

---

## 13. Solver Kernel

**File:** `multifil_jax/kernels/solver.py`

### Unified solver (fixed and dynamic lattice spacing)

`solve_equilibrium(state, constants, topology, K_lat=None, d_ref=None) → (State, residual, new_ls, n_iters)`

Returns a 4-tuple: the equilibrated state, scalar max residual (pN), the
lattice spacing used (solved `d` in dynamic mode, `constants.lattice_spacing` in
fixed mode), and the number of Newton iterations used. `K_lat is None` selects
fixed LS mode at trace time (no runtime branch).

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

### Tolerance floor

```python
tolerance = max(tolerance, thick_k * 1e-4, MIN_FLOAT32_TOLERANCE)
```

Prevents the while_loop from chasing an unreachable target at stiff parameter values.

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
    pre_solve_thick_pos, force, solver_residual, newton_iters, dt
)
```

Returns a `MetricsDict` with ~46 keys (same keys every call). Always computed —
no selection needed.

**Metric groups:**
- Protocol: `axial_force`, `solver_residual`, `z_line`, `pCa`, `lattice_spacing`
- XB counts: `n_bound`, `n_xb_drx`, `n_xb_loose`, `n_xb_tight_1`, `n_xb_tight_2`, `n_xb_free_2`, `n_xb_srx`
- XB fractions: `frac_xb_bound`, `frac_xb_drx`, `frac_xb_loose`, `frac_xb_tight_1`, `frac_xb_tight_2`, `frac_xb_free_2`, `frac_xb_srx`
- TM counts: `n_tm_state_0` through `n_tm_state_3`
- TM fractions: `frac_tm_state_0` through `frac_tm_state_3`, `actin_permissiveness`
- Transitions: `atp_consumed`, `newly_bound`
- Displacement: `thick_displace_mean/max/min/std`, `thin_displace_mean/max/min/std`
- Energy: `thick_energy_first_avg`, `thick_energy_first_delta_avg`, `titin_energy_avg`, `titin_energy_delta_avg`
- Work: `work_thick`, `work_thick_mean`, `work_per_atp`
- ATP expected: `atp_expected_p`, `atp_expected_q`
- Solver: `newton_iters`

---

## 15. Appendix: Key File Reference

| File | Purpose |
|------|---------|
| `multifil_jax/simulation.py` | `run()`, `SimulationResult`, `_run_sim_kernel` |
| `multifil_jax/timestep.py` | `kinetics_step()`, `timestep()` — single step orchestrator |
| `multifil_jax/metrics_fn.py` | `compute_all_metrics()` — ~46-metric MetricsDict |
| `multifil_jax/core/state.py` | State hierarchy, `realize_state()`, `Drivers`, `resolve_value()`, `MetricsDict` |
| `multifil_jax/core/params.py` | `StaticParams`, `DynamicParams`, `get_skeletal_params()`, `get_cardiac_params()` |
| `multifil_jax/core/sarc_geometry.py` | `SarcTopology` — PyTree topology |
| `multifil_jax/kernels/cooperativity.py` | `update_cooperativity()` |
| `multifil_jax/kernels/geometry.py` | `update_nearest_neighbors()` |
| `multifil_jax/kernels/transitions.py` | `thin_transitions()`, `thick_transitions()`, `compute_xb_transition_matrices()` |
| `multifil_jax/kernels/forces.py` | `axial_force_at_mline()`, `compute_forces_vectorized()`, `_xb_radial_force_total()`, `_titin_radial_force_total()` |
| `multifil_jax/kernels/solver.py` | `solve_equilibrium()` (unified fixed/dynamic LS), Thomas algorithm |
| `multifil_jax/kernels/rate_functions.py` | Rate functions (absolute values, no multipliers) |
| `multifil_jax/helper.py` | `count_transitions()` and other utilities |
| `examples/dynamic_lattice_spacing.py` | Dynamic LS demo: isometric, force comparison, K_lat sweep, length ramp |
| `examples/benchmarks/benchmark_minibatch.py` | Minibatch size benchmark CLI |
| `examples/benchmarks/benchmark_dynamic_ls.py` | Dynamic LS performance and lattice scaling benchmark |
| `tests/` | Test suite |

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
- **float16 sampling (cumsum+argmax)**: Systematic one-state-downward bias (~0.088% of samples). No throughput benefit.
