# Code architecture

How a run flows through the code, what each layer owns, and the contracts
between kernels. For how to *use* the package, see [README.md](README.md). Each
function's exact signature lives in its docstring; this file names functions
rather than copying their argument lists.

---

## 1. Tiers

| Tier | Type | Holds | Changing it |
|------|------|-------|-------------|
| 0 | `State(thick, thin)` | pure simulation state — no parameters, no geometry | n/a |
| 1 | `SarcTopology` | structural index maps, and the `StaticParams` it was built from (`topology.static`) | rebuilds the topology and recompiles |
| 2 | `DynamicParams` (alias `Constants`) | physics values as JAX arrays | nothing — sweeps ride the batch axis |
| 3 | `Drivers(pCa, z_line, lattice_spacing)` | per-timestep values | nothing |

**Parameters have one source.** `core/param_table.py` holds one row per
parameter, giving a nominal value, a confidence tag (M/I/G/F), a one-line basis
and, where one can be defended, a range, for each muscle class (`skeletal`,
`cardiac_beta`, `cardiac_alpha`, `insect`). A row's `kind` sends it to
`StaticParams`, `DynamicParams`, or the preset's operating point `(z0, d0)`.
`get_params(cls)` is the only constructor. Neither container has defaults: a
missing or unknown field raises `TypeError`. `resolve(cls)` returns every row
with its tag, basis and range.

**Drivers are not parameters.** `run()` requires `pCa`, `z_line`,
`lattice_spacing`, `dt` and `duration_ms` and rejects non-finite input. Below
the orchestration layer (`kinetics_step`, `timestep`, `compute_all_metrics`,
`KineticsTrace`), each kernel takes the driver scalars it uses as explicit
arguments right after `topology`, so its signature states its driver dependence.

---

## 2. From `run()` to the kernel

```
run() → validate & build SimBatch → pad / chunk → _run_sim_kernel
                                                    vmap(run_single_sim) → lax.scan(step)
                                                     ↑ batch                 ↑ time
```

**Input classification** (`simulation.run`). Everything is validated before
anything compiles:

- `params` is one `DynamicParams` or a list of them. A list becomes the
  `'candidates'` axis, and the fields that differ between candidates are
  recorded in `result.metadata['candidates']`.
- Keyword arguments named after a `DynamicParams` field: a scalar overrides the
  field, a **list** becomes a sweep axis. An array or tuple raises ("use a
  list"); so does a `StaticParams` field, an unknown name, an empty list, or a
  field kwarg on a params list whose members differ in that field.
- Drivers: a scalar is constant, a list is a sweep axis, an array of length
  `n_steps` is a time course. `lattice` picks the spacing mode; `K_lat` and
  `nu` are ordinary fields.
- Every list is **crossed** with every other. Axis order: `z_line`, `pCa`,
  `lattice_spacing`, `candidates`, field lists (kwarg order),
  `subpopulation`, then `replicates` and `time`.

**SimBatch.** `run()` packs every per-sim input into one `SimBatch` NamedTuple
(`params, z, pCa, ls, rng_keys, subpop`), each leaf carrying a
leading batch axis. Padding, minibatch chunking and trimming are each a single
`tree_map` over it, so no per-sim input can be missed by one of them.

- **Padding**: the batch is rounded up to the next `BATCH_BUCKETS` size, so
  sweeps of similar size share a compiled kernel. Padding sims copy sim 0 and are
  trimmed off.
- **Chunking**: `minibatch_size="auto"` splits large batches into chunks that
  reuse one compiled kernel. Chunks accumulate on the host (numpy), so the GPU
  holds one chunk at a time.

**`_run_sim_kernel`** is the one `@jax.jit` in the package. Static arguments are
`dt`, `unroll`, `is_dynamic_ls`, the solver caps (read from `topology.static`),
`subpop_mode` and `scaled_field_names`; changing any of them recompiles.
Parameter values (`K_lat` and `nu` included), driver values and subpopulation scales and
masks are traced, so they do not. The topology enters by closure, not as a
vmapped argument.

**Every simulation starts relaxed.** `realize_state` builds the structural state
(tropomyosin blocked, no head bound) and the equilibrium is solved on it. Then
`transitions.xb_rest_states` draws every head from the stationary distribution of
its own generator at pCa 9 on blocked units, where attachment is zero. The
detached states {0 DRX, 4 Free_2, 5 SRX} form a closed class, so the draw solves
`πQ = 0` on that 3×3 block. `transitions.tm_rest_states` then draws every Tm unit
from its own uncoupled generator at pCa 9. A run whose first pCa is not 9 is therefore a step from rest, and no
settling time is needed.

**Scan carry** is `(state, rng_key, current_ls)`. The third element is the
solved lattice spacing in dynamic-LS mode and a passthrough otherwise. Every
metric is computed every step: the scan body must return the same pytree on
every iteration, and making the set selectable would put it in the JIT
signature.

---

## 3. One timestep

`timestep.py` exposes two functions:

| Step | Function | Phase |
|------|----------|-------|
| 1 | `kernels.geometry.update_nearest_neighbors` | kinetics |
| 2 | `kernels.transitions.thin_transitions` | kinetics |
| 3 | `xb_binned_generator`, then `thick_transitions` | kinetics |
| 4 | `kernels.solver.solve_equilibrium` | mechanics |

`kinetics_step()` runs steps 1–3 and leaves the lattice out of force balance.
`timestep()` runs all four. It returns
`(state, key, residual, new_ls, n_iters, trace, residual_norm, tol_axial)`;
the caps `n_cg_steps` / `n_newton_steps` are required keywords.

**KineticsTrace** (`core/state.py`) carries what the metrics need to describe the
step that actually happened:

| Field | What it is |
|---|---|
| `state` | the **mid** state — after `thin_transitions`, before `thick_transitions` |
| `drivers` | the **pre-solve** drivers the rates were evaluated at |
| `torn` | per-head mask of heads tropomyosin tore off this step |
| `xb_bins` | the `XBBins` the step sampled from, with both exponentials |

`state` is the mid state because that is what `thick_transitions` builds its
generator from. `torn` is carried rather than re-derived because a torn head
lands in a state an ordinary head also reaches. The trace holds a whole `State`,
so it is a within-step value only, never a scan carry or output.

The scan body builds a second, **post-solve** `Drivers` (with the solved
spacing) for the mechanical metrics, and passes both to `compute_all_metrics`.
Each is right for its own question; do not unify them.

---

## 4. State

`core/state.py`:

```
State(
  thick = ThickState(
    displacement   (n_thick, n_crowns)                  crown displacement from rest, nm
    xb_states      (n_thick, n_crowns, n_xb_per_crown)  int8, 0-5
    xb_bound_to    (n_thick, n_crowns, n_xb_per_crown)  bound MONOMER index, -1 unbound
    xb_nearest_bs  (n_thick, n_crowns, n_xb_per_crown)  nearest candidate MONOMER index
    xb_distances   (..., 2)                              (axial, radial) to that site
  ),
  thin = ThinState(
    displacement   (n_thin, n_nodes)   NODE displacement from rest, nm
    tm_states      (n_thin, n_tm)      int8, 0-3
    bound_to       (n_thin, n_cand)    head address per binding candidate, -1 unbound
  ),
)
```

| XB state | Name | Meaning |
|---|---|---|
| 0 | DRX | disordered relaxed, detached |
| 1 | Loose | weakly bound |
| 2 | Tight_1 | strongly bound, post-stroke, phosphate released |
| 3 | Tight_2 | strongly bound, ADP-release-competent |
| 4 | Free_2 | detached after ADP release, ATP bound |
| 5 | SRX | super-relaxed, parked on the backbone |

Bound heads are states 1–3. Rate names follow the indices (`xb_r01_coeff`,
`xb_r12_coeff`, …).

**Coordinates are displacements.** Rest frames live on the topology:
`crown_offsets` (from the M-line) and `node_offsets` (from the Z-disc). Backbone
forces are `diff(u) * k`, with no rest lengths and no `z_line`. Only the
crossbridge, titin, binding-search, hiding-line and overlap paths need absolute
positions; they reconstruct them with `thick_axial(u_thick, topo)` and
`monomer_axial(u_thin, topo, z_line)`, both of which take displacement arrays.
The thin frame follows the Z-disc rigidly, so a moving `z_line` needs no state
update.

States are immutable NamedTuples: update with `._replace()`.

---

## 5. Topology

`SarcTopology.create(nrows, ncols, static, *, periodic, thin_starts,
thick_starts)` runs once in NumPy and returns a registered pytree. Integer sizes
and `static` are aux data; index arrays are children. The field set is declared
once, in `_AUX` / `_CHILDREN`.

**The thin filament is three layers on one monomer list:**

| Layer | Fields | Role |
|---|---|---|
| Monomers | `n_mono`, `mono_offsets`, `mono_node`, `mono_xi`, `mono_tm` | every actin; `monomer_axial` interpolates node displacements, and the XB reaction scatters back with the same weights |
| Binding candidates | `n_cand`, `cand_mono`, `mono_cand`, `face_to_monomers`, `xb_to_mono_indices` | monomers a head can reach; `thin.bound_to` is indexed by candidate |
| Mechanical nodes | `n_nodes`, `node_offsets` | an even grid every `static.thin_node_spacing` nm; segment stiffness from `thin_EA` via `core.state.thin_segment_k` (the only EA→k conversion) |
| Tropomyosin units | `n_tm`, `tm_chains`, `tm_prev_neighbor`, `tm_next_neighbor`, `tm_rep_mono` | one unit per `static.tm_monomers_per_unit` actins; `None` gives one unit per candidate |

**Geometry.** `actin_geometry='vertebrate'` gives 1 thick : 2 thin and 3 faces
per thin; `'invertebrate'` gives 1:3 and 2 faces. `crown_offsets` are
per-filament and differ only when `n_superlattice_classes > 1`. Filament starts
default to a deterministic coprime-step spread; invertebrate thin starts default
to zero because the 3-fold actin registration class supplies the phase variation.

**`create()` raises** when, under `periodic=True`, a thick face has no real thin
partner (invertebrate geometry needs an even `nrows`), and when
`n_superlattice_classes=3` is used without `nrows` even and `ncols` a multiple of 3.

**`xb_valid` gate.** `xb_to_thin_id` / `xb_to_thin_face` are dense arrays, so a
head with no real partner holds a placeholder `(0, 0)` that looks like a real
target. Kernels gate on `xb_valid`; analysis code must too. Use
`topo.valid_xb_targets()` for the masked view. At insect geometry a large share
of slots are invalid.

---

## 6. Kernels

### Geometry — `kernels/geometry.py`

`update_nearest_neighbors` finds each head's nearest binding candidate with a
fixed-width gather over `xb_to_mono_indices`. The search uses the head position
(crown + `static.xb_head_reach`); the stored distance is from the crown. Sites
past the hiding line, where the thin filament tip crosses the M-line, are masked
invalid rather than clamped.

### Tropomyosin — `transitions.thin_transitions`

- A 6-state unit, `state = 2·position + has_calcium` (position 0 blocked,
  1 closed, 2 open; `tm_position`). Position edges blocked ↔ closed ↔ open at each
  calcium level, calcium edges at each position, no blocked ↔ open edge; the
  calcium affinities at closed and open are forced by detailed balance
  (`rate_functions.tm_ca_affinity_ratios`).
- Symmetric Ising coupling on position to the two same-strand neighbours:
  weights W = (−1, 0, +1), `h = tm_J·ΣW(neighbours)`, and a position step s → t
  gets `exp((W(t) − W(s))·h/2)`. Calcium edges get exp(0) = 1.
- Rates depend only on the neighbour composition `(n_blocked, n_closed, n_open)`
  (27 configurations) and the number of STRONG heads bound on the unit
  (0…`tm_max_heads`), so `27 × (tm_max_heads + 1)` 6×6 matrices serve every unit.
  They come from `_compute_unique_tm_Q_matrices`, are exponentiated in one batch,
  and each unit gathers its row.
- **The crossbridge lock is a rate.** With n strong heads bound, the exits from
  open (4 → 2, 5 → 3) are divided by `(1 + xb_tm_K2)^n` inside Q, which
  reproduces the equilibrium ratio at any `dt`. Weak heads do not lock.
  `xb_tm_K2 = jnp.inf` gives the hard lock exactly.
- **Tears** (`torn`): a unit that leaves open tears its Tight_1/Tight_2 heads to
  Free_2, charged one ATP; a unit that enters blocked tears its Loose heads to
  DRX, owing nothing. A Loose head on a unit that only closes stays bound.

### Crossbridges — `transitions.xb_binned_generator` + `thick_transitions`

- Rates depend on the head's axial distance to its target and the target's
  block, so `_build_xb_Q_bins` builds one blocked 6×6 generator plus `4 × n_xb_bins`:
  closed, closed-screened, open, open-screened (screened = the thin–thin
  double-overlap zone, where `r01` is scaled by `1 − thin_thin_overlap_screening`).
  Blocked needs only one: nothing binds there and the unbound rows do not depend on
  distance, and no bound head ever reads a blocked unit (it is torn first).
  Two gates: `attach` (closed or open) multiplies `r01`, `stroke` (open) multiplies
  `r12`; each reverse rate uses its ungated forward rate. A bound head reads the
  unit under its bound monomer, an unbound head the unit under its nearest
  (`xb_gate_position`). Each head's key is its bin plus its block, a function of
  geometry, the gate and the hiding line, never of rates.
- `_xb_Q_resolved` applies subpopulations (`mean_field` blends generators,
  `explicit` stacks them). It takes no exponentials, so sampling and metrics
  share one physics.
- `xb_binned_generator` takes **one** exponential per step and returns
  `XBBins(Q, P, G, key, labels)`, where `P = exp(Q·dt)` and
  `G = ∫₀^dt exp(Qt)dt` (computed alongside P in the same scaling-and-squaring).
- `thick_transitions` samples from `P`, then updates both binding records.
  An SRX head's row first takes the backbone-stress exit: `kinetics_step` computes,
  per crown, the release probability from the tension on that crown's M-line side
  (`rate_functions.xb_rate_50_stress`), and the row becomes `(1 − p)·P[5] + p·e_DRX`.
  `G` does not see this exit.
  Same-step collisions on one site are resolved by scatter-min over head index;
  the loser returns to DRX.
- Invalid heads are forced into the blocked block, where `r01` is exactly zero.
- Every rate law lives in `kernels/rate_functions.py`, one function per
  transition; the builders only assemble.

### Forces — `kernels/forces.py`

- **Two-spring primitives** at the top of the module (`xb_geometry`,
  `xb_springs_for_state`, `xb_elastic_energy`, `xb_polar_forces`,
  `polar_to_filament`) define the crossbridge force law once. The axial solver
  path, the by-state accounting, the radial path and the load-dependent rate
  `r34` all call them.
- `compute_forces_vectorized` returns the per-node axial residual: thick and thin
  backbone springs, crossbridges (the reaction on a monomer is split between its
  two bracketing nodes), and titin. Physics values come from `constants`; the
  titin count from `topology.static.n_titin_per_thick`.
- `axial_force_at_mline` is the reported force: the strain in the first thick
  backbone segment, summed over thick filaments. It includes titin.
- `_xb_radial_force_total` / `_titin_radial_force_total` are the radial sums for
  dynamic lattice spacing. They must stay differentiable in `lattice_spacing`.

### Solver — `kernels/solver.py`

`solve_equilibrium` returns
`(state, residual, new_ls, n_iters, residual_norm, tol_axial)` for both
lattice-spacing modes; `K_lat is None` selects fixed spacing at trace time.

- **Newton–CG.** An outer `while_loop` exits at convergence or at
  `n_newton_steps`. The inner CG runs `n_cg_steps` iterations as a Python loop,
  unrolled so XLA can fuse it. A branch-free backtracking line search
  (λ = 1 tried first) breaks the period-2 cycles undamped Newton can fall into.
- **Preconditioner.** A tridiagonal (Thomas) preconditioner is factored once
  before the scan. `thomas_solve` uses `jax.lax.associative_scan`.
- **Tolerance is scale-free and per block:**
  `tol = solver_atol + solver_rtol · scale`, where `scale` is the RMS backbone
  spring force on axial rows and `|K_lat|·d_ref` on the radial row. Converged
  means `max(|F| / tol) ≤ 1`, which is the `solver_residual_norm` metric.
- **Dynamic lattice spacing.** `d` becomes an extra unknown. The augmented
  residual is `[f_axial(x, d), f_radial(x, d)]` with
  `f_radial = −K_lat·(d − d_ref) − f_xb − f_titin`, and the JVP supplies every
  cross term. The `d` block is preconditioned with the exact `∂f_radial/∂d`, and
  `d` is projected to stay above 1 nm. `run()` scales the per-filament `K_lat` by
  `n_thick`.

### Lattice-spacing modes

| `run(lattice=...)` | Reads | Mode |
|---|---|---|
| `'fixed'` | — | fixed spacing |
| `'poisson'` | `nu` | `poisson_spacing(d0, z0, z, nu, static.filament_radii_sum)` computed as a time course before the scan |
| `'dynamic'` | `K_lat`, `nu` | `d` solved from radial force balance every step, about the Poisson reference |

`is_dynamic_ls` is static, so the two solver paths compile separately.

---

## 7. Metrics

`metrics_fn.compute_all_metrics` returns a `MetricsDict` (a dict with attribute
access) with the same keys on every call. **Every key is independent**: none can
be computed from the others plus the topology. The module docstring describes
each group.

| Group | Keys |
|---|---|
| Protocol & solver | `axial_force`, `solver_residual`, `solver_residual_norm`, `solver_tolerance`, `lattice_spacing`, `newton_iters` |
| XB counts | `n_xb_drx`, `n_xb_loose`, `n_xb_tight_1`, `n_xb_tight_2`, `n_xb_free_2`, `n_xb_srx` |
| XB force by state | `force_xb_loose`, `force_xb_tight_1`, `force_xb_tight_2` |
| TM counts | `n_tm_state_0` … `n_tm_state_5` |
| TM overlap zone | `frac_tm_closed_overlap`, `frac_tm_open_overlap`, `n_overlap_sites` |
| Events | `atp_consumed`, `newly_bound`, `closure_detach_free`, `closure_detach_atp`, `n_xb_loose_closed` |
| Displacement | `thick_displace_{mean,max,min,std}`, `thin_displace_{mean,max,min,std}` |
| Energy | `thick_energy_first_avg`, `titin_energy_avg`, `titin_energy_delta_avg` |
| Work | `xb_work_on_filaments` |
| ATP & cycle fluxes | `atp_expected`, `xb_detach_free`, `xb_give_up`, `atp_net_pi_release`, `atp_net_hydrolysis` |

**ATP.** `atp_expected` is the exact expected number of 3→4 crossings,
`Σ q34 · G[s, 3]` over every head from its own mid state, plus the realised
strong closure tears. It reads the very `XBBins` the sampler used. The one
approximation is that Q is held constant over the step. `atp_consumed` is a
realised count of heads observed to leave Tight_2, and it undercounts heads that
cycle more than once in a step.

**Overlap-zone fractions** average only over sites a head could reach: within
`static.xb_head_reach` of the crowns and past the hiding line. Their denominator
moves with the geometry, which is why they are exported and the all-site
fractions are not.

**Derived quantities** (compute these from the result rather than looking for a
key):

| To get | Compute |
|---|---|
| prescribed z_line / pCa | `result.z_line`, `result.pCa` |
| fraction of heads in a state | count / `topology.total_xbs` |
| fraction of Tm units in a state | count / (`n_thin × n_tm`) |
| bound heads | `n_xb_loose + n_xb_tight_1 + n_xb_tight_2` |
| available Tm in the overlap zone | `frac_tm_closed_overlap + frac_tm_open_overlap` |
| ATP from the 3→4 route alone | `atp_expected − closure_detach_atp` |
| external work at the z-line | `−0.5·(F[..., :-1] + F[..., 1:]) · diff(z)` |
| work per ATP over a window | `sum(xb_work_on_filaments) / sum(atp_expected)` — never per step |

`titin_energy_delta_avg` is not `diff(titin_energy_avg)`: both endpoints are
evaluated at the current step's drivers, so it isolates filament motion when
`z_line` or `d` moves.

---

## 8. Subpopulations

`core/subpopulation.py`, plumbed `run()` → `timestep` → `transitions` /
`metrics_fn`. A `Subpopulation` is a list of K populations (index 0 = wild
type), each a dict of multiplicative scales on `xb_*` / `tm_*` rate fields.
Mechanics always use the base parameters.

| Constructor | Mechanism |
|---|---|
| `mean_field(fraction, **scales)` | generator blend `Q_eff = Σ f_k Q_k`, one exponential; deterministic |
| `random(fraction, seed, **scales)` | per-head / per-unit labels, redrawn per sim from `seed + index` |
| `c_zone(topo, *, c_zone_min_nm, c_zone_max_nm, z_line=None, **scales)` | labels by crown axial band; `tm_*` scales need `z_line` |

There is one code path. No subpopulation means the mean-field case with K = 1,
fraction 1.0 and no scaled field, which `transitions._populations` handles
directly. A list of subpopulations is a sweep axis: all entries must share a mode
and K, and a swept `random` list must share one seed. `fraction = 0` reproduces
the wild type and `fraction = 1` a global scale, exactly.

---

## 9. Files

| File | Contents |
|---|---|
| `simulation.py` | `run()`, `SimulationResult`, `_run_sim_kernel`, batching |
| `timestep.py` | `kinetics_step()`, `timestep()` |
| `metrics_fn.py` | `compute_all_metrics()` |
| `core/param_table.py` | the parameter table |
| `core/params.py` | `StaticParams`, `DynamicParams`, `get_params`, `resolve`, `draw`, `sample_params`, `poisson_spacing` |
| `core/state.py` | `State`, `Drivers`, `KineticsTrace`, `MetricsDict`, `realize_state`, coordinate helpers |
| `core/sarc_geometry.py` | `SarcTopology` |
| `core/subpopulation.py` | `Subpopulation` |
| `kernels/geometry.py` | nearest-site search |
| `kernels/transitions.py` | tropomyosin and crossbridge transitions, matrix exponentials, expected crossings |
| `kernels/rate_functions.py` | every rate law |
| `kernels/forces.py` | force law, residual, M-line force, radial forces |
| `kernels/solver.py` | Newton–CG, Thomas preconditioner, dynamic-LS solve |
| `helper.py` | independent validators (`validate_forces_numerical`, `validate_equilibrium`, `count_transitions`) |
| `utils/hardware.py` | GPU detection, XLA cache |

---

## 10. Performance constraints

These shape the code and should survive refactors:

- Only `_run_sim_kernel` is jitted. A `@jax.jit` inside `scan`/`vmap` adds
  boundaries that block fusion.
- Inner loops that XLA must fuse (CG, Thomas) are Python loops or
  `associative_scan`, not `fori_loop` / `while_loop`.
- Matrix exponentials are taken on the small set of distinct generators, then
  gathered per unit; nothing is exponentiated per head.
- Rate matrices are rebuilt each step rather than carried in the scan carry.
