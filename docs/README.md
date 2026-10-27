# multifil_jax user guide

This guide is for someone comfortable with NumPy who has not used JAX, GPU
computing or JIT compilation. It covers what is simulated, how to run and sweep
it, what comes back, and the physics a user needs to read the results. For how
the code is organised internally, see [CODE_Architecture.md](CODE_Architecture.md).

1. [What is simulated](#1-what-is-simulated)
2. [JAX in one page](#2-jax-in-one-page)
3. [Parameters and topology](#3-parameters-and-topology)
4. [Running a simulation](#4-running-a-simulation)
5. [Sweeps](#5-sweeps)
6. [Reading the result](#6-reading-the-result)
7. [Metrics](#7-metrics)
8. [Recompilation and batching](#8-recompilation-and-batching)
9. [Inside one timestep](#9-inside-one-timestep)
10. [Tropomyosin cooperativity](#10-tropomyosin-cooperativity)
11. [Subpopulations](#11-subpopulations)
12. [Lattice spacing](#12-lattice-spacing)

---

## 1. What is simulated

A **half-sarcomere**: the repeating unit of striated muscle between the M-line
and the Z-disc. It holds two interleaved filament sets:

- **Thick filaments** (myosin), anchored at the M-line. Each carries crowns of
  myosin heads — the crossbridges (XBs).
- **Thin filaments** (actin), anchored at the Z-disc and decorated with
  tropomyosin (Tm), the regulatory strand that covers myosin's binding sites
  until calcium moves it aside.

Every head is a stochastic state machine that binds actin, strokes and detaches.
Every Tm unit is a stochastic state machine driven by calcium and by its
neighbours. Every filament is a chain of springs. Each timestep the kinetics are
sampled, and then the filament lattice is solved back into force balance. A 4×4
lattice tracks thousands of mechanical degrees of freedom and thousands of
individual heads at once.

---

## 2. JAX in one page

JAX looks like NumPy (`jnp.sum`, `jnp.where`, …). Three differences matter here:

- **JIT compilation.** The first `run()` with a given configuration compiles the
  whole simulation loop into one GPU program, which takes from 30 seconds to a
  few minutes. Later calls with the same shapes reuse it. Compiled kernels are
  also cached on disk (`~/.cache/multifil_jax/xla/`), so a new Python session
  loads them in seconds.
- **vmap.** A sweep of many conditions runs as one GPU computation in which
  every simulation advances at once, rather than as a Python loop.
- **Immutable arrays.** Arrays are never modified in place; code writes
  `x.at[i].set(v)` and gets a new array.

JAX dispatches work asynchronously. When timing, wait for the result with
`jax.block_until_ready(result.axial_force)`.

---

## 3. Parameters and topology

### Muscle classes

Every physical, structural and numerical value comes from one table,
`multifil_jax/core/param_table.py`. Each row gives, per muscle class, a nominal
value, a confidence tag and a one-line basis. Where one can be defended, it also
gives a range.

| Class | What it is |
|---|---|
| `skeletal` | fast-twitch skeletal, ~26 °C |
| `cardiac_beta` | β-myosin cardiac (large mammals, human) |
| `cardiac_alpha` | α-myosin cardiac (rodents): `cardiac_beta` with the crossbridge rates that have a matched α/β measurement scaled |
| `insect` | insect flight muscle (Lethocerus): 1:3 lattice, 4 heads per crown, longer filaments |

```python
from multifil_jax import get_params

static, dynamic, z0, d0 = get_params('cardiac_beta')
```

`get_params` returns four things:

- `static` (`StaticParams`) — structure: crowns per filament, heads per crown,
  the actin helix, binning resolution, solver iteration caps. Frozen; changing it
  means a new topology and a recompile.
- `dynamic` (`DynamicParams`) — physics: stiffnesses, rates, free energies,
  couplings, solver tolerances. JAX arrays, so they sweep without recompiling.
- `z0`, `d0` — the class's natural operating point: half-sarcomere length
  (`z_line`, nm from the M-line) and lattice spacing (nm).

**There are no defaults.** Neither container can be built field by field: a
missing or unknown field raises `TypeError`. To change a value, copy:

```python
dynamic = dynamic.copy(thick_k=9000.0, xb_r05=0.15)
static = static.replace(n_superlattice_classes=3)     # the Drosophila superlattice
```

**Confidence tags.** Every row is tagged M (measured), I (inferred from measured
values), G (guess or carried over from another muscle), or F (fitted to an
emergent measurement). Muscle models are underdetermined, and many values are G.
To inspect a class:

```python
from multifil_jax.core.params import resolve
rows = resolve('cardiac_beta')
rows['xb_r34_coeff']        # nominal, tag, basis, lo, hi, scale, source, unit, kind
```

### Topology

```python
from multifil_jax import SarcTopology
topo = SarcTopology.create(4, 4, static)
```

`SarcTopology.create(nrows, ncols, static)` builds every structural index map
once, on the CPU. It records which heads can reach which actin monomers, how
titin connects, and which Tm units neighbour each other. The topology keeps
`static` (`topo.static`), so `run()` reads the structure and solver caps from
it. A 2×2 lattice has 4 thick and 8 thin filaments (vertebrate); larger lattices
add GPU threads rather than wall time until the GPU is saturated.

Some shapes are rejected: invertebrate geometry needs an even `nrows`, and the
3-class superlattice also needs `ncols` divisible by 3 (so 4×4 fails; use 4×6 or
6×6).

**The thin filament has three layers on one monomer list.** Every actin monomer
is kept. The *binding candidates* are the monomers that face a thick filament,
within `target_zone_wiggle`; only those can be bound. The *mechanical nodes* are
an even grid every `thin_node_spacing` nm, with stiffness `thin_EA` per unit
length; each monomer moves with the two nodes around it. The *Tm units* are the
regulatory switches: one per binding candidate by default, or 7 monomers each
with `tm_monomers_per_unit=7`.

**Pitfall: `xb_valid`.** Every head slot needs a target index, because array
shapes are fixed. A head with no real partner holds the placeholder "thin 0,
face 0", which looks like a real target if you read `topo.xb_to_thin_id`
directly. The kernels gate on `topo.xb_valid`; analysis code must too. Use
`topo.valid_xb_targets()`, which returns the masked
`(xb_index, thin_id, thin_face)`. At insect geometry more than half the slots
can be placeholders.

---

## 4. Running a simulation

```python
from multifil_jax import get_params, SarcTopology, run

static, dynamic, z0, d0 = get_params('skeletal')
topo = SarcTopology.create(2, 2, static)

result = run(topo, dynamic, pCa=4.5, z_line=z0, lattice_spacing=d0,
             dt=1.0, duration_ms=1000)
print(result.summary())
```

`run(topology, params, *, pCa, z_line, lattice_spacing, dt, duration_ms, ...)`:

- `params` is required: one `DynamicParams`, or a list of them (see §5).
- The three **drivers** (`pCa`, `z_line`, `lattice_spacing`), `dt` and
  `duration_ms` are required keywords. Drivers must be finite.
- `pCa` is −log10[Ca²⁺]: 9 is relaxed, 4.5 is near-saturating.
- `z_line` is the M-line-to-Z-disc distance in nm; `dt` is in ms. Every `z_line`
  value must exceed the thick-filament tip (`max(topology.crown_offsets)`), or
  `run()` raises before compiling.
- A sim whose solve does not converge is still returned; `run()` warns and names
  it, and `solver_residual_norm` > 1 marks its steps.

Every simulation **starts relaxed**: each head is drawn from its resting
distribution at pCa 9, and so is every Tm unit. A run at pCa 4.5 is
therefore a step activation from rest, and needs no settling period.

Anything malformed — an unknown keyword, an array where a list is meant, a
`StaticParams` field passed to `run()` — raises before compilation starts.

---

## 5. Sweeps

**A list is a sweep axis, and every list is crossed with every other.**

```python
drv = dict(z_line=z0, lattice_spacing=d0, dt=1.0, duration_ms=1000)

# one axis
r = run(topo, dynamic, pCa=[9.0, 6.0, 5.5, 4.5], **drv, replicates=5)
# r.axial_force.shape == (4, 5, 1000)      (pCa, replicates, time)

# two driver axes, crossed
r = run(topo, dynamic, pCa=[9.0, 4.5], z_line=[900.0, 1000.0, 1100.0],
        lattice_spacing=d0, dt=1.0, duration_ms=1000)
# (3, 2, 1, 1000)                         (z_line, pCa, replicates, time)
```

**Parameter fields as keywords.** Any `DynamicParams` field can be passed to
`run()` by name. A scalar overrides that field for every simulation; a list makes
it a sweep axis. Either way it is applied on top of the `params` you passed, so
the rest of the preset is kept:

```python
r = run(topo, dynamic, pCa=4.5, **drv,
        thick_k=[5000.0, 7500.0, 10000.0],   # sweep axis
        thin_EA=[48000.0, 66000.0],          # second axis, crossed
        titin_rest=200.0)                    # override, no axis
# (3, 2, 1, 1000)
```

**Candidate lists.** Passing a list of `DynamicParams` as `params` makes one
`'candidates'` axis. Each candidate is a complete parameter set, so this is how
to vary several fields *together*:

```python
cands = [dynamic, dynamic.copy(xb_r12_coeff=0.3, xb_r34_coeff=0.1)]
r = run(topo, cands, pCa=4.5, **drv)
r.metadata['candidates']        # {field: [value per candidate]} for fields that differ
```

The two cardiac isoforms side by side:

```python
_, beta, z0, d0 = get_params('cardiac_beta')
_, alpha, _, _ = get_params('cardiac_alpha')
r = run(topo, [beta, alpha], pCa=[9.0, 6.0, 4.5], z_line=z0, lattice_spacing=d0,
        dt=1.0, duration_ms=1500)
```

A field keyword on a candidate list is allowed only if the candidates agree on
that field; otherwise it raises.

**Random sampling.** Rows with a range can be sampled, uniformly in log space
by default and linearly where the row says so:

```python
from multifil_jax import draw, sample_params

# one field: a plain list, so it becomes a sweep axis
r = run(topo, dynamic, pCa=4.5, **drv,
        xb_r34_coeff=draw(dynamic, 'xb_r34_coeff', 8, seed=0))

# several fields drawn jointly: a candidate list
ens = sample_params(dynamic, 64, seed=0, vary='GF')    # every ranged G or F row
ens = sample_params(dynamic, 64, seed=0, vary=['xb_r12_coeff', 'tm_J'])
r = run(topo, ens, pCa=4.5, **drv)
```

A field with no range in its class's table cannot be drawn and raises.

**Time courses.** An array of length `n_steps` passed for a driver is applied
step by step:

```python
import numpy as np
t = np.arange(1000)
pCa_trace = 9.0 - 4.5 * np.exp(-t / 100)       # a calcium transient
r = run(topo, dynamic, pCa=pCa_trace, z_line=z0, lattice_spacing=d0,
        dt=1.0, duration_ms=1000)
```

Scalars, lists and time courses can be mixed across different arguments.

**Axis order is fixed**, not the order you typed: `z_line`, `pCa`,
`lattice_spacing`, `candidates`, field keywords (in keyword order, `K_lat` and
`nu` among them), `subpopulation`, then `replicates`, then `time`. Read
`result._axis_names` rather than assuming, and select by value with `.sel()`.

---

## 6. Reading the result

`run()` returns a `SimulationResult`. Every array has shape
`(sweep axes..., replicates, time)`.

| Attribute | What it is |
|---|---|
| `result.axial_force` | force at the M-line, pN (= `result.metrics['axial_force']`) |
| `result.metrics` | every metric (§7); dict and attribute access both work |
| `result.z_line`, `result.pCa`, `result.lattice_spacing` | the driver traces applied (`lattice_spacing` is the prescribed d0; the solved value is a metric) |
| `result._axis_names`, `result.coords` | axis layout, and the values along each sweep axis |
| `result.grid_shape`, `result.dt` | shape without time; the timestep |
| `result.topology_config` | the topology's integer sizes (`n_thick`, `total_xbs`, `n_tm`, …) |
| `result.metadata` | `master_seed`, and `candidates` when a params list was passed |

| Method | Does |
|---|---|
| `.mean()`, `.std()` | collapse the replicates axis |
| `.sel(pCa=4.5)` | slice by coordinate value |
| `result[i, ...]` | integer / slice indexing per axis |
| `SimulationResult.stack(results, axis_name, axis_values)` | combine independent runs (different topologies allowed) under a new outer axis |
| `.summary()` | text summary |

Everything after `run()` is ordinary array math:

```python
F = result.axial_force[..., -300:].mean(axis=-1)       # steady state per sim
F_mean = F.mean(axis=-1)                               # over replicates
```

**Check convergence.** `result.metrics['solver_residual_norm']` is the force
imbalance divided by the solver's own tolerance: ≤ 1 means converged, in every
lattice-spacing mode. `run()` warns after the run if it exceeded 1 anywhere.
`solver_tolerance` gives the tolerance in pN; the raw `solver_residual` is not
meaningful on its own, because "small" depends on how much force the lattice
carries. `newton_iters` is typically 1–4.

**Subtract the passive baseline.** `axial_force` includes titin, which can
exceed the active force at long lengths. Active force is the force at a given
pCa minus the force at pCa 9, at the same length.

---

## 7. Metrics

Every metric is a per-timestep scalar per simulation, and every key is
independent — nothing in `result.metrics` can be computed from the other keys.
The `compute_all_metrics` docstring (`multifil_jax/metrics_fn.py`) documents each
one in detail.

| Key(s) | Meaning |
|---|---|
| `axial_force` | M-line force, pN (includes titin) |
| `solver_residual`, `solver_residual_norm`, `solver_tolerance`, `newton_iters` | solver diagnostics (§6) |
| `lattice_spacing` | the spacing actually used — the solved value under dynamic LS |
| `n_xb_drx`, `n_xb_loose`, `n_xb_tight_1`, `n_xb_tight_2`, `n_xb_free_2`, `n_xb_srx` | heads in each state (table below) |
| `force_xb_loose`, `force_xb_tight_1`, `force_xb_tight_2` | axial crossbridge force by bound state, pN |
| `n_tm_state_0` … `n_tm_state_5` | Tm units in each state (§10) |
| `frac_tm_closed_overlap`, `frac_tm_open_overlap`, `n_overlap_sites` | closed / open Tm fractions over sites a head can reach |
| `atp_expected` | expected ATP used this step — **use this for ATP** |
| `atp_consumed` | realised count of ATP-consuming detachments |
| `newly_bound` | heads that attached this step |
| `closure_detach_free`, `closure_detach_atp` | heads torn off by closing Tm: free (weak) / one ATP each (strong) |
| `n_xb_loose_closed` | weak heads sitting on closed Tm units |
| `xb_detach_free`, `xb_give_up`, `atp_net_pi_release`, `atp_net_hydrolysis` | cycle fluxes for checking the ATP books |
| `thick_displace_*`, `thin_displace_*` | mean/max/min/std node displacement, nm |
| `thick_energy_first_avg`, `titin_energy_avg`, `titin_energy_delta_avg` | stored elastic energy, pN·nm |
| `xb_work_on_filaments` | work done on the lattice by the crossbridges this step, pN·nm |

| State | Name | Meaning |
|---|---|---|
| 0 | DRX | detached, primed, available |
| 1 | Loose | weakly bound |
| 2 | Tight_1 | strongly bound, phosphate released |
| 3 | Tight_2 | strongly bound, able to release ADP |
| 4 | Free_2 | detached after ADP release, ATP bound |
| 5 | SRX | super-relaxed: parked on the backbone, cannot bind until recruited |

**ATP.** `atp_expected` is the exact expected number of ATP-consuming crossings
in the step. It counts heads that cycle more than once within one step, and adds
the strong closure tears. `atp_consumed` counts only realised endpoints and
undercounts, by about 5% on the cardiac presets and 25–28% on skeletal at
`dt = 1 ms`. `xb_give_up` is the strain-gated 2→1 reversal: it costs no ATP and
is **not** a detachment, since the head stays weakly bound.

**Overlap-zone fractions.** An all-site Tm fraction averages over sites no head
can ever reach (opposite the bare zone, beyond the thick filament tip), so it
changes whenever filament lengths change. Prefer the `_overlap` keys when
comparing geometries.

**Derived quantities:**

| To get | Compute |
|---|---|
| bound heads | `n_xb_loose + n_xb_tight_1 + n_xb_tight_2` |
| fraction of heads in a state | count / `result.topology_config['total_xbs']` |
| fraction of Tm units in a state | count / (`n_thin × n_tm` from `topology_config`) |
| available Tm in the overlap zone | `frac_tm_closed_overlap + frac_tm_open_overlap` |
| ATP from cycling alone | `atp_expected − closure_detach_atp` |
| change in first-spring energy | `np.diff(thick_energy_first_avg, axis=-1)` |
| external work at the Z-line | see below |

```python
F, z = result.axial_force, result.z_line
external_work = -0.5 * (F[..., :-1] + F[..., 1:]) * np.diff(z, axis=-1)
```

External work is zero under an isometric hold; `xb_work_on_filaments` is not,
because the heads keep cycling. `titin_energy_delta_avg` is not
`np.diff(titin_energy_avg)`: it holds the step's drivers fixed, so it isolates
filament motion when `z_line` or the spacing moves.

**Never average a per-step ratio.** `mean(w/a)` is not `sum(w)/sum(a)`; the
bias is the squared coefficient of variation of the denominator and is large at
submaximal calcium. Reduce numerator and denominator over the window (and over
replicates) first, then divide:

```python
w = result.metrics['xb_work_on_filaments'][..., -300:].sum(axis=-1)
a = result.metrics['atp_expected'][..., -300:].sum(axis=-1)
work_per_atp = w / a
```

Tension cost is the same: `sum(atp_expected) / sum(active force)` over the window.

---

## 8. Recompilation and batching

**Free to vary (same compiled kernel):** any `DynamicParams` value; driver
values; replicates; subpopulation fractions, scales
and masks.

**Recompiles:** the topology (lattice size or any `StaticParams` field,
including the solver caps); the number of timesteps (`duration_ms / dt`);
switching between fixed and dynamic lattice spacing; the *shape* of a
subpopulation (how many populations, which fields they scale, which mode).

The rule: runs that would create arrays of different shapes inside the loop
need separate kernels; runs that differ only in values share one.

**Batch buckets.** The total batch (all sweep combinations × replicates) is
padded up to the next power of two (`BATCH_BUCKETS`), so a 9-run and a 15-run
sweep share one compiled kernel. Padding simulations are discarded.

**Minibatching.** `minibatch_size="auto"` (the default) splits very large
batches into chunks that reuse one compiled kernel; pass an int to set the
chunk size or `None` to disable. It bounds GPU memory, which scales as
`chunk × n_steps × len(result.metrics) × 4 bytes × 2`. Chunks are gathered in
host RAM. See [INSTALL.md](INSTALL.md) for sizing.

---

## 9. Inside one timestep

`timestep()` (`multifil_jax/timestep.py`) runs four steps:

1. **Nearest sites** (`update_nearest_neighbors`). Each head finds its nearest
   binding candidate. The search uses the head position (crown +
   `xb_head_reach`); the stored distance is from the crown. Sites that have
   crossed the M-line are masked out. Where the opposite half's thin filaments
   reach past the M-line (the *hiding line*), sites are still found but their
   attachment rate is scaled by `1 − thin_thin_overlap_screening`.
2. **Tropomyosin** (`thin_transitions`). Each Tm unit samples its next state
   from a 6-state Markov chain (§10). Strong heads bias their unit towards
   open. A unit that leaves open anyway tears its strong heads to Free_2, one
   ATP each; a unit that enters blocked tears its weak heads to DRX, free.
3. **Crossbridges** (`thick_transitions`). Each head samples its next state
   from a 6-state chain. Rates depend on its distance to its site and the
   position of the site's Tm unit. The rates are evaluated on a grid of distance
   bins (five blocks: blocked, closed, closed-screened, open, open-screened) and
   exponentiated once, and each head gathers its own row.
4. **Mechanics** (`solve_equilibrium`). Newton–CG moves every filament node
   until net force vanishes. The iteration caps are `static.n_newton_steps` and
   `static.n_cg_steps`.

Steps 1–3 are `kinetics_step()`; step 4 is the mechanical solve. Metrics are
computed after step 4 from the states before and after, plus a trace of the
kinetics, so they describe the step that actually happened.

**The crossbridge cycle.** The force-producing path is
0 → 1 → 2 → 3 → 4 → 0, one ATP per lap. Each bound state has its own two-spring
configuration (a globular spring and an angular converter spring). Most of the
working stroke is on 1 → 2, with a smaller second swing on 2 → 3. Detachment
(3 → 4) is load-dependent and, at the shipped values, a catch bond: resisting
load slows it. SRX is a reserve outside the cycle that calcium recruits from.
Every rate law, with its form and source, is in
`multifil_jax/kernels/rate_functions.py`.

---

## 10. Tropomyosin cooperativity

Each Tm unit is in one of three **positions** — blocked, closed, open — with
or without calcium on its troponin: six states, `state = 2 × position +
has_calcium`.

| position | states | crossbridges |
|---|---|---|
| blocked | 0, 1 | nothing binds |
| closed | 2, 3 | weak attachment (`r01`) only |
| open | 4, 5 | weak attachment and the weak → strong step (`r12`) |

The position equilibria depend on calcium (`tm_Keq_unblock_*`,
`tm_Keq_swing_*`), and detailed balance then fixes the calcium affinity at
closed and open: calcium binds more tightly the further the unit is open. Every
cycle obeys detailed balance.

Force rises far more steeply with calcium than independent units allow. Units
along a strand are coupled as a **symmetric Ising chain on position**. Each
position has a weight W = −1, 0, +1 (blocked, closed, open); a unit's field is
`h = tm_J × ΣW(its two neighbours)`, and a position step s → t is multiplied by
`exp((W(t) − W(s)) × h/2)`, so every equilibrium shifts by the Boltzmann factor.
Calcium edges stay within one position and are untouched.

- `tm_J` (kT) sets the emergent steepness of the force–pCa curve.
- `xb_tm_K2` sets how strongly a **strong** head (Tight_1, Tight_2) holds its
  unit open: n strong heads divide the unit's exits from open by `(1 + K2)^n`.
  Weak heads do not lock. `jnp.inf` is a hard lock.
- A unit that leaves open tears its strong heads to Free_2 (one ATP each); a
  unit that enters blocked tears its weak heads to DRX (free).

---

## 11. Subpopulations

A fraction of the motors (or Tm units) can run modified kinetics — a
heterozygous mutation, or cMyBP-C acting only on the C-zone. A population is a
set of **multiplicative** factors on `xb_*` / `tm_*` rate fields of the `params`
you pass. Mechanics always use the base values.

```python
from multifil_jax import Subpopulation

sp = Subpopulation.mean_field(0.5, xb_r01_coeff=3.0, xb_srx_kmax=0.3)
r = run(topo, dynamic, pCa=4.5, **drv, subpopulation=sp)
```

| Constructor | Assignment | Use |
|---|---|---|
| `mean_field(fraction, **scales)` | none: every head sees the fraction-weighted blend of the rate matrices | fast, deterministic |
| `random(fraction, seed, **scales)` | each head / unit independently, with probability `fraction`, redrawn per simulation | spatial mosaicism; replicates sample arrangements |
| `c_zone(topo, c_zone_min_nm=, c_zone_max_nm=, z_line=None, **scales)` | crowns within an axial band from the M-line (the class's band is `static.c_zone_lo` / `c_zone_hi`); `tm_*` scales need `z_line` | spatially localised effects |

`fraction=0` reproduces the wild type and `fraction=1` a global scale, exactly,
in every mode — useful sanity checks. A list of subpopulations is a sweep axis.
All entries must share one mode and population count, and a swept `random` list
must share one seed. See `examples/subpopulation.py`.

---

## 12. Lattice spacing

The spacing between thick and thin filaments is a driver by default.
`run(..., lattice=...)` picks one of three modes:

| Call | Mode |
|---|---|
| `run(..., lattice_spacing=d0)` | `'fixed'` (the default): the spacing as given |
| `run(..., lattice='poisson')` | the spacing follows `z_line` as a prescribed time course, `poisson_spacing(d0, z0, z, nu, static.filament_radii_sum)` (the centre-to-centre spacing scales as `(z0/z)^nu`) |
| `run(..., lattice='dynamic')` | the spacing is solved every step from radial force balance about that Poisson reference |

`K_lat` and `nu` are `DynamicParams` fields with table values, so they are
overridden, swept or drawn like any other field:
`run(..., lattice='dynamic', K_lat=[50.0, 85.0, 200.0])`. Fixed mode reads
neither, Poisson reads `nu`, dynamic reads both.

In dynamic mode, three radial forces balance:

- the lattice's own restoring spring, `−K_lat · (d − d_ref)`;
- the radial component of every bound crossbridge, from the same two-spring law
  as the axial force; under strong activation this usually pulls filaments
  together;
- titin's radial component, which also pulls inward above its slack length.

`d` is solved jointly with the filament positions in one Newton system.
`K_lat` is per thick filament (pN/nm); `run()` scales it by the filament count,
so lattices of different sizes respond alike. The solved spacing is
`result.metrics['lattice_spacing']`. See `examples/dynamic_lattice_spacing.py`.
