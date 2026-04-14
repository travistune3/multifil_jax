# How multifil_jax Works: 

This guide is written for someone comfortable with NumPy array operations who has
not used JAX, GPU-accelerated computing, or JIT compilation before. It explains
what the codebase does, how you interact with it, what comes back from a
simulation, and why certain things are designed the way they are. Code is kept
minimal.

---

## Table of Contents

1. [What Is Being Simulated?](#1-what-is-being-simulated)
2. [Why JAX Instead of NumPy?](#2-why-jax-instead-of-numpy)
3. [The Four Things You Need to Run a Simulation](#3-the-four-things-you-need-to-run-a-simulation)
4. [Running Your First Simulation](#4-running-your-first-simulation)
5. [Understanding SimulationResult](#5-understanding-simulationresult)
6. [What Is in results.metrics?](#6-what-is-in-resultsmetrics)
7. [Post-Processing with Array Operations](#7-post-processing-with-array-operations)
8. [Running a Parameter Sweep](#8-running-a-parameter-sweep)
9. [How Cartesian Product Sweeps Work Internally](#9-how-cartesian-product-sweeps-work-internally)
10. [Batch Bucketing: Why JIT Cares About Array Sizes](#10-batch-bucketing-why-jit-cares-about-array-sizes)
11. [What Can Be Swept (vmapped) and What Cannot](#11-what-can-be-swept-vmapped-and-what-cannot)
12. [What Happens Inside Each Timestep](#12-what-happens-inside-each-timestep)
13. [Dynamic Lattice Spacing](#13-dynamic-lattice-spacing)
14. [The Tiered Architecture: State, Topology, Constants, Drivers](#14-the-tiered-architecture-state-topology-constants-drivers)
15. [Key File Reference](#15-key-file-reference)

---

## 1. What Is Being Simulated?

The codebase simulates a **half-sarcomere** — the functional unit of a skeletal
muscle fiber. A half-sarcomere consists of two interleaved sets of protein
filaments:

- **Thick filaments** (myosin): anchored at the M-line, each carrying dozens of
  molecular motors called crossbridges (XBs).
- **Thin filaments** (actin + tropomyosin): anchored at the Z-line, decorated
  with a regulatory protein chain called tropomyosin (TM).

Muscle contraction happens when crossbridges attach to actin binding sites on
the thin filament, undergo a conformational change (the power stroke), and then
detach. Calcium controls how accessible those binding sites are via the
tropomyosin regulatory system.

The simulation tracks thousands of individual molecules at every millisecond:
which crossbridges are bound, where each filament node is located, how much
force is being generated, and how calcium regulates the system. A single
simulation of 1000 milliseconds at a 4×4 lattice size involves tracking ~3700
mechanical degrees of freedom and ~2500 individual crossbridge molecular state
machines simultaneously.

---

## 2. Why JAX Instead of NumPy?

If you are used to NumPy, JAX looks nearly identical — it has `jnp.array`,
`jnp.sum`, `jnp.where`, and all the usual operations. The key difference is
what happens behind the scenes.

**JIT compilation ("Just In Time").** The first time you call `run()`, JAX
compiles the entire simulation loop into a single optimized GPU program. This
takes anywhere from 30 seconds to a few minutes. Every subsequent call with the
same configuration (same array shapes, same structure) skips compilation and
runs the pre-compiled GPU kernel directly. This is why benchmarks always
distinguish "first run (with JIT)" from "second run (cached)."

**GPU parallelism.** NumPy runs one operation at a time on the CPU. JAX can
express an entire parameter sweep — say, 225 different physical configurations
running in parallel — as a single GPU computation where all 225 sims execute
simultaneously. This is done via `jax.vmap`, which stands for "vectorizing
map." Where NumPy would require a Python `for` loop over conditions, JAX fuses
them into one kernel.

**Immutable arrays.** JAX arrays cannot be modified in place. Instead of
`array[i] = new_value`, you always create a new array. This is what makes the
`.at[i].set()` syntax appear throughout the codebase. It is unfamiliar at first
but enables JAX to reason about the full computation graph before executing it.

**Persistent compilation cache.** The codebase stores compiled GPU kernels in
`~/.cache/multifil_jax/xla/` (configured in `multifil_jax/utils/hardware.py`,
around line 192). After the first run for a given lattice size and sweep size,
future Python sessions with the same configuration load the compiled kernel in
a few seconds rather than recompiling.

---

## 3. The Four Things You Need to Run a Simulation

Before calling `run()`, you need three objects. Understanding what each one is
— and which ones can be changed without recompiling — is the most important
concept in this codebase.

### StaticParams and DynamicParams (the parameters)

`get_skeletal_params()` and `get_cardiac_params()` (`multifil_jax/core/params.py`) each return two objects:

**StaticParams** is a frozen Python dataclass containing structural configuration:
how many crowns per filament (default 52), how many actin polymers per thin
filament (default 15), the actin geometry type (vertebrate or invertebrate), and
solver settings. These values are baked into the compiled GPU kernel. Changing
any of them means JAX must recompile.

**DynamicParams** is a JAX-aware data structure containing all the actual
physics: spring stiffnesses, rate function coefficients, calcium binding
constants, and default values for pCa, z_line, and lattice spacing. Because
`DynamicParams` is a JAX PyTree, its values can be swept across — you can run
hundreds of physical parameter combinations without recompiling. There are
approximately 45 fields.

### SarcTopology (the geometry)

`SarcTopology.create()` (`multifil_jax/core/sarc_geometry.py`)
builds all the structural index maps that describe how thick and thin filaments
are arranged, which crossbridges can reach which binding sites, how many titin
connections exist, and what the rest spacings are between nodes. This process
runs once on the CPU and produces a large set of arrays.

The topology is determined by two parameters: `nrows` and `ncols`, which define
the size of the myosin lattice. A 2×2 lattice has 4 thick filaments and 8 thin
filaments. A 4×4 has 16 thick and 32 thin. An 8×8 has 64 thick and 128 thin.
Changing `nrows` or `ncols` changes the array shapes and therefore requires
recompilation. All lattice sizes run at approximately the same speed (~5 ms per
timestep on an RTX 3090) because larger lattices simply have more GPU threads
working in parallel.

Once created, the topology should be put on the GPU via `jax.device_put(topo)`.
This copies all index arrays to GPU memory once rather than transferring them
on every call.

---

## 4. Running Your First Simulation

The entry point for all simulations is `run()`, defined in `multifil_jax/simulation.py`.
Here is a minimal example of a single isometric contraction (fixed z-line
length, fixed calcium):

```python
from multifil_jax.core.params import get_skeletal_params
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.simulation import run
import jax

static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000, dt=1.0)
print(result.summary())
```

`pCa=4.5` means a calcium concentration of 10^(-4.5), which is moderate
activation. `pCa=9.0` is essentially no calcium (resting muscle). `z_line=900`
is the distance from the M-line to the Z-line in nanometers. `duration_ms=1000`
runs for one simulated second with `dt=1.0` millisecond steps, giving 1000
timesteps.

The first call to `run()` triggers JIT compilation. Subsequent calls with the
same shape arguments (same number of steps, same batch size) reuse the compiled
kernel. The `block_until_ready()` call on `result.axial_force` is sometimes used
to force the GPU to finish before measuring elapsed time, since JAX dispatches
work asynchronously.

---

## 5. Understanding SimulationResult

`run()` returns a `SimulationResult` object (`multifil_jax/simulation.py`).
This is essentially a structured container holding all simulation outputs as
arrays. Every array inside it shares the same shape convention:

```
(sweep_dim_1, sweep_dim_2, ..., replicates, time)
```

For a single simulation with no sweep and `replicates=1`:
- `result.axial_force` has shape `(1, 1000)` — 1 replicate, 1000 timesteps.
- `result.metrics['n_bound']` has the same shape.

For a pCa sweep over 5 values with 3 replicates and 1000 steps:
- `result.axial_force` has shape `(5, 3, 1000)`.
- `result.metrics['n_bound']` has shape `(5, 3, 1000)`.

The ordering of dimensions always follows the order in which sweep parameters
were provided, with replicates second-to-last and time always last. The
`result._axis_names` list names each dimension, and `result.coords` is a
dictionary mapping each axis name to its actual values.

**Key attributes you will use:**

`result.axial_force` — the M-line force trace in piconewtons. This is the
primary output of a mechanical simulation. It is a property that returns
`result.metrics['axial_force']`.

`result.metrics` — a `MetricsDict` containing ~46 quantities computed
at every timestep (described fully in the next section). `MetricsDict` supports
both dict-style access (`result.metrics['n_bound']`) and attribute access
(`result.metrics.n_bound`).

`result.metrics['solver_residual']` — a diagnostic trace showing how well the
mechanical equilibrium solver converged at each step. Values below 0.5 pN are
considered converged.

`result.metrics['newton_iters']` — the number of Newton iterations the solver
used at each step. Typically 1–2 for standard parameters.

`result.z_line`, `result.pCa`, `result.lattice_spacing` — the input traces
actually used during the simulation, useful for aligning outputs with protocol
timing.

`result.topology_config` — a plain dictionary (no JAX arrays) listing the
structural dimensions: n_thick, n_thin, n_crowns, n_sites, total_xbs, etc.

**Methods:**

`result.mean()` — averages across the replicates axis, returning a new
`SimulationResult` with the replicates dimension removed.

`result.std()` — standard deviation across replicates.

`result.sel(pCa=4.5)` — selects the slice where pCa equals 4.5, returning a
new `SimulationResult` with that sweep dimension removed.

`SimulationResult.stack(list_of_results, axis_name='lattice_size', axis_values=[2, 4, 6])`
— combines results from independent runs (possibly with different topologies)
into a single result with a new outer sweep dimension.

---

## 6. What Is in results.metrics?

Every timestep, after the mechanical state has been updated, `compute_all_metrics()`
(`multifil_jax/metrics_fn.py`) computes ~46 scalar quantities and accumulates
them into arrays. These are returned in `result.metrics` as a `MetricsDict`.
Every key in this dictionary maps to an array with the same shape as
`result.axial_force`.

**Protocol values (what was actually applied or measured each step):**
- `'axial_force'` — total M-line force (pN); also accessible as `result.axial_force`
- `'solver_residual'` — Newton solver max force imbalance (pN); below 0.5 = converged
- `'z_line'` — z-line position trace (nm)
- `'pCa'` — pCa trace
- `'lattice_spacing'` — lattice spacing trace (nm); when using dynamic LS, this is
  the emergent value solved each step, not the input

**Crossbridge state counts** (raw integer counts of crossbridges in each state):
- `'n_bound'` — total number of crossbridges bound to actin (states 2, 3, 4)
- `'n_xb_drx'` — disordered relaxed state (state 1, active but unbound)
- `'n_xb_loose'` — loosely bound / weakly attached (state 2)
- `'n_xb_tight_1'` — first strong-binding state (state 3, post power stroke)
- `'n_xb_tight_2'` — second strong-binding state (state 4, pre-detachment)
- `'n_xb_free_2'` — released state after ATP hydrolysis (state 5)
- `'n_xb_srx'` — super-relaxed / off state (state 6)

**Crossbridge state fractions** (same as above but divided by total XB count):
- `'frac_xb_bound'`, `'frac_xb_drx'`, `'frac_xb_loose'`, etc.

**Tropomyosin (TM) state counts** — TM is a four-state regulatory chain:
- `'n_tm_state_0'` through `'n_tm_state_3'` — counts in each TM state
- `'frac_tm_state_0'` through `'frac_tm_state_3'` — fractions
- `'actin_permissiveness'` — mean float 0–1 indicating how accessible actin
  binding sites are across all thin filaments

**Transition events** (events that occurred in this timestep):
- `'atp_consumed'` — stochastic count of crossbridges that completed the
  power stroke and consumed an ATP molecule this step (state 4 → state 5)
- `'newly_bound'` — count of crossbridges that newly attached to actin
  (state 1 → state 2)
- `'atp_expected_p'` — expected ATP consumption using the P-matrix method
  (a smoother, expected-value estimate rather than stochastic count)
- `'atp_expected_q'` — expected ATP consumption using Q-matrix branching
  ratio method

**Displacement statistics** — how far filament nodes are from their rest positions:
- `'thick_displace_mean'`, `'thick_displace_max'`, `'thick_displace_min'`,
  `'thick_displace_std'` — thick filament displacement statistics (nm)
- `'thin_displace_mean'`, `'thin_displace_max'`, `'thin_displace_min'`,
  `'thin_displace_std'` — thin filament displacement statistics (nm)

**Energy metrics** — elastic stored energy in the system:
- `'thick_energy_first_avg'` — mean elastic energy stored in the first crown
  spring of all thick filaments (pN·nm)
- `'thick_energy_first_delta_avg'` — change in that energy from the previous
  timestep
- `'titin_energy_avg'` — mean titin energy across all connections
- `'titin_energy_delta_avg'` — change in titin energy

**Work metrics** — mechanical work done:
- `'work_thick'` — work done by M-line force over thick filament displacement
  this timestep (pN·nm)
- `'work_thick_mean'` — same, normalized by number of thick filament nodes
- `'work_per_atp'` — ratio of work done to ATP consumed (thermodynamic
  efficiency indicator)

**Solver diagnostics:**
- `'newton_iters'` — number of Newton iterations used by the equilibrium solver
  this step. Typically 1–2 for standard parameters; useful for diagnosing
  convergence at extreme parameter values.

---

## 7. Post-Processing with Array Operations

Because every metric is a NumPy-compatible array with the same shape as
`axial_force`, you can do all of your analysis using standard array operations
after the simulation completes — no Python loops needed.

**Steady-state averaging** — take the mean over the last N timesteps to get
the steady-state value:

```python
# Mean of last 20 timesteps across all pCa values (axis 0) and replicates (axis 1)
steady_force = result.axial_force[..., -20:].mean(axis=-1)
# steady_force shape: (n_pCa, n_replicates)

# Then average over replicates
mean_force = steady_force.mean(axis=-1)  # shape: (n_pCa,)
```

**Cross-metric calculations** — create derived metrics by combining arrays
element-wise. For example, efficiency is work per unit ATP:

```python
# Already computed, but if you wanted it differently:
atp = result.metrics['atp_consumed']         # shape: (..., time)
force = result.axial_force                    # shape: (..., time)

# Cumulative ATP over entire simulation
total_atp = atp.sum(axis=-1)                 # shape: (...,) collapses time

# Cumulative work (rough integral: force × z_line_velocity × dt)
# Or use the pre-computed work_thick
total_work = result.metrics['work_thick'].sum(axis=-1)
efficiency = total_work / (total_atp + 1e-9) # avoid divide-by-zero
```

**Adding custom metrics to the result dictionary** — since `result.metrics`
is a dict subclass, you can add new keys:

```python
result.metrics['efficiency'] = (
    result.metrics['work_thick'] / (result.metrics['atp_expected_p'] + 1e-9)
)
```

**Comparing bound fraction over time across pCa conditions** — for a sweep
over pCa values:

```python
# result.axial_force shape: (n_pCa, replicates, time)
mean_bound = result.metrics['frac_xb_bound'].mean(axis=1)  # average replicates
# mean_bound shape: (n_pCa, time)

# Steady-state value for each pCa condition
ss_bound = mean_bound[..., -20:].mean(axis=-1)  # shape: (n_pCa,)
pCa_values = result.coords['pCa']               # Python list
```

**Selecting a subset of conditions with `.sel()`** — returns a new
`SimulationResult` with only that condition:

```python
low_ca = result.sel(pCa=4.5)   # shape drops one dimension
high_ca = result.sel(pCa=9.0)
ratio = low_ca.axial_force / (high_ca.axial_force + 1e-9)
```

The key principle is: all heavy computation runs on the GPU during `run()`,
and everything you do afterward (averaging, ratios, statistics) is just
NumPy-style array math on the returned results. The arrays can be passed
to `numpy.array()` if you need them as ordinary NumPy arrays for matplotlib
or scipy.

---

## 8. Running a Parameter Sweep

The most powerful feature of the codebase is the ability to run hundreds of
parameter combinations simultaneously on the GPU. This is done by passing a
**list** instead of a scalar to any parameter of `run()`.

**Single axis sweep** — passing a list to `pCa` runs one simulation per
pCa value, all in parallel:

```python
result = run(topo, pCa=[9.0, 7.0, 6.0, 5.0, 4.5, 4.0], z_line=900.0,
             duration_ms=1000, replicates=5)
# result.axial_force shape: (6, 5, 1000)
# axis 0: pCa values, axis 1: replicates, axis 2: time
```

**Multi-axis Cartesian product sweep** — passing lists to multiple parameters
creates a sweep grid. Passing lists of length 3 for both `pCa` and `z_line`
runs 3 × 3 = 9 combinations:

```python
result = run(topo,
             pCa=[9.0, 6.0, 4.5],
             z_line=[850.0, 900.0, 950.0],
             duration_ms=1000)
# result.axial_force shape: (3, 3, 1, 1000)
# (n_pCa, n_z_line, replicates, time)
```

**Physical parameter sweeps** — sweeping over values in `DynamicParams` works
the same way, using the `dynamic_params` argument as a dict:

```python
thick_sweep = [1000, 1500, 2000, 2500, 3000]   # pN/nm
thin_sweep  = [800,  1200, 1600]                # pN/nm

result = run(topo, pCa=4.5, z_line=900.0,
             dynamic_params={'thick_k': thick_sweep, 'thin_k': thin_sweep},
             duration_ms=1000)
# result.axial_force shape: (5, 3, 1, 1000)
# (n_thick_k, n_thin_k, replicates, time)
```

You can also combine protocol sweeps with physical parameter sweeps:

```python
result = run(topo,
             pCa=[9.0, 4.5],
             dynamic_params={'thick_k': [1000, 2000, 3000]},
             replicates=3,
             duration_ms=500)
# result.axial_force shape: (2, 3, 3, 500)
# (n_pCa, n_thick_k, replicates, time)
```

**Time-varying inputs** — passing a NumPy array of length `n_steps` applies
a different value at every timestep (a time trace rather than a scalar):

```python
import numpy as np
t = np.arange(1000)
# Calcium transient: starts high, decays exponentially
pCa_trace = 4.0 + (9.0 - 4.0) * (1 - np.exp(-t / 100))

result = run(topo, pCa=pCa_trace, z_line=900.0, duration_ms=1000)
```

Arrays and lists cannot be mixed for the same parameter (one is a sweep axis,
the other is a time trace). However, you can mix scalar, sweep list, and time
trace across different parameters in the same call.

---

## 9. How Cartesian Product Sweeps Work Internally

This section explains the internal mechanism, which is important for
understanding what can and cannot be parallelized.

When `run()` receives lists for multiple parameters, it builds a Cartesian
product grid using `jnp.meshgrid`. If you pass 5 values for `thick_k` and 3
values for `thin_k`, `meshgrid` produces a 5×3 grid of all combinations. The
grid is then "flattened" to a single batch dimension of size 15. Replicates are
tiled on top, giving a total batch of `15 × replicates` independent simulations.

Each simulation in the batch gets its own entry in every input array. The batch
dimension is arranged so that element 0 has `(thick_k[0], thin_k[0])`, element
1 has `(thick_k[0], thin_k[1])`, element 2 has `(thick_k[0], thin_k[2])`,
element 3 has `(thick_k[1], thin_k[0])`, and so on.

These 15 (or 45, with replicates=3) simulations are then run using
`jax.vmap`, a JAX function that takes a function designed for a single
simulation and transforms it into one that runs across all batch elements
simultaneously. Think of vmap as saying: "take this function that operates on
one simulation, and run it on all rows of my batch array at once" — similar
to how NumPy broadcasting eliminates explicit loops.

The vmapped function is called `run_single_sim` and wraps a `jax.lax.scan`
— a JAX construct for sequential operations (the time loop). The architecture
is therefore:

```
vmap over batch dimension (all parameter combinations in parallel on GPU)
  └── lax.scan over time dimension (sequential timesteps, one after another)
        └── timestep() called at each step
```

vmap handles the spatial parallelism (many simulations at once), and scan
handles the temporal iteration (one timestep follows the next, carrying state
forward). XLA fuses these together into a single GPU kernel, minimizing the
number of separate GPU operations.

After the batch completes, `run()` reshapes the flat batch dimension back into
the original grid shape — from `(15, 1000)` back to `(5, 3, 1000)` for the
5×3 sweep — and returns the `SimulationResult` with the correct axis labels.

---

## 10. Batch Bucketing: Why JIT Cares About Array Sizes

JAX's JIT compilation produces a GPU kernel that is specific to the **exact
shape** of every input array. If you run a sweep of 15 combinations and then
another sweep of 9 combinations, JAX would normally compile a new kernel for
each, even though the computation is identical — only the number of parallel
threads differs.

Compilation takes 30–400 seconds. Running two sweeps of slightly different
sizes should not require two full compilations.

To solve this, `run()` rounds up every batch size to the nearest value in a
fixed list of "buckets" (`multifil_jax/simulation.py`):

```
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
```

A sweep of 15 simulations is padded to 16 (the next bucket). A sweep of 225
is padded to 256. A sweep of 1125 is padded to 2048.

Padding means adding "dummy" simulations with the same parameters as the last
real simulation. The dummy results are discarded before returning. The cost is
that you may run a handful of redundant simulations (at most you run twice as
many as needed, since the bucket sizes double). The benefit is that a 15-sim
sweep and a 9-sim sweep both compile as 16, so the second call uses the cached
kernel from the first.

This bucketing is handled by `get_bucket_size()` and is completely transparent
— you pass the exact number of conditions you want, and the padding happens
internally.

The persistent XLA cache (`~/.cache/multifil_jax/xla/`) stores compiled kernels
across Python sessions. The first time you run a 256-bucket 2×2 lattice sweep
might take 3 minutes to compile. The next session, it loads in a few seconds.
Different configurations (lattice sizes, `StaticParams` values, batch sizes)
each get their own cache entry automatically — you do not need to clear the
cache when switching between them. Clear it only after upgrading JAX, since
compiled kernels from an older version are incompatible:
`rm -rf ~/.cache/multifil_jax/xla/`

**Minibatching.** For very large sweeps (≥16384 runs), the `minibatch_size`
parameter lets `run()` split the padded batch into smaller chunks. Each chunk
calls the same compiled kernel, so there is no recompilation. The default
`"auto"` setting chunks batches of 16384+ into groups of 4096, which
benchmarks show is ~2% faster due to better L2 cache utilization. The primary
reason to use minibatching is to bound peak GPU VRAM on memory-constrained GPUs
(e.g. 8 GB): peak VRAM ≈ minibatch_size × n_steps × 46 × 4 bytes × 2.

---

## 11. What Can Be Swept (vmapped) and What Cannot

Understanding what triggers recompilation and what does not is essential for
efficient use of this codebase.

**Things you CAN change without recompilation (sweep freely):**

Any field in `DynamicParams` can be swept. These are JAX arrays that vmap
treats as regular data. You can run 1000 different spring stiffness values in
one call and JAX sees it as a single batch computation. The list of sweepable
physics parameters includes all spring constants (`thick_k`, `thin_k`,
`titin_b`, `titin_a`), all rate function parameters, calcium binding
constants, and the default pCa/z_line/lattice_spacing.

`K_lat` and `nu` (lattice stiffness and Poisson exponent) can also be swept
as lists. They join the Cartesian product grid alongside other sweep parameters.

The number of **replicates** can change freely — it just changes the batch
size, which is handled by bucket rounding.

The **simulation duration** (`duration_ms`) and **timestep** (`dt`) control the
number of scan steps. In principle these affect the compiled kernel, but in
practice `n_steps` is a static argument, meaning different durations require
different compiled kernels. Bucket rounding does NOT apply to timestep count.

**Things you CANNOT change without recompilation:**

The **topology** (`SarcTopology`) encodes array sizes (number of crowns, number
of sites, number of titin connections, etc.). JAX compiles based on these
shapes. A 2×2 lattice and a 4×4 lattice require separate compiled kernels.
This is why `SarcTopology` is constructed once and reused across many `run()`
calls.

`StaticParams` fields (n_crowns, actin_geometry, n_newton_steps, n_cg_steps)
are embedded into the compiled kernel at trace time. Changing any of these
requires creating a new topology and recompiling.

Switching between **fixed LS and dynamic LS** modes requires recompilation,
because `is_dynamic_ls` is a JIT static argument. However, different `K_lat`
values within dynamic LS mode share the same compiled kernel (K_lat is traced,
not static).

**The fundamental rule:** if two runs would produce arrays of different shapes
anywhere inside the simulation loop, they require separate compiled kernels.
If they produce the same shapes with different values, they share a kernel.

---

## 12. What Happens Inside Each Timestep

Each millisecond of simulated time, `timestep()` (`multifil_jax/timestep.py`)
is called once. It takes the current mechanical state, the physical parameters,
the time-varying driver values, and a random number key, and returns a 5-tuple:
`(new_state, new_rng_key, solver_residual, new_lattice_spacing, newton_iterations)`.
Here is what happens, in order:

**Step 0: Resolve drivers.** The simulation has two ways to specify values like
pCa, z_line, and lattice spacing: as static defaults in `DynamicParams` (Tier 2)
or as per-step values in `Drivers` (Tier 3). `resolve_value()` (`multifil_jax/core/state.py`)
merges these: if the driver value is a valid number (not NaN), use
it; otherwise use the constant default. This allows a parameter to be constant
for most of a simulation but overridden at specific timesteps without branching.
A merged `resolved_constants` object is built via `DynamicParams.with_drivers()`
(`multifil_jax/core/params.py`) and used for all subsequent steps.

**Step 1: Calculate thin filament internal forces.** Before updating cooperativity,
the codebase computes the internal tension forces along each thin filament spring
chain using `calculate_thin_forces_for_cooperativity()` (`multifil_jax/kernels/forces.py`).
This gives the raw spring forces at each actin site, which determine how far
the cooperative activation signal can spread.

**Step 2: Update cooperativity.** `update_cooperativity()` (`multifil_jax/kernels/cooperativity.py`)
implements the mechanical feedback mechanism for calcium regulation. Tropomyosin
cooperativity means that when one region of the thin filament is active (calcium
bound), it mechanically biases neighboring regions toward activation. The "span"
of this influence depends on the local filament tension: higher tension → longer
cooperative span. This step updates which actin sites are cooperatively
activated and their permissiveness values (how likely they are to accept a
crossbridge).

**Step 3: Update nearest binding sites.** For every crossbridge, `update_nearest_neighbors()`
(`multifil_jax/kernels/geometry.py`) computes the exact axial and radial
distance to the nearest available actin binding site. These distances are the
geometric inputs to the rate functions that determine how likely a crossbridge
is to bind or detach. This is recomputed every step because filament positions
change after each equilibrium solve.

**Step 4: Thin filament (tropomyosin) transitions.** `thin_transitions()` (`multifil_jax/kernels/transitions.py`)
applies a four-state stochastic Markov model to the tropomyosin chain on each
thin filament. The four states represent different positions of the tropomyosin
strand relative to actin, ranging from fully blocking crossbridge access (state 0)
to fully open (state 3). The transition rate matrix Q is computed based on
current calcium concentration and cooperativity, converted to a transition
probability matrix P via a matrix exponential, and then used to draw stochastic
transitions for each site. The matrix exponential is computed using a
Padé approximation with scaling-and-squaring via `expm_pade6_batch()`.

**Step 5: Thick filament (crossbridge) transitions.** `thick_transitions()` (`multifil_jax/kernels/transitions.py`)
applies a six-state Markov model to every individual crossbridge. The six states
are: super-relaxed off (state 6), disordered relaxed (state 1), weakly bound
(state 2), and three strongly-bound states (3, 4, 5) representing the power
stroke and ATP hydrolysis cycle. Transition rates depend on the exact axial and
radial distances computed in Step 3, as well as the current permissiveness from
Step 2. Each crossbridge gets its own transition matrix computed from its exact
geometry — no approximations or binning. The shared helper `compute_xb_transition_matrices()`
returns one (6×6) Q and P matrix per crossbridge.

**Step 6: Solve mechanical equilibrium.** After crossbridges have attached or
detached, the filament network is no longer at mechanical equilibrium — the
force balance has changed. `solve_equilibrium()` (`multifil_jax/kernels/solver.py`)
runs a Newton-CG iterative solver to find new filament node positions that
satisfy force balance everywhere. The Newton loop uses `jax.lax.while_loop`
and exits as soon as the maximum force imbalance falls below a tolerance
threshold (or after a maximum number of iterations). A tridiagonal preconditioner
factored before the scan loop (via `build_prefactored_preconditioner()`,
`multifil_jax/kernels/solver.py`) accelerates the inner conjugate-gradient
solve. This step is by far the most computationally intensive.

Steps 0–5 are encapsulated in `kinetics_step()`, which returns the post-kinetics
state and `resolved_constants`. This separation exists to support future
finite-element coupling: run kinetics across all coupled sarcomeres independently,
then perform a coupled mechanical equilibration.

After all steps complete, the new state, solver residual, emergent lattice
spacing, and iteration count are returned. The scan loop in `run_single_sim`
carries the state forward to the next timestep. Immediately after `timestep()`
returns, `compute_all_metrics()` is called inside the scan body, comparing the
state before and after the step to produce all ~46 scalar metrics. These are
accumulated as arrays across time and returned as `result.metrics`.

---

## 13. Dynamic Lattice Spacing

By default, the lattice spacing (the distance from a thick filament to its
neighboring thin filaments, typically ~14 nm) is either held constant or
prescribed externally via a Poisson ratio relation to z-line length. The
**dynamic lattice spacing** feature makes this distance an emergent quantity
solved self-consistently from radial force balance at each timestep.

### The physics

Three forces act in the radial (cross-filament) direction:

1. **Lattice stiffness** (`K_lat`): a restoring spring that resists deviation
   from the Poisson-scaled reference spacing. `F_lat = -K_lat * (d - d_ref)`.
   This represents the combined stiffness of the Z-disc, M-line, and surrounding
   cytoskeletal structures.

2. **Crossbridge radial force**: every bound crossbridge has a two-spring model
   (globular + converter springs) whose geometry depends on the lattice spacing.
   When the total XB length exceeds the rest length, XBs pull filaments together
   (compressive). When shorter than rest, they push apart (expansive). During
   full activation (many strong-state XBs), the net XB force is typically
   compressive.

3. **Titin radial force**: titin runs diagonally from thick filament to Z-disc.
   The radial component of its tension always pulls thick toward thin
   (compressive).

At equilibrium: `F_lat + F_xb_radial + F_titin_radial = 0`, and the emergent
`d` satisfies this balance.

### How the solver works

Rather than iterating between axial and radial solves, `d` is appended as one
extra degree of freedom to the existing Newton-CG position vector:

```
augmented positions:  [thick_crown_1, ..., thin_site_N, d]    (n+1 elements)
augmented residual:   [f_axial_1,     ..., f_axial_N,  f_rad]  (n+1 elements)
```

JAX's automatic differentiation computes the full Jacobian-vector products for
this augmented system, including all cross-coupling terms (how changing `d`
affects axial forces, and how changing axial positions affects the radial
residual). The preconditioner is block-diagonal: Thomas tridiagonal for the
axial block, and the exact Jacobian diagonal inverse for the d block.

### Three modes in `run()`

| Call | Mode | What happens |
|------|------|-------------|
| `run(topo, pCa=4.5)` | Fixed LS | `lattice_spacing` held constant at 14.0 nm |
| `run(topo, pCa=4.5, nu=0.5)` | Poisson LS | `ls = d0*(z0/z)^0.5` pre-computed as trace |
| `run(topo, pCa=4.5, K_lat=5.0, nu=0.5)` | Dynamic LS | `d` solved from radial force balance |

`K_lat` is specified as per-filament stiffness (pN/nm per thick filament).
`run()` internally multiplies by `n_thick` so the lattice spacing deviation is
independent of lattice size. This means a 2×2 and an 8×8 lattice with the same
`K_lat` show the same mean deviation from `d_ref`.

### What to expect

At `K_lat=5.0 pN/nm` and `pCa=4.5` (full activation), the lattice typically
compresses by ~6 nm from the 14 nm reference, settling around ~8 nm. This is
driven by strong-state crossbridges pulling filaments together. The magnitude
of compression depends on `K_lat`: stiffer lattice → smaller deviation.

`result.metrics['lattice_spacing']` contains the emergent lattice spacing at
each timestep, so you can track how the lattice responds to activation, length
changes, or stiffness sweeps.

---

## 14. The Tiered Architecture: State, Topology, Constants, Drivers

The codebase separates simulation data into four tiers to control what can
change, when, and at what cost. Understanding this is helpful when you want
to customize the simulation.

**Tier 0 — State** (`multifil_jax/core/state.py`): The pure mechanical
state of the system at a single point in time. Contains thick filament node
positions, thin filament node positions, crossbridge states, bound-to
indices, and tropomyosin states. The state carries no physics parameters and
no geometry information — it is purely the current configuration of the system.
State is created via `realize_state()` at the start of a simulation and then
updated immutably at each step via `._replace()`.

**Tier 1 — Topology** (`multifil_jax/core/sarc_geometry.py`): The structural
connectivity of the sarcomere. Contains pre-computed index maps: which thick
filament faces which thin filament, what the rest spacings between crowns are,
how titin connects thick to thin, etc. This never changes during a simulation.
It is the only tier that requires recompilation when changed.

**Tier 2 — Constants / DynamicParams** (`multifil_jax/core/params.py`):
All physical parameters — spring stiffnesses, rate coefficients, energy
parameters, calcium sensitivity constants. Also contains the "default" values
of pCa, z_line, and lattice_spacing that are used when no per-step driver is
provided. These can be swept freely across the batch dimension. The alias
`Constants = DynamicParams` is defined in `params.py`.

**Tier 3 — Drivers** (`multifil_jax/core/state.py`, the `Drivers` NamedTuple):
Per-timestep overrides for pCa, z_line, and lattice_spacing. At each step in
the scan, the current values from the trace arrays are packaged into a `Drivers`
object and passed to `timestep()`. The `resolve_value()` function merges Tier 3
into Tier 2: if the driver is not NaN, use it; otherwise fall back to the
constant. This allows a driver to be "off" (NaN) for most of the simulation
and "active" only during specific steps.

---

## 15. Key File Reference

| File | What It Contains |
|------|-----------------|
| `multifil_jax/simulation.py` | `run()` — the main entry point for all simulations |
| `multifil_jax/simulation.py` | `SimulationResult` — the result container |
| `multifil_jax/simulation.py` | `BATCH_BUCKETS`, `get_bucket_size()`, `_run_sim_kernel()` |
| `multifil_jax/timestep.py` | `kinetics_step()` — stochastic phase (steps 0–5) |
| `multifil_jax/timestep.py` | `timestep()` — full step orchestrator (kinetics + solve) |
| `multifil_jax/metrics_fn.py` | `compute_all_metrics()` — ~46-metric MetricsDict |
| `multifil_jax/core/state.py` | `State`, `realize_state()`, `Drivers`, `resolve_value()`, `MetricsDict` |
| `multifil_jax/core/params.py` | `StaticParams`, `DynamicParams`, `get_skeletal_params()`, `get_cardiac_params()` |
| `multifil_jax/core/sarc_geometry.py` | `SarcTopology.create()` — topology builder |
| `multifil_jax/kernels/cooperativity.py` | `update_cooperativity()` — TM cooperative activation |
| `multifil_jax/kernels/geometry.py` | `update_nearest_neighbors()` — XB-to-BS distances |
| `multifil_jax/kernels/transitions.py` | `thin_transitions()`, `thick_transitions()`, `compute_xb_transition_matrices()` |
| `multifil_jax/kernels/forces.py` | `axial_force_at_mline()`, `compute_forces_vectorized()`, `_xb_radial_force_total()`, `_titin_radial_force_total()` |
| `multifil_jax/kernels/solver.py` | `solve_equilibrium()` (unified fixed/dynamic LS), Thomas algorithm |
| `multifil_jax/kernels/rate_functions.py` | Crossbridge rate functions (geometry-dependent) |
| `multifil_jax/utils/hardware.py` | GPU detection, XLA cache configuration |
| `examples/quickstart.py` | Worked examples: isometric, sweeps, transients, structural stack |
| `examples/dynamic_lattice_spacing.py` | Dynamic LS demo: isometric, force comparison, K_lat sweep, length ramp |
| `examples/benchmarks/benchmark_minibatch.py` | Minibatch size benchmark CLI |
| `examples/benchmarks/benchmark_dynamic_ls.py` | Dynamic LS performance and lattice scaling benchmark |
| `tests/` | Test suite |
