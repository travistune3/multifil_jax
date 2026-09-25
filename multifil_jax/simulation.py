"""
The simulation engine and the public run() entry point.

ARCHITECTURE: VMAP OUTSIDE, SCAN INSIDE
---------------------------------------
The single most important structural decision in this codebase:

    run() -> vmap(run_single_sim) -> lax.scan(timestep)
              over the batch axis      over the time axis

Batch outside, time inside. The batch axis is embarrassingly parallel — every
simulation is independent — while the time axis is inherently sequential, since
each step depends on the last. Putting vmap outermost lets XLA fuse the entire
computation into one GPU kernel in which all batch elements advance in lockstep.
The alternative, looping over simulations and scanning inside each, would launch
one kernel per simulation and leave the GPU almost idle.

The practical consequence is that a sweep of hundreds of conditions costs barely
more wall-clock time than a single simulation, up to the point where the GPU
saturates. Sweeping is close to free; running many separate calls is not.

SWEEPS AS INTEGER LOOKUPS
-------------------------
Any input can be a scalar, a list, or a time trace:

    scalar          the same value for every simulation and every step
    list            a sweep axis, Cartesian-producted with every other list
    array           a per-timestep trace

Internally each sweepable input keeps a small table of its distinct values, and
the sweep grid is a Cartesian product of plain integer row numbers. Each
simulation looks up its row in each table. That one mechanism covers scalar
sweeps, trace sweeps, lattice-stiffness sweeps, parameter-field sweeps and
candidate lists uniformly, instead of a special case for each.

Result arrays come back shaped (sweep_1, ..., sweep_N, replicates, time). Sweep
axes appear in a FIXED order — z_line, pCa, lattice_spacing, K_lat, nu, then
dynamic_params fields, then subpopulation — NOT the order the caller passed
them. Select with .sel(name=value) rather than by axis position.

BATCH BUCKETING
---------------
XLA compiles for exact array shapes, so a 225-simulation sweep and a
256-simulation sweep would otherwise be two separate compilations of identical
code. Batch sizes are therefore padded up to the next power of two before
dispatch and trimmed afterwards. Compilation is expensive; running a few extra
padded simulations is not.

WHAT TRIGGERS RECOMPILATION
---------------------------
Free: any DynamicParams value, the number of replicates, K_lat and nu, and the
number of sweep points within a bucket.

Not free: the topology (it defines every array shape), StaticParams solver
settings, the duration, switching between fixed and dynamic lattice spacing, and
the SHAPE of a subpopulation configuration — though not its values.

Usage:
    from multifil_jax.simulation import run
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.params import get_skeletal_params

    static, dynamic, z0, d0 = get_skeletal_params()
    topo = SarcTopology.create(nrows=2, ncols=2, static_params=static,
                               dynamic_params=dynamic)

    result = run(topo, pCa=4.5, z_line=z0, lattice_spacing=d0, duration_ms=1000,
                 dynamic_params=dynamic, static_params=static)
    print(result.summary())
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Dict, Tuple, List, Optional, Union, NamedTuple

from multifil_jax.core.params import (
    StaticParams, DynamicParams, get_skeletal_params, DYNAMIC_FIELDS,
    poisson_spacing
)
from multifil_jax.core.state import realize_state, State, Drivers, MetricsDict, build_preconditioner_params, thin_segment_k
from multifil_jax.kernels.solver import build_prefactored_preconditioner
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.kernels.geometry import update_nearest_neighbors
from multifil_jax.kernels.solver import solve_equilibrium
from multifil_jax.kernels.transitions import xb_rest_states
from multifil_jax.kernels.forces import axial_force_at_mline
from multifil_jax.timestep import timestep
from multifil_jax.metrics_fn import compute_all_metrics
from multifil_jax.core.subpopulation import Subpopulation, generate_random_masks_batch


# =============================================================================
# SIMULATION RESULT
# =============================================================================

class SimulationResult:
    """Result container from run() with visualization and grid support.

    Consolidates all simulation outputs including:
    - Force and metrics time traces (force lives in metrics['axial_force'])
    - Input replay (z_line, pCa, lattice_spacing traces used)
    - Grid metadata for parameter sweeps (coords, slicing, mean/std)

    Data Cube Convention:
        Shape = (Sweep_1, Sweep_2, ..., Replicates, Time)
        - Sweep dimensions only appear if that input was a list
        - Replicates axis always present for consistency (even if replicates=1)

    Attributes:
        metrics: MetricsDict of metric arrays from metrics_fn (includes
                 'axial_force' and 'solver_residual' keys)
        rng_key: Final RNG key state for continuation
        z_line: (..., replicates, n_steps) Z-line position trace
        pCa: (..., replicates, n_steps) pCa trace
        lattice_spacing: (..., replicates, n_steps) lattice spacing trace
        metadata: Dict with the master seed
        dt: Timestep in milliseconds
        name: Simulation/experiment name
        _axis_names: List: ['pCa', 'thick_k', 'replicates', 'time']
        coords: Dict: {'pCa': [...], 'thick_k': [...], ...}
        topology_config: the topology's structural integers (+ K_lat / nu in
            dynamic-LS mode)

    Every derived result (mean, std, indexing, sel, stack) is built by _map,
    which applies ONE function to every metric and the three driver traces and
    carries everything else across, so no field can be dropped by one of them.
    """

    __slots__ = (
        'metrics', 'rng_key',
        'z_line', 'pCa', 'lattice_spacing',
        'metadata', 'dt', 'name',
        '_axis_names', 'coords',
        'topology_config',
    )

    def __init__(
        self,
        metrics: 'MetricsDict',
        rng_key: jnp.ndarray,
        z_line: jnp.ndarray,
        pCa: jnp.ndarray,
        lattice_spacing: jnp.ndarray,
        metadata: Optional[Dict] = None,
        dt: float = 1.0,
        name: str = "",
        axis_names: List[str] = None,
        coords: Dict[str, List] = None,
        topology_config: Optional[Dict] = None,
    ):
        self.metrics = metrics if isinstance(metrics, MetricsDict) else MetricsDict(metrics)
        self.rng_key = rng_key
        self.z_line = z_line
        self.pCa = pCa
        self.lattice_spacing = lattice_spacing
        self.metadata = metadata if metadata is not None else {}
        self.dt = float(dt)
        self.name = str(name)
        self._axis_names = axis_names if axis_names is not None else []
        self.coords = coords if coords is not None else {}
        self.topology_config = topology_config if topology_config is not None else {}

    _TRACES = ('z_line', 'pCa', 'lattice_spacing')

    def _with(self, metrics, traces, axis_names, coords, suffix='') -> 'SimulationResult':
        """New result with these arrays and axes; every other field carried over."""
        return SimulationResult(
            metrics=MetricsDict(metrics), rng_key=self.rng_key, **traces,
            metadata=self.metadata, dt=self.dt, name=self.name + suffix,
            axis_names=axis_names, coords=coords, topology_config=self.topology_config,
        )

    def _map(self, fn, axis_names, coords, suffix: str = '') -> 'SimulationResult':
        """New result with fn applied to every metric and driver trace."""
        return self._with({k: fn(v) for k, v in self.metrics.items()},
                          {t: fn(getattr(self, t)) for t in self._TRACES},
                          axis_names, coords, suffix)

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape without the time axis."""
        return tuple(self.metrics['axial_force'].shape[:-1])

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the simulation."""
        return self.metrics['axial_force'].shape[-1]

    @property
    def replicate_axis(self) -> Optional[int]:
        """Index of replicate axis, derived from axis_names."""
        if 'replicates' in self._axis_names:
            return self._axis_names.index('replicates')
        return None

    @property
    def time(self) -> jnp.ndarray:
        """Time array in milliseconds."""
        return jnp.arange(self.n_steps) * self.dt

    @property
    def axial_force(self) -> jnp.ndarray:
        """Axial force at the M-line (pN). Equivalent to metrics['axial_force']."""
        return self.metrics['axial_force']

    @property
    def mean_force(self) -> float:
        """Mean axial force over simulation."""
        return float(jnp.mean(self.metrics['axial_force']))

    @property
    def steady_state_force(self) -> float:
        """Mean force over last 20% of simulation."""
        n_avg = max(1, self.n_steps // 5)
        return float(jnp.mean(self.metrics['axial_force'][..., -n_avg:]))

    def _reduce_replicates(self, reduce_fn, suffix: str) -> 'SimulationResult':
        """Collapse the replicate axis (axis -2) with reduce_fn.

        The drivers are identical across replicates by construction, so
        reduce_fn is applied to them too: their mean is the trace and their
        std is 0.
        """
        if self.replicate_axis is None:
            raise ValueError("No replicate axis to reduce.")
        return self._map(
            lambda v: reduce_fn(v, axis=-2),
            [n for n in self._axis_names if n != 'replicates'],
            {k: v for k, v in self.coords.items() if k != 'replicates'},
            suffix)

    def mean(self) -> 'SimulationResult':
        """Mean across the replicate axis (axis -2)."""
        return self._reduce_replicates(jnp.mean, "_mean")

    def std(self) -> 'SimulationResult':
        """Standard deviation across the replicate axis (axis -2)."""
        return self._reduce_replicates(jnp.std, "_std")

    def __getitem__(self, key) -> 'SimulationResult':
        """Slice all tensors identically, return new SimulationResult.

        Accepts an int, a slice, or a tuple of them, applied to the leading
        axes. An int drops its axis from `_axis_names`/`coords`; a slice
        subsets that axis's coords. Anything else (a mask, an array, Ellipsis,
        None) raises: the axis bookkeeping cannot follow it.
        """
        key_tuple = key if isinstance(key, tuple) else (key,)
        if len(key_tuple) > len(self._axis_names):
            raise IndexError(f"{len(key_tuple)} indices for axes {self._axis_names}")
        axis_names = list(self._axis_names)
        coords = dict(self.coords)
        for name, k in zip(self._axis_names, key_tuple):
            if isinstance(k, (int, np.integer)):
                axis_names.remove(name)
                coords.pop(name, None)
            elif isinstance(k, slice):
                if name in coords:
                    coords[name] = list(coords[name])[k]
            else:
                raise TypeError(
                    f"SimulationResult index must be int or slice per axis, got {type(k).__name__}")
        return self._map(lambda v: v[key], axis_names, coords)

    def summary(self) -> str:
        """Return text summary of simulation results."""
        lines = [
            f"SimulationResult: {self.name}",
            f"  Shape: {self.axial_force.shape}",
            f"  Duration: {self.n_steps * self.dt:.1f} ms ({self.n_steps} steps @ dt={self.dt}ms)",
            f"  Mean force: {self.mean_force:.2f} pN",
            f"  Steady-state force: {self.steady_state_force:.2f} pN",
            f"  Metrics: {list(self.metrics.keys()) if self.metrics else 'none'}",
            f"  Max solver residual: {float(jnp.max(self.metrics['solver_residual'])):.4f} pN",
        ]
        if self._axis_names:
            lines.append(f"  Grid axes: {self._axis_names}")
        if self.coords:
            lines.append(f"  Coords: {list(self.coords.keys())}")
        return '\n'.join(lines)

    def sel(self, **kwargs) -> 'SimulationResult':
        """Coordinate-based slicing (e.g., result.sel(pCa=5.0)).

        Args:
            **kwargs: axis_name=value pairs. Value must exist in coords.

        Returns:
            Sliced SimulationResult
        """
        idx = [slice(None)] * len(self._axis_names)
        for axis_name, value in kwargs.items():
            if axis_name not in self.coords:
                raise ValueError(f"Unknown axis '{axis_name}'. Available: {list(self.coords.keys())}")
            coord_list = self.coords[axis_name]
            if value not in coord_list:
                raise ValueError(f"Value {value} not in {axis_name} coords: {coord_list}")
            idx[self._axis_names.index(axis_name)] = coord_list.index(value)
        return self[tuple(idx)]

    @classmethod
    def stack(cls, results: List['SimulationResult'], axis_name: str = 'structural',
              axis_values: Optional[List] = None) -> 'SimulationResult':
        """Stack multiple SimulationResults along a new named axis.

        Useful for structural sweeps (different topologies) that cannot be
        batched within a single JIT call.

        Args:
            results: List of SimulationResult with compatible shapes
            axis_name: Name for the new axis (e.g., 'nrows', 'structural')
            axis_values: Optional coordinate values for the new axis

        Returns:
            Stacked SimulationResult with new leading axis
        """
        if not results:
            raise ValueError("Cannot stack empty list of results")
        out = results[0]._with(
            {k: jnp.stack([r.metrics[k] for r in results]) for k in results[0].metrics},
            {t: jnp.stack([getattr(r, t) for r in results]) for t in cls._TRACES},
            [axis_name] + results[0]._axis_names,
            {**results[0].coords,
             axis_name: axis_values if axis_values is not None else list(range(len(results)))},
            "_stacked")
        out.rng_key = results[-1].rng_key
        return out

    def __repr__(self) -> str:
        force = self.metrics.get('axial_force')
        shape_str = str(force.shape) if force is not None and hasattr(force, 'shape') else '?'
        return f"SimulationResult(name='{self.name}', shape={shape_str}, dt={self.dt})"


# =============================================================================
# BATCH PADDING (avoids recompilation for different sweep sizes)
# =============================================================================

BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def _pad(arr, pad_n):
    """Pad arr along axis 0 by replicating arr[:1] pad_n times."""
    return jnp.concatenate([arr, jnp.broadcast_to(arr[:1], (pad_n,) + arr.shape[1:])])


def get_bucket_size(actual_size: int) -> int:
    """Round a batch size up to the next power-of-two bucket.

    XLA compiles per exact shape, so without this every distinct sweep size
    would pay a fresh compilation for identical code. Padding to buckets means a
    225-point sweep and a 144-point sweep both compile once, as 256, and the
    second reuses the first's kernel.

    The cost is running at most twice as many simulations as requested; the
    padded ones are trimmed off before the result is returned. Compilation takes
    minutes, so this is overwhelmingly the right trade.
    """
    for bucket in BATCH_BUCKETS:
        if bucket >= actual_size:
            return bucket
    return actual_size  # Larger than all buckets — use exact size


# When to split a padded batch into sequential chunks: (min_batch, chunk_size).
#
# Chunking is primarily a MEMORY control, not a speed one. Every simulation
# accumulates all 43 metrics at every timestep, so peak GPU memory scales as
#
#     peak VRAM (GB) ~ minibatch_size * n_steps * 43 * 4 bytes * 2 / 1e9
#
# A CHUNKED RUN ACCUMULATES ON THE HOST, NOT THE DEVICE — see the chunk loop in
# run(). Chunking alone never bounded the total: every chunk stayed resident and
# the concatenate then built a second full copy, so a big sweep finished its
# compute and died assembling the result. 43 metrics x 4 bytes = 172 bytes per
# sim per timestep, so 67k sims over 1200 steps wants 13.9 GB, twice, on a
# 24 GB card. (It was 252 bytes over 63 metrics until 2026-09-19, when the 20
# reconstructible ones were deleted — a 1.5x reduction, not a fix.)
#
# >>> THE HOST FIX TRADES A GPU LIMIT FOR A HOST-RAM LIMIT, AND THAT IS NOT A
#     REAL SOLUTION. A 236k-sim x 1200-step grid still needs ~49 GB of system
#     RAM to hold traces the caller usually reduces to one scalar per sim and
#     throws away. It works here (125 GB under WSL2) and it will not work on a
#     normal machine. The durable fix is to stop materialising what nobody
#     wants — reduce each chunk's time axis before accumulating it — which is
#     deliberately NOT done here because doing it inside the kernel changes its
#     output signature and reintroduces the per-metric-list recompile that got
#     explicit metric selection removed in the first place. Revisit with that
#     constraint in mind. <<<
#
# — roughly 0.7 GB of metrics for 4096 simulations over 1000 steps, and about
# twice that in total. On an 8 GB card a long simulation at large batch will run
# out of memory without chunking. Since each chunk calls the same compiled
# kernel, splitting costs no recompilation.
#
# The speed effect is small and only appears at large batch, where a chunk that
# fits in L2 cache measured a couple of percent faster than the full batch on an
# RTX 3090. Below ~16384 the full batch was as fast or faster, so no chunking is
# applied there. The exact crossover has not been measured on other hardware.
_MINIBATCH_HEURISTIC = (
    (16384, 4096),   # batch ≥ 16384 → chunk to 4096 (benchmarked optimal)
)


def _auto_minibatch_size(padded_batch: int) -> Optional[int]:
    """Return the auto-selected chunk size for a given padded batch, or None."""
    for threshold, chunk in _MINIBATCH_HEURISTIC:
        if padded_batch >= threshold:
            return chunk
    return None


# =============================================================================
# SUBPOPULATION RESOLUTION (host-side, before the kernel)
# =============================================================================

def _resolve_explicit_masks(subpops, topology, total_batch, idx, fractions_b):
    """Per-sim INT-label masks for explicit modes (random / c_zone).

    ``idx`` is the (total_batch,) gather index into ``subpops`` when swept as a
    list axis, or None for a single (unswept) object broadcast to every sim.
    """
    mode = subpops[0].mode
    if mode == 'c_zone':
        xb_variants = jnp.stack([jnp.asarray(sp.xb_mask) for sp in subpops])  # (V, total_xbs)
        tm_variants = jnp.stack([jnp.asarray(sp.tm_mask) for sp in subpops])  # (V, n_thin * n_tm)
        if idx is not None:
            xb_mask_b = xb_variants[idx]
            tm_mask_b = tm_variants[idx]
        else:
            xb_mask_b = jnp.broadcast_to(xb_variants[0], (total_batch,) + xb_variants.shape[1:])
            tm_mask_b = jnp.broadcast_to(tm_variants[0], (total_batch,) + tm_variants.shape[1:])
    elif mode == 'random':
        seeds = {int(sp.seed or 0) for sp in subpops}
        if len(seeds) > 1:
            raise ValueError(
                f"All swept 'random' subpopulations must share one seed; got {sorted(seeds)}"
            )
        xb_mask_b, tm_mask_b = generate_random_masks_batch(
            subpops[0].seed or 0, fractions_b[:, 1], topology, total_batch)
    else:
        raise ValueError(f"Unknown explicit subpopulation mode '{mode}'")
    return xb_mask_b, tm_mask_b


def _resolve_subpopulation(subpopulation, topology, total_batch, flat_idx,
                           is_list_axis):
    """Resolve a Subpopulation (single object or list) into the kernel's two
    static arguments and its per-sim batched arrays.

    Returns:
        (mode, scaled_field_names, subpop_arrays)

    mode is 'mean_field' or 'explicit'. subpop_arrays is a dict of
    (total_batch, ...) arrays: {'scale_matrix' (·,K,F), 'fractions' (·,K)} for
    mean-field, or {'scale_matrix', 'xb_mask' (·,total_xbs),
    'tm_mask' (·,n_thin * n_tm)} for explicit modes.

    No subpopulation is the mean-field case with K = 1, fraction 1.0 and no
    scaled field — the same path, not a separate one.
    """
    if subpopulation is None:
        return ('mean_field', (),
                {'scale_matrix': jnp.ones((total_batch, 1, 0), jnp.float32),
                 'fractions': jnp.ones((total_batch, 1), jnp.float32)})

    subpops = subpopulation if is_list_axis else [subpopulation]

    modes = {sp.mode for sp in subpops}
    if len(modes) > 1:
        raise ValueError(f"All swept subpopulations must share one mode; got {sorted(modes)}")
    n_pops_set = {sp.K for sp in subpops}
    if len(n_pops_set) > 1:
        raise ValueError(f"All swept subpopulations must have the same K; got {sorted(n_pops_set)}")
    mode = subpops[0].mode
    n_pops = subpops[0].K

    # Union of scaled fields (must all be xb_* or tm_* — mechanics use the WT base).
    field_set = set()
    for sp in subpops:
        field_set |= set(sp.scaled_fields())
    bad = [f for f in field_set if not (f.startswith('xb_') or f.startswith('tm_'))]
    if bad:
        raise ValueError(
            f"Subpopulation scales must be xb_* or tm_* fields; got {sorted(bad)}"
        )
    scaled_field_names = tuple(sorted(field_set))

    # Per-variant scale/fraction tables → gather (or broadcast) per sim.
    scale_variants = jnp.stack([sp.scale_array(scaled_field_names) for sp in subpops])  # (V,K,F)
    frac_variants = jnp.stack([sp.fractions for sp in subpops])                          # (V,K)
    idx = flat_idx['subpopulation'] if is_list_axis else None
    if idx is not None:
        scale_matrix_b = scale_variants[idx]                       # (total_batch, K, F)
        fractions_b = frac_variants[idx]                           # (total_batch, K)
    else:
        scale_matrix_b = jnp.broadcast_to(
            scale_variants[0], (total_batch,) + scale_variants.shape[1:])
        fractions_b = jnp.broadcast_to(frac_variants[0], (total_batch, n_pops))

    if mode == 'mean_field':
        return mode, scaled_field_names, {'scale_matrix': scale_matrix_b,
                                          'fractions': fractions_b}
    xb_mask_b, tm_mask_b = _resolve_explicit_masks(
        subpops, topology, total_batch, idx, fractions_b)
    return 'explicit', scaled_field_names, {'scale_matrix': scale_matrix_b,
                                            'xb_mask': xb_mask_b, 'tm_mask': tm_mask_b}


# =============================================================================
# MODULE-LEVEL SIMULATION KERNEL
# =============================================================================

class SimBatch(NamedTuple):
    """Everything that differs between the simulations of one run(), each
    leaf with a leading (batch,) axis. Padding, chunking, trimming and the
    kernel's vmap all treat it as one pytree, so no per-sim input can be
    forgotten by one of them.

    Fields:
        params: DynamicParams, every field (batch,)
        z, pCa, ls: (batch, n_steps) driver traces
        K_lat: (batch,) lattice stiffness, already x n_thick (dynamic LS only)
        nu: (batch,) Poisson exponent
        rng_keys: (batch, 2) one PRNG key per simulation
        subpop: dict of per-sim subpopulation arrays (see _resolve_subpopulation)
    """
    params: DynamicParams
    z: jnp.ndarray
    pCa: jnp.ndarray
    ls: jnp.ndarray
    K_lat: jnp.ndarray
    nu: jnp.ndarray
    rng_keys: jnp.ndarray
    subpop: object

@partial(jax.jit, static_argnames=[
    'dt', 'unroll', 'is_dynamic_ls', 'n_cg_steps', 'n_newton_steps',
    'subpop_mode', 'scaled_field_names'])
def _run_sim_kernel(
    topology: SarcTopology,
    batch: SimBatch,
    dt: float,
    unroll: int,
    n_cg_steps: int,
    n_newton_steps: int,
    is_dynamic_ls: bool = False,
    subpop_mode: str = 'mean_field',
    scaled_field_names: tuple = (),
):
    """JIT-compiled simulation kernel (unified fixed + dynamic LS).

    Always computes all metrics via compute_all_metrics(). No metrics/manifest
    in static_argnames — changing metric selection never triggers recompilation.

    Args:
        topology: SarcTopology with pre-computed index maps (broadcast via closure)
        batch: SimBatch, every leaf (batch, ...); vmapped whole with in_axes=0
        dt: Timestep in ms (static)
        unroll: Scan unrolling factor (static)
        is_dynamic_ls: If True, solve lattice spacing as a DOF (static)
        subpop_mode: 'mean_field' or 'explicit' (static)
        scaled_field_names: the xb_*/tm_* fields the subpopulations scale,
            the columns of batch.subpop['scale_matrix'] (static)

    Returns:
        MetricsDict with all metric scalars, shape (batch, time).
        Includes 'axial_force', 'solver_residual', and 'newton_iters' keys.
    """

    def create_and_equilibrate(constants, z0, pCa0, ls0):
        """Create state from topology + constants and solve equilibrium.

        The thin frame is anchored on the Z-disc, so every reconstruction of an
        absolute position needs the Z-line this state is built at: z0, the
        first sample of the z_line trace.
        """
        state = realize_state(topology, constants, z0, pCa0, ls0)
        state = update_nearest_neighbors(state, topology, z0, ls0)
        state, _residual, _, _, _, _ = solve_equilibrium(
            state, constants, topology, z0, ls0,
            n_cg_steps=n_cg_steps,
            n_newton_steps=n_newton_steps,
        )
        return state

    def run_single_sim(state, b):
        """Run simulation with scan inside vmap. b is one simulation's SimBatch."""
        constants, z_trace, pCa_trace, ls_trace = b.params, b.z, b.pCa, b.ls
        K_lat_val, nu_val, subpop = b.K_lat, b.nu, b.subpop
        n_thick, n_crowns = state.thick.displacement.shape
        n_thin, n_nodes = state.thin.displacement.shape
        precond_params = build_preconditioner_params(
            n_thick, n_crowns, n_thin, n_nodes,
            constants.thick_k, thin_segment_k(constants, topology),
        )
        prefactored_precond = build_prefactored_preconditioner(precond_params)

        l0 = z_trace[0]  # reference z for Poisson scaling

        # Subpopulation: the K population constants, built once per sim (rate
        # scales are per-sim; the drivers reach the kernels per step). A kernel
        # whose rates no population scales gets None = the wild type alone.
        scale_matrix = subpop['scale_matrix']  # (K, F)
        constants_k = [
            constants.copy(**{name: getattr(constants, name) * scale_matrix[k, j]
                              for j, name in enumerate(scaled_field_names)})
            for k in range(scale_matrix.shape[0])
        ]

        def _pops(prefix, mask):
            if not any(f.startswith(prefix) for f in scaled_field_names):
                return None
            if subpop_mode == 'mean_field':
                return ('mean_field', constants_k, subpop['fractions'])
            return ('explicit', constants_k, subpop[mask])

        xb_subpop, tm_subpop = _pops('xb_', 'xb_mask'), _pops('tm_', 'tm_mask')

        # Start relaxed: every head drawn from its own resting distribution. The
        # draw has its own stream, folded off the sim's key, so the scan's key
        # is untouched.
        state = xb_rest_states(state, constants, topology, z_trace[0], ls_trace[0],
                               jax.random.fold_in(b.rng_keys, 0), xb_subpop)

        def scan_fn(carry, inputs):
            old_state, k, current_ls = carry
            z_val, pCa_val, ls_val = inputs

            # NO THIN POSITION SHIFT, AND NO dz. Thin displacements are
            # measured from a Z-disc-anchored rest frame, so the filament moves
            # rigidly with the Z-line by construction and there is nothing to
            # update. The per-step z-line displacement is not carried either:
            # its only reader was `sarcomere_work`, and that is reconstructed
            # after the fact from the axial_force and z_line traces.

            if is_dynamic_ls:
                drivers = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=current_ls)
                d_ref = poisson_spacing(ls_val, l0, z_val, nu_val)
            else:
                drivers = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=ls_val)
                d_ref = None

            (new_state, new_k, solver_residual, new_ls, n_iters, trace,
             residual_norm, solver_tolerance) = timestep(
                old_state, constants, drivers, topology, k, dt=dt,
                K_lat=K_lat_val if is_dynamic_ls else None,
                d_ref=d_ref,
                n_cg_steps=n_cg_steps,
                n_newton_steps=n_newton_steps,
                precond_params=precond_params,
                prefactored_precond=prefactored_precond,
                xb_subpop=xb_subpop,
                tm_subpop=tm_subpop,
            )

            # POST-SOLVE drivers: these carry the emergent new_ls, and must,
            # because force and the reported lattice_spacing are post-solve
            # quantities. The Q-matrix metrics deliberately do NOT use them —
            # they read trace.drivers, which carries the PRE-solve spacing the
            # rates were actually evaluated at. Both are right for their own
            # question; do not unify them. See compute_all_metrics.
            drivers_for_metrics = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=new_ls)
            force = axial_force_at_mline(new_state, constants, topology)

            all_metrics = compute_all_metrics(
                old_state, new_state, constants, drivers_for_metrics,
                topology, force, solver_residual, n_iters, dt, trace,
                residual_norm, solver_tolerance,
            )

            return (new_state, new_k, new_ls), all_metrics

        _, metrics_out = jax.lax.scan(
            scan_fn,
            (state, b.rng_keys, ls_trace[0]),
            (z_trace, pCa_trace, ls_trace),
            unroll=unroll,
        )
        return metrics_out

    batched_states = jax.vmap(create_and_equilibrate)(
        batch.params, batch.z[:, 0], batch.pCa[:, 0], batch.ls[:, 0])
    return jax.vmap(run_single_sim)(batched_states, batch)


# =============================================================================
# TOP-LEVEL run() API
# =============================================================================

def run(
    topology: SarcTopology,
    *,
    pCa: Union[float, List[float], jnp.ndarray],
    z_line: Union[float, List[float], jnp.ndarray],
    lattice_spacing: Union[float, List[float], jnp.ndarray],
    duration_ms: float = 1000.0,
    dt: float = 1.0,
    K_lat: Union[float, List[float], None] = None,
    nu: Union[float, List[float]] = 0.0,
    dynamic_params: Union[DynamicParams, Dict[str, Union[float, List[float]]]] = None,
    static_params: 'StaticParams' = None,
    replicates: int = 1,
    rng_seed: int = 0,
    unroll: int = 1,
    minibatch_size: Optional[int] = "auto",
    verbose: bool = False,
    subpopulation=None,
) -> SimulationResult:
    """Run a muscle simulation with the given topology.

    This is the primary API. Accepts a pre-constructed SarcTopology
    (topology defines structural configuration; changing it requires
    recompilation). All other parameters are sweepable without recompile.

    Batch padding: sweep sizes are rounded up to the nearest bucket
    (1, 2, 4, ..., 16384), so a 225-run sweep and a 256-run sweep
    share the same compiled kernel.

    Everything after `topology` is keyword-only. The three drivers (pCa, z_line,
    lattice_spacing) are REQUIRED and have no defaults: they are not physics
    constants and do not live in DynamicParams. Each preset returns its natural
    z_line and lattice_spacing as z0, d0 — pass those when you have no other
    operating point in mind.

    Args:
        topology: Pre-constructed SarcTopology (from SarcTopology.create())
        pCa: Calcium as -log10([Ca]) -- float, list (sweep), or array (trace)
        z_line: Z-line position (nm) -- float, list (sweep), or array (trace)
        lattice_spacing: Lattice spacing (nm) -- float, list, or array. In
            dynamic LS mode (K_lat set) it is the reference d0 and initial guess.
        duration_ms: Simulation duration in milliseconds
        dt: Timestep in milliseconds
        K_lat: Lattice stiffness per thick filament (pN/nm). None = fixed LS.
               Float or list (sweep). Internally scaled by n_thick.
        nu: Poisson exponent. Applied to the CENTRE-TO-CENTRE spacing, since
            the filament radii are fixed while the lattice closes:
                d(z) = (d0 + FILAMENT_RADII_SUM)*(z0/z)^nu - FILAMENT_RADII_SUM
            with d the thick-to-thin SURFACE gap. Float or list (sweep).
            If K_lat is None and nu>0, pre-computes Poisson LS trace.
        dynamic_params: DynamicParams, dict of overrides/sweeps, or list of DynamicParams.
                        A list creates a 'candidates' sweep axis (one element per candidate),
                        Cartesian-producted with other sweep axes. Useful for batching CMA-ES
                        population evaluations: run(topo, z_line=traces, dynamic_params=[dp0..dpN])
                        gives result shape (N, n_freq, replicates, time).

                        CAUTION — WHICH BASELINE THE UNSWEPT FIELDS COME FROM.
                        None and the DICT form both build on a bare, SKELETAL
                        DynamicParams(). They do NOT inherit the preset the
                        topology was built with: SarcTopology.create() consumes
                        dynamic_params for geometry and does not retain it, so
                        run() has no way to recover a cardiac or IFM preset from
                        the topology. Passing dynamic_params={'tm_J_M': [...]}
                        against a cardiac topology therefore sweeps tm_J_M with
                        SKELETAL kinetics everywhere else, silently. To sweep one
                        field of a non-skeletal preset, use the candidate-LIST
                        form instead:
                            run(topo, pCa=..., z_line=z0, lattice_spacing=d0,
                                dynamic_params=[dynamic.copy(tm_J_M=v)
                                                for v in values])
                        (This has already invalidated one cardiac calibration.)
        replicates: Number of statistical replicates per sweep point
        rng_seed: Base random seed
        unroll: Scan unrolling factor
        minibatch_size: Chunk size for splitting the padded batch across multiple
            _run_sim_kernel calls. "auto" (default) applies _MINIBATCH_HEURISTIC
            based on padded batch size. None disables chunking. An explicit int
            overrides the heuristic; snapped down to the nearest power-of-2 bucket.
            Primary use: bounding peak GPU VRAM on memory-constrained GPUs (e.g.
            8 GB RTX 4060). Rule of thumb: peak VRAM (GB) ≈
            minibatch_size × n_steps × 45 × 4 bytes × 2 / 1e9. For a 4060 at
            1000 steps, minibatch_size=4096 uses ~3 GB; 8192 uses ~6 GB.
        verbose: Print progress info
        subpopulation: Optional Subpopulation (or list of them) modelling a
            fraction of XB motors / TM units with scaled kinetics. mean_field
            blends the K population rate matrices (Q_eff = Σ f_k Q_k); random /
            c_zone assign per-XB / per-site integer labels and select per unit.
            A list is a sweep axis (mutually exclusive with a dynamic_params
            candidate list), supported for all three modes — mean_field,
            random, and c_zone. A swept 'random' list must share one seed
            across every entry (severity/fraction may still vary).
            subpopulation=None leaves every path unchanged.

    Returns:
        SimulationResult with shape (sweep_1, ..., replicates, time)

    Example:
        from multifil_jax.core.sarc_geometry import SarcTopology
        from multifil_jax.core.params import get_skeletal_params, StaticParams

        static, dynamic, z0, d0 = get_skeletal_params()
        topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

        result = run(topo, pCa=4.5, z_line=z0, lattice_spacing=d0, duration_ms=100)
        print(result.summary())
    """
    n_steps = int(duration_ms / dt)
    topology = jax.device_put(topology)

    if static_params is None:
        from multifil_jax.core.params import StaticParams as _StaticParams
        static_params = _StaticParams()

    # =========================================================================
    # Build batched kernel inputs via integer row-number lookup.
    #
    # Every sweepable input keeps a small "variant table" of its possible values.
    # The sweep grid is a Cartesian product of plain integer row-numbers (one axis
    # per swept input); each simulation looks up its row in every table. This one
    # idea unifies scalars, time-traces, list-of-trace sweeps, K_lat/nu sweeps,
    # DynamicParams field sweeps, and candidate lists.
    # =========================================================================

    # Base DynamicParams supplies scalar values for un-swept fields; dict/list
    # forms carry their own per-field overrides below.
    if dynamic_params is None or isinstance(dynamic_params, (dict, list)):
        base_dynamic = DynamicParams()
    elif isinstance(dynamic_params, DynamicParams):
        base_dynamic = dynamic_params
    else:
        raise ValueError(
            f"dynamic_params must be DynamicParams, dict, or list, got {type(dynamic_params)}"
        )

    is_dynamic_ls = K_lat is not None

    def _as_trace(v):
        """Normalize one waveform variant to a length-n_steps trace."""
        a = np.asarray(v, dtype=float)
        return np.full(n_steps, float(a)) if a.ndim == 0 else a

    sweep_axes = []        # (name, coords) in Cartesian-product / axis order
    waveform_tables = {}   # name -> (n_variants, n_steps)
    wf_sweeps = set()      # waveform names that are sweep axes
    param_tables = {}      # DYNAMIC_FIELDS name -> (n_variants,) lookup table
    param_axis = {}        # field name -> axis name to index it with

    # 1. Waveform inputs (z_line, pCa, lattice_spacing) — one table each.
    #    Each must be finite everywhere: there is no fallback value behind them.
    for name, value in [('z_line', z_line), ('pCa', pCa), ('lattice_spacing', lattice_spacing)]:
        elems = value if isinstance(value, list) else [value]
        if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in elems):
            raise ValueError(f"{name} contains non-finite values")
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{name} sweep list is empty")
            waveform_tables[name] = jnp.asarray(np.stack([_as_trace(v) for v in value]))
            is_trace_sweep = any(
                not isinstance(v, (int, float)) and np.asarray(v).ndim == 1
                and len(np.asarray(v)) == n_steps for v in value
            )
            coords = ([np.asarray(waveform_tables[name][i]) for i in range(len(value))]
                      if is_trace_sweep else list(value))
            sweep_axes.append((name, coords))
            wf_sweeps.add(name)
        else:
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] != n_steps:
                raise ValueError(f"{name} array length {arr.shape[0]} != n_steps {n_steps}")
            waveform_tables[name] = jnp.asarray(_as_trace(value))[None, :]  # (1, n_steps)

    # 2. K_lat / nu — per-sim scalars (never traces).
    if isinstance(K_lat, list):
        sweep_axes.append(('K_lat', list(K_lat)))
    if isinstance(nu, list):
        sweep_axes.append(('nu', list(nu)))

    # 3. DynamicParams sweeps / overrides — one (n_variants,) table per swept field.
    if isinstance(dynamic_params, list):
        if not dynamic_params:
            raise ValueError("dynamic_params candidate list is empty")
        sweep_axes.append(('candidates', list(range(len(dynamic_params)))))
        for name in DYNAMIC_FIELDS:
            param_tables[name] = jnp.stack(
                [jnp.asarray(getattr(dp, name), jnp.float32) for dp in dynamic_params]
            )
            param_axis[name] = 'candidates'
    elif isinstance(dynamic_params, dict):
        for pname, pval in dynamic_params.items():
            if pname not in DYNAMIC_FIELDS:
                raise ValueError(
                    f"Unknown dynamic_params field '{pname}'. Valid: {list(DYNAMIC_FIELDS)}"
                )
            if isinstance(pval, list):
                sweep_axes.append((pname, list(pval)))
                param_tables[pname] = jnp.asarray(pval, jnp.float32)
                param_axis[pname] = pname
    elif isinstance(dynamic_params, DynamicParams):
        for name in DYNAMIC_FIELDS:
            arr = jnp.asarray(getattr(dynamic_params, name))
            if arr.ndim > 0 and arr.shape[0] > 1:
                sweep_axes.append((name, list(arr.tolist())))
                param_tables[name] = arr.astype(jnp.float32)
                param_axis[name] = name

    # 4. Subpopulation list = a candidate-style sweep axis. At most one
    #    candidate-style axis is allowed (each element is a whole config).
    if isinstance(subpopulation, list):
        if isinstance(dynamic_params, list):
            raise ValueError(
                "Cannot sweep a dynamic_params list and a subpopulation list "
                "simultaneously (both are candidate-style axes)."
            )
        if not subpopulation:
            raise ValueError("subpopulation list is empty")
        sweep_axes.append(('subpopulation', list(range(len(subpopulation)))))

    # Cartesian product over integer row-numbers; tile for replicates.
    if sweep_axes:
        grids = jnp.meshgrid(*[jnp.arange(len(c)) for _, c in sweep_axes], indexing='ij')
        batch_size = grids[0].size
        flat_idx = {name: jnp.repeat(g.reshape(-1), replicates)
                    for (name, _), g in zip(sweep_axes, grids)}
    else:
        batch_size = 1
        flat_idx = {}

    total_batch = batch_size * replicates
    grid_shape = tuple(len(c) for _, c in sweep_axes)
    axis_names = [name for name, _ in sweep_axes] + ['replicates', 'time']
    coords = {name: c for name, c in sweep_axes}
    coords['replicates'] = list(range(replicates))
    coords['time'] = (jnp.arange(n_steps) * dt).tolist()

    if verbose:
        print(f"Grid shape: {grid_shape}, axes: {axis_names}, "
              f"batch_size: {batch_size}, total: {total_batch}")
        if is_dynamic_ls:
            print(f"Dynamic LS: K_lat={K_lat}, nu={nu}")

    # Materialize waveforms → (total_batch, n_steps).
    def _waveform(name):
        table = waveform_tables[name]
        if name in wf_sweeps:
            return table[flat_idx[name]]
        return jnp.broadcast_to(table[0], (total_batch, n_steps))

    z_batched = _waveform('z_line')
    pCa_batched = _waveform('pCa')
    ls_batched = _waveform('lattice_spacing')

    # K_lat / nu batched scalars (total_batch,).
    if isinstance(K_lat, list):
        K_lat_batched = jnp.asarray(K_lat, jnp.float32)[flat_idx['K_lat']]
    elif is_dynamic_ls:
        K_lat_batched = jnp.full(total_batch, float(K_lat))
    else:
        K_lat_batched = jnp.zeros(total_batch)

    if isinstance(nu, list):
        nu_batched = jnp.asarray(nu, jnp.float32)[flat_idx['nu']]
    else:
        nu_batched = jnp.full(total_batch, float(nu))

    if is_dynamic_ls:
        K_lat_batched = K_lat_batched * topology.n_thick

    # Poisson pre-computation (K_lat=None, nu>0): scale LS from its own d0.
    ls_is_trace = (not isinstance(lattice_spacing, list)
                   and np.asarray(lattice_spacing).ndim > 0)
    has_nonzero_nu = (any(v != 0.0 for v in nu) if isinstance(nu, list)
                      else float(nu) != 0.0)
    if not is_dynamic_ls and not ls_is_trace and has_nonzero_nu:
        ls_batched = poisson_spacing(ls_batched[:, 0:1], z_batched[:, 0:1],
                                     z_batched, nu_batched[:, None])

    # Batched DynamicParams — one lookup per field.
    def _param(name):
        if name in param_axis:
            return param_tables[name][flat_idx[param_axis[name]]]
        if isinstance(dynamic_params, dict) and name in dynamic_params:
            return jnp.full(total_batch, float(dynamic_params[name]))
        return jnp.full(total_batch, float(getattr(base_dynamic, name)))

    batched_params = DynamicParams(**{name: _param(name) for name in DYNAMIC_FIELDS})

    # Subpopulation resolution → two static args + per-sim batched arrays.
    subpop_mode, scaled_field_names, subpop_arrays = _resolve_subpopulation(
        subpopulation, topology, total_batch, flat_idx,
        is_list_axis=isinstance(subpopulation, list),
    )

    batch = SimBatch(
        params=batched_params, z=z_batched, pCa=pCa_batched, ls=ls_batched,
        K_lat=K_lat_batched, nu=nu_batched,
        rng_keys=jax.random.split(jax.random.PRNGKey(rng_seed), total_batch),
        subpop=subpop_arrays)

    # Pad batch to bucket size. The padding sims copy sim 0 and are trimmed
    # off below; every sim is independent under vmap, so they change nothing.
    padded_batch = get_bucket_size(total_batch)
    if padded_batch > total_batch:
        batch = jax.tree_util.tree_map(lambda x: _pad(x, padded_batch - total_batch), batch)

    if verbose:
        print(f"Running simulation kernel (batch={total_batch}, padded={padded_batch})...")

    resolved_minibatch = _auto_minibatch_size(padded_batch) if minibatch_size == "auto" else minibatch_size
    # Snap to largest BATCH_BUCKETS value that (a) is <= resolved_minibatch and
    # (b) divides padded_batch evenly. Since padded_batch is always a power of 2,
    # any power-of-2 bucket <= padded_batch divides it. This prevents a non-bucket
    # minibatch_size from producing a shorter last chunk and triggering a JIT recompile.
    if resolved_minibatch is not None and resolved_minibatch not in BATCH_BUCKETS:
        valid = [b for b in BATCH_BUCKETS if b <= resolved_minibatch and b <= padded_batch]
        resolved_minibatch = max(valid) if valid else None
    use_minibatch = (resolved_minibatch is not None) and (resolved_minibatch < padded_batch)

    kernel_kwargs = dict(
        dt=dt,
        unroll=unroll,
        is_dynamic_ls=is_dynamic_ls,
        n_cg_steps=static_params.n_cg_steps,
        n_newton_steps=static_params.n_newton_steps,
        subpop_mode=subpop_mode,
        scaled_field_names=scaled_field_names,
    )

    # One chunk = the whole padded batch when not minibatching.
    chunk_size = resolved_minibatch if use_minibatch else padded_batch
    if use_minibatch and verbose:
        print(f"Minibatching: {padded_batch // chunk_size} chunks of {chunk_size}")
    starts = list(range(0, padded_batch, chunk_size))

    def _run_chunk(start):
        return _run_sim_kernel(
            topology=topology,
            batch=jax.tree_util.tree_map(lambda x: x[start:start + chunk_size], batch),
            **kernel_kwargs,
        )

    # ONE CHUNK: unchanged, and stays on the device. The overwhelming majority
    # of runs land here, and a host round-trip would tax all of them to fix a
    # problem only the chunked path has.
    chunk_maxima = None
    if len(starts) == 1:
        batched_metrics = _run_chunk(starts[0])
    else:
        # MANY CHUNKS: accumulate on the HOST, into a buffer allocated ONCE.
        # Preallocation is the whole trick. Appending to a list and calling
        # np.concatenate at the end would peak at twice the final size, which
        # is the exact bug this replaces, moved one level down.
        batched_metrics = None
        chunk_maxima = {k: -float('inf') for k in ('solver_residual',
                                                   'solver_residual_norm')}
        for start in starts:
            chunk = _run_chunk(start)
            if batched_metrics is None:
                batched_metrics = MetricsDict({
                    k: np.empty((padded_batch,) + v.shape[1:], dtype=v.dtype)
                    for k, v in chunk.items()})
            # The convergence maxima are taken PER CHUNK, on the device, while
            # the chunk is still there. Reducing the assembled host array later
            # would drag every byte back across the bus.
            for k in chunk_maxima:
                chunk_maxima[k] = max(chunk_maxima[k], float(jnp.max(chunk[k])))
            for k, v in chunk.items():
                host = np.asarray(v)
                batched_metrics[k][start:start + host.shape[0]] = host
            del chunk, host      # release the device buffer before the next one

    # Trim the padding and reshape to the data cube, metrics and drivers alike.
    final_shape = grid_shape + (replicates, n_steps)
    reshaped_metrics, (reshaped_z, reshaped_pCa, reshaped_ls) = jax.tree_util.tree_map(
        lambda v: v[:total_batch].reshape(final_shape),
        (dict(batched_metrics), (batch.z, batch.pCa, batch.ls)))
    reshaped_metrics = MetricsDict(reshaped_metrics)

    # Post-run solver convergence check. The test is the NORMALIZED residual,
    # which is <= 1 exactly when the solve converged — in both LS modes, and at
    # any stiffness. The old absolute-pN threshold was a third copy of the
    # float32 coordinate floor and moved with thick_k rather than with the
    # physics; see kernels/solver._convergence_tolerance.
    if chunk_maxima is not None:
        # Already reduced per chunk, on the device, during accumulation.
        max_residual = chunk_maxima['solver_residual']
        max_norm = chunk_maxima['solver_residual_norm']
    else:
        max_residual = float(jnp.max(reshaped_metrics['solver_residual']))
        max_norm = float(jnp.max(reshaped_metrics['solver_residual_norm']))
    if max_norm > 1.0:
        import warnings
        warnings.warn(
            f"Solver did not converge everywhere: max normalized residual "
            f"{max_norm:.3g} > 1 (max raw residual {max_residual:.4g} pN). "
            f"Tune DynamicParams.solver_rtol/solver_atol, or raise "
            f"StaticParams.n_newton_steps."
        )

    if verbose:
        print(f"Result shape: {reshaped_metrics['axial_force'].shape}")
        print(f"Max solver residual: {max_residual:.4f} pN "
              f"(normalized {max_norm:.4f})")

    topology_config = {name: getattr(topology, name) for name in topology._AUX}
    if is_dynamic_ls:
        topology_config['K_lat'] = K_lat
        topology_config['K_lat_eff'] = float(K_lat * topology.n_thick) if not isinstance(K_lat, list) else [float(k * topology.n_thick) for k in K_lat]
        topology_config['nu'] = nu

    # grid_shape / axis_names are first-class SimulationResult fields and K_lat /
    # nu live in topology_config; metadata keeps only what nothing else stores.
    metadata = {'master_seed': rng_seed}

    return SimulationResult(
        metrics=reshaped_metrics,
        rng_key=batch.rng_keys[total_batch - 1],
        z_line=reshaped_z,
        pCa=reshaped_pCa,
        lattice_spacing=reshaped_ls,
        metadata=metadata,
        dt=dt,
        name="run",
        axis_names=axis_names,
        coords=coords,
        topology_config=topology_config,
    )

