"""
Unified Simulation Engine for JAX Half-Sarcomere

This module provides the core simulation engine with proper vmap-outside-scan
architecture for maximum GPU utilization through XLA kernel fusion.

Architecture Principle:
    vmap OUTSIDE, scan INSIDE = single fused XLA kernel per batch

Primary API:
    run(topology, pCa=..., z_line=..., ...) -> SimulationResult

Usage:
    from multifil_jax.simulation import run, get_skeletal_params
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.params import StaticParams

    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

    # Simple isometric simulation
    result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000)
    print(result.summary())

    # Parameter sweep (list = sweep axis)
    result = run(topo, pCa=[4.5, 5.0, 6.0], replicates=5)

    # DynamicParams sweep
    result = run(topo, pCa=4.5, dynamic_params={'thick_k': [1000, 2000, 3000]})
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Dict, Tuple, List, Optional, Union, TYPE_CHECKING

from multifil_jax.core.params import (
    StaticParams, DynamicParams, get_skeletal_params, DYNAMIC_FIELDS
)
from multifil_jax.core.state import realize_state, State, Drivers, MetricsDict, build_preconditioner_params
from multifil_jax.kernels.solver import build_prefactored_preconditioner
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.kernels.geometry import update_nearest_neighbors
from multifil_jax.kernels.solver import solve_equilibrium
from multifil_jax.kernels.forces import axial_force_at_mline
from multifil_jax.timestep import timestep
from multifil_jax.metrics_fn import compute_all_metrics


# =============================================================================
# SIMULATION RESULT
# =============================================================================

@jax.tree_util.register_pytree_node_class
class SimulationResult:
    """Result container from run() with visualization and grid support.

    Consolidates all simulation outputs including:
    - Force and metrics time traces (force lives in metrics['axial_force'])
    - Input replay (z_line, pCa, lattice_spacing traces used)
    - Final state for continuation
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
        final_state: Nested dict or None if return_final_state=False
        metadata: Dict with params, geometry, etc.
        dt: Timestep in milliseconds
        name: Simulation/experiment name
        _grid_shape: Tuple: shape without time
        _axis_names: List: ['pCa', 'thick_k', 'replicates', 'time']
        coords: Dict: {'pCa': [...], 'thick_k': [...], ...}
    """

    __slots__ = (
        'metrics', 'rng_key',
        'z_line', 'pCa', 'lattice_spacing',
        'final_state', 'metadata', 'dt', 'name',
        '_grid_shape', '_axis_names', 'coords',
        'topology_config',
    )

    def __init__(
        self,
        metrics: 'MetricsDict',
        rng_key: jnp.ndarray,
        z_line: jnp.ndarray,
        pCa: jnp.ndarray,
        lattice_spacing: jnp.ndarray,
        final_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        dt: float = 1.0,
        name: str = "",
        grid_shape: Tuple[int, ...] = None,
        axis_names: List[str] = None,
        coords: Dict[str, List] = None,
        topology_config: Optional[Dict] = None,
    ):
        self.metrics = metrics if isinstance(metrics, MetricsDict) else MetricsDict(metrics)
        self.rng_key = rng_key
        self.z_line = z_line
        self.pCa = pCa
        self.lattice_spacing = lattice_spacing
        self.final_state = final_state
        self.metadata = metadata if metadata is not None else {}
        self.dt = float(dt)
        self.name = str(name)
        self._grid_shape = grid_shape
        self._axis_names = axis_names if axis_names is not None else []
        self.coords = coords if coords is not None else {}
        self.topology_config = topology_config if topology_config is not None else {}

    def tree_flatten(self) -> Tuple[Tuple, Tuple]:
        """Flatten for JAX tree operations."""
        children = (
            self.metrics, self.rng_key,
            self.z_line, self.pCa, self.lattice_spacing,
            self.final_state,
        )
        aux_data = (
            self.metadata, self.dt, self.name,
            self._grid_shape, self._axis_names, self.coords,
            self.topology_config,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple, children: Tuple) -> 'SimulationResult':
        """Reconstruct SimulationResult from flattened representation."""
        (metrics, rng_key, z_line, pCa, lattice_spacing, final_state) = children
        (metadata, dt, name, grid_shape, axis_names, coords,
         topology_config) = aux_data
        return cls(
            metrics=MetricsDict(metrics), rng_key=rng_key,
            z_line=z_line, pCa=pCa, lattice_spacing=lattice_spacing,
            final_state=final_state, metadata=metadata, dt=dt, name=name,
            grid_shape=grid_shape, axis_names=axis_names, coords=coords,
            topology_config=topology_config,
        )

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

    def mean(self) -> 'SimulationResult':
        """Reduce across replicate axis (axis -2)."""
        if self.replicate_axis is None:
            raise ValueError("No replicate axis to reduce.")

        new_metrics = MetricsDict({k: jnp.mean(v, axis=-2) for k, v in self.metrics.items()})
        new_z = jnp.mean(self.z_line, axis=-2)
        new_pCa = jnp.mean(self.pCa, axis=-2)
        new_ls = jnp.mean(self.lattice_spacing, axis=-2)

        new_coords = {k: v for k, v in self.coords.items() if k != 'replicates'}
        new_axis_names = [n for n in self._axis_names if n != 'replicates']
        new_force_shape = new_metrics['axial_force'].shape

        return SimulationResult(
            metrics=new_metrics, rng_key=self.rng_key,
            z_line=new_z, pCa=new_pCa, lattice_spacing=new_ls,
            final_state=None, metadata=self.metadata, dt=self.dt,
            name=self.name + "_mean",
            grid_shape=new_force_shape[:-1] if new_metrics['axial_force'].ndim > 1 else None,
            axis_names=new_axis_names, coords=new_coords,
        )

    def std(self) -> 'SimulationResult':
        """Standard deviation across replicate axis (axis -2)."""
        if self.replicate_axis is None:
            raise ValueError("No replicate axis to reduce.")

        new_metrics = MetricsDict({k: jnp.std(v, axis=-2) for k, v in self.metrics.items()})
        new_z = jnp.std(self.z_line, axis=-2)
        new_pCa = jnp.std(self.pCa, axis=-2)
        new_ls = jnp.std(self.lattice_spacing, axis=-2)

        new_coords = {k: v for k, v in self.coords.items() if k != 'replicates'}
        new_axis_names = [n for n in self._axis_names if n != 'replicates']

        return SimulationResult(
            metrics=new_metrics, rng_key=self.rng_key,
            z_line=new_z, pCa=new_pCa, lattice_spacing=new_ls,
            final_state=None, metadata=self.metadata, dt=self.dt,
            name=self.name + "_std",
            grid_shape=new_metrics['axial_force'].shape[:-1] if new_metrics['axial_force'].ndim > 1 else None,
            axis_names=new_axis_names, coords=new_coords,
        )

    def __getitem__(self, key) -> 'SimulationResult':
        """Slice all tensors identically, return new SimulationResult."""
        new_metrics = MetricsDict({k: v[key] for k, v in self.metrics.items()})
        new_z = self.z_line[key]
        new_pCa = self.pCa[key]
        new_ls = self.lattice_spacing[key]
        new_force = new_metrics['axial_force']

        return SimulationResult(
            metrics=new_metrics, rng_key=self.rng_key,
            z_line=new_z, pCa=new_pCa, lattice_spacing=new_ls,
            final_state=None, metadata=self.metadata, dt=self.dt, name=self.name,
            grid_shape=new_force.shape[:-1] if new_force.ndim > 1 else None,
            axis_names=self._axis_names, coords=self.coords,
        )

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
        idx = [slice(None)] * self.axial_force.ndim
        new_axis_names = list(self._axis_names)
        new_coords = dict(self.coords)

        for axis_name, value in kwargs.items():
            if axis_name not in self.coords:
                raise ValueError(f"Unknown axis '{axis_name}'. Available: {list(self.coords.keys())}")
            coord_list = self.coords[axis_name]
            if value not in coord_list:
                raise ValueError(f"Value {value} not in {axis_name} coords: {coord_list}")
            axis_idx = self._axis_names.index(axis_name)
            val_idx = coord_list.index(value)
            idx[axis_idx] = val_idx

        result = self[tuple(idx)]
        # Remove sliced axes from names/coords
        for axis_name in kwargs:
            if axis_name in new_axis_names:
                new_axis_names.remove(axis_name)
            new_coords.pop(axis_name, None)

        result._axis_names = new_axis_names
        result.coords = new_coords
        return result

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

        stacked_z = jnp.stack([r.z_line for r in results])
        stacked_pCa = jnp.stack([r.pCa for r in results])
        stacked_ls = jnp.stack([r.lattice_spacing for r in results])

        # Stack metrics (includes axial_force and solver_residual)
        metric_keys = list(results[0].metrics.keys())
        stacked_metrics = MetricsDict({
            k: jnp.stack([r.metrics[k] for r in results]) for k in metric_keys
        })

        # Build new axis names and coords
        new_axis_names = [axis_name] + results[0]._axis_names
        new_coords = dict(results[0].coords)
        new_coords[axis_name] = axis_values if axis_values is not None else list(range(len(results)))

        return cls(
            metrics=stacked_metrics,
            rng_key=results[-1].rng_key,
            z_line=stacked_z,
            pCa=stacked_pCa,
            lattice_spacing=stacked_ls,
            final_state=None,
            metadata=results[0].metadata,
            dt=results[0].dt,
            name=results[0].name + "_stacked",
            grid_shape=stacked_metrics['axial_force'].shape[:-1],
            axis_names=new_axis_names,
            coords=new_coords,
        )

    def __repr__(self) -> str:
        force = self.metrics.get('axial_force')
        shape_str = str(force.shape) if force is not None and hasattr(force, 'shape') else '?'
        return f"SimulationResult(name='{self.name}', shape={shape_str}, dt={self.dt})"


# =============================================================================
# BATCH PADDING (avoids recompilation for different sweep sizes)
# =============================================================================

BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def get_bucket_size(actual_size: int) -> int:
    """Round up to the next bucket size for JIT cache reuse.

    A 15x15 sweep (225) and a 12x12 sweep (144) both compile as 256.
    """
    for bucket in BATCH_BUCKETS:
        if bucket >= actual_size:
            return bucket
    return actual_size  # Larger than all buckets — use exact size


# Minibatch heuristic table: (min_padded_batch, chunk_size)
# Benchmarked on RTX 3090, 2x2 lattice, 100ms (see benchmarking/benchmark_minibatch.py).
#   batch=256:   no-minibatch is optimal (monotone degradation as M shrinks)
#   batch=16384: M=4096 is 2.2% faster than no-minibatch (L2 cache fit)
# Crossover between 256 and 16384 is not yet benchmarked — update this table
# after running: python benchmarking/benchmark_minibatch.py --total_runs <N> --min_m 256
#
# Memory note (relevant for 8 GB GPUs, e.g. RTX 4060):
#   Peak VRAM (GB) ≈ minibatch_size × n_steps × 45 × 4 bytes × 2 / 1e9
#   Example: minibatch_size=4096, n_steps=1000 → ≈ 1.5 GB for metrics alone; ~3 GB total.
#   If you hit OOM at large batch, reduce minibatch_size or pass it explicitly.
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
# MODULE-LEVEL SIMULATION KERNEL
# =============================================================================

@partial(jax.jit, static_argnames=['dt', 'unroll', 'is_dynamic_ls', 'n_cg_steps', 'n_newton_steps'])
def _run_sim_kernel(
    topology: SarcTopology,
    batched_params: DynamicParams,
    z_batched: jnp.ndarray,
    pCa_batched: jnp.ndarray,
    ls_batched: jnp.ndarray,
    rng_keys: jnp.ndarray,
    dt: float,
    unroll: int,
    is_dynamic_ls: bool = False,
    K_lat_batched: jnp.ndarray = None,
    nu_batched: jnp.ndarray = None,
    n_cg_steps: int = 1,
    n_newton_steps: int = 16,
):
    """JIT-compiled simulation kernel (unified fixed + dynamic LS).

    Always computes all metrics via compute_all_metrics(). No metrics/manifest
    in static_argnames — changing metric selection never triggers recompilation.

    Args:
        topology: SarcTopology with pre-computed index maps (broadcast via closure)
        batched_params: DynamicParams with batch dimension
        z_batched: (batch, time) z-line values
        pCa_batched: (batch, time) pCa values
        ls_batched: (batch, time) lattice spacing values
        rng_keys: (batch,) RNG keys
        dt: Timestep in ms (static)
        unroll: Scan unrolling factor (static)
        is_dynamic_ls: If True, solve lattice spacing as a DOF (static)
        K_lat_batched: (batch,) per-sim lattice stiffness (ignored if not is_dynamic_ls)
        nu_batched: (batch,) per-sim Poisson exponent (ignored if not is_dynamic_ls)

    Returns:
        MetricsDict with all metric scalars, shape (batch, time).
        Includes 'axial_force', 'solver_residual', and 'newton_iters' keys.
    """

    def create_and_equilibrate(constants, z0, pCa0, ls0):
        """Create state from topology + constants and solve equilibrium."""
        state = realize_state(topology, constants, z0, pCa0, ls0)
        state = update_nearest_neighbors(state, constants, topology)
        state, _residual, _, _ = solve_equilibrium(
            state, constants, topology,
            n_cg_steps=n_cg_steps,
            n_newton_steps=n_newton_steps,
        )
        return state

    def run_single_sim(state, constants, key, z_trace, pCa_trace, ls_trace, K_lat_val, nu_val):
        """Run simulation with scan inside vmap."""
        n_thick, n_crowns = state.thick.axial.shape
        n_thin, n_sites = state.thin.axial.shape
        precond_params = build_preconditioner_params(
            n_thick, n_crowns, n_thin, n_sites,
            constants.thick_k, constants.thin_k,
        )
        prefactored_precond = build_prefactored_preconditioner(precond_params)

        delta_z = jnp.concatenate([jnp.zeros(1), jnp.diff(z_trace)])
        l0 = z_trace[0]  # reference z for Poisson scaling

        def scan_fn(carry, inputs):
            old_state, k, current_ls = carry
            z_val, pCa_val, ls_val, dz = inputs

            old_state = old_state._replace(
                thin=old_state.thin._replace(
                    axial=old_state.thin.axial + dz
                )
            )

            pre_solve_thick_pos = old_state.thick.axial

            if is_dynamic_ls:
                drivers = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=current_ls)
                d_ref = ls_val * (l0 / z_val) ** nu_val
            else:
                drivers = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=ls_val)
                d_ref = None

            new_state, new_k, solver_residual, new_ls, n_iters = timestep(
                old_state, constants, drivers, topology, k, dt=dt,
                K_lat=K_lat_val if is_dynamic_ls else None,
                d_ref=d_ref,
                n_cg_steps=n_cg_steps,
                n_newton_steps=n_newton_steps,
                precond_params=precond_params,
                prefactored_precond=prefactored_precond,
            )

            # Build metrics with emergent new_ls for correct force computation
            constants_for_metrics = constants.with_drivers(pCa_val, z_val, new_ls)
            drivers_for_metrics = Drivers(pCa=pCa_val, z_line=z_val, lattice_spacing=new_ls)
            force = axial_force_at_mline(new_state, constants_for_metrics, topology)

            all_metrics = compute_all_metrics(
                old_state, new_state, constants_for_metrics, drivers_for_metrics,
                topology, pre_solve_thick_pos, force, solver_residual, n_iters, dt,
            )

            return (new_state, new_k, new_ls), all_metrics

        _, metrics_out = jax.lax.scan(
            scan_fn,
            (state, key, ls_trace[0]),
            (z_trace, pCa_trace, ls_trace, delta_z),
            unroll=unroll,
        )
        return metrics_out

    batched_states = jax.vmap(create_and_equilibrate, in_axes=(0, 0, 0, 0))(
        batched_params,
        z_batched[:, 0], pCa_batched[:, 0], ls_batched[:, 0],
    )

    batched_metrics = jax.vmap(
        run_single_sim, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    )(
        batched_states, batched_params, rng_keys,
        z_batched, pCa_batched, ls_batched, K_lat_batched, nu_batched,
    )

    return batched_metrics


# =============================================================================
# TOP-LEVEL run() API
# =============================================================================

def run(
    topology: SarcTopology,
    duration_ms: float = 1000.0,
    dt: float = 1.0,
    pCa: Union[float, List[float], jnp.ndarray] = 4.5,
    z_line: Union[float, List[float], jnp.ndarray] = 900.0,
    lattice_spacing: Union[float, List[float], jnp.ndarray] = 14.0,
    K_lat: Union[float, List[float], None] = None,
    nu: Union[float, List[float]] = 0.0,
    dynamic_params: Union[DynamicParams, Dict[str, Union[float, List[float]]]] = None,
    static_params: 'StaticParams' = None,
    replicates: int = 1,
    rng_seed: int = 0,
    unroll: int = 1,
    minibatch_size: Optional[int] = "auto",
    verbose: bool = False,
) -> SimulationResult:
    """Run a muscle simulation with the given topology.

    This is the primary API. Accepts a pre-constructed SarcTopology
    (topology defines structural configuration; changing it requires
    recompilation). All other parameters are sweepable without recompile.

    Batch padding: sweep sizes are rounded up to the nearest bucket
    (1, 2, 4, ..., 16384), so a 225-run sweep and a 256-run sweep
    share the same compiled kernel.

    Args:
        topology: Pre-constructed SarcTopology (from SarcTopology.create())
        duration_ms: Simulation duration in milliseconds
        dt: Timestep in milliseconds
        pCa: Calcium as -log10([Ca]) -- float, list (sweep), or array (trace)
        z_line: Z-line position (nm) -- float, list (sweep), or array (trace)
        lattice_spacing: Lattice spacing (nm) -- float, list, or array
        K_lat: Lattice stiffness per thick filament (pN/nm). None = fixed LS.
               Float or list (sweep). Internally scaled by n_thick.
        nu: Poisson exponent for d_ref(z) = d0*(z0/z)^nu. Float or list (sweep).
            If K_lat is None and nu>0, pre-computes Poisson LS trace.
        dynamic_params: DynamicParams, dict of overrides/sweeps, or list of DynamicParams.
                        A list creates a 'candidates' sweep axis (one element per candidate),
                        Cartesian-producted with other sweep axes. Useful for batching CMA-ES
                        population evaluations: run(topo, z_line=traces, dynamic_params=[dp0..dpN])
                        gives result shape (N, n_freq, replicates, time).
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

    Returns:
        SimulationResult with shape (sweep_1, ..., replicates, time)

    Example:
        from multifil_jax.core.sarc_geometry import SarcTopology
        from multifil_jax.core.params import get_skeletal_params, StaticParams

        static, dynamic = get_skeletal_params()
        topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)

        result = run(topo, pCa=4.5, z_line=900.0, duration_ms=100)
        print(result.summary())
    """
    n_steps = int(duration_ms / dt)
    topology = jax.device_put(topology)

    if static_params is None:
        from multifil_jax.core.params import StaticParams as _StaticParams
        static_params = _StaticParams()

    # Resolve base dynamic params
    if dynamic_params is None:
        base_dynamic = DynamicParams()
    elif isinstance(dynamic_params, DynamicParams):
        base_dynamic = dynamic_params
    elif isinstance(dynamic_params, dict):
        base_dynamic = DynamicParams()
    elif isinstance(dynamic_params, list):
        base_dynamic = DynamicParams()
    else:
        raise ValueError(f"dynamic_params must be DynamicParams, dict, or list of DynamicParams, got {type(dynamic_params)}")

    # Mode selection
    is_dynamic_ls = K_lat is not None

    # Build sweep grid
    sweep_axes = []
    scalar_values = {}
    trace_values = {}
    trace_sweep_stacks = {}  # name -> (n_cond, n_steps) for list-of-trace sweeps
    param_sweep_names = set()

    for name, value in [('z_line', z_line), ('pCa', pCa), ('lattice_spacing', lattice_spacing)]:
        if isinstance(value, list):
            # Detect list-of-traces: each element is a 1D array of length n_steps
            first = value[0] if value else None
            if (first is not None
                    and hasattr(first, '__len__')
                    and not isinstance(first, (int, float))
                    and np.asarray(first).ndim == 1
                    and len(first) == n_steps):
                # Store stacked traces; use integer indices through meshgrid
                trace_sweep_stacks[name] = jnp.array(
                    np.stack([np.asarray(v) for v in value])
                )  # (n_cond, n_steps)
                sweep_axes.append((name, list(range(len(value)))))
            else:
                sweep_axes.append((name, value))
        elif hasattr(value, 'shape') and hasattr(value, 'ndim') and value.ndim > 0:
            if value.shape[0] != n_steps:
                raise ValueError(f"{name} array length {value.shape[0]} != n_steps {n_steps}")
            trace_values[name] = jnp.asarray(value)
        else:
            scalar_values[name] = float(value)

    # K_lat and nu sweep classification (scalars or lists, never time-series)
    K_lat_sweep_values = None
    nu_sweep_values = None
    if isinstance(K_lat, list):
        sweep_axes.append(('K_lat', K_lat))
        K_lat_sweep_values = K_lat
    if isinstance(nu, list):
        sweep_axes.append(('nu', nu))
        nu_sweep_values = nu

    if isinstance(dynamic_params, dict):
        for param_name, param_values in dynamic_params.items():
            if isinstance(param_values, list):
                sweep_axes.append((param_name, param_values))
                param_sweep_names.add(param_name)
    elif isinstance(dynamic_params, DynamicParams):
        for name in DYNAMIC_FIELDS:
            val = getattr(dynamic_params, name)
            arr = jnp.asarray(val)
            if arr.ndim > 0 and arr.shape[0] > 1:
                sweep_axes.append((name, list(arr.tolist())))
                param_sweep_names.add(name)
    elif isinstance(dynamic_params, list):
        # List of DynamicParams = one coupled candidate per element.
        # Creates a single 'candidates' axis; Cartesian-producted with any other
        # sweep axes (e.g. z_line frequency traces) to give candidate × freq grid.
        sweep_axes.append(('candidates', list(range(len(dynamic_params)))))

    # Build Cartesian product
    if sweep_axes:
        param_arrays = [jnp.array(vals) for _, vals in sweep_axes]
        grids = jnp.meshgrid(*param_arrays, indexing='ij')
        batch_size = grids[0].size
        flat_grids = {name: g.flatten() for (name, _), g in zip(sweep_axes, grids)}
    else:
        batch_size = 1
        flat_grids = {}

    total_batch = batch_size * replicates
    grid_shape = tuple(len(vals) for _, vals in sweep_axes)

    axis_names = [name for name, _ in sweep_axes] + ['replicates', 'time']
    coords = {}
    for name, vals in sweep_axes:
        if name in trace_sweep_stacks:
            # Store actual traces, not the integer indices used internally
            coords[name] = [np.array(trace_sweep_stacks[name][i]) for i in range(len(vals))]
        else:
            coords[name] = list(vals)
    coords['replicates'] = list(range(replicates))
    coords['time'] = (jnp.arange(n_steps) * dt).tolist()

    if verbose:
        print(f"Grid shape: {grid_shape}, axes: {axis_names}, "
              f"batch_size: {batch_size}, total: {total_batch}")
        if is_dynamic_ls:
            print(f"Dynamic LS: K_lat={K_lat}, nu={nu}")

    # Tile for replicates
    flat_grids_tiled = {name: jnp.repeat(arr, replicates) for name, arr in flat_grids.items()}

    # Broadcast waveforms to (total_batch, n_steps)
    def broadcast_waveform(name, default_val):
        if name in flat_grids_tiled:
            if name in trace_sweep_stacks:
                # Integer indices into stacked traces → (total_batch, n_steps)
                indices = flat_grids_tiled[name].astype(int)
                return trace_sweep_stacks[name][indices]
            return jnp.broadcast_to(flat_grids_tiled[name][:, None], (total_batch, n_steps))
        elif name in trace_values:
            return jnp.broadcast_to(trace_values[name][None, :], (total_batch, n_steps))
        elif name in scalar_values:
            return jnp.full((total_batch, n_steps), scalar_values[name])
        else:
            return jnp.full((total_batch, n_steps), default_val)

    z_batched = broadcast_waveform('z_line', 900.0)
    pCa_batched = broadcast_waveform('pCa', 4.5)
    ls_batched = broadcast_waveform('lattice_spacing', 14.0)

    # Build K_lat_batched and nu_batched (shape: (total_batch,))
    if K_lat_sweep_values is not None:
        K_lat_batched = flat_grids_tiled['K_lat']
    elif is_dynamic_ls:
        K_lat_batched = jnp.full(total_batch, float(K_lat))
    else:
        K_lat_batched = jnp.zeros(total_batch)  # dummy, ignored at trace time

    if nu_sweep_values is not None:
        nu_batched = flat_grids_tiled['nu']
    elif isinstance(nu, (int, float)):
        nu_batched = jnp.full(total_batch, float(nu))
    else:
        nu_batched = jnp.zeros(total_batch)

    # K_lat × n_thick scaling: per-filament → effective lattice stiffness
    if is_dynamic_ls:
        K_lat_batched = K_lat_batched * topology.n_thick

    # Poisson pre-computation (K_lat=None, nu>0): convert to LS time-series
    if not is_dynamic_ls and 'lattice_spacing' not in trace_values:
        # Check if any nu value is non-zero
        has_nonzero_nu = False
        if isinstance(nu, list):
            has_nonzero_nu = any(v != 0.0 for v in nu)
        elif isinstance(nu, (int, float)):
            has_nonzero_nu = nu != 0.0

        if has_nonzero_nu:
            # d0 is the per-sim reference lattice spacing
            if 'lattice_spacing' in flat_grids_tiled:
                d0_batched = flat_grids_tiled['lattice_spacing'][:, None]  # (batch, 1)
            else:
                d0_val = scalar_values.get('lattice_spacing', 14.0)
                d0_batched = jnp.full((total_batch, 1), d0_val)

            l0 = z_batched[:, 0:1]  # (batch, 1) - reference z
            nu_col = nu_batched[:, None]  # (batch, 1)
            ls_batched = d0_batched * (l0 / z_batched) ** nu_col  # (batch, n_steps)

    # Build batched DynamicParams
    batched_param_kwargs = {}
    if isinstance(dynamic_params, list):
        # Index into the candidate list using the 'candidates' flat grid.
        # .tolist() does ONE device-to-host transfer (vs 49 fields × 48 elements
        # = 2352 syncs if we iterate over a JAX array element-by-element).
        cand_idx_list = flat_grids_tiled['candidates'].astype(int).tolist()
        for name in DYNAMIC_FIELDS:
            batched_param_kwargs[name] = jnp.array([
                float(getattr(dynamic_params[i], name)) for i in cand_idx_list
            ])
    else:
        for name in DYNAMIC_FIELDS:
            if name in param_sweep_names and name in flat_grids_tiled:
                batched_param_kwargs[name] = flat_grids_tiled[name]
            elif isinstance(dynamic_params, dict) and name in dynamic_params:
                val = dynamic_params[name]
                if not isinstance(val, list):
                    batched_param_kwargs[name] = jnp.full(total_batch, float(val))
                else:
                    batched_param_kwargs[name] = flat_grids_tiled[name]
            else:
                batched_param_kwargs[name] = jnp.full(total_batch, float(getattr(base_dynamic, name)))
    batched_params = DynamicParams(**batched_param_kwargs)

    # Generate unique RNG keys
    rng_keys = jax.random.split(jax.random.PRNGKey(rng_seed), total_batch)

    # Pad batch to bucket size
    padded_batch = get_bucket_size(total_batch)
    if padded_batch > total_batch:
        pad_n = padded_batch - total_batch
        z_batched = jnp.concatenate([z_batched, jnp.broadcast_to(z_batched[:1], (pad_n, n_steps))])
        pCa_batched = jnp.concatenate([pCa_batched, jnp.broadcast_to(pCa_batched[:1], (pad_n, n_steps))])
        ls_batched = jnp.concatenate([ls_batched, jnp.broadcast_to(ls_batched[:1], (pad_n, n_steps))])
        K_lat_batched = jnp.concatenate([K_lat_batched, jnp.broadcast_to(K_lat_batched[:1], (pad_n,))])
        nu_batched = jnp.concatenate([nu_batched, jnp.broadcast_to(nu_batched[:1], (pad_n,))])
        rng_keys = jnp.concatenate([rng_keys, jax.random.split(jax.random.PRNGKey(rng_seed + 1), pad_n)])
        pad_kwargs = {}
        for name in DYNAMIC_FIELDS:
            val = getattr(batched_params, name)
            pad_kwargs[name] = jnp.concatenate([val, jnp.broadcast_to(val[:1], (pad_n,))])
        batched_params = DynamicParams(**pad_kwargs)

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
    )

    if not use_minibatch:
        batched_metrics = _run_sim_kernel(
            topology=topology,
            batched_params=batched_params,
            z_batched=z_batched,
            pCa_batched=pCa_batched,
            ls_batched=ls_batched,
            rng_keys=rng_keys,
            K_lat_batched=K_lat_batched,
            nu_batched=nu_batched,
            **kernel_kwargs,
        )
    else:
        if verbose:
            print(f"Minibatching: {padded_batch // resolved_minibatch} chunks of {resolved_minibatch}")
        chunks = []
        for start in range(0, padded_batch, resolved_minibatch):
            end = start + resolved_minibatch
            chunks.append(_run_sim_kernel(
                topology=topology,
                batched_params=jax.tree_util.tree_map(lambda x: x[start:end], batched_params),
                z_batched=z_batched[start:end],
                pCa_batched=pCa_batched[start:end],
                ls_batched=ls_batched[start:end],
                rng_keys=rng_keys[start:end],
                K_lat_batched=K_lat_batched[start:end],
                nu_batched=nu_batched[start:end],
                **kernel_kwargs,
            ))
        batched_metrics = MetricsDict({
            k: jnp.concatenate([c[k] for c in chunks], axis=0)
            for k in chunks[0]
        })

    # Slice back to actual batch size
    if padded_batch > total_batch:
        batched_metrics = MetricsDict({k: v[:total_batch] for k, v in batched_metrics.items()})
        z_batched = z_batched[:total_batch]
        pCa_batched = pCa_batched[:total_batch]
        ls_batched = ls_batched[:total_batch]

    # Reshape to data cube
    final_shape = grid_shape + (replicates, n_steps)
    reshaped_metrics = MetricsDict({key: val.reshape(final_shape) for key, val in batched_metrics.items()})
    reshaped_z = z_batched.reshape(final_shape)
    reshaped_pCa = pCa_batched.reshape(final_shape)
    reshaped_ls = ls_batched.reshape(final_shape)

    # Post-run solver residual validation
    max_residual = float(jnp.max(reshaped_metrics['solver_residual']))
    tol = StaticParams().solver_residual_tol
    if max_residual > tol:
        import warnings
        warnings.warn(
            f"Solver max residual {max_residual:.2f} pN exceeds "
            f"threshold {tol} pN (set via StaticParams.solver_residual_tol)"
        )

    if verbose:
        print(f"Result shape: {reshaped_metrics['axial_force'].shape}")
        print(f"Max solver residual: {max_residual:.4f} pN")

    topology_config = {
        'n_thick': topology.n_thick,
        'n_crowns': topology.n_crowns,
        'n_thin': topology.n_thin,
        'n_sites': topology.n_sites,
        'n_titin': topology.n_titin,
        'total_xbs': topology.total_xbs,
        'n_faces_per_thin': topology.n_faces_per_thin,
    }
    if is_dynamic_ls:
        topology_config['K_lat'] = K_lat
        topology_config['K_lat_eff'] = float(K_lat * topology.n_thick) if not isinstance(K_lat, list) else [float(k * topology.n_thick) for k in K_lat]
        topology_config['nu'] = nu

    metadata = {
        'grid_shape': grid_shape,
        'axis_names': axis_names,
        'master_seed': rng_seed,
    }
    if is_dynamic_ls:
        metadata['K_lat'] = K_lat
        metadata['nu'] = nu

    return SimulationResult(
        metrics=reshaped_metrics,
        rng_key=rng_keys[-1],
        z_line=reshaped_z,
        pCa=reshaped_pCa,
        lattice_spacing=reshaped_ls,
        final_state=None,
        metadata=metadata,
        dt=dt,
        name="run",
        grid_shape=grid_shape,
        axis_names=axis_names,
        coords=coords,
        topology_config=topology_config,
    )

