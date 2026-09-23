"""
One timestep of the half-sarcomere simulation.

A timestep has two halves, and they are separated deliberately:

    KINETICS   stochastic — tropomyosin sites and myosin heads draw new states
    MECHANICS  deterministic — the lattice is solved back to force balance

That ordering matters. Chemistry is evaluated at the CURRENT geometry, then the
mechanics catch up to the new chemistry. Doing it the other way would let heads
bind at positions the lattice has not actually reached.

The split is also why kinetics_step() is a public function rather than an
implementation detail. In a multi-sarcomere model the chemistry of every
sarcomere is independent and can be advanced separately, but the mechanics are
coupled — the sarcomeres share filaments and must be equilibrated together. That
requires running all the kinetics first, then one joint solve, which this
structure permits without modification.

STEP ORDER

    1. update nearest sites   recompute each head's target and its strain, since
                              the lattice moved during the previous solve
    2. thin transitions       tropomyosin sites sample new states, coupled to
                              their chain neighbours
    3. thick transitions      heads sample new states; binding bookkeeping is
                              updated on both filaments
    --- kinetics_step() returns here, with a KineticsTrace of what it saw ---
    4. solve equilibrium      Newton-CG until net force on every node vanishes

The drivers (pCa, z_line, lattice_spacing) arrive as one Drivers bundle and are
handed to each kernel as the explicit scalars it uses. In the kinetics phase
z_line is read only to place the thin sites for the nearest-site search. Ising
cooperativity takes its neighbour information straight from tm_states, so
nothing here depends on filament tension — which is what lets a multi-sarcomere extension run every
sarcomere's chemistry independently before one joint solve.

A NOTE ON Z-LINE CHANGES. Thin displacements are measured from a frame anchored
on the Z-disc, so when z_line moves between steps the carried-over displacements
place the whole thin filament rigidly translated with it — no explicit shift, and
the known rigid-body part is never left for the solver to discover. Kinetics then
runs on that rigidly translated configuration, so this step's binding search and
strain-dependent rates see the translated XB strains. Only the solve that follows
lets the thin filament stretch under its bound crossbridges and move off the
Z-disc's rigid motion.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, TYPE_CHECKING

from multifil_jax.kernels.geometry import update_nearest_neighbors
from multifil_jax.kernels.transitions import (thin_transitions, thick_transitions,
                                              xb_binned_generator)
from multifil_jax.kernels.solver import solve_equilibrium
from multifil_jax.core.state import Drivers, KineticsTrace

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


# ============================================================================
# KINETICS PHASE
# ============================================================================

def kinetics_step(state: 'State',
                  constants: 'DynamicParams',
                  drivers: Drivers,
                  topology: 'SarcTopology',
                  rng_key: jnp.ndarray,
                  *,
                  dt: float,
                  xb_subpop=None,
                  tm_subpop=None) -> Tuple['State', jnp.ndarray, KineticsTrace]:
    """Run the stochastic half of a timestep: everything except the force solve.

    Updates crossbridge geometry, then samples new
    tropomyosin and crossbridge states. Leaves the lattice out of equilibrium —
    the caller is responsible for solving it, either immediately (timestep()) or
    after gathering several coupled sarcomeres.

    Args:
        state: Current State NamedTuple (pure state, no embedded params)
        constants: DynamicParams/Constants with physics values
        drivers: Drivers NamedTuple with this step's pCa/z_line/lattice_spacing
        topology: SarcTopology for indexing
        rng_key: JAX random key for stochastic transitions
        dt: Timestep size (ms) -- keyword-only, JIT static

    Returns:
        (state_after_kinetics, new_rng_key, trace)

        trace is a KineticsTrace: the MID state (after thin_transitions, before
        thick_transitions), the drivers it ran at, the subpopulation tuple,
        and the closure-tear mask. Everything a metric
        needs to describe the step that actually happened, gathered once here
        rather than re-derived — see core/state.KineticsTrace.

        trace.drivers carries the pre-solve drivers. A coupled solver may
        substitute a different z_line before equilibrating, via
        trace.drivers._replace(z_line=...).

        THE TRACE IS A WITHIN-STEP VALUE. It holds a whole State, so it must
        never be put in a scan carry or a scan output.
    """
    # Step 1: Update nearest binding sites using topology
    state = update_nearest_neighbors(state, topology, drivers.z_line,
                                     drivers.lattice_spacing)

    # Step 2: Thin filament transitions. Neighbour states are counted inside
    # thin_transitions from the current tm_states, so there is nothing to
    # precompute here.
    rng_key, thin_key = jax.random.split(rng_key)
    state, _P_thin, torn = thin_transitions(
        state, constants, topology, drivers.pCa, thin_key, dt, tm_subpop=tm_subpop)

    # Step 3: build the crossbridge generator and exponentiate it — ONCE.
    # `state` is at this instant exactly what the generator must be built from.
    # Both consumers read these same bins: thick_transitions samples from
    # bins.P, and the ATP metrics read expected crossing counts out of bins.G.
    # Until 2026-09-11 metrics_fn took a SECOND exponential of its own, of a
    # different (absorbing) generator; the exact estimator needs no such
    # generator, so the duplication went with it.
    bins = xb_binned_generator(state, constants, topology, drivers.pCa,
                               drivers.lattice_spacing, dt, xb_subpop=xb_subpop)

    # Capture the trace HERE, between the two transition calls, because `state`
    # and `bins` together are exactly the step thick_transitions is about to
    # take. A metric that describes that step has to read these, or it describes
    # a step that never happened.
    trace = KineticsTrace(state=state, drivers=drivers,
                          xb_subpop=xb_subpop, torn=torn, xb_bins=bins)

    # Step 4: Thick filament transitions
    rng_key, thick_key = jax.random.split(rng_key)
    state = thick_transitions(state, bins, topology, thick_key)

    return state, rng_key, trace


# ============================================================================
# MAIN TIMESTEP FUNCTION
# ============================================================================

def timestep(state: 'State',
             constants: 'DynamicParams',
             drivers: Drivers,
             topology: 'SarcTopology',
             rng_key: jnp.ndarray,
             *,
             dt: float,
             K_lat=None,
             d_ref=None,
             n_cg_steps: int,
             n_newton_steps: int,
             precond_params=None,
             prefactored_precond=None,
             xb_subpop=None,
             tm_subpop=None) -> Tuple['State', jnp.ndarray, jnp.ndarray, float,
                                       int, KineticsTrace, jnp.ndarray,
                                       jnp.ndarray]:
    """Execute one timestep of the half-sarcomere simulation.

    Tiered Architecture:
        state: Pure simulation state (Tier 0)
        constants: Physics parameters (Tier 2)
        drivers: This step's pCa/z_line/lattice_spacing (Tier 3)
        topology: Structural index data (Tier 1)

    Args:
        state: Current State NamedTuple (pure state, no embedded params)
        constants: DynamicParams/Constants with physics values
        drivers: Drivers NamedTuple with this step's pCa/z_line/lattice_spacing
        topology: SarcTopology for indexing
        rng_key: JAX random key for stochastic transitions
        dt: Timestep size (ms) -- keyword-only, JIT static
        K_lat: Lattice stiffness (pN/nm). None = fixed LS mode.
        d_ref: Poisson-scaled reference lattice spacing (nm). Required if K_lat is not None.
        precond_params: Pre-built PreconditionerParams (optional)
        prefactored_precond: Pre-factored Thomas data (optional)

    Returns:
        new_state: State after both kinetics and equilibration
        new_rng_key: Advanced RNG key
        solver_residual: Largest remaining net force on any node (pN). Raw, so
            it carries units and no scale. Read it against solver_tolerance, or
            read solver_residual_norm instead.
        new_ls: Lattice spacing used. The solved value in dynamic mode, the
            prescribed one otherwise.
        n_iters: Newton iterations taken, useful for spotting configurations
            where the solve is struggling.
        trace: KineticsTrace for this step — the mid state, the pre-solve
            drivers, the subpopulation tuple and the closure-tear mask. Feed it straight to compute_all_metrics. Within-step only;
            never carry it through a scan. See kinetics_step.
        solver_residual_norm: max(|F| / tol_vec), dimensionless. <= 1 means
            converged, in BOTH lattice-spacing modes.
        solver_tolerance: the axial convergence tolerance actually used (pN).
    """
    state, rng_key, trace = kinetics_step(
        state, constants, drivers, topology, rng_key, dt=dt,
        xb_subpop=xb_subpop, tm_subpop=tm_subpop,
    )

    (new_state, solver_residual, new_ls, n_iters,
     residual_norm, tol_axial) = solve_equilibrium(
        state, constants, topology, drivers.z_line, drivers.lattice_spacing,
        K_lat=K_lat, d_ref=d_ref,
        n_cg_steps=n_cg_steps,
        n_newton_steps=n_newton_steps,
        precond_params=precond_params,
        prefactored_precond=prefactored_precond,
    )

    return (new_state, rng_key, solver_residual, new_ls, n_iters, trace,
            residual_norm, tol_axial)
