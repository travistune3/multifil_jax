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

    1. resolve drivers        merge per-step pCa / z_line / lattice_spacing over
                              the constant defaults
    2. update nearest sites   recompute each head's target and its strain, since
                              the lattice moved during the previous solve
    3. thin transitions       tropomyosin sites sample new states, coupled to
                              their chain neighbours
    4. thick transitions      heads sample new states; binding bookkeeping is
                              updated on both filaments
    --- kinetics_step() returns here ---
    5. solve equilibrium      Newton-CG until net force on every node vanishes

Note that no step of the kinetics phase reads z_line. Ising cooperativity takes
its neighbour information straight from tm_states, so nothing here depends on
filament tension — which is what lets a multi-sarcomere extension run every
sarcomere's chemistry independently before one joint solve.

A NOTE ON Z-LINE CHANGES. When z_line moves between steps, the thin filament
node positions are shifted before this function is called (see simulation.py's
scan body), not left for the solver to discover. Handing the solver a lattice
where the boundary has jumped but nothing else has moved would make it resolve
the entire imposed displacement in one Newton solve, which is both slower and
less accurate than applying the known rigid-body part analytically first.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, TYPE_CHECKING

from multifil_jax.kernels.geometry import update_nearest_neighbors
from multifil_jax.kernels.transitions import thin_transitions, thick_transitions
from multifil_jax.kernels.solver import solve_equilibrium
from multifil_jax.core.state import Drivers, resolve_value

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
                  tm_subpop=None) -> Tuple['State', jnp.ndarray, 'DynamicParams']:
    """Run the stochastic half of a timestep: everything except the force solve.

    Resolves the drivers, updates crossbridge geometry, then samples new
    tropomyosin and crossbridge states. Leaves the lattice out of equilibrium —
    the caller is responsible for solving it, either immediately (timestep()) or
    after gathering several coupled sarcomeres.

    Args:
        state: Current State NamedTuple (pure state, no embedded params)
        constants: DynamicParams/Constants with physics values
        drivers: Drivers NamedTuple with per-step pCa/z_line/ls (NaN = use constant)
        topology: SarcTopology for indexing
        rng_key: JAX random key for stochastic transitions
        dt: Timestep size (ms) -- keyword-only, JIT static

    Returns:
        (state_after_kinetics, new_rng_key, resolved_constants)

        resolved_constants carries the driver values baked in, so callers do not
        have to redo the NaN-merge. A coupled solver may substitute a different
        z_line before equilibrating, via constants.with_drivers(...).
    """
    # Step 0: Resolve drivers -- merge time-varying overrides with constants
    pCa = resolve_value(drivers.pCa, constants.pCa)
    z_line = resolve_value(drivers.z_line, constants.z_line)
    lattice_spacing = resolve_value(drivers.lattice_spacing, constants.lattice_spacing)

    # Build a resolved constants with current driver values
    resolved_constants = constants.with_drivers(pCa, z_line, lattice_spacing)

    # Resolve subpopulation constants_k with the same drivers (rate scales are
    # already baked in; with_drivers only sets pCa/z_line/lattice_spacing).
    def _resolve_subpop(sp):
        if sp is None:
            return None
        mode, constants_k, extra = sp
        return (mode, [ck.with_drivers(pCa, z_line, lattice_spacing) for ck in constants_k], extra)
    xb_subpop_r = _resolve_subpop(xb_subpop)
    tm_subpop_r = _resolve_subpop(tm_subpop)

    # Step 1: Update nearest binding sites using topology
    state = update_nearest_neighbors(state, resolved_constants, topology)

    # Step 2: Thin filament transitions. Neighbour states are counted inside
    # thin_transitions from the current tm_states, so there is nothing to
    # precompute here.
    rng_key, thin_key = jax.random.split(rng_key)
    state, _P_thin = thin_transitions(state, resolved_constants, topology, thin_key, dt,
                                      tm_subpop=tm_subpop_r)

    # Step 3: Thick filament transitions
    rng_key, thick_key = jax.random.split(rng_key)
    state = thick_transitions(state, resolved_constants, topology, thick_key, dt,
                              xb_subpop=xb_subpop_r)

    return state, rng_key, resolved_constants


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
             solver_tol: Optional[float] = None,
             n_cg_steps: int = 6,
             n_newton_steps: int = 16,
             precond_params=None,
             prefactored_precond=None,
             xb_subpop=None,
             tm_subpop=None) -> Tuple['State', jnp.ndarray, jnp.ndarray, float, int]:
    """Execute one timestep of the half-sarcomere simulation.

    Tiered Architecture:
        state: Pure simulation state (Tier 0)
        constants: Physics parameters including default pCa/z_line/ls (Tier 2)
        drivers: Time-varying overrides for pCa/z_line/ls (Tier 3)
        topology: Structural index data (Tier 1)

    Args:
        state: Current State NamedTuple (pure state, no embedded params)
        constants: DynamicParams/Constants with physics values
        drivers: Drivers NamedTuple with per-step pCa/z_line/ls (NaN = use constant)
        topology: SarcTopology for indexing
        rng_key: JAX random key for stochastic transitions
        dt: Timestep size (ms) -- keyword-only, JIT static
        K_lat: Lattice stiffness (pN/nm). None = fixed LS mode.
        d_ref: Poisson-scaled reference lattice spacing (nm). Required if K_lat is not None.
        solver_tol: Convergence tolerance (pN). If None, uses constants.solver_tol
        precond_params: Pre-built PreconditionerParams (optional)
        prefactored_precond: Pre-factored Thomas data (optional)

    Returns:
        new_state: State after both kinetics and equilibration
        new_rng_key: Advanced RNG key
        solver_residual: Largest remaining net force on any node (pN). Worth
            checking — a large value means the reported force is not an
            equilibrium force.
        new_ls: Lattice spacing used. The solved value in dynamic mode, the
            prescribed one otherwise.
        n_iters: Newton iterations taken, useful for spotting configurations
            where the solve is struggling.
    """
    state, rng_key, resolved_constants = kinetics_step(
        state, constants, drivers, topology, rng_key, dt=dt,
        xb_subpop=xb_subpop, tm_subpop=tm_subpop,
    )

    new_state, solver_residual, new_ls, n_iters = solve_equilibrium(
        state, resolved_constants, topology,
        K_lat=K_lat, d_ref=d_ref,
        tolerance=solver_tol,
        n_cg_steps=n_cg_steps,
        n_newton_steps=n_newton_steps,
        precond_params=precond_params,
        prefactored_precond=prefactored_precond,
    )

    return new_state, rng_key, solver_residual, new_ls, n_iters
