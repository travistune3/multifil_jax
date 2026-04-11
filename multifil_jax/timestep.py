"""
Single Timestep Module for JAX Half-Sarcomere

Tiered Architecture:
    kernel(state, constants, drivers, topology, rng_key, *, dt)

Main functions:
    kinetics_step() - Execute the stochastic kinetics phase only (steps 1-6)
    timestep()      - Execute one full timestep (kinetics + equilibrium solve)

Workflow:
    1. Resolve drivers (pCa, z_line, lattice_spacing)
    2. Calculate INTERNAL thin filament forces (for cooperativity)
    3. Update cooperativity based on internal filament tension
    4. Update nearest binding sites for crossbridges
    5. Thin filament transitions (tropomyosin states)
    6. Thick filament transitions (crossbridge states + binding)
    --- kinetics_step() returns here ---
    7. Solve for new equilibrium positions (Newton solver)
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, TYPE_CHECKING

from multifil_jax.kernels.cooperativity import update_cooperativity
from multifil_jax.kernels.geometry import update_nearest_neighbors
from multifil_jax.kernels.transitions import thin_transitions, thick_transitions
from multifil_jax.kernels.solver import solve_equilibrium
from multifil_jax.kernels.forces import (
    calculate_thin_forces_for_cooperativity,
)
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
                  dt: float) -> Tuple['State', jnp.ndarray, 'DynamicParams']:
    """Execute the stochastic kinetics phase of one timestep.

    Performs steps 1-6 of the full timestep workflow: resolves drivers, updates
    cooperativity and nearest neighbors, and runs stochastic thin/thick filament
    transitions. Does NOT solve for mechanical equilibrium.

    This is separated from timestep() to allow a future FE solver to run kinetics
    across all coupled sarcomeres before performing a coupled mechanical equilibration.

    Args:
        state: Current State NamedTuple (pure state, no embedded params)
        constants: DynamicParams/Constants with physics values
        drivers: Drivers NamedTuple with per-step pCa/z_line/ls (NaN = use constant)
        topology: SarcTopology for indexing
        rng_key: JAX random key for stochastic transitions
        dt: Timestep size (ms) -- keyword-only, JIT static

    Returns:
        (state_after_kinetics, new_rng_key, resolved_constants)
        resolved_constants is the DynamicParams with driver values (pCa, z_line,
        lattice_spacing) baked in. The FE solver can replace z_line before passing
        to solve_equilibrium via constants.with_drivers(...).
    """
    # Step 0: Resolve drivers -- merge time-varying overrides with constants
    pCa = resolve_value(drivers.pCa, constants.pCa)
    z_line = resolve_value(drivers.z_line, constants.z_line)
    lattice_spacing = resolve_value(drivers.lattice_spacing, constants.lattice_spacing)

    # Build a resolved constants with current driver values
    resolved_constants = constants.with_drivers(pCa, z_line, lattice_spacing)

    # Step 1: Calculate INTERNAL thin filament forces for cooperativity
    thin_internal_forces = calculate_thin_forces_for_cooperativity(state, resolved_constants, topology)

    # Step 2: Update cooperativity based on INTERNAL filament tension
    state = update_cooperativity(state, resolved_constants, thin_internal_forces, topology)

    # Step 3: Update nearest binding sites using topology
    state = update_nearest_neighbors(state, resolved_constants, topology)

    # Step 4: Thin filament transitions
    rng_key, thin_key = jax.random.split(rng_key)
    state, _P_thin = thin_transitions(state, resolved_constants, topology, thin_key, dt)

    # Step 5: Thick filament transitions
    rng_key, thick_key = jax.random.split(rng_key)
    state = thick_transitions(state, resolved_constants, topology, thick_key, dt)

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
             n_cg_steps: int = 1,
             n_newton_steps: int = 16,
             precond_params=None,
             prefactored_precond=None) -> Tuple['State', jnp.ndarray, jnp.ndarray, float, int]:
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
        (new_state, new_rng_key, solver_residual, new_ls, n_iters)
    """
    state, rng_key, resolved_constants = kinetics_step(
        state, constants, drivers, topology, rng_key, dt=dt
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
