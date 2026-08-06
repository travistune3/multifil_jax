"""
Force calculations for the half-sarcomere.

WHAT IS BEING BALANCED
----------------------
The half-sarcomere is a network of springs. Each thick filament is a chain of
crowns connected by backbone springs; each thin filament is a chain of binding
sites connected likewise. Crossbridges bridge between them, and titin tethers
the thick filament to the Z-disc. Every node has a position, and mechanical
equilibrium means the net axial force on every node is zero.

This module computes those forces. The solver (kernels/solver.py) then moves the
nodes until the residual vanishes:

    F(x) = F_backbone(x) + F_crossbridge(x) + F_titin(x) = 0

Note what is NOT here: inertia and viscosity. The system is solved to
equilibrium at every timestep rather than integrated forward in time. At
sarcomere scale, viscous relaxation is far faster than the chemistry, so the
mechanics can be treated as instantaneously equilibrated after each kinetic
step. This is why there is a Newton solve in the loop rather than an ODE
integrator.

SIGN CONVENTION
---------------
Positions increase from the M-line (0) toward the Z-line. A positive force on a
node pushes it toward larger coordinates, i.e. away from the M-line. Contractile
force therefore appears as a NEGATIVE force on thin filament nodes — the thin
filament is being pulled inward.

AXIAL AND RADIAL
----------------
Most of this module computes AXIAL forces, which is what the equilibrium solver
balances and what "muscle force" means. Two functions at the end compute RADIAL
forces instead: crossbridges and titin both act at an angle, so they squeeze the
filament lattice together as well as pulling along it. Those are used only in
dynamic lattice spacing mode, where the spacing itself is solved as an unknown.

FORCE TYPE OVERVIEW:
====================

EXTERNAL FORCES (from crossbridge attachments):
    - crossbridge_force_single() - Force from one XB (reference implementation)
    - compute_xb_forces_vectorized() - All XB forces (vectorized, used by solver)
    - calculate_crossbridge_forces_on_thin() - XB forces aggregated per binding site

PASSIVE FORCES (from filament deformation):
    - compute_thick_passive_forces_single() - One thick filament (with titin)
    - compute_thick_passive_forces_vectorized() - All thick filaments
    - compute_thin_passive_forces_single() - One thin filament
    - compute_thin_passive_forces_vectorized() - All thin filaments

COMBINED FORCES (for equilibrium solver):
    - compute_forces_vectorized() - Complete force residual F(x) for solver
    - compute_forces_from_state_vectorized() - Convenience wrapper

OUTPUT METRICS (for measurements):
    - axial_force_at_mline() - Total force at M-line (primary force output)

USAGE BY MODULE:
================
    solver.py - Uses compute_forces_vectorized() for Newton solver
    diagnostics - Uses axial_force_at_mline() for force measurements
    debugging - Uses calculate_crossbridge_forces_on_thin() for XB force analysis
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Union, TYPE_CHECKING
from functools import partial

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


# ============================================================================
# THICK FILAMENT PASSIVE FORCES (with Titin)
# ============================================================================

def compute_thick_passive_forces_single(
    positions: jnp.ndarray,
    rests: jnp.ndarray,
    thick_k: float,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    n_titin: int = 6
) -> jnp.ndarray:
    """Net axial force on each crown of one thick filament: backbone + titin.

    The thick filament is a chain of springs running from the M-line out to the
    tip. The M-line itself is treated as a fixed anchor at position 0 — it is the
    mirror plane where the two half-sarcomeres meet, so by symmetry it does not
    move.

    Net force on an interior crown is the difference between the springs on
    either side of it, so a uniformly stretched filament has zero net force
    everywhere except at its ends, as it should.

    TITIN acts only on the LAST crown, the filament tip, because that is where
    it attaches. It is a one-sided exponential spring reaching diagonally to the
    Z-disc: its length is sqrt(axial^2 + lattice_spacing^2), and only the axial
    component enters here. It pulls the tip toward the Z-line, and it is
    one-sided — compression produces no force, since a protein tether cannot
    push.

    Args:
        positions: (n_crowns,) crown axial positions (nm)
        rests: (n_crowns,) rest spacing between each crown and the previous node
        thick_k: Backbone spring constant per segment (pN/nm)
        z_line: Z-line position (nm), the far anchor for titin
        lattice_spacing: Radial thick-to-thin distance (nm), the other leg of
            titin's diagonal
        titin_a, titin_b, titin_rest: Exponential spring parameters,
            F = titin_a * exp(titin_b * (L - titin_rest))
        n_titin: Titin molecules per thick filament. 6 matches the vertebrate
            sixfold arrangement; unverified for the 1:3 insect lattice, where it
            scales total passive force directly.

    Returns:
        forces: (n_crowns,) net axial force on each crown (pN)
    """
    # Prepend M-line position (0) to crown positions
    axial_with_mline = jnp.concatenate([jnp.array([0.0]), positions])

    # Calculate distances between adjacent nodes
    dists = jnp.diff(axial_with_mline)

    # Spring forces: F = k * (actual - rest)
    spring_forces = (dists - rests) * thick_k

    # Calculate titin force for last crown
    myo_loc = positions[-1]
    axial_dist = z_line - myo_loc

    # Total titin length (Pythagorean theorem)
    titin_length = jnp.sqrt(axial_dist**2 + lattice_spacing**2)

    # Exponential force (with clipping for numerical stability)
    exp_arg = titin_b * (titin_length - titin_rest)
    exp_arg = jnp.clip(exp_arg, -100.0, 100.0)
    titin_force_total = titin_a * jnp.exp(exp_arg)
    titin_force_total = jnp.maximum(titin_force_total, 0.0)

    # Axial component
    cos_angle = jnp.where(titin_length > 0, axial_dist / titin_length, 0.0)
    titin_force_axial = titin_force_total * cos_angle

    # Multiply by number of titin filaments
    total_titin_force = n_titin * titin_force_axial

    # Append titin force at the end
    spring_forces_with_titin = jnp.concatenate([spring_forces, jnp.array([total_titin_force])])

    # Net force at each crown is diff of spring forces
    net_forces = jnp.diff(spring_forces_with_titin)

    return net_forces


def compute_thick_passive_forces_vectorized(
    positions_thick: jnp.ndarray,
    rests_thick: jnp.ndarray,
    thick_k: float,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    n_titin_per_thick: int = 6
) -> jnp.ndarray:
    """Vectorized thick filament passive forces for all thick filaments.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        rests_thick: (n_thick, n_crowns) rest spacings
        thick_k: Thick filament spring constant
        z_line: Z-line position
        lattice_spacing: Lattice spacing
        titin_a, titin_b, titin_rest: Titin parameters
        n_titin_per_thick: Number of titin per thick filament

    Returns:
        forces: (n_thick, n_crowns) net force on each crown
    """
    compute_fn = partial(
        compute_thick_passive_forces_single,
        thick_k=thick_k,
        z_line=z_line,
        lattice_spacing=lattice_spacing,
        titin_a=titin_a,
        titin_b=titin_b,
        titin_rest=titin_rest,
        n_titin=n_titin_per_thick
    )

    forces = jax.vmap(compute_fn)(positions_thick, rests_thick)

    return forces


# ============================================================================
# THIN FILAMENT PASSIVE FORCES
# ============================================================================

def compute_thin_passive_forces_single(
    positions: jnp.ndarray,
    rests: jnp.ndarray,
    thin_k: float,
    z_line: float
) -> jnp.ndarray:
    """Compute net force on each binding site of one thin filament.

    Args:
        positions: (n_sites,) binding site axial positions
        rests: (n_sites,) rest spacings between sites
        thin_k: Thin filament spring constant (pN/nm)
        z_line: Z-line position (nm)

    Returns:
        forces: (n_sites,) net force on each binding site
    """
    # Append z-line to positions
    axial_with_zline = jnp.concatenate([positions, jnp.array([z_line])])

    # Calculate distances between adjacent nodes
    dists = jnp.diff(axial_with_zline)

    # Spring forces
    spring_forces = (dists - rests) * thin_k

    # Prepend 0 (first site has no spring on M-line side)
    spring_forces_with_zero = jnp.concatenate([jnp.array([0.0]), spring_forces])

    # Net force at each site is diff of spring forces
    net_forces = jnp.diff(spring_forces_with_zero)

    return net_forces


def compute_thin_passive_forces_vectorized(
    positions_thin: jnp.ndarray,
    rests_thin: jnp.ndarray,
    thin_k: float,
    z_line: float
) -> jnp.ndarray:
    """Vectorized thin filament passive forces for all thin filaments.

    Args:
        positions_thin: (n_thin, n_sites) binding site positions
        rests_thin: (n_thin, n_sites) rest spacings
        thin_k: Thin filament spring constant
        z_line: Z-line position

    Returns:
        forces: (n_thin, n_sites) net force on each site
    """
    compute_fn = partial(
        compute_thin_passive_forces_single,
        thin_k=thin_k,
        z_line=z_line
    )

    forces = jax.vmap(compute_fn)(positions_thin, rests_thin)

    return forces


# ============================================================================
# CROSSBRIDGE FORCES (Fully Vectorized)
# ============================================================================

def compute_xb_forces_vectorized(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    lattice_spacing: float,
    params: 'DynamicParams',
    geometry: 'SarcTopology'
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Axial force every attached crossbridge exerts on its crown and its site.

    Each attached head is a two-spring element (see core/params.py for the
    geometry and where the rest configurations come from). Given the head's
    current offset x to its bound site and the lattice spacing d, its length and
    angle are r = sqrt(x^2 + d^2) and theta = atan2(d, x), and the axial force
    follows from differentiating the two-spring potential with respect to the
    crown's position:

        F = g_k*(r - g_rest)*cos(theta) - (c_k/r)*(theta - c_rest)*sin(theta)
              globular, linear                converter, angular

    THE MINUS SIGN ON THE ANGULAR TERM IS NOT COSMETIC. The two springs pull the
    head in competing directions: extending the linear spring resists
    lengthening, while winding the angular spring past its rest angle drives the
    head the other way. Flipping that sign inverts the converter's contribution,
    and because c_k can dominate g_k in the strong state, it can silently
    reverse the sign of total muscle force.

    Which rest configuration applies depends on the head's state: strong
    (states 2 and 3) or weak (states 1 and 4). Heads in states 0 or 5 are
    detached and contribute nothing.

    Forces are equal and opposite: whatever a head does to its crown, it does the
    negative of to its binding site. Site accumulation uses segment_sum because
    many heads can share one site, and atomics avoid materializing an
    (n_xb x n_sites) scatter matrix.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        positions_thin: (n_thin, n_sites) binding site positions
        xb_states: (n_thick, n_crowns, n_xb_per_crown) XB states (0-5)
        xb_bound_to: (n_thick, n_crowns, n_xb_per_crown) site indices
            Just site_idx (-1 if unbound), thin from geometry
        lattice_spacing: Lattice spacing (nm)
        params: Parameter dictionary with XB spring parameters
        geometry: SarcTopology with xb_to_thin_id (required).

    Returns:
        forces_on_thick: (n_thick, n_crowns) net XB force on each crown
        forces_on_thin: (n_thin, n_sites) net XB force on each binding site
    """
    n_thick, n_crowns = positions_thick.shape
    n_thin, n_sites = positions_thin.shape
    n_xb_per_crown = xb_states.shape[2]
    n_crowns_total = n_thick * n_crowns
    n_sites_total = n_thin * n_sites

    # Flatten all arrays for vectorized computation
    xb_states_flat = xb_states.reshape(-1)
    xb_bound_flat = xb_bound_to.reshape(-1)  # Just site_idx now

    # Expand positions to match XB array shape
    xb_positions_flat = jnp.repeat(
        positions_thick.reshape(-1),
        n_xb_per_crown
    )

    # Flatten thin positions for indexing
    positions_thin_flat = positions_thin.reshape(-1)

    # Check bound status
    is_bound = (xb_states_flat >= 1) & (xb_states_flat <= 3) & (xb_bound_flat >= 0)

    # Use geometry for thin lookup - NO DIVISION
    thin_idx = geometry.xb_to_thin_id  # Static from topology
    site_idx = xb_bound_flat           # Runtime state (or -1 if unbound)

    # Handle unbound XBs: clip to valid range (forces will be zeroed anyway)
    thin_idx_safe = jnp.clip(thin_idx, 0, n_thin - 1)
    site_idx_safe = jnp.clip(site_idx, 0, n_sites - 1)
    thin_flat_idx = thin_idx_safe * n_sites + site_idx_safe

    # Get binding site positions using geometry lookup
    bs_positions = positions_thin_flat[thin_flat_idx]

    # Calculate distances
    x_dist = bs_positions - xb_positions_flat

    # Crossbridge force calculation (two-spring model)
    r = jnp.sqrt(x_dist**2 + lattice_spacing**2)

    # OPTIMIZED: Algebraic substitution for trig functions
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    cos_theta = x_dist / r_safe
    sin_theta = lattice_spacing / r_safe
    theta = jnp.arctan2(lattice_spacing, x_dist)

    # Get spring parameters based on state
    is_strong = (xb_states_flat == 2) | (xb_states_flat == 3)

    # Extract scalar params (attribute access)
    c_rest_strong = params.xb_c_rest_strong
    c_rest_weak = params.xb_c_rest_weak
    c_k_strong = params.xb_c_k_strong
    c_k_weak = params.xb_c_k_weak
    g_rest_strong = params.xb_g_rest_strong
    g_rest_weak = params.xb_g_rest_weak
    g_k_strong = params.xb_g_k_strong
    g_k_weak = params.xb_g_k_weak

    # Converter domain (angular spring)
    c_rest = jnp.where(is_strong, c_rest_strong, c_rest_weak)
    c_k = jnp.where(is_strong, c_k_strong, c_k_weak)

    # Globular domain (linear spring)
    g_rest = jnp.where(is_strong, g_rest_strong, g_rest_weak)
    g_k = jnp.where(is_strong, g_k_strong, g_k_weak)

    # Calculate axial force using algebraic trig
    # Sign: F_crown_x = -∂U_g/∂x_crown - ∂U_c/∂x_crown
    #   = +g_k*(r-g_rest)*cos_theta - (c_k/r)*(theta-c_rest)*sin_theta
    f_axial = (g_k * (r - g_rest) * cos_theta -
               (1.0 / r_safe) * c_k * (theta - c_rest) * sin_theta)

    # Zero force for unbound XBs
    forces = jnp.where(is_bound, f_axial, 0.0)

    # Accumulate thick filament forces (reshape+sum - unchanged, regular pattern)
    forces_per_crown = forces.reshape(n_crowns_total, n_xb_per_crown)
    forces_on_thick_flat = forces_per_crown.sum(axis=-1)
    forces_on_thick = forces_on_thick_flat.reshape(n_thick, n_crowns)

    # REFACTORED: Use segment_sum instead of one_hot matmul
    # segment_sum uses GPU atomic operations - no huge buffer allocation
    forces_on_thin_flat = jax.ops.segment_sum(
        -forces,
        thin_flat_idx,
        num_segments=n_sites_total
    )
    forces_on_thin = forces_on_thin_flat.reshape(n_thin, n_sites)

    return forces_on_thick, forces_on_thin


# ============================================================================
# COMBINED FORCE CALCULATION
# ============================================================================

def compute_forces_vectorized(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    thick_k: float,
    thin_k: float,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params: 'DynamicParams',
    geometry: 'SarcTopology'
) -> jnp.ndarray:
    """Complete vectorized force residual calculation.

    This is the JAX-native replacement for compute_forces_from_positions().
    It is fully JIT-compilable and GPU-ready.

    The residual is: F(x) = 0 at equilibrium

    Rest spacings are sourced from geometry (topology):
        thick: geometry.crown_rests (n_thick, n_crowns), per-filament
        thin:  geometry.binding_rests (n_thin, n_sites)

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        positions_thin: (n_thin, n_sites) binding site positions
        thick_k: Thick filament spring constant
        thin_k: Thin filament spring constant
        z_line: Z-line position
        lattice_spacing: Lattice spacing
        titin_a, titin_b, titin_rest: Titin parameters
        xb_states: Crossbridge states
        xb_bound_to: Crossbridge binding info (site_idx only, thin from geometry)
        params: Parameter dictionary
        geometry: SarcTopology with xb_to_thin_id, crown_rests, binding_rests.

    Returns:
        forces: (n_thick_nodes + n_thin_nodes,) flattened force residual
    """
    rests_thick = geometry.crown_rests
    rests_thin = geometry.binding_rests

    # 1. Thick filament passive forces (with titin)
    forces_thick = compute_thick_passive_forces_vectorized(
        positions_thick, rests_thick,
        thick_k, z_line, lattice_spacing,
        titin_a, titin_b, titin_rest
    )

    # 2. Thin filament passive forces
    forces_thin = compute_thin_passive_forces_vectorized(
        positions_thin, rests_thin,
        thin_k, z_line
    )

    # 3. Crossbridge forces (pass geometry for optimized indexing)
    xb_forces_thick, xb_forces_thin = compute_xb_forces_vectorized(
        positions_thick, positions_thin,
        xb_states, xb_bound_to,
        lattice_spacing, params, geometry
    )

    # Combine passive and XB forces
    total_forces_thick = forces_thick + xb_forces_thick
    total_forces_thin = forces_thin + xb_forces_thin

    # Flatten and concatenate
    forces_flat = jnp.concatenate([
        total_forces_thick.flatten(),
        total_forces_thin.flatten()
    ])

    return forces_flat


def compute_forces_from_state_vectorized(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
) -> jnp.ndarray:
    """Convenience function to compute forces directly from state + constants + topology.

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with physics values
        topology: SarcTopology with structural index maps

    Returns:
        forces: Flattened force residual array
    """
    return compute_forces_vectorized(
        positions_thick=state.thick.axial,
        positions_thin=state.thin.axial,
        thick_k=constants.thick_k,
        thin_k=constants.thin_k,
        z_line=constants.z_line,
        lattice_spacing=constants.lattice_spacing,
        titin_a=constants.titin_a,
        titin_b=constants.titin_b,
        titin_rest=constants.titin_rest,
        xb_states=state.thick.xb_states,
        xb_bound_to=state.thick.xb_bound_to,
        params=constants,
        geometry=topology,
    )


# ============================================================================
# TOTAL THICK FILAMENT FORCES (for work_thick metric)
# ============================================================================

def compute_thick_forces_vectorized(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
) -> jnp.ndarray:
    """Compute net force on each thick filament node.

    Combines passive spring forces (including titin) and crossbridge forces.
    Used for work_thick metric calculation: Work = F_residual × dx

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with physics values
        topology: SarcTopology with structural index maps

    Returns:
        Forces on thick nodes: (n_thick, n_crowns)
    """
    # 1. Passive forces from thick filament springs (including titin)
    f_passive = compute_thick_passive_forces_vectorized(
        state.thick.axial,
        topology.crown_rests,
        constants.thick_k,
        constants.z_line,
        constants.lattice_spacing,
        constants.titin_a,
        constants.titin_b,
        constants.titin_rest,
    )

    # 2. Crossbridge forces on thick filament
    xb_forces_thick, _ = compute_xb_forces_vectorized(
        state.thick.axial,
        state.thin.axial,
        state.thick.xb_states,
        state.thick.xb_bound_to,
        constants.lattice_spacing,
        constants,
        topology,
    )

    # Combined forces
    return f_passive + xb_forces_thick


# ============================================================================
# OUTPUT METRICS
# ============================================================================

def axial_force_at_mline(state: 'State', constants: 'DynamicParams', topology: 'SarcTopology') -> float:
    """Total axial force delivered to the M-line. The model's primary output.

    This is what "muscle force" means for this simulation, and what a
    force transducer attached to the preparation would read.

    IT IS MEASURED, NOT SUMMED. The force is read from the strain in the first
    backbone spring of each thick filament — the segment between the M-line and
    the first crown:

        force = sum over thick filaments of (crown[0] - bare_zone) * thick_k

    Deliberately NOT the sum of individual crossbridge forces. Once the solver
    has equilibrated the lattice, every force generated anywhere on the filament
    must be transmitted through that first segment to reach the M-line, so its
    strain is the honest total — including the contributions of titin and of
    filament compliance, and correctly excluding any internal forces that cancel.
    Summing crossbridge forces directly would double-count strain that the
    backbone is already carrying.

    A corollary worth remembering: this reading is only meaningful once the
    solver has converged. Reading it mid-solve gives the force implied by a
    not-yet-equilibrated configuration.

    Also note that at rest this is NOT zero — titin alone can dominate it at
    long sarcomere lengths. Subtract a relaxed (pCa 9) baseline before
    interpreting active force.


    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with thick_k
        topology: SarcTopology with crown_offsets[:, 0] = per-filament bare_zone distance

    Returns:
        force: Total axial force at M-line (pN)
    """
    bare_zone = topology.crown_offsets[:, 0]
    force_per_thick = (state.thick.axial[:, 0] - bare_zone) * constants.thick_k
    return jnp.sum(force_per_thick)


# ============================================================================
# RADIAL FORCE FUNCTIONS (for dynamic lattice spacing solver)
# ============================================================================

# Titin molecules per thick filament, for the radial force path.
#
# MUST match the n_titin_per_thick default used on the axial path
# (compute_thick_passive_forces_vectorized). The two paths describe the same
# physical tethers resolved along different axes; if they disagree on how many
# there are, the axial and radial force magnitudes become mutually inconsistent
# and the dynamic-lattice-spacing solve balances against a fiction.
#
# 6 matches the vertebrate sixfold arrangement of titin around each thick
# filament. It is unverified for the 1:3 invertebrate lattice — no source giving
# a per-thick-filament connecting-filament count for insect flight muscle was
# located — and it scales passive force linearly, so it matters.
_N_TITIN_PER_THICK = 6


def _xb_radial_force_total(
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    lattice_spacing: float,
    params,
    topology,
) -> float:
    """Total XB radial force on the lattice from all bound crossbridges.

    Replicates the geometry from compute_xb_forces_vectorized but accumulates
    the radial component instead of axial. Differentiable w.r.t. lattice_spacing
    for JVP in the bordered Newton solver.

    Args:
        xb_states: (n_thick, n_crowns, n_xb_per_crown) XB states (0-5)
        xb_bound_to: (n_thick, n_crowns, n_xb_per_crown) site indices (-1 unbound)
        positions_thick: (n_thick, n_crowns) crown axial positions
        positions_thin: (n_thin, n_sites) binding site axial positions
        lattice_spacing: Current lattice spacing d (nm)
        params: DynamicParams with XB spring constants
        topology: SarcTopology with xb_to_thin_id

    Returns:
        Scalar total radial force (pN). Positive = outward (increasing d).
    """
    n_thin, n_sites = positions_thin.shape
    n_xb_per_crown = xb_states.shape[2]

    xb_states_flat = xb_states.reshape(-1)
    xb_bound_flat = xb_bound_to.reshape(-1)

    xb_positions_flat = jnp.repeat(positions_thick.reshape(-1), n_xb_per_crown)
    positions_thin_flat = positions_thin.reshape(-1)

    is_bound = (xb_states_flat >= 1) & (xb_states_flat <= 3) & (xb_bound_flat >= 0)

    thin_idx = topology.xb_to_thin_id
    site_idx = xb_bound_flat
    thin_idx_safe = jnp.clip(thin_idx, 0, n_thin - 1)
    site_idx_safe = jnp.clip(site_idx, 0, n_sites - 1)
    thin_flat_idx = thin_idx_safe * n_sites + site_idx_safe

    bs_positions = positions_thin_flat[thin_flat_idx]
    x_dist = bs_positions - xb_positions_flat

    r = jnp.sqrt(x_dist**2 + lattice_spacing**2)
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    cos_theta = x_dist / r_safe
    sin_theta = lattice_spacing / r_safe
    theta = jnp.arctan2(lattice_spacing, x_dist)

    is_strong = (xb_states_flat == 2) | (xb_states_flat == 3)
    c_rest = jnp.where(is_strong, params.xb_c_rest_strong, params.xb_c_rest_weak)
    c_k = jnp.where(is_strong, params.xb_c_k_strong, params.xb_c_k_weak)
    g_rest = jnp.where(is_strong, params.xb_g_rest_strong, params.xb_g_rest_weak)
    g_k = jnp.where(is_strong, params.xb_g_k_strong, params.xb_g_k_weak)

    f_radial = (g_k * (r - g_rest) * sin_theta +
                (1.0 / r_safe) * c_k * (theta - c_rest) * cos_theta)

    forces_radial = jnp.where(is_bound, f_radial, 0.0)
    return jnp.sum(forces_radial)


def _titin_radial_force_total(
    positions_thick: jnp.ndarray,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
) -> float:
    """Total titin radial force from all thick filaments.

    Replicates geometry from compute_thick_passive_forces_single but returns
    the radial component. Differentiable w.r.t. lattice_spacing.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        z_line: Z-line position (nm)
        lattice_spacing: Current lattice spacing d (nm)
        titin_a, titin_b, titin_rest: Titin exponential spring parameters

    Returns:
        Scalar total radial titin force (pN). Positive = outward (increasing d).
    """
    myo_loc = positions_thick[:, -1]
    axial_dist = z_line - myo_loc

    titin_length = jnp.sqrt(axial_dist**2 + lattice_spacing**2)
    titin_length_safe = jnp.where(titin_length > 1e-10, titin_length, 1e-10)

    exp_arg = jnp.clip(titin_b * (titin_length - titin_rest), -100.0, 100.0)
    titin_force = jnp.maximum(titin_a * jnp.exp(exp_arg), 0.0)

    sin_angle = lattice_spacing / titin_length_safe
    titin_radial_per_thick = _N_TITIN_PER_THICK * titin_force * sin_angle

    return jnp.sum(titin_radial_per_thick)

