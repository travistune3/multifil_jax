"""
Force Calculations for JAX Half-Sarcomere

This module provides ALL force calculations for the multifil_jax simulation.
Functions are organized by their purpose and the type of forces calculated.

FORCE TYPE OVERVIEW:
====================

INTERNAL FORCES (from filament backbone springs, NOT crossbridges):
    - thin_filament_internal_forces() - Spring forces within thin filament backbone
    - calculate_thin_forces_for_cooperativity() - Wrapper for cooperativity updates

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
    timestep.py - Uses calculate_thin_forces_for_cooperativity() for cooperativity
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
# CROSSBRIDGE FORCES - Single Element
# ============================================================================

@jax.jit
def crossbridge_force_single(xb_distance: jnp.ndarray,
                             lattice_spacing: float,
                             state: int,
                             params: dict) -> Tuple[float, float]:
    """Calculate force from a single crossbridge.

    This implements the two-spring model from mh.py Head class.

    Args:
        xb_distance: (x, y) distance from XB base to binding site
            x = axial distance (nm)
            y = lattice spacing (nm)
        lattice_spacing: Lattice spacing (nm)
        state: XB state (1=DRX, 2=loose, 3=tight_1, 4=tight_2, 5=free_2, 6=SRX)
        params: Parameter dictionary with spring constants and rest lengths

    Returns:
        f_axial: Axial force (pN)
        f_radial: Radial force (pN)
    """
    x, y = xb_distance[0], xb_distance[1]

    # Convert to polar coordinates
    r = jnp.sqrt(x**2 + y**2)

    # OPTIMIZED: Algebraic substitution for trig functions
    # cos(theta) = x/r, sin(theta) = y/r
    # Safe division to avoid r=0 issues
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    cos_theta = x / r_safe
    sin_theta = y / r_safe

    # Keep arctan2 ONLY for angular spring displacement (theta - c_rest)
    theta = jnp.arctan2(y, x)

    # Get spring parameters based on state
    # States 3,4 use strong (bound) springs, others use weak
    is_strong = (state == 3) | (state == 4)

    # Converter domain (angular spring)
    c_rest = jnp.where(is_strong,
                       params.xb_c_rest_strong,
                       params.xb_c_rest_weak)
    c_k = jnp.where(is_strong,
                    params.xb_c_k_strong,
                    params.xb_c_k_weak)

    # Globular domain (linear spring)
    g_rest = jnp.where(is_strong,
                       params.xb_g_rest_strong,
                       params.xb_g_rest_weak)
    g_k = jnp.where(is_strong,
                    params.xb_g_k_strong,
                    params.xb_g_k_weak)

    # Resolve into axial and radial components using algebraic trig
    f_axial = (g_k * (r - g_rest) * cos_theta +
               (1.0 / r_safe) * c_k * (theta - c_rest) * sin_theta)

    f_radial = (g_k * (r - g_rest) * sin_theta +
                (1.0 / r_safe) * c_k * (theta - c_rest) * cos_theta)

    # Only bound states (2,3,4) produce force
    is_bound = (state >= 2) & (state <= 4)
    f_axial = jnp.where(is_bound, f_axial, 0.0)
    f_radial = jnp.where(is_bound, f_radial, 0.0)

    return f_axial, f_radial


# ============================================================================
# THICK FILAMENT PASSIVE FORCES (with Titin)
# ============================================================================

@jax.jit
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
    """Compute net force on each crown of one thick filament.

    Args:
        positions: (n_crowns,) crown axial positions
        rests: (n_crowns,) rest spacings between crowns
        thick_k: Thick filament spring constant (pN/nm)
        z_line: Z-line position (nm)
        lattice_spacing: Lattice spacing (nm)
        titin_a, titin_b, titin_rest: Titin parameters
        n_titin: Number of titin filaments attached to this thick (default 6)

    Returns:
        forces: (n_crowns,) net force on each crown
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


@jax.jit
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

@jax.jit
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


@jax.jit
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
# INTERNAL THIN FILAMENT FORCES (for cooperativity)
# ============================================================================

@jax.jit
def thin_filament_internal_forces(axial_positions: jnp.ndarray,
                                   rest_spacings: jnp.ndarray,
                                   k: float,
                                   z_line: float) -> jnp.ndarray:
    """Calculate NET force on each thin filament binding site node.

    This is the JAX equivalent of NumPy's af.py::_axial_thin_filament_forces().

    Returns NET force on each node = force from right spring - force from left spring.
    When cumsummed with np.triu(), gives cumulative tension at each site.

    Args:
        axial_positions: (n_thin, n_sites) binding site positions (nm)
        rest_spacings: (n_thin, n_sites) rest spacings between sites (nm)
        k: Thin filament spring constant (pN/nm)
        z_line: Z-line position (nm)

    Returns:
        net_forces: (n_thin, n_sites) NET force on each binding site node (pN)
    """
    n_thin, n_sites = axial_positions.shape

    # Step 1: Append z_line to axial positions
    z_line_col = jnp.full((n_thin, 1), z_line)
    axial_with_zline = jnp.concatenate([axial_positions, z_line_col], axis=1)

    # Step 2: Calculate distances between consecutive positions
    dists = jnp.diff(axial_with_zline, axis=1)

    # Step 3: Calculate spring forces for ALL n springs
    spring_forces = (dists - rest_spacings) * k

    # Step 4: Prepend zero (first node has no spring to M-line side)
    padded_springs = jnp.concatenate([
        jnp.zeros((n_thin, 1)),
        spring_forces
    ], axis=1)

    # Step 5: Net force on each node = diff of padded spring forces
    net_forces = jnp.diff(padded_springs, axis=1)

    return net_forces


def calculate_thin_forces_for_cooperativity(state: 'State', constants: 'DynamicParams') -> jnp.ndarray:
    """Calculate INTERNAL thin filament forces for cooperativity calculations.

    PURPOSE: These forces are used by update_cooperativity() to determine
    the cooperative span of tropomyosin based on filament tension.

    FORCE TYPE: INTERNAL spring forces from thin filament backbone deformation.
    These are NOT crossbridge forces - they represent the tension transmitted
    through the actin filament springs.

    CRITICAL DISTINCTION: The NumPy OOP version (af._axial_thin_filament_forces)
    explicitly states "sans cross-bridges" - meaning internal spring forces only.
    Using crossbridge forces here would be INCORRECT for cooperativity.

    Args:
        state: State NamedTuple containing:
            - thin.axial: (n_thin, n_sites) binding site positions
            - thin.rests: (n_thin, n_sites) rest spacings
        constants: DynamicParams with:
            - thin_k: thin filament spring constant
            - z_line: Z-line position

    Returns:
        forces: (n_thin, n_sites) internal spring forces on each binding site (pN)

    See Also:
        - thin_filament_internal_forces() - The underlying calculation
        - calculate_crossbridge_forces_on_thin() - EXTERNAL forces (different purpose)
        - NumPy reference: multifil/af.py:732-753 (_axial_thin_filament_forces)
    """
    thin_axial = state.thin.axial
    thin_rests = state.thin.rests

    return thin_filament_internal_forces(
        thin_axial,
        thin_rests,
        constants.thin_k,
        constants.z_line
    )


@jax.jit
def calculate_crossbridge_forces_on_thin(state: 'State', constants: 'DynamicParams', topology: 'SarcTopology') -> jnp.ndarray:
    """Calculate EXTERNAL forces on thin filament sites from bound crossbridges.

    PURPOSE: Aggregate all crossbridge forces onto their nearest binding sites.
    Used for diagnostics, debugging, and calculating total force output.

    FORCE TYPE: EXTERNAL forces from crossbridge attachments. These are the
    forces that crossbridges exert ON the thin filament at each binding site.

    DO NOT USE FOR COOPERATIVITY: Cooperativity depends on INTERNAL filament
    tension, not external crossbridge forces. Use calculate_thin_forces_for_cooperativity()
    for cooperativity calculations.

    Uses topology.xb_to_thin_id for thin filament lookup (no division).
    Uses jax.ops.segment_sum for efficient GPU accumulation.

    Args:
        state: State NamedTuple containing:
            - thin.axial: (n_thin, n_sites) binding site positions
            - thick.axial: (n_thick, n_crowns) crown positions
            - thick.xb_states: (n_thick, n_crowns, 3) XB states (1-6)
            - thick.xb_bound_to: (n_thick, n_crowns, 3) binding info (site_idx only, or -1)
        constants: DynamicParams with:
            - lattice_spacing: lattice spacing (nm)
            - xb_* spring parameters
        topology: SarcTopology with xb_to_thin_id (required).

    Returns:
        forces: (n_thin, n_sites) total XB force on each binding site (pN)
            Positive = towards Z-line, Negative = towards M-line

    See Also:
        - calculate_thin_forces_for_cooperativity() - INTERNAL forces (different purpose)
        - compute_xb_forces_vectorized() - Returns forces on both thick and thin
    """
    thin_axial = state.thin.axial
    xb_states = state.thick.xb_states
    xb_bound_to = state.thick.xb_bound_to
    xb_positions = state.thick.axial
    bs_positions = state.thin.axial
    lattice_spacing = constants.lattice_spacing

    n_thin, n_sites = thin_axial.shape
    n_sites_total = n_thin * n_sites
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape

    # Flatten all crossbridges for vectorized computation
    xb_states_flat = xb_states.reshape(-1)
    xb_bound_to_flat = xb_bound_to.reshape(-1)  # Now just site_idx!

    # Repeat crown positions for each XB in the crown
    xb_positions_expanded = jnp.repeat(xb_positions[:, :, None], n_xb_per_crown, axis=2)
    xb_positions_flat = xb_positions_expanded.reshape(-1)

    # Use topology for thin filament lookup - NO DIVISION
    thin_idx = topology.xb_to_thin_id  # (total_xbs,) - static from topology
    site_idx = xb_bound_to_flat  # (total_xbs,) - runtime binding state (or -1 if unbound)

    # Check if bound (states 2,3,4 are bound, states 1,5,6 are unbound)
    is_bound = (xb_states_flat >= 2) & (xb_states_flat <= 4) & (xb_bound_to_flat >= 0)

    # Clip site indices to valid range for unbound XBs (forces will be zeroed anyway)
    site_idx_safe = jnp.clip(site_idx, 0, n_sites - 1)
    thin_idx_safe = jnp.clip(thin_idx, 0, n_thin - 1)

    # Compute flat index for segment_sum
    thin_flat_idx = thin_idx_safe * n_sites + site_idx_safe

    # Get binding site positions
    positions_thin_flat = bs_positions.reshape(-1)
    bs_pos = positions_thin_flat[thin_flat_idx]

    # Calculate distances
    x_dist = bs_pos - xb_positions_flat

    # Crossbridge force calculation (two-spring model)
    r = jnp.sqrt(x_dist**2 + lattice_spacing**2)

    # OPTIMIZED: Algebraic substitution for trig functions
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    cos_theta = x_dist / r_safe
    sin_theta = lattice_spacing / r_safe
    theta = jnp.arctan2(lattice_spacing, x_dist)

    # Get spring parameters based on state
    is_strong = (xb_states_flat == 3) | (xb_states_flat == 4)

    c_rest = jnp.where(is_strong, constants.xb_c_rest_strong, constants.xb_c_rest_weak)
    c_k = jnp.where(is_strong, constants.xb_c_k_strong, constants.xb_c_k_weak)
    g_rest = jnp.where(is_strong, constants.xb_g_rest_strong, constants.xb_g_rest_weak)
    g_k = jnp.where(is_strong, constants.xb_g_k_strong, constants.xb_g_k_weak)

    # Calculate axial force
    f_axial = (g_k * (r - g_rest) * cos_theta +
               (1.0 / r_safe) * c_k * (theta - c_rest) * sin_theta)

    # Zero force for unbound XBs
    forces_contributions = jnp.where(is_bound, f_axial, 0.0)

    # REFACTORED: Use segment_sum instead of jnp.zeros + scatter-add
    # segment_sum uses GPU atomic operations for efficient accumulation
    forces_on_thin_flat = jax.ops.segment_sum(
        forces_contributions,
        thin_flat_idx,
        num_segments=n_sites_total
    )

    return forces_on_thin_flat.reshape(n_thin, n_sites)


# ============================================================================
# CROSSBRIDGE FORCES (Fully Vectorized)
# ============================================================================

def compute_xb_forces_vectorized(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    lattice_spacing: float,
    params: Dict,
    geometry: 'SarcTopology'
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fully vectorized crossbridge force calculation.

    Uses geometry.xb_to_thin_id for thin filament lookup (no division).
    Uses segment_sum for efficient GPU accumulation.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        positions_thin: (n_thin, n_sites) binding site positions
        xb_states: (n_thick, n_crowns, n_xb_per_crown) XB states (1-6)
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
    is_bound = (xb_states_flat >= 2) & (xb_states_flat <= 4) & (xb_bound_flat >= 0)

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
    is_strong = (xb_states_flat == 3) | (xb_states_flat == 4)

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
    f_axial = (g_k * (r - g_rest) * cos_theta +
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

@jax.jit
def compute_forces_vectorized(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    rests_thick: jnp.ndarray,
    rests_thin: jnp.ndarray,
    thick_k: float,
    thin_k: float,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params: Dict,
    geometry: 'SarcTopology'
) -> jnp.ndarray:
    """Complete vectorized force residual calculation.

    This is the JAX-native replacement for compute_forces_from_positions().
    It is fully JIT-compilable and GPU-ready.

    The residual is: F(x) = 0 at equilibrium

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        positions_thin: (n_thin, n_sites) binding site positions
        rests_thick: (n_thick, n_crowns) rest spacings
        rests_thin: (n_thin, n_sites) rest spacings
        thick_k: Thick filament spring constant
        thin_k: Thin filament spring constant
        z_line: Z-line position
        lattice_spacing: Lattice spacing
        titin_a, titin_b, titin_rest: Titin parameters
        xb_states: Crossbridge states
        xb_bound_to: Crossbridge binding info (site_idx only, thin from geometry)
        params: Parameter dictionary
        geometry: SarcTopology with xb_to_thin_id (required).

    Returns:
        forces: (n_thick_nodes + n_thin_nodes,) flattened force residual
    """
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
        rests_thick=state.thick.rests,
        rests_thin=state.thin.rests,
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

@jax.jit
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
        state.thick.rests,
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

@jax.jit
def axial_force_at_mline(state: 'State', constants: 'DynamicParams') -> float:
    """Calculate total axial force at the M-line (effective axial force).

    This is the key output for force-length and force-velocity measurements.

    IMPORTANT: This is the SPRING force transmitted through the thick filament
    backbone to the M-line, NOT the sum of all crossbridge forces.

    Formula: force = sum_over_thick_filaments( (crown[0] - bare_zone) * thick_k )

    Matches OOP mf.effective_axial_force() at mf.py:562.

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with thick_k and thick_bare_zone

    Returns:
        force: Total axial force at M-line (pN)
    """
    crown_positions = state.thick.axial
    bare_zone = constants.thick_bare_zone
    thick_k = constants.thick_k

    force_per_thick = (crown_positions[:, 0] - bare_zone) * thick_k
    force_at_mline = jnp.sum(force_per_thick)

    return force_at_mline


# ============================================================================
# RADIAL FORCE FUNCTIONS (for dynamic lattice spacing solver)
# ============================================================================

# n_titin_per_thick used in radial force calculation.
# Must match the default in compute_thick_passive_forces_vectorized (which uses
# n_titin_per_thick=6). Changing this without changing that function would produce
# inconsistent axial/radial force magnitudes.
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
        xb_states: (n_thick, n_crowns, n_xb_per_crown) XB states (1-6)
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

    is_bound = (xb_states_flat >= 2) & (xb_states_flat <= 4) & (xb_bound_flat >= 0)

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

    is_strong = (xb_states_flat == 3) | (xb_states_flat == 4)
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

