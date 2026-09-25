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
    - compute_thin_passive_forces() - All thin filaments

COMBINED FORCES (for equilibrium solver):
    - compute_forces_vectorized() - Complete force residual F(x) for solver

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

from multifil_jax.core.state import thick_axial, monomer_axial, thin_segment_k

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


# ============================================================================
# THICK FILAMENT PASSIVE FORCES (with Titin)
# ============================================================================

def compute_thick_passive_forces_single(
    u: jnp.ndarray,
    offsets: jnp.ndarray,
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

    COORDINATES ARE DISPLACEMENTS. `u` is each crown's offset from its rest
    position, so the backbone law needs no rest lengths: `crown_offsets` is the
    cumsum of `crown_rests`, and the two cancel algebraically. What survives is
    diff(u) * k, in which nothing large is subtracted from anything large. See
    core/state.py for why that matters. Titin still needs a TRUE position, so
    the rest frame is passed in as `offsets` and added back for the tip alone.

    Args:
        u: (n_crowns,) crown displacements from rest (nm)
        offsets: (n_crowns,) rest-frame crown positions, topology.crown_offsets
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
    # Spring extension IS the displacement difference: the M-line is a fixed
    # anchor at u = 0 and the rest lengths cancel. Exactly zero at rest.
    spring_forces = jnp.diff(jnp.concatenate([jnp.array([0.0]), u])) * thick_k

    # Calculate titin force for last crown — a true axial position is needed
    myo_loc = offsets[-1] + u[-1]
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
    u_thick: jnp.ndarray,
    offsets_thick: jnp.ndarray,
    z_line: float,
    lattice_spacing: float,
    constants: 'DynamicParams',
) -> jnp.ndarray:
    """Vectorized thick filament passive forces for all thick filaments.

    Args:
        u_thick: (n_thick, n_crowns) crown displacements from rest
        offsets_thick: (n_thick, n_crowns) rest frame, topology.crown_offsets
        z_line: Z-line position
        lattice_spacing: Lattice spacing
        constants: DynamicParams carrying thick_k and titin_a/b/rest

    Returns:
        forces: (n_thick, n_crowns) net force on each crown
    """
    compute_fn = partial(
        compute_thick_passive_forces_single,
        thick_k=constants.thick_k,
        z_line=z_line,
        lattice_spacing=lattice_spacing,
        titin_a=constants.titin_a,
        titin_b=constants.titin_b,
        titin_rest=constants.titin_rest,
        n_titin=_N_TITIN_PER_THICK,
    )
    return jax.vmap(compute_fn)(u_thick, offsets_thick)


# ============================================================================
# THIN FILAMENT PASSIVE FORCES
# ============================================================================

def compute_thin_passive_forces(u_thin: jnp.ndarray, thin_k) -> jnp.ndarray:
    """Net backbone force on each mechanical node of every thin filament.

    COORDINATES ARE DISPLACEMENTS, and the Z-line leaves this function
    entirely. The thin frame is anchored on the Z-disc (absolute position is
    `z_line - node_offsets + u`), and `node_rests` is exactly the diff of
    `node_offsets`, so both the rest lengths and z_line cancel. The filament
    therefore follows the Z-disc rigidly with no force and no state update —
    which is why the simulation loop no longer shifts thin positions when the
    Z-line moves. See core/state.py.

    Row-wise: the Z-disc is the fixed anchor u = 0 past the last node, and the
    pointed-end node has no spring on its M-line side.

    Args:
        u_thin: (n_thin, n_nodes) node displacements from rest (nm)
        thin_k: segment stiffness (pN/nm), core.state.thin_segment_k

    Returns:
        forces: (n_thin, n_nodes) net force on each node
    """
    spring_forces = jnp.diff(u_thin, axis=1, append=0.0) * thin_k
    return jnp.diff(spring_forces, axis=1, prepend=0.0)


# ============================================================================
# CROSSBRIDGE TWO-SPRING PRIMITIVES
# ============================================================================
#
# The head is a linear (globular) spring of length r in series with an angular
# (converter) spring at angle theta, anchored on the thick filament and reaching
# to a binding site on the thin filament a lattice spacing d away:
#
#     U(r, theta) = 0.5*g_k*(r - g_rest)^2 + 0.5*c_k*(theta - c_rest)^2
#
# In the head's OWN polar frame that potential exerts two orthogonal forces:
#
#     F_r     = dU/dr        = g_k*(r - g_rest)          along the head
#     F_theta = (1/r) dU/dtheta = (c_k/r)*(theta - c_rest)   across it
#
# and the forces the filaments feel are that pair ROTATED into the filament
# frame by the head's own angle:
#
#     [F_axial ]   [cos(theta)  -sin(theta)] [F_r    ]
#     [F_radial] = [sin(theta)   cos(theta)] [F_theta]
#
# with F_axial = dU/dx and F_radial = dU/dd. Writing it as a rotation is the
# point: a rotation cannot lose a projection or flip one term's sign without
# breaking orthonormality, and the invariant
#
#     F_axial^2 + F_radial^2 == F_r^2 + F_theta^2
#
# catches that whole class of error. It exists because the class was real —
# the load-dependent rate path in transitions.py fed the Bell exponent
# `F_r + F_theta`, which is neither an axial force nor any other force: both
# projections dropped and the converter term's sign flipped relative to the
# axial law. Every consumer of this potential now goes through these five
# functions, so a change to the physics reaches all of them or none.
#
# All five are ELEMENTWISE. The force path calls them on (n_xb_total,) arrays,
# the rate path on (n_bins,) grid geometries; nothing here reduces or reshapes.


def xb_geometry(x_dist, lattice_spacing):
    """Head polar geometry from its axial offset and the lattice spacing.

    Args:
        x_dist: Axial offset from crown to binding site (nm), site minus crown
        lattice_spacing: Radial offset (nm) — a scalar, or per-element

    Returns:
        (r, theta, cos_theta, sin_theta). r is floored at 1e-10 in the
        divisions and the trig ratios so a head sitting exactly on its site
        cannot produce a NaN; theta comes from atan2 and needs no floor.
    """
    r = jnp.sqrt(x_dist**2 + lattice_spacing**2)
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    cos_theta = x_dist / r_safe
    sin_theta = lattice_spacing / r_safe
    theta = jnp.arctan2(lattice_spacing, x_dist)
    return r, theta, cos_theta, sin_theta


def xb_springs_for_state(xb_states_flat, params):
    """The two-spring parameters of whichever configuration a head is in.

    THREE configurations, not two: state 1 Loose (*_weak), state 2 Tight_1
    (*_tight_1) and state 3 Tight_2 (*_strong). Tight_1 gained its own rest
    configuration with the split stroke (S129) — see the _DYNAMIC_DEFAULTS
    block on xb_c_rest_tight_1. A plain three-way select, no interpolation:
    there is no fraction to forget.

    Detached heads (states 0, 4, 5) fall through to the weak block. Nothing
    reads their force — every caller masks on `is_bound` — so the value is
    arbitrary; what matters is that it is finite and needs no branch.

    Args:
        xb_states_flat: (n,) integer crossbridge states
        params: DynamicParams carrying the xb_{g,c}_{k,rest}_* fields

    Returns:
        (g_k, g_rest, c_k, c_rest), each (n,)
    """
    is_t2 = (xb_states_flat == 3)
    is_t1 = (xb_states_flat == 2)

    # Globular domain (linear spring)
    g_k = jnp.where(is_t2, params.xb_g_k_strong,
                    jnp.where(is_t1, params.xb_g_k_tight_1, params.xb_g_k_weak))
    g_rest = jnp.where(is_t2, params.xb_g_rest_strong,
                       jnp.where(is_t1, params.xb_g_rest_tight_1, params.xb_g_rest_weak))

    # Converter domain (angular spring)
    c_k = jnp.where(is_t2, params.xb_c_k_strong,
                    jnp.where(is_t1, params.xb_c_k_tight_1, params.xb_c_k_weak))
    c_rest = jnp.where(is_t2, params.xb_c_rest_strong,
                       jnp.where(is_t1, params.xb_c_rest_tight_1, params.xb_c_rest_weak))

    return g_k, g_rest, c_k, c_rest


def xb_elastic_energy(r, theta, g_k, g_rest, c_k, c_rest):
    """Elastic energy stored in the two springs at this geometry.

        U = 0.5*g_k*(r - g_rest)^2 + 0.5*c_k*(theta - c_rest)^2

    Returns pN*nm. The rate path divides by kT to get kT units; the caller
    does that, not this function, so the mechanical and chemical sides cannot
    disagree about what a joule is.
    """
    return 0.5 * g_k * (r - g_rest)**2 + 0.5 * c_k * (theta - c_rest)**2


def xb_polar_forces(r, theta, g_k, g_rest, c_k, c_rest):
    """The two spring forces in the head's own frame.

        F_r     = g_k*(r - g_rest)             along the head
        F_theta = (c_k/r)*(theta - c_rest)     perpendicular to it

    These are the gradients of `xb_elastic_energy` in polar coordinates. They
    are NOT forces on a filament and must never be added together: they are
    orthogonal components, and summing them is the bug this module's header
    describes. Rotate them with `polar_to_filament` first.

    `f . r_hat` — the load along the head, if a rate ever wants it — is the
    first return value, no extra function needed.
    """
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    F_r = g_k * (r - g_rest)
    # Written (1/r)*c_k*dtheta rather than (c_k/r)*dtheta: same quantity, but
    # the reciprocal-first association is what the axial and radial laws used
    # before they were consolidated, and keeping it makes the consolidation
    # bit-exact against the golden master instead of last-ulp-different.
    F_theta = (1.0 / r_safe) * c_k * (theta - c_rest)
    return F_r, F_theta


def polar_to_filament(F_r, F_theta, cos_theta, sin_theta):
    """Rotate the head's polar forces into the filament frame.

        F_axial  = F_r*cos(theta) - F_theta*sin(theta)   ( = dU/dx )
        F_radial = F_r*sin(theta) + F_theta*cos(theta)   ( = dU/dd )

    THE MINUS ON THE AXIAL CONVERTER TERM IS THE S50 SIGN AND IS NOT
    NEGOTIABLE (see the DO-NOT-REGRESS list). It is not a convention: the
    rotation is orthonormal, and flipping it breaks
    F_axial^2 + F_radial^2 == F_r^2 + F_theta^2.

    Returns:
        (F_axial, F_radial), same shape as the inputs.
    """
    F_axial = F_r * cos_theta - F_theta * sin_theta
    F_radial = F_r * sin_theta + F_theta * cos_theta
    return F_axial, F_radial


# ============================================================================
# CROSSBRIDGE FORCES (Fully Vectorized)
# ============================================================================

def compute_xb_forces_vectorized(
    positions_thick: jnp.ndarray,
    positions_mono: jnp.ndarray,
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
    (states 2 and 3), weak otherwise. Only state 1 is both weak AND bound;
    states 0, 4 and 5 are detached and carry no spring (rate_functions.py module
    docstring), and n_bound counts states 1-3 only (metrics_fn.py). They are
    given the weak parameter set by the `jnp.where` below purely because it
    needs some value -- force is gated on xb_bound_to (-1 when unbound), not on
    state, so a detached head contributes nothing regardless. An earlier version
    of this sentence listed state 4 as weak-bound, which contradicted both.

    Forces are equal and opposite: whatever a head does to its crown, it does the
    negative of to its bound MONOMER, and the monomer hands that reaction to the
    two thin nodes it is interpolated between, with the same two weights
    core.state.monomer_axial blends their displacements with. The scatter is the
    transpose of that gather, which is what keeps the thin rows of the residual
    the exact gradient of the energy. Accumulation uses segment_sum because many
    heads can share a node, and atomics avoid materializing an
    (n_xb x n_nodes) scatter matrix.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        positions_mono: (n_thin, n_mono) monomer positions, from monomer_axial
        xb_states: (n_thick, n_crowns, n_xb_per_crown) XB states (0-5)
        xb_bound_to: (n_thick, n_crowns, n_xb_per_crown) monomer indices
            (-1 if unbound), thin from geometry
        lattice_spacing: Lattice spacing (nm)
        params: Parameter dictionary with XB spring parameters
        geometry: SarcTopology with xb_to_thin_id, mono_node, mono_xi.

    Returns:
        forces_on_thick: (n_thick, n_crowns) net XB force on each crown
        forces_on_thin: (n_thin, n_nodes) net XB force on each thin node
    """
    forces, mono_flat_idx = _xb_axial_forces_flat(
        positions_thick, positions_mono, xb_states, xb_bound_to,
        lattice_spacing, params, geometry)

    n_thick, n_crowns = positions_thick.shape
    n_thin, n_mono = positions_mono.shape
    n_xb_per_crown = xb_states.shape[2]
    n_crowns_total = n_thick * n_crowns

    # Accumulate thick filament forces (reshape+sum - unchanged, regular pattern)
    forces_per_crown = forces.reshape(n_crowns_total, n_xb_per_crown)
    forces_on_thick_flat = forces_per_crown.sum(axis=-1)
    forces_on_thick = forces_on_thick_flat.reshape(n_thick, n_crowns)

    # Split each reaction between node e and node e + 1. Segment n_nodes of
    # each filament is the Z-disc anchor: it takes the share of a monomer past
    # the last node, and is dropped, because the Z-disc is not an unknown.
    n_seg = geometry.n_nodes + 1
    e = geometry.mono_node.reshape(-1)[mono_flat_idx]
    xi = geometry.mono_xi.reshape(-1)[mono_flat_idx]
    # One scatter over both shares, not one per share: a segment_sum is a
    # GPU scatter, and this runs at every force evaluation of the Newton/CG loop.
    # UNBOUND HEADS ARE SENT PAST THE END AND DROPPED. They carry zero force, but
    # left in place they all clip to monomer 0 of their thin, and invalid heads
    # share the (0, 0) placeholder, so over half the updates write ONE segment.
    # A GPU scatter-add that hammers one slot is several times slower than the
    # same scatter with spread indices.
    node_flat = jnp.where(xb_bound_to.reshape(-1) >= 0,
                          (mono_flat_idx // n_mono) * n_seg + e, n_thin * n_seg)
    forces_on_nodes = jax.ops.segment_sum(
        jnp.concatenate([-(1.0 - xi) * forces, -xi * forces]),
        jnp.concatenate([node_flat, node_flat + 1]),
        num_segments=n_thin * n_seg)
    forces_on_thin = forces_on_nodes.reshape(n_thin, n_seg)[:, :-1]

    return forces_on_thick, forces_on_thin


def xb_axial_force_by_state(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    lattice_spacing: float,
    params: 'DynamicParams',
    geometry: 'SarcTopology'
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Total axial crossbridge force, split by bound state.

    Answers "did force rise because MORE heads are strongly bound, or because
    EACH strongly bound head is carrying more?" -- the state occupancies alone
    cannot separate those. Dividing each sum by the matching state count gives
    the mean force a head in that state is exerting; if two conditions differ in
    the sums but not in the per-head means, the change is pure occupancy.

    Uses the same `_xb_axial_forces_flat` as `compute_xb_forces_vectorized`, so
    the two can never disagree about the force law, and

        weak + tight_1 + tight_2 == compute_xb_forces_vectorized(...)[0].sum()

    exactly. Note that is the total XB force on the thick filaments, which is
    NOT `axial_force` at the M-line -- that is read from backbone spring strain
    and includes titin (see metrics_fn module docstring).

    Returns:
        (f_weak, f_tight_1, f_tight_2) scalar summed axial force (pN) from
        crossbridges in states 1, 2 and 3 respectively.
    """
    forces, _ = _xb_axial_forces_flat(
        positions_thick, positions_thin, xb_states, xb_bound_to,
        lattice_spacing, params, geometry)
    s = xb_states.reshape(-1)
    return (jnp.sum(jnp.where(s == 1, forces, 0.0)),
            jnp.sum(jnp.where(s == 2, forces, 0.0)),
            jnp.sum(jnp.where(s == 3, forces, 0.0)))


def _xb_axial_forces_flat(
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    lattice_spacing: float,
    params: 'DynamicParams',
    geometry: 'SarcTopology'
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Axial force exerted by EVERY crossbridge, flat, plus its thin-site index.

    The two-spring force law lives in the primitives at the top of this module
    and only there. This function is the flat axial view of it; the solver path
    (`compute_xb_forces_vectorized`), the by-state accounting
    (`xb_axial_force_by_state`), the radial path (`_xb_radial_force_total`) and
    THE LOAD-DEPENDENT RATE PATH (`transitions.xb_rate_matrix`, which imports
    `xb_polar_forces` and `polar_to_filament` directly) all read the same
    primitives, so a change to the physics cannot reach one and miss another.
    The rate path is the one that did miss: it built its own scalar inline and
    was wrong from the initial commit until 2026-09-09. See the primitives
    header for the geometry, the sign conventions and the invariant.

    `positions_thin` is MONOMER positions (core.state.monomer_axial), not nodes.

    Returns:
        forces: (n_xb_total,) axial force per crossbridge, 0 where unbound
        thin_flat_idx: (n_xb_total,) flat (thin * n_mono + monomer) index of the
            monomer each XB acts on
    """
    n_thick, n_crowns = positions_thick.shape
    n_thin, n_mono = positions_thin.shape
    n_xb_per_crown = xb_states.shape[2]

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
    mono_idx = xb_bound_flat           # Runtime state (or -1 if unbound)

    # Handle unbound XBs: clip to valid range (forces will be zeroed anyway)
    thin_idx_safe = jnp.clip(thin_idx, 0, n_thin - 1)
    mono_idx_safe = jnp.clip(mono_idx, 0, n_mono - 1)
    thin_flat_idx = thin_idx_safe * n_mono + mono_idx_safe

    # Get binding site positions using geometry lookup
    bs_positions = positions_thin_flat[thin_flat_idx]

    # Calculate distances
    x_dist = bs_positions - xb_positions_flat

    # Two-spring force law, via the shared primitives at the top of this module.
    r, theta, cos_theta, sin_theta = xb_geometry(x_dist, lattice_spacing)
    g_k, g_rest, c_k, c_rest = xb_springs_for_state(xb_states_flat, params)
    F_r, F_theta = xb_polar_forces(r, theta, g_k, g_rest, c_k, c_rest)
    f_axial = polar_to_filament(F_r, F_theta, cos_theta, sin_theta)[0]

    # Zero force for unbound XBs
    forces = jnp.where(is_bound, f_axial, 0.0)

    return forces, thin_flat_idx


def xb_axial_work(
    pos_thick_old: jnp.ndarray,
    pos_thin_old: jnp.ndarray,
    ls_old,
    pos_thick_new: jnp.ndarray,
    pos_thin_new: jnp.ndarray,
    ls_new,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params: 'DynamicParams',
    geometry: 'SarcTopology',
) -> jnp.ndarray:
    """Work done ON the filament lattice BY the crossbridges over one step.

    This is the quantity a tension-cost study wants in the numerator of an
    efficiency: what the motors actually delivered, per ATP they actually spent.
    It is NOT the work the half-sarcomere does externally — under an isometric
    hold that is exactly zero while heads are still cycling and spending. Use
    `sarcomere_work` for the external quantity; the two are different questions
    and metrics_fn reports both.

    SIGN CONVENTION, derived from this module and not chosen here.
    `x = site - crown` (see `_xb_axial_forces_flat`) and `F_axial = dU/dx` (see
    `polar_to_filament`), so the force the SPRING exerts is -F_axial and the
    work a head does on the lattice is `-F_axial * dx`. A head pulling its site
    towards the M-line as the lattice moves that way returns a POSITIVE number.

    THE ATTACHED POPULATION IS THE NEW ONE, and both force evaluations use it.
    States change in the kinetics phase and positions change in the solve, in
    that order, so the heads that were attached DURING the displacement are the
    heads `xb_states` names after kinetics. Passing the old states instead would
    credit a head for a displacement it was not attached for.

    Trapezoid in force: `0.5*(f_old + f_new) * dx`. The force law is nonlinear
    in position, so this is second-order accurate in the step, not exact — the
    error falls quadratically as dt shrinks, which
    `local_projects/regression/xb_force_law_invariant.py` checks against
    `xb_elastic_energy` directly.

    >>> AXIAL COMPONENT ONLY. In dynamic-LS mode the lattice spacing `d` also
        changes over the step, and the radial force does real work that this
        does not capture. `F_radial = dU/dd` is available from the same
        primitives if that term is ever wanted; it is deliberately not folded in
        here, because then the number would stop being comparable between fixed
        and dynamic LS runs without saying so.

    Args:
        pos_thick_old/pos_thin_old/ls_old: lattice before the equilibrium solve
        pos_thick_new/pos_thin_new/ls_new: lattice after it
        xb_states, xb_bound_to: the post-kinetics population (see above)
        params: DynamicParams carrying the xb_{g,c}_{k,rest}_* fields
        geometry: SarcTopology, for xb_to_thin_id

    Returns:
        Scalar work in pN*nm, summed over every crossbridge. Unbound heads carry
        zero force at both ends, so no mask is applied or needed.
    """
    f_old, thin_flat_idx = _xb_axial_forces_flat(
        pos_thick_old, pos_thin_old, xb_states, xb_bound_to,
        ls_old, params, geometry)
    f_new, _ = _xb_axial_forces_flat(
        pos_thick_new, pos_thin_new, xb_states, xb_bound_to,
        ls_new, params, geometry)

    # Reuse the site mapping the force helper already resolved rather than
    # rebuilding it: it is the same clip-and-flatten, and two copies could drift.
    n_xb_per_crown = xb_states.shape[2]

    def _x(pos_thick, pos_thin):
        return (pos_thin.reshape(-1)[thin_flat_idx]
                - jnp.repeat(pos_thick.reshape(-1), n_xb_per_crown))

    dx = _x(pos_thick_new, pos_thin_new) - _x(pos_thick_old, pos_thin_old)
    return -jnp.sum(0.5 * (f_old + f_new) * dx)


# ============================================================================
# COMBINED FORCE CALCULATION
# ============================================================================

def compute_forces_vectorized(
    u_thick: jnp.ndarray,
    u_thin: jnp.ndarray,
    z_line: float,
    lattice_spacing: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    constants: 'DynamicParams',
    topology: 'SarcTopology',
) -> jnp.ndarray:
    """Complete vectorized force residual calculation.

    This is the JAX-native replacement for compute_forces_from_positions().
    It is fully JIT-compilable and GPU-ready.

    The residual is: F(u) = 0 at equilibrium

    THE DEGREES OF FREEDOM ARE DISPLACEMENTS, not absolute positions (see
    core/state.py). The backbone laws use them directly and need no rest
    lengths. The crossbridge and titin paths need true axial coordinates, so
    this function reconstructs them from the topology rest frames and hands
    those on. Those two paths subtract quantities that are O(10-25 nm) and set
    by lattice geometry rather than by force balance, so they neither shrink
    under a stiffness sweep nor cancel catastrophically.

    Args:
        u_thick: (n_thick, n_crowns) crown displacements from rest
        u_thin: (n_thin, n_nodes) thin node displacements from rest
        z_line: Z-line position
        lattice_spacing: Lattice spacing
        xb_states: Crossbridge states
        xb_bound_to: Crossbridge binding info (site_idx only, thin from topology)
        constants: DynamicParams — backbone, titin and crossbridge springs
        topology: SarcTopology with xb_to_thin_id, crown_offsets and the
            monomer interpolation (mono_offsets, mono_node, mono_xi).

    Returns:
        forces: (n_thick_nodes + n_thin_nodes,) flattened force residual
    """
    # 1. Thick filament passive forces (with titin)
    forces_thick = compute_thick_passive_forces_vectorized(
        u_thick, topology.crown_offsets, z_line, lattice_spacing, constants)

    # 2. Thin filament passive forces
    forces_thin = compute_thin_passive_forces(u_thin, thin_segment_k(constants, topology))

    # 3. Crossbridge forces — these need true axial coordinates
    xb_forces_thick, xb_forces_thin = compute_xb_forces_vectorized(
        thick_axial(u_thick, topology), monomer_axial(u_thin, topology, z_line),
        xb_states, xb_bound_to,
        lattice_spacing, constants, topology
    )

    return jnp.concatenate([
        (forces_thick + xb_forces_thick).flatten(),
        (forces_thin + xb_forces_thin).flatten(),
    ])


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

        force = sum over thick filaments of u[0] * thick_k

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
        topology: unused; kept so the signature matches the other metric
            readers. The bare-zone distance IS crown_offsets[:, 0], which is
            exactly what the stored displacement is measured from.

    Returns:
        force: Total axial force at M-line (pN)
    """
    force_per_thick = state.thick.displacement[:, 0] * constants.thick_k
    return jnp.sum(force_per_thick)


# ============================================================================
# RADIAL FORCE FUNCTIONS (for dynamic lattice spacing solver)
# ============================================================================

# Titin molecules per thick filament, read by BOTH the axial path
# (compute_thick_passive_forces_vectorized) and the radial path below. The two
# paths describe the same physical tethers resolved along different axes; one
# constant keeps the axial and radial magnitudes mutually consistent.
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
        xb_bound_to: (n_thick, n_crowns, n_xb_per_crown) monomer indices (-1 unbound)
        positions_thick: (n_thick, n_crowns) crown axial positions
        positions_thin: (n_thin, n_mono) monomer axial positions (monomer_axial)
        lattice_spacing: Current lattice spacing d (nm)
        params: DynamicParams with XB spring constants
        topology: SarcTopology with xb_to_thin_id

    Returns:
        Scalar total radial force (pN). Positive = outward (increasing d).
    """
    n_thin, n_mono = positions_thin.shape
    n_xb_per_crown = xb_states.shape[2]

    xb_states_flat = xb_states.reshape(-1)
    xb_bound_flat = xb_bound_to.reshape(-1)

    xb_positions_flat = jnp.repeat(positions_thick.reshape(-1), n_xb_per_crown)
    positions_thin_flat = positions_thin.reshape(-1)

    is_bound = (xb_states_flat >= 1) & (xb_states_flat <= 3) & (xb_bound_flat >= 0)

    thin_idx = topology.xb_to_thin_id
    mono_idx = xb_bound_flat
    thin_idx_safe = jnp.clip(thin_idx, 0, n_thin - 1)
    mono_idx_safe = jnp.clip(mono_idx, 0, n_mono - 1)
    thin_flat_idx = thin_idx_safe * n_mono + mono_idx_safe

    bs_positions = positions_thin_flat[thin_flat_idx]
    x_dist = bs_positions - xb_positions_flat

    # The SAME primitives as _xb_axial_forces_flat, taking the other component
    # of the same rotation. The S129 prototype patch could not reach this
    # function (it rewrote one function's source); sharing the law here removes
    # the axial/radial mismatch that would otherwise exist for every state-2
    # head under dynamic lattice spacing.
    r, theta, cos_theta, sin_theta = xb_geometry(x_dist, lattice_spacing)
    g_k, g_rest, c_k, c_rest = xb_springs_for_state(xb_states_flat, params)
    F_r, F_theta = xb_polar_forces(r, theta, g_k, g_rest, c_k, c_rest)
    f_radial = polar_to_filament(F_r, F_theta, cos_theta, sin_theta)[1]

    forces_radial = jnp.where(is_bound, f_radial, 0.0)
    return jnp.sum(forces_radial)


def _titin_radial_force_total(
    positions_thick: jnp.ndarray,
    z_line: float,
    lattice_spacing: float,
    constants: 'DynamicParams',
) -> float:
    """Total titin radial force from all thick filaments.

    Replicates geometry from compute_thick_passive_forces_single but returns
    the radial component. Differentiable w.r.t. lattice_spacing.

    Args:
        positions_thick: (n_thick, n_crowns) crown positions
        z_line: Z-line position (nm)
        lattice_spacing: Current lattice spacing d (nm)
        constants: DynamicParams carrying titin_a/b/rest

    Returns:
        Scalar total radial titin force (pN). Positive = outward (increasing d).
    """
    myo_loc = positions_thick[:, -1]
    axial_dist = z_line - myo_loc

    titin_length = jnp.sqrt(axial_dist**2 + lattice_spacing**2)
    titin_length_safe = jnp.where(titin_length > 1e-10, titin_length, 1e-10)

    exp_arg = jnp.clip(constants.titin_b * (titin_length - constants.titin_rest), -100.0, 100.0)
    titin_force = jnp.maximum(constants.titin_a * jnp.exp(exp_arg), 0.0)

    sin_angle = lattice_spacing / titin_length_safe
    titin_radial_per_thick = _N_TITIN_PER_THICK * titin_force * sin_angle

    return jnp.sum(titin_radial_per_thick)

