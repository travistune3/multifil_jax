"""
Simulation state: what changes from one timestep to the next.

STATE IS DELIBERATELY EMPTY OF EVERYTHING ELSE
----------------------------------------------
A State holds node positions and molecular states, and nothing more. No
parameters, no geometry, no calcium level, no spring constants. Those live in
DynamicParams (Tier 2), SarcTopology (Tier 1) and Drivers (Tier 3) and are
passed alongside.

The separation is what makes parameter sweeps cheap. If stiffness lived inside
State, a sweep over stiffness would mean a distinct State per sweep point, and
vmap would have to carry a copy of everything. Because parameters are separate,
one State layout serves every sweep point and only the small parameter object is
batched.

It also keeps the simulation honest about what is genuinely time-varying. Rest
lengths, connectivity and crown positions cannot drift during a run, because
they are not in the object that gets updated.

NAMEDTUPLES, NOT DATACLASSES
----------------------------
NamedTuple is a JAX pytree out of the box, is immutable, and has no __dict__ for
JAX to trip over. Updates use ._replace(), which builds a new object sharing the
unchanged arrays:

    new_state = state._replace(thick=state.thick._replace(displacement=new_u))

Nesting means a change to thick filament positions leaves the thin filament
arrays untouched by reference, with no copying.

POSITIONS ARE STORED AS DISPLACEMENTS FROM REST
-----------------------------------------------
`thick.displacement` and `thin.displacement` are offsets from the rest frame,
not absolute axial coordinates. Absolute positions are reconstructed on demand
by thick_axial() / monomer_axial() below.

The reason is float32. Absolute node positions run to ~1000 nm, where
consecutive float32 values are ~1e-4 nm apart, while the strain a backbone
spring actually carries is F/k — about 0.03 nm at default stiffness, and
SMALLER the stiffer the filament. Storing absolute positions means every
backbone force is extracted by cancelling two ~1000 nm floats, so the force
error is k * ulp(1000 nm): it grows linearly with stiffness while the signal
does not. Measured, that was 0.69 pN of spurious net force per node at rest
with nothing attached, and a 14.9% force error at 16x the default stiffness
against a float64 reference.

In displacement coordinates the rest lengths cancel algebraically — both
`crown_offsets` and `node_offsets` ARE the cumulated rest frames — so the
backbone laws collapse to diff(u) * k with no subtraction of large numbers at
all. The error becomes relative (~float32 eps) and stiffness-independent. At
rest with no load the residual is now exactly zero.

STATE INDEX CONVENTIONS
-----------------------
Crossbridges use 0-5 (0 DRX, 1 Loose, 2 Tight_1, 3 Tight_2, 4 Free_2, 5 SRX)
and tropomyosin 0-3 (0 Ca-free blocking, 1 Ca-bound blocking, 2 closed, 3 open).
Both are stored as int8 — with hundreds of thousands of units these arrays are
large, and 8 bits is ample for six states. Beware that jnp.argmax returns int32,
so every sampling site must cast back explicitly on assignment or the arrays
silently widen.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from .params import DynamicParams, StaticParams
from .sarc_geometry import SarcTopology


# =============================================================================
# NAMEDTUPLE STATE HIERARCHY
# =============================================================================
# NamedTuples are immutable JAX-compatible PyTrees that enable efficient
# compilation by avoiding Python dict overhead. Use ._replace() for updates.
#
# State is PURE simulation state: no embedded params, geometry, or constants.
# Filament stiffnesses (thick_k, thin_EA) live in DynamicParams.
# bare_zone (StaticParams) is baked into topology.crown_offsets (Topology, Tier 1).
# Topology (SarcTopology/SarcTopology) passed as separate argument.

class ThickState(NamedTuple):
    """Thick filament state arrays.

    All arrays have leading dimension n_thick (number of thick filaments).
    Spring constant (k) moved to Constants (Tier 2). bare_zone (StaticParams)
    is baked into topology.crown_offsets, not stored as a separate value.
    Structural arrays (crown_starts, connectivity, crown_rests) moved to Topology (Tier 1).
    """
    displacement: jnp.ndarray    # (n_thick, n_crowns) crown offset from rest;
                                 # absolute = topology.crown_offsets + this
    xb_states: jnp.ndarray       # (n_thick, n_crowns, n_xb_per_crown) crossbridge states (0-5), int8
    xb_bound_to: jnp.ndarray     # (n_thick, n_crowns, n_xb_per_crown) bound MONOMER index (-1 if unbound)
    xb_nearest_bs: jnp.ndarray   # (n_thick, n_crowns, n_xb_per_crown) nearest candidate MONOMER index
    xb_distances: jnp.ndarray    # (n_thick, n_crowns, n_xb_per_crown, 2) distances to that monomer


class ThinState(NamedTuple):
    """Thin filament state arrays, one per layer (see SarcTopology).

    All arrays have leading dimension n_thin (number of thin filaments).
    Spring constant in Constants (Tier 2); structure in Topology.
    permissiveness is derived on-demand: (tm_states == 3).astype(float32)
    """
    displacement: jnp.ndarray    # (n_thin, n_nodes) NODE offset from rest;
                                 # absolute = z_line - topology.node_offsets + this
    tm_states: jnp.ndarray       # (n_thin, n_tm) tropomyosin UNIT states (0-3), int8
    bound_to: jnp.ndarray        # (n_thin, n_cand) XB bound to this CANDIDATE (-1 if unbound)


class State(NamedTuple):
    """Pure simulation state — no embedded params, geometry, or constants.

    Use state._replace(field=new_value) for immutable updates.
    For nested updates: state._replace(thick=state.thick._replace(displacement=new_u))

    Removed fields (moved to other tiers):
        - params → Constants (Tier 2), passed as separate arg
        - geometry → SarcTopology (Tier 1), passed as separate arg
        - z_line, pCa, lattice_spacing → Drivers (Tier 3) or Constants
        - preconditioner → rebuilt from Topology + Constants at solve time
        - titin → constants.titin_a/b/rest + topology.titin_connections
        - thick.crown_starts, thick.connectivity → Topology
        - thin.tm_chains, thin.connectivity, thin.face_to_monomers → Topology
    """
    thick: ThickState
    thin: ThinState


def thick_axial(u_thick: jnp.ndarray, topology: 'SarcTopology') -> jnp.ndarray:
    """Absolute crown positions (nm) from the stored displacements.

    `topology.crown_offsets` is the cumulative sum of `crown_rests`, so this is
    the rest frame plus the strain the state is actually carrying. Call it
    wherever a TRUE axial coordinate is needed — crossbridge reach, titin's
    diagonal to the Z-disc, the M-line visibility gate, the overlap window.
    Backbone forces must NOT go through here: they read the displacements
    directly, which is the entire point of the coordinate change.

    Takes the displacement ARRAY, not a State, because the solver's residual
    has displacements and no State.

    Args:
        u_thick: (n_thick, n_crowns) crown displacements, state.thick.displacement

    Returns:
        (n_thick, n_crowns) absolute axial positions, nm from the M-line.
    """
    return topology.crown_offsets + u_thick


def monomer_axial(u_thin: jnp.ndarray, topology: 'SarcTopology', z_line) -> jnp.ndarray:
    """Absolute actin MONOMER positions (nm) from the thin node displacements.

    A monomer between node e = mono_node and the next node toward the Z-disc
    moves with the linear blend (1 - xi) * u[e] + xi * u[e + 1], xi = mono_xi;
    past the last node the Z-disc itself (u = 0) is node e + 1. This is the ONE
    place monomer positions are made — the crossbridge force and work paths, the
    radial path, the binding search and the overlap metrics all call it — and
    compute_xb_forces_vectorized scatters with the same two weights, so the
    thin rows of the residual stay the exact gradient of the energy.

    DISPLACEMENTS ARE INTERPOLATED, NEVER ABSOLUTE POSITIONS. mono_offsets is
    each monomer's own helix rest offset, so a monomer at rest sits exactly
    there, and no ~1000 nm float32 value is ever blended (see POSITIONS ARE
    STORED AS DISPLACEMENTS FROM REST above). Where a monomer coincides with a
    node, xi is exactly 0 and this is bit-for-bit that node's position.

    Args:
        u_thin: (n_thin, n_nodes) node displacements from rest
        z_line: current Z-line position (nm). Resolve drivers before calling.

    Returns:
        (n_thin, n_mono) absolute axial positions, nm from the M-line.
    """
    u_z = jnp.concatenate([u_thin, jnp.zeros((u_thin.shape[0], 1), u_thin.dtype)], axis=1)
    e = topology.mono_node
    xi = topology.mono_xi
    u_mono = ((1.0 - xi) * jnp.take_along_axis(u_z, e, axis=1)
              + xi * jnp.take_along_axis(u_z, e + 1, axis=1))
    return z_line - topology.mono_offsets + u_mono


def thin_segment_k(constants: DynamicParams, topology: 'SarcTopology'):
    """Stiffness (pN/nm) of one thin backbone segment: thin_EA / thin_node_spacing.

    The one place the material rigidity becomes a spring constant. All node
    segments share the same rest length, so one value serves every segment and
    one thin factorization serves every filament in the preconditioner.
    """
    return constants.thin_EA / topology.thin_node_spacing


class Drivers(NamedTuple):
    """Time-varying inputs for the simulation (Tier 3).

    Per-step values, always finite; there is no fallback. The drivers are not
    in DynamicParams. The bundle exists only at the orchestration layer
    (kinetics_step, timestep, compute_all_metrics, KineticsTrace); leaf kernels
    take the individual scalars they use as explicit arguments.
    """
    pCa: jnp.ndarray             # scalar per timestep
    z_line: jnp.ndarray          # scalar per timestep, nm from the M-line
    lattice_spacing: jnp.ndarray # scalar per timestep, nm surface-to-surface

class KineticsTrace(NamedTuple):
    """What the kinetics phase saw, handed forward so metrics need not guess.

    A WITHIN-STEP VALUE ONLY. It carries a whole `State`, so it must never be
    put in a scan carry or a scan output — `kinetics_step` builds it and
    `compute_all_metrics` consumes it inside the same scan body, and nothing
    stacks it over time.

    WHY `state` IS THE MID STATE AND NOT `old_state`. `thick_transitions`
    samples its 6x6 generator from the sarcomere as it stands AFTER
    `update_nearest_neighbors` and `thin_transitions` have run. Metrics that
    rebuild that generator — `atp_expected` and the exported cycle fluxes
    (`xb_detach_atp`, `xb_detach_free`, `xb_give_up`, `atp_net_pi_release`,
    `atp_net_hydrolysis`) — must build it from the same state, or they describe
    a step that never happened. Read
    off `old_state` instead, the error is small (0.06%-0.46% measured, cardiac
    and skeletal, dt 1.0 and 0.1) but it is systematic and free to avoid.

    `drivers` are the ones the kinetics phase ran at, carrying the PRE-solve
    lattice spacing. The mechanics path deliberately does NOT use them:
    `axial_force_at_mline` and the reported `lattice_spacing` are post-solve
    quantities and must be read at the SOLVED spacing. Both are correct; they
    are answering different questions.

    Fields:
        state: post-thin_transitions, pre-thick_transitions State
        drivers: the Drivers the kinetics ran at (pre-solve lattice spacing)
        torn: (n_thick, n_crowns, n_xb_per_crown) bool, heads tropomyosin tore
            off this step. Not recoverable from the before/after states — a torn
            head lands where an ordinary one does — so it is carried, not
            re-derived. See transitions.thin_transitions.
        xb_bins: transitions.XBBins — the binned crossbridge generator the step
            was taken with, and both exponentials of it. Carried rather than
            rebuilt so the sampler and the metrics cannot disagree about the
            step, and so only ONE matrix exponential is taken per step. The bin
            grid is (2 * n_xb_bins, 6, 6) = (400, 6, 6) plus one index per head
            — small, and a within-step value, which is what this trace is for.
    """
    state: 'State'
    drivers: Drivers
    torn: jnp.ndarray
    xb_bins: object


class MetricsDict(dict):
    """dict subclass with attribute access.

    result.metrics.axial_force == result.metrics['axial_force']
    Fully backwards-compatible with plain dict subscript access.
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


jax.tree_util.register_pytree_node(
    MetricsDict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: MetricsDict(zip(keys, values)),
)


# JAX: NamedTuple is a valid pytree leaf that JAX can JIT through.
# Unlike regular classes, NamedTuple is immutable and has no __dict__,
# making it safe for compilation. This is preferred over dataclass
# when you need to store values that will be passed through JIT functions.
class PreconditionerParams(NamedTuple):
    """Storable preconditioner parameters (valid JAX PyTree leaf).

    Stores single-filament tridiagonal arrays for thick and thin filament types.
    All thick filaments share the same base tridiagonal matrix (same spring
    constant, same boundary conditions), and likewise for thin filaments.
    Factor once per type, broadcast across filaments via vmap in_axes=(None, 0).

    Future: per-filament modifications (XB binding stiffness, titin) can be
    applied by building per-filament diagonal arrays and re-factoring via vmap.

    Attributes:
        lower_thick: (n_crowns-1,) sub-diagonal for thick filament type
        diag_thick: (n_crowns,) main diagonal for thick filament type
        upper_thick: (n_crowns-1,) super-diagonal for thick filament type
        lower_thin: (n_nodes-1,) sub-diagonal for thin filament type
        diag_thin: (n_nodes,) main diagonal for thin filament type
        upper_thin: (n_nodes-1,) super-diagonal for thin filament type
        n_thick: Number of thick filaments
        n_crowns: Number of crowns per thick filament
        n_thin: Number of thin filaments
        n_nodes: Number of mechanical nodes per thin filament
    """
    lower_thick: jnp.ndarray
    diag_thick: jnp.ndarray
    upper_thick: jnp.ndarray
    lower_thin: jnp.ndarray
    diag_thin: jnp.ndarray
    upper_thin: jnp.ndarray
    n_thick: int
    n_crowns: int
    n_thin: int
    n_nodes: int




# =============================================================================
# REALIZE STATE (TOPOLOGY-BASED STATE CREATION)
# =============================================================================

def realize_state(
    topology: 'SarcTopology',
    constants: DynamicParams,
    z_line: float,
    pCa: float,
    lattice_spacing: float,
) -> State:
    """Create pure State from topology and constants.

    Returns a State NamedTuple with NO embedded params/geometry/constants.
    These are passed separately to kernels.

    Args:
        topology: SarcTopology (SarcTopology) from SarcTopology.create()
        constants: Constants (DynamicParams) with current physics values
        z_line: Z-line position (nm)
        pCa: Calcium concentration as -log10([Ca])
        lattice_spacing: Lattice spacing (nm)

    Returns:
        state: Pure State NamedTuple compatible with JAX vmap/scan
    """
    n_thick = topology.n_thick
    n_crowns = topology.n_crowns
    n_thin = topology.n_thin
    n_xb_per_crown = topology.n_xb_per_crown

    # =========================================================================
    # THICK FILAMENT STATE (no k or bare_zone — those are in Constants)
    # Structural arrays (crown_starts, connectivity) are in Topology.
    # =========================================================================
    # Rest frame, exactly: crown_offsets IS the cumsum of crown_rests, so the
    # freshly created state carries zero strain and zero backbone force.
    thick_displacement = jnp.zeros((n_thick, n_crowns), dtype=jnp.float32)

    xb_states = jnp.zeros((n_thick, n_crowns, n_xb_per_crown), dtype=jnp.int8)
    xb_bound_to = jnp.full((n_thick, n_crowns, n_xb_per_crown), -1, dtype=jnp.int32)
    xb_nearest_bs = jnp.zeros((n_thick, n_crowns, n_xb_per_crown), dtype=jnp.int32)
    xb_distances = jnp.zeros((n_thick, n_crowns, n_xb_per_crown, 2), dtype=jnp.float32)

    thick_state = ThickState(
        displacement=thick_displacement,
        xb_states=xb_states,
        xb_bound_to=xb_bound_to,
        xb_nearest_bs=xb_nearest_bs,
        xb_distances=xb_distances,
    )

    # =========================================================================
    # THIN FILAMENT STATE (no k — that's in Constants)
    # Structural arrays (tm_chains, connectivity, face_to_monomers) are in Topology.
    # =========================================================================
    # Likewise: node_rests is the diff of node_offsets, so zero here is
    # the exact force-free thin backbone, independent of where the Z-line is.
    thin_displacement = jnp.zeros((n_thin, topology.n_nodes), dtype=jnp.float32)
    tm_states = jnp.zeros((n_thin, topology.n_tm), dtype=jnp.int8)
    bound_to = jnp.full((n_thin, topology.n_cand), -1, dtype=jnp.int32)

    thin_state = ThinState(
        displacement=thin_displacement,
        tm_states=tm_states,
        bound_to=bound_to,
    )

    # =========================================================================
    # ASSEMBLE PURE STATE
    # Titin parameters are in Constants (titin_a/b/rest).
    # Titin connections are in Topology (titin_connections).
    # =========================================================================
    state = State(
        thick=thick_state,
        thin=thin_state,
    )

    return state


# =============================================================================
# PRECONDITIONER PARAMETERS
# =============================================================================

def build_preconditioner_params(
    n_thick: int, n_crowns: int, n_thin: int, n_nodes: int,
    thick_k: float, thin_k: float
) -> PreconditionerParams:
    """Build preconditioner parameters from topology and spring constants.

    Creates single-filament tridiagonal arrays for each filament type.
    All thick filaments share the same tridiagonal matrix (same k, same
    boundary conditions), and likewise for thin. Factor once per type,
    broadcast across filaments at apply time.

    Args:
        n_thick: Number of thick filaments
        n_crowns: Number of crowns per thick filament
        n_thin: Number of thin filaments
        n_nodes: Number of mechanical nodes per thin filament
        thick_k: Thick filament spring constant (pN/nm)
        thin_k: Thin filament spring constant (pN/nm)

    Returns:
        PreconditionerParams with single-filament arrays
    """
    # Single thick filament tridiagonal: diag=-2k (interior), -k (boundary)
    diag_thick = jnp.full((n_crowns,), -2.0 * thick_k)
    diag_thick = diag_thick.at[-1].set(-1.0 * thick_k)  # Last crown boundary

    lower_thick = jnp.full((n_crowns - 1,), thick_k)
    upper_thick = jnp.full((n_crowns - 1,), thick_k)

    # Single thin filament tridiagonal: diag=-k (first boundary), -2k (rest)
    diag_thin = jnp.full((n_nodes,), -2.0 * thin_k)
    diag_thin = diag_thin.at[0].set(-1.0 * thin_k)  # First node boundary

    lower_thin = jnp.full((n_nodes - 1,), thin_k)
    upper_thin = jnp.full((n_nodes - 1,), thin_k)

    return PreconditionerParams(
        lower_thick=lower_thick,
        diag_thick=diag_thick,
        upper_thick=upper_thick,
        lower_thin=lower_thin,
        diag_thin=diag_thin,
        upper_thin=upper_thin,
        n_thick=n_thick,
        n_crowns=n_crowns,
        n_thin=n_thin,
        n_nodes=n_nodes
    )


def get_ca_concentration(pCa: float) -> float:
    """Calculate calcium concentration from pCa.

    Args:
        pCa: -log10([Ca]) value

    Returns:
        ca_concentration: Calcium concentration in Molar (10**(-pCa))
    """
    return 10.0 ** (-pCa)
