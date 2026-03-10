"""
State Structure for JAX Half-Sarcomere

Tiered Architecture:
    Tier 1 (Topology): Shape-determining data. Changing requires recompile.
    Tier 2 (Constants): Physics values. Sweepable without recompile.
    Tier 3 (Drivers): Time-varying inputs (pCa, z_line, lattice_spacing).
    State: Pure simulation state — no embedded params, geometry, or constants.

Kernel signature: kernel(state, constants, drivers, topology, rng_key, *, dt)

Usage:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import realize_state

    topology = SarcTopology.create(nrows, ncols, static_params, constants)
    topology = jax.device_put(topology)

    state = realize_state(topology, constants, z_line, pCa, lattice_spacing)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from .params import DynamicParams, StaticParams, get_default_params
from .sarc_geometry import SarcTopology


# =============================================================================
# NAMEDTUPLE STATE HIERARCHY
# =============================================================================
# NamedTuples are immutable JAX-compatible PyTrees that enable efficient
# compilation by avoiding Python dict overhead. Use ._replace() for updates.
#
# State is PURE simulation state: no embedded params, geometry, or constants.
# Constants (thick_k, thin_k, bare_zone) moved to DynamicParams/Constants.
# Topology (SarcTopology/SarcTopology) passed as separate argument.

class ThickState(NamedTuple):
    """Thick filament state arrays.

    All arrays have leading dimension n_thick (number of thick filaments).
    Spring constant (k) and bare_zone moved to Constants (Tier 2).
    Structural arrays (crown_starts, connectivity) moved to Topology (Tier 1).
    """
    axial: jnp.ndarray           # (n_thick, n_crowns) crown axial positions
    rests: jnp.ndarray           # (n_thick, n_crowns) rest spacings between crowns
    xb_states: jnp.ndarray       # (n_thick, n_crowns, 3) crossbridge states (1-6)
    xb_bound_to: jnp.ndarray     # (n_thick, n_crowns, 3) bound site indices (-1 if unbound)
    xb_nearest_bs: jnp.ndarray   # (n_thick, n_crowns, 3) nearest binding site indices
    xb_distances: jnp.ndarray    # (n_thick, n_crowns, 3, 2) distances to nearest BS


class ThinState(NamedTuple):
    """Thin filament state arrays.

    All arrays have leading dimension n_thin (number of thin filaments).
    Spring constant (k) moved to Constants (Tier 2).
    Structural arrays (tm_chains, connectivity, face_to_sites, n_sites_per_face)
    moved to Topology (Tier 1).
    """
    axial: jnp.ndarray           # (n_thin, n_sites) binding site axial positions
    rests: jnp.ndarray           # (n_thin, n_sites) rest spacings between sites
    tm_states: jnp.ndarray       # (n_thin, n_sites) tropomyosin states (0-3)
    permissiveness: jnp.ndarray  # (n_thin, n_sites) permissiveness (float 0-1)
    subject_to_coop: jnp.ndarray # (n_thin, n_sites) cooperative status (bool)
    bound_to: jnp.ndarray        # (n_thin, n_sites) XB bound to this site (-1 if unbound)


class State(NamedTuple):
    """Pure simulation state — no embedded params, geometry, or constants.

    Use state._replace(field=new_value) for immutable updates.
    For nested updates: state._replace(thick=state.thick._replace(axial=new_axial))

    Removed fields (moved to other tiers):
        - params → Constants (Tier 2), passed as separate arg
        - geometry → SarcTopology (Tier 1), passed as separate arg
        - z_line, pCa, lattice_spacing → Drivers (Tier 3) or Constants
        - preconditioner → rebuilt from Topology + Constants at solve time
        - titin → constants.titin_a/b/rest + topology.titin_connections
        - thick.crown_starts, thick.connectivity → Topology
        - thin.tm_chains, thin.connectivity, thin.face_to_sites → Topology
    """
    thick: ThickState
    thin: ThinState


class Drivers(NamedTuple):
    """Time-varying inputs for the simulation (Tier 3).

    When a driver is constant/swept, its value is in Constants and
    the Drivers field is NaN (sentinel).
    """
    pCa: jnp.ndarray             # scalar per timestep, NaN if in Constants
    z_line: jnp.ndarray          # scalar per timestep, NaN if in Constants
    lattice_spacing: jnp.ndarray # scalar per timestep, NaN if in Constants

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
        lower_thin: (n_sites-1,) sub-diagonal for thin filament type
        diag_thin: (n_sites,) main diagonal for thin filament type
        upper_thin: (n_sites-1,) super-diagonal for thin filament type
        n_thick: Number of thick filaments
        n_crowns: Number of crowns per thick filament
        n_thin: Number of thin filaments
        n_sites: Number of binding sites per thin filament
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
    n_sites: int




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
    n_sites = topology.n_sites

    # =========================================================================
    # THICK FILAMENT STATE (no k or bare_zone — those are in Constants)
    # Structural arrays (crown_starts, connectivity) are in Topology.
    # =========================================================================
    thick_axial = jnp.broadcast_to(
        topology.crown_offsets[None, :],
        (n_thick, n_crowns)
    ).copy()

    thick_rests = jnp.broadcast_to(
        topology.crown_rests[None, :],
        (n_thick, n_crowns)
    ).copy()

    xb_states = jnp.ones((n_thick, n_crowns, 3), dtype=jnp.int32)
    xb_bound_to = jnp.full((n_thick, n_crowns, 3), -1, dtype=jnp.int32)
    xb_nearest_bs = jnp.zeros((n_thick, n_crowns, 3), dtype=jnp.int32)
    xb_distances = jnp.zeros((n_thick, n_crowns, 3, 2), dtype=jnp.float32)

    thick_state = ThickState(
        axial=thick_axial,
        rests=thick_rests,
        xb_states=xb_states,
        xb_bound_to=xb_bound_to,
        xb_nearest_bs=xb_nearest_bs,
        xb_distances=xb_distances,
    )

    # =========================================================================
    # THIN FILAMENT STATE (no k — that's in Constants)
    # Structural arrays (tm_chains, connectivity, face_to_sites) are in Topology.
    # =========================================================================
    thin_axial = z_line - topology.binding_offsets
    thin_rests = topology.binding_rests
    tm_states = jnp.zeros((n_thin, n_sites), dtype=jnp.int32)
    permissiveness = jnp.zeros((n_thin, n_sites), dtype=jnp.float32)
    subject_to_coop = jnp.zeros((n_thin, n_sites), dtype=jnp.bool_)
    bound_to = jnp.full((n_thin, n_sites), -1, dtype=jnp.int32)

    thin_state = ThinState(
        axial=thin_axial,
        rests=thin_rests,
        tm_states=tm_states,
        permissiveness=permissiveness,
        subject_to_coop=subject_to_coop,
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
    n_thick: int, n_crowns: int, n_thin: int, n_sites: int,
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
        n_sites: Number of binding sites per thin filament
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
    diag_thin = jnp.full((n_sites,), -2.0 * thin_k)
    diag_thin = diag_thin.at[0].set(-1.0 * thin_k)  # First site boundary

    lower_thin = jnp.full((n_sites - 1,), thin_k)
    upper_thin = jnp.full((n_sites - 1,), thin_k)

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
        n_sites=n_sites
    )


def get_state_summary(state: State) -> str:
    """Get human-readable summary of state structure.

    Args:
        state: State NamedTuple

    Returns:
        summary: String describing the state
    """
    thick = state.thick
    thin = state.thin
    n_thick = thick.axial.shape[0]
    n_crowns = thick.axial.shape[1]
    n_thin = thin.axial.shape[0]
    n_sites = thin.axial.shape[1]
    thick_axial_shape = thick.axial.shape
    thick_xb_shape = thick.xb_states.shape
    thin_axial_shape = thin.axial.shape
    thin_tm_shape = thin.tm_states.shape

    total_xbs = n_thick * n_crowns * 3

    summary = f"""
Half-Sarcomere State Summary
{'='*60}
Thick Filaments: {n_thick}
  Crowns per filament: {n_crowns}
  Total crossbridges: {total_xbs}

Thin Filaments: {n_thin}
  Binding sites per filament: {n_sites}
  Total binding sites: {n_thin * n_sites}

State Arrays:
  thick.axial: {thick_axial_shape}
  thick.xb_states: {thick_xb_shape}
  thin.axial: {thin_axial_shape}
  thin.tm_states: {thin_tm_shape}
{'='*60}
"""
    return summary


def get_ca_concentration(pCa: float) -> float:
    """Calculate calcium concentration from pCa.

    Args:
        pCa: -log10([Ca]) value

    Returns:
        ca_concentration: Calcium concentration in Molar (10**(-pCa))
    """
    return 10.0 ** (-pCa)


def resolve_value(driver_val: jnp.ndarray, constant_val: jnp.ndarray) -> jnp.ndarray:
    """Resolve a driver value: use driver if valid, else fall back to constant.

    This keeps physics kernels clean — no conditional branching, just jnp.where.

    Args:
        driver_val: Value from Drivers (NaN sentinel if not a trace)
        constant_val: Fallback value from Constants

    Returns:
        Resolved value (driver if not NaN, else constant)
    """
    return jnp.where(jnp.isnan(driver_val), constant_val, driver_val)


def encode_xb_address(thick_idx: int, crown_idx: int, xb_idx: int) -> int:
    """Encode crossbridge address as single integer.

    Used for thin filament 'bound_to' array.

    Args:
        thick_idx: Thick filament index
        crown_idx: Crown index within thick filament
        xb_idx: Crossbridge index within crown (0-2)

    Returns:
        encoded: Single integer encoding all three indices
    """
    # Pack into single int: thick_idx*1000000 + crown_idx*100 + xb_idx
    return thick_idx * 1000000 + crown_idx * 100 + xb_idx


def decode_xb_address(encoded: int) -> Tuple[int, int, int]:
    """Decode crossbridge address from single integer.

    Args:
        encoded: Encoded address from encode_xb_address()

    Returns:
        thick_idx, crown_idx, xb_idx: The three indices
    """
    xb_idx = encoded % 100
    crown_idx = (encoded // 100) % 10000
    thick_idx = encoded // 1000000
    return thick_idx, crown_idx, xb_idx

