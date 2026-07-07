"""
Cooperativity Calculations for JAX Half-Sarcomere

This module handles tropomyosin cooperative activation.
Key mechanism: Sites in state 2 (closed) activate neighboring sites,
making them more likely to transition to permissive states. Note that 'neighboring' sites are face aware, meaning only sites on the same tm-chain are considered. 

From current code:
- hs.py: set_subject_to_cooperativity()
- af.py: TmSite cooperativity logic

Key concepts:
- Span: Distance over which cooperative effect acts
- Force-dependent: Span decreases with force (tension turns off cooperativity)
- State-dependent: Only sites in state 2 exert cooperative effect
- Tropomyosin-chain: Every other binding site is considered on the same tm chain, and cooperativiy calculations are tm-chain aware.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams
    from multifil_jax.core.sarc_geometry import SarcTopology


# ============================================================================
# COOPERATIVE SPAN CALCULATION
# ============================================================================

def calculate_cooperative_span(tension: jnp.ndarray,
                               span_base: float = 62.0,
                               span_force50: float = -8.0,
                               span_steep: float = 1.0) -> jnp.ndarray:
    """Calculate cooperative span based on local tension.

    The span increases with increasing force.

    Args:
        tension: (n_sites,) local force/tension at each site (pN)
        span_base: Maximum span at zero force (nm)
        span_force50: Force at which span is half-maximum (pN)
        span_steep: Steepness of force effect

    Returns:
        span: (n_sites,) cooperative activation span (nm)

    How to modify:
        - Asymmetric response: Different functions for compression/tension
        - Multi-factor: Add calcium dependence
        - Saturation: Add upper limit on span
        - Hysteresis: Make span depend on previous state

    Formula (from hs.py line 1623):
        span = 0.5 * base * (1 + tanh(steep * (F50 + F)))
    """

    span = 0.5 * span_base * (1.0 + jnp.tanh(span_steep * (span_force50 + tension)))

    # Ensure span is non-negative
    span = jnp.maximum(span, 0.0)

    return span


def get_site_tensions(thin_forces: jnp.ndarray) -> jnp.ndarray:
    """Calculate cumulative tension at each site from crossbridge forces.

    Tension = sum of forces from this site to M-line (end of array).
    tensions[i] = sum of forces[i:]

    This matches OOP: np.triu(np.ones(n)) @ forces

    Args:
        thin_forces: (n_thin, n_sites) forces on each site (pN)

    Returns:
        tensions: (n_thin, n_sites) cumulative tension (pN)

    How to modify:
        - Local vs global: Use local force instead of cumulative
        - Smoothing: Apply spatial filter to tensions
        - Time-averaging: Include history of forces
    """
    # OPTIMIZED: O(N) cumsum instead of O(N^2) einsum with upper triangular mask
    # Reverse cumsum: flip, cumsum, flip back
    # ~3.8x faster than the original einsum implementation
    tensions = jnp.flip(jnp.cumsum(jnp.flip(thin_forces, axis=1), axis=1), axis=1)

    return tensions


# ============================================================================
# COOPERATIVE REGION DETERMINATION
# ============================================================================

def _find_cooperative_sites_single_chain(chain_states: jnp.ndarray,
                                         chain_positions: jnp.ndarray,
                                         chain_spans: jnp.ndarray) -> jnp.ndarray:
    """Find cooperative sites within a single TM chain.

    This is a helper function that processes one TM chain at a time.
    Each thin filament has 2 TM chains (assigned by monomer index parity).

    Args:
        chain_states: (n_chain_sites,) TM states for this chain only
        chain_positions: (n_chain_sites,) axial positions for this chain only
        chain_spans: (n_chain_sites,) spans for this chain only

    Returns:
        is_cooperative: (n_chain_sites,) cooperative status for this chain
    """
    # Distance matrix between sites in THIS chain only
    distances = jnp.abs(chain_positions[:, None] - chain_positions[None, :])

    # Within span matrix: within[i,j] = True if site j is within span of site i
    within_span = distances < chain_spans[:, None]

    # State 2 (Ca²⁺-activated, no XB) and state 3 (XB-bound) both activate neighbors.
    # State 3 units are physically held open by the XB and propagate cooperativity at
    # least as strongly as Ca²⁺-only units (McKillop & Geeves 1993; Mijailovich 2020).
    is_activator = (chain_states == 2) | (chain_states == 3)

    # Cooperative if any activator site in this chain is within span
    is_cooperative = jnp.any(within_span & is_activator[None, :], axis=1)

    return is_cooperative


def find_cooperative_sites_with_chains(tm_states: jnp.ndarray,
                                       tm_positions: jnp.ndarray,
                                       spans: jnp.ndarray,
                                       tm_chains: jnp.ndarray) -> jnp.ndarray:
    """Determine which sites are cooperative using stored chain assignments.

    OPTIMIZED: Uses windowed approach instead of full O(n^2) distance matrix.
    Since cooperative span is limited (~62nm max) and site spacing is ~24.8nm,
    only ~5 neighbors on each side need to be checked.

    Each thin filament has 2 TM chains (two-start helix). The chain assignment
    is determined by the original monomer index in the 390-monomer helix, NOT
    by binding site index. This is stored in tm_chains (0 or 1 for each site).

    Cooperativity only spreads within the SAME chain.

    Args:
        tm_states: (n_sites,) tropomyosin states (0-3)
        tm_positions: (n_sites,) axial positions of TM sites (nm)
        spans: (n_sites,) cooperative span for each site (nm)
        tm_chains: (n_sites,) chain assignment (0 or 1) for each site

    Returns:
        is_cooperative: (n_sites,) boolean array of cooperative sites
    """
    n_sites = tm_states.shape[0]

    # Windowed approach: max_window is a JAX static-shape ceiling; the physical
    # filter is `span` (set per-site by tension). Sized so the window never bites
    # for spans up to ~500 nm (well past TM persistence length ~50–100 nm).
    max_window = 20

    # State 2 (Ca²⁺-activated, no XB) and state 3 (XB-bound) both activate neighbors.
    # State 3 units are physically held open by the XB and propagate cooperativity at
    # least as strongly as Ca²⁺-only units (McKillop & Geeves 1993; Mijailovich 2020).
    is_chain_0 = (tm_chains == 0)
    is_chain_1 = (tm_chains == 1)
    is_activator = (tm_states == 2) | (tm_states == 3)

    def check_site_cooperative(i):
        """Check if site i is subject to cooperativity."""
        my_chain = tm_chains[i]
        my_pos = tm_positions[i]
        my_span = spans[i]

        # Create neighbor indices (clipped to valid range)
        neighbor_offsets = jnp.arange(-max_window, max_window + 1)
        neighbor_indices = jnp.clip(i + neighbor_offsets, 0, n_sites - 1)

        # Get neighbor properties
        neighbor_activators = is_activator[neighbor_indices]
        neighbor_positions = tm_positions[neighbor_indices]
        neighbor_chains = tm_chains[neighbor_indices]

        # Check conditions: activator state, same chain, within span
        same_chain = (neighbor_chains == my_chain)
        within_span = jnp.abs(neighbor_positions - my_pos) < my_span

        # Any neighbor that meets all conditions activates this site
        return jnp.any(neighbor_activators & same_chain & within_span)

    # Vectorize over all sites
    is_cooperative = jax.vmap(check_site_cooperative)(jnp.arange(n_sites))

    return is_cooperative


# ============================================================================
# FULL COOPERATIVITY UPDATE
# ============================================================================

def update_cooperativity(state: 'State',
                        constants: 'DynamicParams',
                        thin_forces: jnp.ndarray,
                        topology: 'SarcTopology' = None) -> 'State':
    """Update which TM sites are subject to cooperative activation.

    This should be called before thin_transitions() to determine
    which sites get the cooperative rate boost.

    Args:
        state: State NamedTuple (must have state.thin with tm_states, axial)
        constants: DynamicParams object with attribute access for span parameters
        thin_forces: (n_thin, n_sites) forces on thin filament sites
        topology: SarcTopology with tm_chains (structural data)

    Returns:
        new_state: State with updated 'subject_to_coop' field

    How to modify:
        - Per-filament parameters: Different span_base per thin filament
        - Time dependence: Smooth transitions in/out of cooperative regions
        - All-or-none: Sites are either fully cooperative or not

    From hs.py set_subject_to_cooperativity() (lines 1585-1638)
    """
    tm_states = state.thin.tm_states      # (n_thin, n_sites)
    thin_axial = state.thin.axial         # (n_thin, n_sites)
    tm_chains = topology.tm_chains        # (n_thin, n_sites) - 0 or 1, from Topology

    n_thin, n_sites = tm_states.shape

    # Get span parameters from constants
    span_base = constants.tm_span_base
    span_force50 = constants.tm_span_force50
    span_steep = constants.tm_span_steep

    # Calculate tensions at each site
    tensions = get_site_tensions(thin_forces)  # (n_thin, n_sites)

    # Calculate spans based on tension
    # OPTIMIZED: Flatten, compute once, reshape (instead of nested vmap)
    tensions_flat = tensions.reshape(-1)
    spans_flat = calculate_cooperative_span(tensions_flat, span_base, span_force50, span_steep)
    spans = spans_flat.reshape(n_thin, n_sites)  # (n_thin, n_sites)

    # Find cooperative sites for each filament using actual chain assignments
    # This is critical: cooperativity only spreads within the same TM chain
    is_cooperative = jax.vmap(find_cooperative_sites_with_chains)(
        tm_states,
        thin_axial,
        spans,
        tm_chains
    )  # (n_thin, n_sites)

    # Update state
    new_thin = state.thin._replace(subject_to_coop=is_cooperative)
    new_state = state._replace(thin=new_thin)

    return new_state


# ============================================================================
# SYMMETRIC ISING COOPERATIVITY — neighbor-state count helper
# ============================================================================
#
# For each TM site, count same-chain neighbors within a fixed span that are in
# state 2 (Ca-open) and state 3 (XB-bound) separately. Counts are capped at 2
# per channel so the index space stays (3, 3) for Q-matrix gather.
# The closed-neighbor count is derived in the transitions kernel as
# n_closed = max(0, N_TYPICAL - n_2 - n_3), so this helper only returns the two
# open-channel counts.
#
# Used by transitions.thin_transitions_ising when ising_coop=True is passed to run().
# No state field is written — counts are consumed inline.

def count_neighbor_states_split(tm_states: jnp.ndarray,
                                 tm_positions: jnp.ndarray,
                                 span: float,
                                 tm_chains: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Count same-chain neighbors in state 2, state 3, and closed (state 0/1)
    within a fixed span. Each count is capped at 2 for tractable Q-matrix indexing.

    Args:
        tm_states:    (n_sites,) TM states (0-3)
        tm_positions: (n_sites,) axial positions of TM sites (nm)
        span:         scalar nm, fixed cooperative reach (radius)
        tm_chains:    (n_sites,) chain assignment (0 or 1) for each site

    Returns:
        n_2:      (n_sites,) int32 same-chain state-2 neighbors, capped at 2
        n_3:      (n_sites,) int32 same-chain state-3 neighbors, capped at 2
        n_closed: (n_sites,) int32 same-chain state-{0,1} neighbors, capped at 2
    """
    n_sites = tm_states.shape[0]
    max_window = 20  # JAX static-shape ceiling; physical filter is `span`

    is_2 = (tm_states == 2)
    is_3 = (tm_states == 3)
    is_closed = (tm_states == 0) | (tm_states == 1)

    def count_site(i):
        my_chain = tm_chains[i]
        my_pos = tm_positions[i]

        offsets = jnp.arange(-max_window, max_window + 1)
        idx = jnp.clip(i + offsets, 0, n_sites - 1)
        not_self = (offsets != 0)

        nb_pos = tm_positions[idx]
        nb_chain = tm_chains[idx]
        nb_is_2 = is_2[idx]
        nb_is_3 = is_3[idx]
        nb_is_closed = is_closed[idx]

        same_chain = (nb_chain == my_chain)
        within = jnp.abs(nb_pos - my_pos) < span
        mask = same_chain & within & not_self

        n2 = jnp.sum(jnp.where(mask & nb_is_2, 1, 0))
        n3 = jnp.sum(jnp.where(mask & nb_is_3, 1, 0))
        nc = jnp.sum(jnp.where(mask & nb_is_closed, 1, 0))
        return (jnp.minimum(n2, 2).astype(jnp.int32),
                jnp.minimum(n3, 2).astype(jnp.int32),
                jnp.minimum(nc, 2).astype(jnp.int32))

    n_2_arr, n_3_arr, n_c_arr = jax.vmap(count_site)(jnp.arange(n_sites))
    return n_2_arr, n_3_arr, n_c_arr
