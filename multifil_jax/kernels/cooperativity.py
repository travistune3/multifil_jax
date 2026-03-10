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

@jax.jit
def calculate_cooperative_span(tension: jnp.ndarray,
                               span_base: float = 62.0,
                               span_force50: float = -8.0,
                               span_steep: float = 1.0) -> jnp.ndarray:
    """Calculate cooperative span based on local tension.

    The span decreases with increasing force (tension opposes cooperativity).

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


@jax.jit
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

    # State 2 activators in this chain
    is_state_2 = (chain_states == 2)

    # Cooperative if any state-2 site in this chain is within span
    is_cooperative = jnp.any(within_span & is_state_2[None, :], axis=1)

    return is_cooperative


@jax.jit
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

    # Windowed approach: check only nearby sites
    # Max span ~62nm, site spacing ~24.8nm, so max window of ~5 sites each direction
    max_window = 6

    # Precompute state-2 mask for each chain
    is_chain_0 = (tm_chains == 0)
    is_chain_1 = (tm_chains == 1)
    is_state_2 = (tm_states == 2)

    def check_site_cooperative(i):
        """Check if site i is subject to cooperativity."""
        my_chain = tm_chains[i]
        my_pos = tm_positions[i]
        my_span = spans[i]

        # Create neighbor indices (clipped to valid range)
        neighbor_offsets = jnp.arange(-max_window, max_window + 1)
        neighbor_indices = jnp.clip(i + neighbor_offsets, 0, n_sites - 1)

        # Get neighbor properties
        neighbor_states = tm_states[neighbor_indices]
        neighbor_positions = tm_positions[neighbor_indices]
        neighbor_chains = tm_chains[neighbor_indices]

        # Check conditions: state 2, same chain, within span
        is_neighbor_state_2 = (neighbor_states == 2)
        same_chain = (neighbor_chains == my_chain)
        within_span = jnp.abs(neighbor_positions - my_pos) < my_span

        # Any neighbor that meets all conditions activates this site
        return jnp.any(is_neighbor_state_2 & same_chain & within_span)

    # Vectorize over all sites
    is_cooperative = jax.vmap(check_site_cooperative)(jnp.arange(n_sites))

    return is_cooperative


# ============================================================================
# FULL COOPERATIVITY UPDATE
# ============================================================================

@jax.jit
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
# HELPER FUNCTIONS
# ============================================================================

@jax.jit
def get_cooperativity_stats(state: 'State') -> Dict[str, float]:
    """Get statistics about cooperativity state.

    Useful for diagnostics and visualization.

    Args:
        state: State NamedTuple (must have state.thin with subject_to_coop, tm_states)

    Returns:
        stats: Dictionary with cooperativity statistics
    """
    is_coop = state.thin.subject_to_coop
    tm_states = state.thin.tm_states

    stats = {
        'fraction_cooperative': jnp.mean(is_coop.astype(jnp.float32)),
        'fraction_state_2': jnp.mean((tm_states == 2).astype(jnp.float32)),
        'fraction_state_3': jnp.mean((tm_states == 3).astype(jnp.float32)),
        'n_cooperative_sites': jnp.sum(is_coop),
        'n_state_2_sites': jnp.sum(tm_states == 2),
        'n_state_3_sites': jnp.sum(tm_states == 3),
    }

    return stats


