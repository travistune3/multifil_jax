"""
Tropomyosin cooperativity — the LEGACY tension-span model, plus a shared helper.

WHY COOPERATIVITY EXISTS IN THE MODEL
-------------------------------------
Calcium binding at one troponin does not regulate one binding site in isolation.
Tropomyosin is a continuous strand lying along the actin filament, so when one
stretch of it moves aside, the adjoining stretches are mechanically dragged
toward moving too. This is what makes muscle force rise far more steeply with
calcium than independent binding could explain — a measured Hill coefficient
around 3, where independent sites would give 1.

Whatever the model, cooperativity is always between sites ON THE SAME
TROPOMYOSIN STRAND. A thin filament carries two strands, one along each groove
of the actin double helix, and they do not couple to each other. Every function
here respects that.

WHICH MODEL RUNS
----------------
The simulation has two cooperativity implementations, and this module serves
both, though unequally:

  DEFAULT — symmetric Ising coupling. Lives in kernels/transitions.py
    (thin_transitions_ising). From this module it uses only
    count_neighbor_states_split() at the bottom.

  LEGACY — tension-dependent span, selected by run(legacy_coop=True). This is
    everything else in this module: compute the tension carried along the thin
    filament, convert it into a spatial span in nanometres, and mark every site
    within that span of an active site as cooperatively activated.

The legacy model is deprecated. It is retained so that parameter sets fitted
under it remain reproducible. Its distinguishing feature — that cooperativity
reaches a physical distance which grows with filament tension, rather than a
fixed number of neighbours — makes it mechanically appealing but harder to
reason about thermodynamically, since the resulting rate boost is applied to
forward rates only.
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

    Formula:
        span = 0.5 * span_base * (1 + tanh(span_steep * (span_force50 + tension)))

    A tanh gives a smooth saturating switch: the span approaches zero at low
    tension, span_base at high tension, and transitions around span_force50.
    Smoothness matters for the solver — a hard threshold here would make force
    a discontinuous function of position.

    Note span_force50 is NEGATIVE by default (-8 pN), so the midpoint sits at a
    compressive load and typical tensile loads land on the upper plateau.
    """

    span = 0.5 * span_base * (1.0 + jnp.tanh(span_steep * (span_force50 + tension)))

    # Ensure span is non-negative
    span = jnp.maximum(span, 0.0)

    return span


def get_site_tensions(thin_forces: jnp.ndarray) -> jnp.ndarray:
    """Calculate cumulative tension at each site from crossbridge forces.

    Tension = sum of forces from this site to M-line (end of array).
    tensions[i] = sum of forces[i:]

    Equivalent to an upper-triangular matrix product, but computed as a
    reversed cumulative sum, which is O(n) rather than O(n^2).

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
    """Mark which sites fall inside a cooperative span — LEGACY path only.

    Must run before thin_transitions(), which consumes the flag this writes.
    The default Ising path never calls this: it derives neighbour information
    directly from tm_states, which is why the default kinetics phase can skip
    both this step and the thin-filament force calculation feeding it.

    Three stages:
      1. Convert per-node spring forces into cumulative TENSION at each site
         (get_site_tensions) — the load actually borne at that point along the
         filament, which is the mechanically meaningful quantity for straining
         tropomyosin.
      2. Convert tension into a span in nanometres (calculate_cooperative_span).
         Higher tension gives a longer reach.
      3. Mark every site lying within its span of an activating site, staying
         within each tropomyosin chain (find_cooperative_sites_with_chains).

    Writes a boolean per site into state.thin.subject_to_coop. The magnitude of
    the resulting rate boost is not decided here — that is tm_coop_magnitude,
    applied in transitions.py.

    Args:
        state: Current State (reads thin.tm_states and thin.axial)
        constants: DynamicParams with tm_span_base / tm_span_force50 /
            tm_span_steep
        thin_forces: (n_thin, n_sites) internal backbone spring forces, from
            forces.calculate_thin_forces_for_cooperativity(). NOT crossbridge
            forces — see that function for why the distinction matters.
        topology: SarcTopology with tm_chains, assigning each site to a strand

    Returns:
        new_state: State with subject_to_coop updated
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
# The one piece of this module the DEFAULT (Ising) path uses.
#
# For each tropomyosin site, count how many of its same-chain neighbours are in
# state 2 (Ca-open) and state 3 (crossbridge-bound), keeping the two counts
# separate so they can carry different coupling strengths (J_C and J_M).
#
# NEIGHBOURS ARE STRUCTURAL, NOT SPATIAL. A site couples to its immediate
# predecessor and successor in its own chain's axial ordering, precomputed once
# when the topology is built (topology.tm_prev_neighbor / tm_next_neighbor).
# There is no distance threshold and no length scale to tune, which is what
# makes the coupling a genuine nearest-neighbour Ising chain rather than a
# windowed approximation to one.
#
# Endpoints self-reference rather than carrying a sentinel index, so every
# gather is valid and no masking is needed. A site therefore has at most two
# real neighbours by construction, and the counts fall in {0, 1, 2} naturally —
# the jnp.minimum(..., 2) below is defensive, kept because the downstream
# Q-matrix gather assumes that range structurally.
#
# Nothing is written to state: the counts are consumed inline by
# transitions.thin_transitions_ising().

def count_neighbor_states_split(tm_states: jnp.ndarray,
                                 tm_prev_neighbor: jnp.ndarray,
                                 tm_next_neighbor: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Count same-chain neighbors in state 2, state 3, and closed (state 0/1),
    using the fixed (<=2) topological same-chain neighbor set per site.

    Args:
        tm_states:        (n_sites,) TM states (0-3)
        tm_prev_neighbor: (n_sites,) nearest same-chain predecessor site index
            (self-referencing at chain endpoints/padding)
        tm_next_neighbor: (n_sites,) nearest same-chain successor site index
            (self-referencing at chain endpoints/padding)

    Returns:
        n_2:      (n_sites,) int32 same-chain state-2 neighbors, in {0,1,2}
        n_3:      (n_sites,) int32 same-chain state-3 neighbors, in {0,1,2}
        n_closed: (n_sites,) int32 same-chain state-{0,1} neighbors, in {0,1,2}
    """
    n_sites = tm_states.shape[0]
    site_idx = jnp.arange(n_sites)

    is_2 = (tm_states == 2)
    is_3 = (tm_states == 3)
    is_closed = (tm_states == 0) | (tm_states == 1)

    prev_real = (tm_prev_neighbor != site_idx)
    next_real = (tm_next_neighbor != site_idx)

    n_2 = (is_2[tm_prev_neighbor] & prev_real).astype(jnp.int32) + (is_2[tm_next_neighbor] & next_real).astype(jnp.int32)
    n_3 = (is_3[tm_prev_neighbor] & prev_real).astype(jnp.int32) + (is_3[tm_next_neighbor] & next_real).astype(jnp.int32)
    n_c = (is_closed[tm_prev_neighbor] & prev_real).astype(jnp.int32) + (is_closed[tm_next_neighbor] & next_real).astype(jnp.int32)

    return jnp.minimum(n_2, 2), jnp.minimum(n_3, 2), jnp.minimum(n_c, 2)
