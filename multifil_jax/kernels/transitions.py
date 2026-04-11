"""
State Transitions for JAX Half-Sarcomere (Tiered Architecture)

This module handles all state transitions:
1. Tropomyosin transitions (thin filaments)
2. Crossbridge transitions (thick filaments)
3. Binding/unbinding events

Tiered Architecture:
    - State (Tier 0): Pure simulation arrays (thick, thin, titin NamedTuples).
      No embedded params, geometry, pCa, lattice_spacing, or bin arrays.
    - Constants (Tier 2): DynamicParams object with .pCa, .lattice_spacing, etc.
    - Topology (Tier 1): SarcTopology with .eye_4, .eye_6, .xb_to_thin_id, etc.

Key features:
- JIT-compiled for speed
- Vectorized over all sites/crossbridges
- Functional random number generation (JAX style)
- Clear modification points

PERFORMANCE OPTIMIZATIONS:

1. TM Locked Transitions (thin_transitions):
   - Reduced O(N) matrix exponentials to O(2) unique matrices
   - Sites binned by cooperativity (coop vs non-coop) -> 2 unique P matrices
   - Locked sites (state 3 AND bound) force probability [0,0,0,1]
   - Speedup: ~3000x reduction in matrix exponential calls

2. Per-XB Transition Matrices (thick_transitions):
   - Computes one Q→P matrix per crossbridge using exact distances
   - No distance binning — each XB uses its actual (axial, radial) position
   - Ensures correctness under varying lattice spacing and spring constant sweeps

3. Matrix Exponential (expm_pade6_batch):
   - 6th-order Pade approximation with scaling/squaring
   - Fixed 18 iterations for XLA fusion (no fori_loop)
   - Row normalization to ensure valid probability matrices
   - Optional identity parameter to avoid XLA materializing copies in vmap

Rate functions are now in rate_functions.py for easy customization.
Parameters use consolidated absolute values from DynamicParams.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from typing import Tuple, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams

# Import rate functions and energy calculations
from .rate_functions import (
    tm_rate_12, tm_rate_21, tm_rate_23, tm_rate_32,
    tm_rate_34, tm_rate_43, tm_rate_41,
    xb_rate_12, xb_rate_21, xb_rate_23, xb_rate_32,
    xb_rate_34, xb_rate_43, xb_rate_45, xb_rate_54,
    xb_rate_51, xb_rate_15, xb_rate_61, xb_rate_16,
    compute_xb_energies, compute_xb_free_energies, compute_xb_force,
)


# ============================================================================
# PADE COEFFICIENTS FOR 6TH ORDER MATRIX EXPONENTIAL
# ============================================================================

# Pade(6,6) coefficients from Higham (2005) Table 10.2
PADE6_B = jnp.array([
    1.0,                    # b0
    1.0/2.0,               # b1 = 1/2
    1.0/9.0,               # b2 = 1/9
    1.0/72.0,              # b3 = 1/72
    1.0/1008.0,            # b4 = 1/1008
    1.0/30240.0,           # b5 = 1/30240
    1.0/1209600.0          # b6 = 1/1209600
], dtype=jnp.float32)


# ============================================================================
# MATRIX EXPONENTIAL (6th order Pade with scaling/squaring - OPTIMIZED)
# ============================================================================

def expm_pade6_batch(
    A_batch: jnp.ndarray,
    identity: jnp.ndarray,
) -> jnp.ndarray:
    """6th order Pade approximation for batch of matrices.

    OPTIMIZED: Replaces 3rd order with 6th order for better accuracy.
    Uses FIXED iteration count (16) for XLA kernel fusion.

    Args:
        A_batch: (batch, n, n) array of matrices
        identity: (n, n) identity matrix from SarcTopology (eye_4 or eye_6).
            Avoids creating jnp.eye(n) inside the function, which can cause
            XLA to materialize copies during vmap/parameter sweeps.

    Returns:
        exp(A): (batch, n, n) matrix exponentials
    """
    batch_size, n, _ = A_batch.shape

    # Step 1: Compute infinity norm for each matrix
    a_norms = jnp.max(jnp.sum(jnp.abs(A_batch), axis=2), axis=1)

    # Step 2: Determine scaling factors (for ||A/2^s|| <= 0.5)
    s = jnp.maximum(0, jnp.ceil(jnp.log2(a_norms / 0.5 + 1e-10)).astype(jnp.int32))

    # Step 3: Scale matrices
    scale_factors = jnp.power(2.0, s)[:, None, None]
    A_scaled = A_batch / scale_factors

    # Step 4: Compute matrix powers (vectorized)
    I = identity
    A2 = jnp.einsum('...ij,...jk->...ik', A_scaled, A_scaled)
    A4 = jnp.einsum('...ij,...jk->...ik', A2, A2)
    A6 = jnp.einsum('...ij,...jk->...ik', A4, A2)

    # Step 5: Compute U and V for Pade approximant
    b = PADE6_B

    # U = A * (b1*I + b3*A2 + b5*A4)
    inner = b[1]*I + b[3]*A2 + b[5]*A4
    U = jnp.einsum('...ij,...jk->...ik', A_scaled, inner)

    # V = b0*I + b2*A2 + b4*A4 + b6*A6
    V = b[0]*I + b[2]*A2 + b[4]*A4 + b[6]*A6

    # Step 6: Solve (V - U) @ R = (V + U)
    result = jnp.linalg.solve(V - U, V + U)

    # Step 7: Square s times using fori_loop for XLA fusion
    # Handles ||A|| up to 2^18 = 262144; 2 extra no-op iters for typical norms
    def _square_step(i, result):
        should_square = i < s
        squared = jnp.einsum('...ij,...jk->...ik', result, result)
        return jnp.where(should_square[:, None, None], squared, result)

    result = jax.lax.fori_loop(0, 18, _square_step, result)

    # Step 8: Row normalization (fix float32 drift)
    row_sums = jnp.sum(result, axis=2, keepdims=True)
    result = result / row_sums

    # Guard against NaN
    result = jnp.where(jnp.isnan(result), 1.0/n, result)

    return result


# ============================================================================
# OPTIMIZED RATE MATRIX CONSTRUCTION
# ============================================================================

def _build_tm_Q_matrix_optimized(k_11, k_12, k_14, k_21, k_22, k_23,
                                  k_32, k_33, k_34, k_41, k_43, k_44):
    """Build TM rate matrices efficiently using jnp.stack.

    This replaces the chain of .at[].set() calls with a single construction.
    Much more efficient for JAX compilation.

    Args:
        k_ij: (n_sites,) arrays of rate coefficients

    Returns:
        Q: (n_sites, 4, 4) rate matrices
    """
    n_sites = k_11.shape[0]
    zeros = jnp.zeros(n_sites)

    # Build each row as (n_sites, 4)
    row0 = jnp.stack([k_11, k_12, zeros, k_14], axis=1)
    row1 = jnp.stack([k_21, k_22, k_23, zeros], axis=1)
    row2 = jnp.stack([zeros, k_32, k_33, k_34], axis=1)
    row3 = jnp.stack([k_41, zeros, k_43, k_44], axis=1)

    # Stack rows to form (n_sites, 4, 4)
    Q = jnp.stack([row0, row1, row2, row3], axis=1)

    return Q


def _build_xb_Q_matrix_optimized(r11, r12, r15, r16, r21, r22, r23,
                                  r32, r33, r34, r43, r44, r45,
                                  r51, r54, r55, r61, r66):
    """Build XB rate matrices efficiently using jnp.stack.

    This replaces the chain of .at[].set() calls with a single construction.

    Args:
        r_ij: (n_xb,) arrays of rate coefficients

    Returns:
        Q: (n_xb, 6, 6) rate matrices
    """
    n_xb = r11.shape[0]
    zeros = jnp.zeros(n_xb)

    # Build each row as (n_xb, 6)
    # Row 0 (state 1 = DRX): can go to states 2, 5, 6
    row0 = jnp.stack([r11, r12, zeros, zeros, r15, r16], axis=1)

    # Row 1 (state 2 = loose): can go to states 1, 3
    row1 = jnp.stack([r21, r22, r23, zeros, zeros, zeros], axis=1)

    # Row 2 (state 3 = tight_1): can go to states 2, 4
    row2 = jnp.stack([zeros, r32, r33, r34, zeros, zeros], axis=1)

    # Row 3 (state 4 = tight_2): can go to states 3, 5
    row3 = jnp.stack([zeros, zeros, r43, r44, r45, zeros], axis=1)

    # Row 4 (state 5 = free_2): can go to states 1, 4
    row4 = jnp.stack([r51, zeros, zeros, r54, r55, zeros], axis=1)

    # Row 5 (state 6 = SRX): can go to state 1
    row5 = jnp.stack([r61, zeros, zeros, zeros, zeros, r66], axis=1)

    # Stack rows to form (n_xb, 6, 6)
    Q = jnp.stack([row0, row1, row2, row3, row4, row5], axis=1)

    return Q


# ============================================================================
# MATRIX EXPONENTIAL
# ============================================================================

def matrix_exponential_batch(
    Q: jnp.ndarray,
    dt: float,
    identity: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Compute matrix exponential for batch of rate matrices.

    For transition matrices Q, computes P = exp(Q * dt) for each matrix.

    OPTIMIZED: Uses 6th order Pade approximation with fixed 16 iterations
    for scaling/squaring. This enables XLA kernel fusion.

    Args:
        Q: (n_matrices, n_states, n_states) rate matrices
        dt: Timestep length (ms)
        identity: Optional (n_states, n_states) identity matrix from template.
            Pass state['template'].eye_4 for TM or eye_6 for XB transitions.
            Avoids XLA materializing copies during vmap/parameter sweeps.

    Returns:
        P: (n_matrices, n_states, n_states) probability matrices
    """
    # Scale Q by dt and compute exp using Pade6
    # Row normalization is done inside expm_pade6_batch
    return expm_pade6_batch(Q * dt, identity=identity)


# ============================================================================
# TROPOMYOSIN TRANSITIONS
# ============================================================================

def _compute_unique_tm_Q_matrices(ca_concentration: float,
                                   coop_magnitude: float,
                                   params) -> jnp.ndarray:
    """Compute the 2 unique Q rate matrices for TM sites.

    TM sites only have 2 unique non-locked configurations:
    1. Non-cooperative (coop_factor = 1.0)
    2. Cooperative (coop_factor = coop_magnitude)

    Locked sites are handled separately via masking in thin_transitions.

    Args:
        ca_concentration: Calcium concentration (M)
        coop_magnitude: Cooperativity multiplier (e.g., 100.0)
        params: DynamicParams object with TM rate constants

    Returns:
        Q_unique: (2, 4, 4) array where:
            Q_unique[0] = non-cooperative Q matrix
            Q_unique[1] = cooperative Q matrix
    """
    # Get rate constants from params (attribute access)
    K1 = params.tm_K1
    K2 = params.tm_K2
    K3 = params.tm_K3
    k_12_base = params.tm_k_12
    k_23_base = params.tm_k_23
    k_34_base = params.tm_k_34
    k_41_base = params.tm_k_41

    def build_Q_matrix(coop_factor):
        """Build a single 4x4 Q matrix for given cooperativity."""
        # Forward rates
        k_12 = k_12_base * ca_concentration * coop_factor
        k_23 = k_23_base * coop_factor
        k_34 = k_34_base * coop_factor
        k_41 = k_41_base  # Return rate is not affected by coop

        # Backward rates (detailed balance)
        k_21 = k_12_base / K1
        k_32 = k_23_base / K2
        k_43 = k_34_base / K3

        # Diagonal rates
        k_11 = -k_12
        k_22 = -(k_21 + k_23)
        k_33 = -(k_32 + k_34)
        k_44 = -(k_41 + k_43)

        # Build Q matrix
        return jnp.array([
            [k_11, k_12, 0.0, 0.0],
            [k_21, k_22, k_23, 0.0],
            [0.0, k_32, k_33, k_34],
            [k_41, 0.0, k_43, k_44]
        ], dtype=jnp.float32)

    # Build Q matrices for both configurations
    Q_non_coop = build_Q_matrix(1.0)
    Q_coop = build_Q_matrix(coop_magnitude)

    return jnp.stack([Q_non_coop, Q_coop])


def thin_transitions(state: 'State',
                    constants: 'DynamicParams',
                    topology: 'SarcTopology',
                    rng_key: jax.random.PRNGKey,
                    dt: float,
                    random_values: Optional[jnp.ndarray] = None) -> Tuple['State', jnp.ndarray]:
    """Perform tropomyosin state transitions for all binding sites.

    OPTIMIZED: Instead of computing N matrix exponentials (one per site),
    computes only 2 unique P matrices and indexes into them.

    Locked sites (tm_state==3 AND bound) are handled via masking:
    their probability vector is forced to [0, 0, 0, 1] (stay in state 3).

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with physics values (pCa, tm_* rates)
        topology: SarcTopology with eye_4 identity matrix
        rng_key: JAX random key
        dt: Timestep length (ms)
        random_values: Optional pre-generated random values (for testing)

    Returns:
        new_state: Updated state with new tm_states
        P_unique: (2, 4, 4) the unique P matrices used (for validation)
    """
    tm_states = state.thin.tm_states
    is_cooperative = state.thin.subject_to_coop
    bound_to = state.thin.bound_to
    eye_4 = topology.eye_4

    is_bound = bound_to >= 0  # (n_thin, n_sites)
    n_thin, n_sites = tm_states.shape
    n_sites_total = n_thin * n_sites

    # Flatten for processing
    tm_states_flat = tm_states.reshape(-1)
    is_coop_flat = is_cooperative.reshape(-1)
    is_bound_flat = is_bound.reshape(-1)

    # Calculate ca_concentration from pCa (pCa is the user-facing variable)
    ca_conc = 10.0 ** (-constants.pCa)
    coop_magnitude = constants.tm_coop_magnitude

    # OPTIMIZATION: Compute only 2 unique Q matrices
    Q_unique = _compute_unique_tm_Q_matrices(ca_conc, coop_magnitude, constants)

    # Compute P = exp(Q * dt) for both configurations
    P_unique = expm_pade6_batch(Q_unique * dt, identity=eye_4)  # (2, 4, 4)

    # Map each site to its P matrix (0 = non-coop, 1 = coop)
    config_idx = is_coop_flat.astype(jnp.int32)  # (n_sites_total,)

    # Get P matrix for each site via indexing
    P_indexed = P_unique[config_idx]  # (n_sites_total, 4, 4)

    # Get probability vectors for current states
    current_states = tm_states_flat.astype(jnp.int32)
    prob_vectors = jax.vmap(lambda P, s: P[s])(P_indexed, current_states)  # (n_sites_total, 4)

    # CRITICAL: Apply locked mask
    # Locked = in state 3 AND bound to crossbridge
    locked_mask = (tm_states_flat == 3) & is_bound_flat  # (n_sites_total,)

    # Locked sites: probability = [0, 0, 0, 1] (100% stay in state 3)
    locked_prob = jnp.array([0.0, 0.0, 0.0, 1.0])
    prob_vectors = jnp.where(locked_mask[:, None], locked_prob, prob_vectors)

    # Sample new states from categorical distribution
    if random_values is None:
        rng_key, subkey = jax.random.split(rng_key)
        random_values = jax.random.uniform(subkey, shape=(n_sites_total,))

    # Standard categorical sampling: cumsum + argmax
    cum_probs = jnp.cumsum(prob_vectors, axis=1)
    new_states = jnp.argmax(random_values[:, None] < cum_probs, axis=1)

    # Reshape back — cast to int8 to match ThinState.tm_states dtype
    new_tm_states = new_states.reshape(n_thin, n_sites).astype(jnp.int8)

    # Update state with _replace
    new_thin = state.thin._replace(
        tm_states=new_tm_states,
    )
    new_state = state._replace(thin=new_thin)

    return new_state, P_unique


# ============================================================================
# CROSSBRIDGE TRANSITIONS
# ============================================================================

def xb_rate_matrix(xb_distances: jnp.ndarray,
                   lattice_spacing: float,
                   spring_constants: jnp.ndarray,
                   permissiveness: jnp.ndarray,
                   ca_concentration: float,
                   temp_celsius: float,
                   params: Dict) -> jnp.ndarray:
    """Construct rate matrices for crossbridge transitions.

    Each crossbridge has a 6x6 rate matrix for its states:
    - State 1: DRX (disordered relaxed, unbound)
    - State 2: Loose (weakly bound)
    - State 3: Tight_1 (strongly bound, pre-power-stroke)
    - State 4: Tight_2 (strongly bound, post-power-stroke)
    - State 5: Free_2 (unbound after power stroke)
    - State 6: SRX (super-relaxed, sequestered)

    Args:
        xb_distances: (n_xb, 2) array of (axial, radial) distances to binding sites
        lattice_spacing: Lattice spacing (nm)
        spring_constants: (n_xb, 8) array of spring constants and rest lengths
            [:, 0:4]: g_k_weak, g_r_weak, c_k_weak, c_r_weak
            [:, 4:8]: g_k_strong, g_r_strong, c_k_strong, c_r_strong
        permissiveness: (n_xb,) binding site permissiveness (0-1)
        ca_concentration: Calcium concentration (M)
        temp_celsius: Temperature (C)
        params: Parameter dictionary with consolidated rate coefficients

    Returns:
        Q: (n_xb, 6, 6) rate matrices

    Note:
        Rate functions are now in rate_functions.py for easy customization.
        Parameters use consolidated absolute values from DynamicParams.
    """

    n_xb = xb_distances.shape[0]

    # Convert to polar coordinates
    x = xb_distances[:, 0]
    y = xb_distances[:, 1]
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)

    # Get spring constants
    g_k_weak = spring_constants[:, 0]
    g_r_weak = spring_constants[:, 1]
    c_k_weak = spring_constants[:, 2]
    c_r_weak = spring_constants[:, 3]
    g_k_strong = spring_constants[:, 4]
    g_r_strong = spring_constants[:, 5]
    c_k_strong = spring_constants[:, 6]
    c_r_strong = spring_constants[:, 7]

    # Calculate potential energies (in units of kT)
    # Note: Use 1.3810 to match OOP exactly (hs.py line 1707)
    k_t = 1.3810e-23 * (temp_celsius + 273.15) * 1e21  # pN*nm

    # Compute energies using helper function (vectorized)
    E_weak = (0.5 * g_k_weak * (r - g_r_weak)**2 +
              0.5 * c_k_weak * (theta - c_r_weak)**2) / k_t
    E_strong = (0.5 * g_k_strong * (r - g_r_strong)**2 +
                0.5 * c_k_strong * (theta - c_r_strong)**2) / k_t

    # Energy difference - use simple subtraction to match OOP exactly
    E_diff = E_weak - E_strong

    # Free energies (from Pate & Cooke 1989, in units of kT)
    U_DRX = params.xb_U_DRX
    U_loose_base = params.xb_U_loose
    U_tight_1_base = params.xb_U_tight_1
    U_tight_2_base = params.xb_U_tight_2

    U_loose = U_loose_base + E_weak
    U_tight_1 = U_tight_1_base + E_strong
    U_tight_2 = U_tight_2_base + E_strong

    # Calculate forces for force-dependent rates
    f_3_4 = g_k_strong * (r - g_r_strong) + (1.0/r) * c_k_strong * (theta - c_r_strong)

    # ========================================================================
    # RATE DEFINITIONS using consolidated params and imported rate functions
    # ========================================================================

    ones = jnp.ones(n_xb)

    # Get consolidated rate coefficients from params (attribute access)
    # These already include the multipliers (e.g., xb_r12_coeff = mh_br * tau)
    r12_coeff = params.xb_r12_coeff
    r23_coeff = params.xb_r23_coeff
    r34_coeff = params.xb_r34_coeff
    r45_coeff = params.xb_r45_coeff
    r51_rate = params.xb_r51
    r15_rate = params.xb_r15
    r16_rate = params.xb_r16

    # SRX parameters
    srx_k0 = params.xb_srx_k0
    srx_kmax = params.xb_srx_kmax
    srx_b = params.xb_srx_b
    srx_ca50 = params.xb_srx_ca50

    # DRX (1) <-> loose (2) using imported rate functions
    r12 = xb_rate_12(permissiveness, r12_coeff, E_weak)
    r21 = xb_rate_21(r12, U_DRX, U_loose)

    # loose (2) <-> tight_1 (3)
    r23 = xb_rate_23(r23_coeff, E_diff)
    r32 = xb_rate_32(r23, U_loose, U_tight_1)

    # tight_1 (3) <-> tight_2 (4)
    r34 = xb_rate_34(r34_coeff, f_3_4, params.xb_delta_34, k_t)
    r43 = xb_rate_43(r34, U_tight_1, U_tight_2)

    # tight_2 (4) <-> free_2 (5)
    r45 = xb_rate_45(r45_coeff, f_3_4, params.xb_delta_45, k_t)
    r54 = xb_rate_54() * ones  # Always 0

    # free_2 (5) <-> DRX (1)
    r51 = xb_rate_51(r51_rate) * ones
    r15 = xb_rate_15(r15_rate) * ones

    # SRX (6) <-> DRX (1)
    r61 = xb_rate_61(ca_concentration, srx_k0, srx_kmax, srx_b, srx_ca50) * ones
    r16 = xb_rate_16(r16_rate) * ones

    # Diagonal rates: row sums must be zero for valid rate matrices
    # Direct arithmetic — avoids vmap+stack overhead (ordered_sum was just jnp.sum)
    r11 = -(r12 + r15 + r16)
    r22 = -(r21 + r23)
    r33 = -(r32 + r34)
    r44 = -(r43 + r45)
    r55 = -(r54 + r51)
    r66 = -r61

    # ========================================================================
    # CONSTRUCT Q MATRICES (optimized - single construction instead of .at[].set())
    # ========================================================================

    Q = _build_xb_Q_matrix_optimized(r11, r12, r15, r16, r21, r22, r23,
                                      r32, r33, r34, r43, r44, r45,
                                      r51, r54, r55, r61, r66)

    return Q


def compute_xb_transition_matrices(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-XB Q, P, and P_abs matrices via binned expm.

    Shared by thick_transitions() (for state sampling) and
    compute_all_metrics() (for ATP expected metrics).

    Instead of computing one expm per XB (n_xb_total calls), computes expm
    for 2*n_xb_bins positions (n_bins AP=0 + n_bins AP=1) then gathers per-XB
    via bin index lookup. Reduces expm calls by ~6× at 4×4, 71% of step time.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        dt: Timestep length (ms)

    Returns:
        Q_all:    (n_xb_total, 6, 6) rate matrices per crossbridge
        P_all:    (n_xb_total, 6, 6) transition probability matrices per crossbridge
        P_abs_all:(n_xb_total, 6, 6) absorbing-state P (row 4 zeroed) for ATP metrics
    """
    xb_states = state.thick.xb_states
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape
    n_xb_total = n_thick * n_crowns * n_xb_per_crown

    # Get axial distances for bin assignment
    xb_distances = state.thick.xb_distances
    lattice_spacing = constants.lattice_spacing

    if xb_distances is not None:
        xb_distances_flat = xb_distances.reshape(-1, 2)
    else:
        xb_distances_flat = jnp.zeros((n_xb_total, 2))
        xb_distances_flat = xb_distances_flat.at[:, 0].set(5.0)
        xb_distances_flat = xb_distances_flat.at[:, 1].set(lattice_spacing)

    # Get permissiveness from nearest binding sites
    xb_nearest_bs = state.thick.xb_nearest_bs
    tm_states = state.thin.tm_states
    n_thin, n_sites = tm_states.shape

    if xb_nearest_bs is not None:
        xb_nearest_bs_flat = xb_nearest_bs.reshape(-1)
        thin_indices = topology.xb_to_thin_id
        site_indices = jnp.clip(xb_nearest_bs_flat, 0, n_sites - 1)
        nearest_tm_states = tm_states[thin_indices, site_indices]
        permissiveness = (nearest_tm_states == 3).astype(jnp.float32)
    else:
        permissiveness = jnp.ones(n_xb_total) * 0.5

    ca_conc = 10.0 ** (-constants.pCa)
    n_bins = topology.xb_bin_centers.shape[0]   # static integer known to XLA
    d = lattice_spacing

    # Build (n_bins, 2) distance grid: [bin_center, lattice_spacing] for each bin
    x_centers = topology.xb_bin_centers                      # (n_bins,)
    dist_grid = jnp.stack([x_centers, jnp.full(n_bins, d)], axis=1)  # (n_bins, 2)

    # Spring constants: same scalar for all bins
    spring_vec = jnp.array([
        constants.xb_g_k_weak,   constants.xb_g_rest_weak,
        constants.xb_c_k_weak,   constants.xb_c_rest_weak,
        constants.xb_g_k_strong, constants.xb_g_rest_strong,
        constants.xb_c_k_strong, constants.xb_c_rest_strong,
    ])
    springs_grid = jnp.broadcast_to(spring_vec, (n_bins, 8))  # (n_bins, 8)

    # Q matrices for AP=0 and AP=1 at each bin position
    Q_ap0 = xb_rate_matrix(dist_grid, d, springs_grid,
                            jnp.zeros(n_bins), ca_conc, constants.temp_celsius, constants)
    Q_ap1 = xb_rate_matrix(dist_grid, d, springs_grid,
                            jnp.ones(n_bins),  ca_conc, constants.temp_celsius, constants)
    # Layout: [0..n_bins-1] = AP=0, [n_bins..2*n_bins-1] = AP=1
    Q_bins = jnp.concatenate([Q_ap0, Q_ap1], axis=0)         # (2*n_bins, 6, 6)

    P_bins = matrix_exponential_batch(Q_bins, dt, identity=topology.eye_6)

    # Absorbing-state version: zero row 4 so state 5 cannot exit (ATP metric)
    Q_abs_bins = Q_bins.at[:, 4, :].set(0.0)
    P_abs_bins = matrix_exponential_batch(Q_abs_bins, dt, identity=topology.eye_6)

    # Assign each XB to a bin via digitize + clip
    x_axial = xb_distances_flat[:, 0]                              # (n_xb_total,)
    bin_idx = jnp.digitize(x_axial, topology.xb_bin_edges) - 1    # in [-1, n_bins]
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    ap  = permissiveness.astype(jnp.int32)                         # 0 or 1
    key = ap * n_bins + bin_idx                                    # in [0, 2*n_bins)

    Q_all     = Q_bins[key]      # (n_xb_total, 6, 6)
    P_all     = P_bins[key]      # (n_xb_total, 6, 6)
    P_abs_all = P_abs_bins[key]  # (n_xb_total, 6, 6)

    return Q_all, P_all, P_abs_all


def thick_transitions(state: 'State',
                     constants: 'DynamicParams',
                     topology: 'SarcTopology',
                     rng_key: jax.random.PRNGKey,
                     dt: float,
                     random_values: Optional[jnp.ndarray] = None):
    """Perform crossbridge state transitions for all crossbridges.

    Computes one Q→P matrix per crossbridge using exact distances.
    Uses topology.xb_to_thin_id for thin filament lookup (no division).

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with physics values (pCa, lattice_spacing, xb_* rates)
        topology: SarcTopology with xb_to_thin_id, eye_6
        rng_key: JAX random key
        dt: Timestep length (ms)
        random_values: Optional pre-generated random values (for testing)

    Returns:
        new_state: Updated State with new crossbridge states and binding
    """
    # Get current xb states
    xb_states = state.thick.xb_states  # (n_thick, n_crowns, 3)
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape

    # Flatten for processing
    xb_states_flat = xb_states.reshape(-1)  # (n_thick * n_crowns * 3,)
    n_xb_total = xb_states_flat.shape[0]

    # Compute per-XB Q/P matrices via shared helper
    _Q_all, P_all, _P_abs = compute_xb_transition_matrices(state, constants, topology, dt)

    # Sample new states (same logic as thin_transitions)
    current_states = (xb_states_flat - 1).astype(jnp.int32)  # States are 1-6, indices are 0-5

    # Get probability vectors — index directly into P_all using current state
    prob_vectors = jax.vmap(lambda P, s: P[s])(P_all, current_states)  # (n_xb_total, 6)

    # Get permissiveness and binding info (needed for binding logic below)
    xb_nearest_bs = state.thick.xb_nearest_bs
    tm_states = state.thin.tm_states
    n_thin, n_sites = tm_states.shape

    if xb_nearest_bs is not None:
        xb_nearest_bs_flat = xb_nearest_bs.reshape(-1)
        thin_indices = topology.xb_to_thin_id
        site_indices = jnp.clip(xb_nearest_bs_flat, 0, n_sites - 1)
        nearest_tm_states = tm_states[thin_indices, site_indices]
        permissiveness = (nearest_tm_states == 3).astype(jnp.float32)
    else:
        permissiveness = jnp.ones(n_xb_total) * 0.5
        xb_nearest_bs_flat = jnp.full(n_xb_total, -1)
        thin_indices = topology.xb_to_thin_id
        site_indices = jnp.zeros(n_xb_total, dtype=jnp.int32)

    # Sample new states
    if random_values is None:
        rng_key, subkey = jax.random.split(rng_key)
        random_values = jax.random.uniform(subkey, shape=(n_xb_total,))

    cum_probs = jnp.cumsum(prob_vectors, axis=1)
    new_states_indices = jnp.argmax(random_values[:, None] < cum_probs, axis=1)
    new_states = new_states_indices + 1  # Convert back to 1-6

    # Reshape back — cast to int8 to match ThickState.xb_states dtype
    new_xb_states = new_states.reshape(n_thick, n_crowns, n_xb_per_crown).astype(jnp.int8)

    # ========================================================================
    # BINDING/UNBINDING LOGIC
    # ========================================================================
    old_states_flat = xb_states_flat
    new_states_flat = new_states

    old_is_bound = (old_states_flat >= 2) & (old_states_flat <= 4)
    new_is_bound = (new_states_flat >= 2) & (new_states_flat <= 4)

    is_binding = (~old_is_bound) & new_is_bound
    is_unbinding = old_is_bound & (~new_is_bound)

    xb_bound_to_flat = state.thick.xb_bound_to.reshape(-1)
    thin_bound_to_flat = state.thin.bound_to.reshape(-1)

    if xb_nearest_bs is not None:
        nearest_site_occupied = thin_bound_to_flat[thin_indices * n_sites + site_indices] >= 0
        can_bind = is_binding & (permissiveness > 0.5) & (~nearest_site_occupied)

        new_xb_bound_to_flat = jnp.where(
            can_bind,
            xb_nearest_bs_flat,
            jnp.where(
                is_unbinding,
                -1,
                xb_bound_to_flat
            )
        )

        xb_indices_arr = jnp.arange(n_xb_total)

        # STEP 1: Clear unbinding sites
        old_thin_indices = topology.xb_to_thin_id
        old_site_indices = jnp.clip(xb_bound_to_flat, 0, n_sites - 1)

        new_thin_bound_to_flat = thin_bound_to_flat.at[old_thin_indices * n_sites + old_site_indices].set(
            jnp.where(is_unbinding & (xb_bound_to_flat >= 0), -1, thin_bound_to_flat[old_thin_indices * n_sites + old_site_indices])
        )

        # STEP 2: Set binding sites
        binding_site_flat_indices = thin_indices * n_sites + site_indices

        n_sites_total = n_thin * n_sites
        binding_counts = jnp.zeros(n_sites_total, dtype=jnp.int32)
        binding_counts = binding_counts.at[binding_site_flat_indices].add(
            can_bind.astype(jnp.int32)
        )

        scatter_values = jnp.where(can_bind, xb_indices_arr, new_thin_bound_to_flat[binding_site_flat_indices])
        new_thin_bound_to_flat = new_thin_bound_to_flat.at[binding_site_flat_indices].set(scatter_values)

        new_xb_bound_to = new_xb_bound_to_flat.reshape(n_thick, n_crowns, n_xb_per_crown)
        new_thin_bound_to = new_thin_bound_to_flat.reshape(n_thin, n_sites)

        # If binding failed (site occupied), revert to DRX state (state 1)
        new_states_flat = jnp.where(
            is_binding & (~can_bind),
            1,
            new_states_flat
        )
        new_xb_states = new_states_flat.reshape(n_thick, n_crowns, n_xb_per_crown).astype(jnp.int8)
    else:
        new_xb_bound_to = state.thick.xb_bound_to
        new_thin_bound_to = state.thin.bound_to

    # Update state
    new_thick = state.thick._replace(
        xb_states=new_xb_states,
        xb_bound_to=new_xb_bound_to
    )
    new_thin = state.thin._replace(
        bound_to=new_thin_bound_to
    )
    new_state = state._replace(thick=new_thick, thin=new_thin)

    return new_state

