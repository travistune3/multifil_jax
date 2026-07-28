"""
Stochastic state transitions for tropomyosin and crossbridges.

This is where the model's chemistry actually happens. Every millisecond, each
tropomyosin site and each myosin head independently draws a new state from a
transition probability distribution that depends on calcium, on its neighbours,
and on its current mechanical strain.

FROM RATES TO A TRANSITION
--------------------------
The rate laws in rate_functions.py give a matrix Q of instantaneous rates, where
Q[i,j] is the rate of going from state i to state j and each row sums to zero.
That is a continuous-time Markov chain. To advance it by a finite timestep dt we
need the transition PROBABILITY matrix

    P = expm(Q * dt)

whose entry P[i,j] is the probability of being in state j after dt given state i
now. Taking the matrix exponential — rather than the cheaper Euler step
P ~ I + Q*dt — matters here: it stays a valid probability matrix at any dt, and
it correctly accounts for units that pass through an intermediate state within a
single timestep. Metrics that count ATP consumption depend on exactly that (see
the absorbing-state construction in _xb_P_from_Q_bins).

Each unit then samples its next state from its own row of P.

WHY THIS IS NOT ONE MATRIX EXPONENTIAL PER UNIT
-----------------------------------------------
A lattice can hold hundreds of thousands of heads and sites, and a 6x6 matrix
exponential each is far too expensive. Both state machines exploit the same
observation: although every unit has its own rate matrix in principle, the rates
depend on only a few DISCRETE quantities, so there are far fewer distinct
matrices than there are units.

  Tropomyosin, Ising path (thin_transitions_ising): a site's rates depend only on
    how many of its two chain neighbours are in each state. That is 27
    combinations, so 27 matrices serve every site on every filament.

  Tropomyosin, legacy path (thin_transitions): rates depend only on whether the
    site is cooperatively activated — 2 matrices.

  Crossbridges (thick_transitions): rates depend on the head's axial distance to
    its target (continuous) and on whether that target is open (binary). The
    continuous axis is discretized into n_xb_bins bins, giving 2 * n_xb_bins
    matrices regardless of lattice size. Heads then gather the row for their own
    bin. Bin resolution is a genuine accuracy/cost tradeoff, controlled by
    StaticParams.n_xb_bins / xb_bin_lo / xb_bin_hi.

In every case: build the small set of distinct matrices, exponentiate them in one
batch, then gather per unit.

TIERED INPUTS
-------------
    State (Tier 0)      pure simulation arrays — positions and states only
    Topology (Tier 1)   index maps, chain neighbours, bin edges, identity matrices
    Constants (Tier 2)  DynamicParams with rates, pCa, lattice_spacing

Rate laws live in rate_functions.py; this module assembles and applies them.
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
    tm_rate_01, tm_rate_10, tm_rate_12, tm_rate_21,
    tm_rate_23, tm_rate_32, tm_rate_30,
    xb_rate_01, xb_rate_10, xb_rate_12, xb_rate_21,
    xb_rate_23, xb_rate_32, xb_rate_34, xb_rate_43,
    xb_rate_40, xb_rate_04, xb_rate_50, xb_rate_05,
    compute_xb_energies,
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
    """Matrix exponential of a batch of small matrices, by scaling-and-squaring.

    Computes exp(A) for each matrix in the batch. Used to turn rate matrices
    Q*dt into transition probability matrices.

    METHOD. A Pade approximant is accurate only for small ||A||, so the standard
    trick is to scale the matrix down until it is small, approximate there, and
    square the result back up:

        1. pick s such that ||A / 2^s|| <= 0.5
        2. approximate exp(A / 2^s) by a 6th-order Pade rational approximation
        3. square the result s times, since exp(A) = (exp(A / 2^s))^(2^s)

    The Pade form is exp(X) ~ (V - U)^-1 (V + U) with U odd and V even in X,
    which needs only matrix powers and one small linear solve — cheaper and
    better conditioned than a Taylor series of comparable accuracy.

    WHY A FIXED 18 SQUARINGS. Each matrix in the batch needs its own s, but a
    data-dependent loop count would prevent XLA from fusing the batch into one
    kernel. Instead the loop always runs 18 times and each step squares only the
    matrices that still need it, selecting with jnp.where. That covers
    ||A|| up to 2^18 ~ 2.6e5, comfortably beyond any physiological rate times a
    sensible timestep. The unused iterations cost a predictable few percent, and
    buy full fusion across the batch.

    NUMERICAL SAFEGUARDS. Rows are renormalized to sum to 1 afterwards, since
    float32 rounding through many squarings leaves them slightly off, and a row
    that does not sum to 1 is not a probability distribution — the sampler would
    silently bias. Any NaN that survives is replaced with a uniform distribution
    rather than propagating.

    Args:
        A_batch: (batch, n, n) matrices to exponentiate (typically Q * dt)
        identity: (n, n) identity from SarcTopology (eye_4 or eye_6). Passed in
            rather than built with jnp.eye(n) inside, which would make XLA
            materialize a fresh copy per batch element under vmap.

    Returns:
        (batch, n, n) matrix exponentials
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

def _build_tm_Q_matrix_optimized(k_00, k_01, k_10, k_11, k_12,
                                  k_21, k_22, k_23, k_30, k_32, k_33):
    """Assemble tropomyosin rate matrices from their individual rates.

    Layout of the 4x4 generator (row = current state, column = destination):

            to:   0        1        2        3
        from 0 [ k_00     k_01      0        0   ]
        from 1 [ k_10     k_11     k_12      0   ]
        from 2 [  0       k_21     k_22     k_23 ]
        from 3 [ k_30      0       k_32     k_33 ]

    The zeros are structural, not merely small. A site cannot jump from blocking
    straight to open (0->3) — calcium must bind first — and it cannot go from
    Ca-bound-blocking straight back to open (1->3). The one asymmetry is the
    3->0 entry, which closes the cycle: calcium dissociates directly from the
    open state rather than retracing the forward path.

    Diagonal entries k_ii are negative and make each row sum to zero, which is
    what makes this a valid generator: probability is conserved.

    Built by stacking rows rather than scattering into a zero matrix with
    .at[].set(), which would emit a separate XLA scatter per entry.

    Args:
        k_ij: (n_configs,) arrays, one entry per distinct neighbour configuration

    Returns:
        Q: (n_configs, 4, 4) rate matrices
    """
    n = k_00.shape[0]
    zeros = jnp.zeros(n)

    # Build each row as (n, 4)
    row0 = jnp.stack([k_00, k_01, zeros, zeros], axis=1)
    row1 = jnp.stack([k_10, k_11, k_12, zeros], axis=1)
    row2 = jnp.stack([zeros, k_21, k_22, k_23], axis=1)
    row3 = jnp.stack([k_30, zeros, k_32, k_33], axis=1)

    # Stack rows to form (n, 4, 4)
    Q = jnp.stack([row0, row1, row2, row3], axis=1)

    return Q


def _build_xb_Q_matrix_optimized(r00, r01, r04, r05, r10, r11, r12,
                                  r21, r22, r23, r32, r33, r34,
                                  r40, r43, r44, r50, r55):
    """Assemble crossbridge rate matrices from their individual rates.

    Layout of the 6x6 generator (row = current state, column = destination),
    with states 0 DRX, 1 Loose, 2 Tight_1, 3 Tight_2, 4 Free_2, 5 SRX:

            to:    0      1      2      3      4      5
        from 0 [ r00    r01     0      0     r04    r05 ]
        from 1 [ r10    r11    r12     0      0      0  ]
        from 2 [  0     r21    r22    r23     0      0  ]
        from 3 [  0      0     r32    r33    r34     0  ]
        from 4 [ r40     0      0     r43    r44     0  ]
        from 5 [ r50     0      0      0      0     r55 ]

    The sparsity encodes the biology. A head must pass through weak binding
    before it can bind strongly (no 0->2), it cannot skip the working stroke
    (no 2->4), and SRX connects only to DRX — a parked head has to rejoin the
    available pool before it can do anything else. r43 is structurally present
    but always zero: re-attaching directly into the post-stroke state would run
    the ATPase backwards.

    Note that state 0 has three exits: forward into binding (r01), backward
    toward the pre-recovery state (r04), and sideways into the SRX reserve
    (r05). That three-way branch at DRX is where thick-filament regulation
    competes with attachment.

    Diagonal entries are negative and make each row sum to zero.

    Args:
        r_ij: (n_xb,) arrays, one entry per crossbridge (or per rate bin)

    Returns:
        Q: (n_xb, 6, 6) rate matrices
    """
    n_xb = r00.shape[0]
    zeros = jnp.zeros(n_xb)

    # Build each row as (n_xb, 6)
    # Row 0 (state 0 = DRX): can go to states 1, 4, 5
    row0 = jnp.stack([r00, r01, zeros, zeros, r04, r05], axis=1)

    # Row 1 (state 1 = loose): can go to states 0, 2
    row1 = jnp.stack([r10, r11, r12, zeros, zeros, zeros], axis=1)

    # Row 2 (state 2 = tight_1): can go to states 1, 3
    row2 = jnp.stack([zeros, r21, r22, r23, zeros, zeros], axis=1)

    # Row 3 (state 3 = tight_2): can go to states 2, 4
    row3 = jnp.stack([zeros, zeros, r32, r33, r34, zeros], axis=1)

    # Row 4 (state 4 = free_2): can go to states 0, 3
    row4 = jnp.stack([r40, zeros, zeros, r43, r44, zeros], axis=1)

    # Row 5 (state 5 = SRX): can go to state 0
    row5 = jnp.stack([r50, zeros, zeros, zeros, zeros, r55], axis=1)

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
    """Convert rate matrices into transition probability matrices over one step.

    P = expm(Q * dt). Entry P[i,j] is the probability that a unit currently in
    state i is in state j after dt has elapsed — including via intermediate
    states, which is what distinguishes this from a first-order Euler step.

    Thin wrapper over expm_pade6_batch(); see there for the algorithm.

    Args:
        Q: (n_matrices, n_states, n_states) rate matrices, rows summing to zero
        dt: Timestep length (ms)
        identity: (n_states, n_states) identity from the topology — topology.eye_4
            for tropomyosin, topology.eye_6 for crossbridges. Passing it in
            avoids XLA materializing a fresh identity per batch element.

    Returns:
        P: (n_matrices, n_states, n_states) row-stochastic probability matrices
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
    """Build the 2 distinct rate matrices of the LEGACY cooperativity model.

    On the legacy path a site's rates depend on exactly one binary property:
    whether it currently falls inside the cooperative span of an active
    neighbour, as determined earlier by kernels/cooperativity.py. So the entire
    filament population reduces to two rate matrices:

        Q_unique[0]  not cooperatively activated (boost factor 1.0)
        Q_unique[1]  cooperatively activated     (boost factor coop_magnitude)

    The boost multiplies the three forward rates and leaves the reverse rates
    alone. That asymmetry is deliberate in this model but does break detailed
    balance — cooperativity here changes the equilibrium as well as the kinetics,
    without an explicit free-energy justification. The Ising path
    (_compute_unique_tm_Q_matrices_ising) was introduced to address exactly that.

    Sites locked by a bound crossbridge are NOT a third configuration; they are
    handled by overriding their probability vector in thin_transitions().

    Args:
        ca_concentration: Calcium concentration (M)
        coop_magnitude: Forward-rate multiplier for activated sites
        params: DynamicParams with the tm_* rate constants

    Returns:
        Q_unique: (2, 4, 4) — index 0 non-cooperative, index 1 cooperative
    """
    # Get rate constants from params (attribute access)
    Keq_01 = params.tm_Keq_01
    Keq_12 = params.tm_Keq_12
    Keq_23 = params.tm_Keq_23
    k_01_base = params.tm_k_01
    k_12_base = params.tm_k_12
    k_23_base = params.tm_k_23
    k_30_base = params.tm_k_30

    # Per-config cooperativity factors (index 0 = non-coop, 1 = coop)
    coop = jnp.array([1.0, coop_magnitude], dtype=jnp.float32)

    # Forward rates (per-config)
    k_01 = k_01_base * ca_concentration * coop
    k_12 = k_12_base * coop
    k_23 = k_23_base * coop
    k_30 = jnp.broadcast_to(k_30_base, (2,))  # not coop-modulated

    # Backward rates (detailed balance, identical across configs)
    k_10 = jnp.broadcast_to(k_01_base / Keq_01, (2,))
    k_21 = jnp.broadcast_to(k_12_base / Keq_12, (2,))
    k_32 = jnp.broadcast_to(k_23_base / Keq_23, (2,))

    # Diagonal rates
    k_00 = -k_01
    k_11 = -(k_10 + k_12)
    k_22 = -(k_21 + k_23)
    k_33 = -(k_30 + k_32)

    return _build_tm_Q_matrix_optimized(
        k_00, k_01, k_10, k_11, k_12,
        k_21, k_22, k_23, k_30, k_32, k_33,
    )


def thin_transitions(state: 'State',
                    constants: 'DynamicParams',
                    topology: 'SarcTopology',
                    rng_key: jax.random.PRNGKey,
                    dt: float,
                    random_values: Optional[jnp.ndarray] = None,
                    tm_subpop=None) -> Tuple['State', jnp.ndarray]:
    """Advance every tropomyosin site one timestep — LEGACY cooperativity path.

    Reached only when run(legacy_coop=True). The default path is
    thin_transitions_ising(); this one is retained so that parameter sets fitted
    under the older cooperativity model remain reproducible.

    Requires state.thin.subject_to_coop to have been filled in first by
    kernels/cooperativity.py — unlike the Ising path, which derives its
    neighbour information itself.

    Only 2 distinct probability matrices exist (cooperative and not), so the
    whole filament population is served by one 2-matrix exponential plus a
    gather, rather than one exponential per site.

    LOCKED SITES. A site in state 3 with a crossbridge attached cannot leave:
    its probability vector is overridden to [0, 0, 0, 1]. Physically the bound
    head holds tropomyosin out of the way, so the site cannot close underneath
    it. This is applied after the matrix exponential rather than by zeroing
    rates, because whether a site is locked changes every timestep while the
    rate matrices do not — folding it into Q would defeat the 2-matrix
    reduction.

    Args:
        state: Current State (reads tm_states, subject_to_coop, bound_to)
        constants: DynamicParams with pCa and the tm_* rates
        topology: SarcTopology, for the eye_4 identity
        rng_key: JAX random key for sampling
        dt: Timestep length (ms)
        random_values: Optional pre-drawn uniforms, for deterministic testing
        tm_subpop: Optional (mode, constants_k, extra) for mixed populations;
            None runs the single-population path verbatim. 'mean_field'
            weight-sums the per-population rate matrices before one exponential;
            'explicit' exponentiates each population and selects per site by
            integer label. See core/subpopulation.py.

    Returns:
        new_state: State with updated tm_states
        P_unique: the distinct probability matrices used, for validation
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

    # Map each site to its config (0 = non-coop, 1 = coop)
    config_idx = is_coop_flat.astype(jnp.int32)  # (n_sites_total,)

    if tm_subpop is None:
        # OPTIMIZATION: Compute only 2 unique Q matrices, then P = exp(Q*dt)
        Q_unique = _compute_unique_tm_Q_matrices(ca_conc, constants.tm_coop_magnitude, constants)
        P_unique = expm_pade6_batch(Q_unique * dt, identity=eye_4)  # (2, 4, 4)
        P_indexed = P_unique[config_idx]  # (n_sites_total, 4, 4)
    else:
        mode, constants_k, extra = tm_subpop
        # Per-population unique-Q matrices (each (2, 4, 4)); coop magnitude is
        # per-population so a subpop may scale tm_coop_magnitude too.
        Q_k = [_compute_unique_tm_Q_matrices(ca_conc, ck.tm_coop_magnitude, ck)
               for ck in constants_k]
        if mode == 'mean_field':
            fractions = extra  # (K,)
            Q_eff = sum(fractions[k] * Q_k[k] for k in range(len(constants_k)))
            P_unique = expm_pade6_batch(Q_eff * dt, identity=eye_4)  # (2, 4, 4)
            P_indexed = P_unique[config_idx]
        else:  # explicit mixture: per-site label select
            labels = extra  # (n_sites_total,) INT in [0, K)
            Q_stack = jnp.stack(Q_k)  # (K, 2, 4, 4)
            Kp = Q_stack.shape[0]
            P_unique = expm_pade6_batch(
                Q_stack.reshape(Kp * 2, 4, 4) * dt, identity=eye_4).reshape(Kp, 2, 4, 4)
            P_indexed = P_unique[labels, config_idx]  # (n_sites_total, 4, 4)

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
# SYMMETRIC ISING TROPOMYOSIN COOPERATIVITY — the default path
# ============================================================================
#
# THE BIOLOGY. Tropomyosin is a continuous strand running the length of the thin
# filament, not a row of independent switches. When one stretch swings away from
# actin, it mechanically strains the adjoining stretches toward doing the same.
# That coupling is why muscle force rises far more steeply with calcium than
# independent binding at each troponin could ever produce.
#
# THE MODEL. Treat each site as a spin coupled to its two nearest neighbours
# ALONG ITS OWN TROPOMYOSIN STRAND. Neighbours are structural, precomputed once
# as topology.tm_prev_neighbor / tm_next_neighbor — a site's immediate
# predecessor and successor in its chain's axial ordering. There is no distance
# threshold and no length scale to tune.
#
# Each site feels a local field, in kT:
#
#     h(i) = J_C * n_2(i) + J_M * n_3(i) - 0.5*(J_C + J_M) * n_closed(i)
#
# where n_2, n_3 and n_closed count how many of site i's neighbours are Ca-open,
# crossbridge-bound, and closed respectively. Each count is in {0, 1, 2}, since
# a site has at most two same-chain neighbours by construction.
#
# WHY THE FIELD IS SPLIT SYMMETRICALLY. Forward rates are multiplied by
# exp(+h/2) and reverse rates by exp(-h/2). The equilibrium constant therefore
# shifts by exactly exp(h) — precisely the Boltzmann factor for a state whose
# energy has been lowered by h kT. This makes the coupling a genuine free-energy
# term rather than an ad hoc rate boost, and detailed balance survives on every
# reversible leg. Multiplying only the forward rates, as the legacy model does,
# would not have that property.
#
# The one-way cycle-closing rate k_30 is deliberately left unscaled: boosting it
# alongside the forward rates produces anti-cooperative runaway rather than
# sharper activation.
#
# LIMITS. At J_C = J_M = 0 the field vanishes and this reduces exactly to the
# uncoupled baseline. At J_C = J_M = J it reduces to the textbook 1D Ising
# chain, h = J * (n_open - n_closed).
#
# IMPLEMENTATION. Since the field depends only on the triple
# (n_2, n_3, n_closed), there are 3^3 = 27 distinct rate matrices for the entire
# system, however large the lattice. Build all 27, exponentiate in one batch,
# then gather per site.


def _compute_unique_tm_Q_matrices_ising(ca_concentration: float,
                                         J_C: float,
                                         J_M: float,
                                         params) -> jnp.ndarray:
    """Build all 27 rate matrices of the Ising cooperativity model.

    One matrix per possible neighbour composition (n_2, n_3, n_closed), each
    count running over {0, 1, 2}. Because that is the only thing a site's rates
    depend on, these 27 matrices cover every site in the system regardless of
    lattice size.

    The local field h is applied as exp(+h/2) on the three forward rates and
    exp(-h/2) on their reverses, leaving the one-way cycle-closing rate k_30 at
    its base value — see the section header above for why.

    Args:
        ca_concentration: Calcium concentration (M)
        J_C: Coupling to Ca-open neighbours (kT)
        J_M: Coupling to crossbridge-bound neighbours (kT)
        params: DynamicParams with the tm_* rates and equilibrium constants

    Returns:
        Q_unique: (3, 3, 3, 4, 4), indexed [n_2, n_3, n_closed]
    """
    Keq_01 = params.tm_Keq_01
    Keq_12 = params.tm_Keq_12
    Keq_23 = params.tm_Keq_23
    k_01_base = params.tm_k_01
    k_12_base = params.tm_k_12
    k_23_base = params.tm_k_23
    k_30_base = params.tm_k_30

    # 27 configurations as flat arrays (index = n_2*9 + n_3*3 + n_closed)
    levels = jnp.array([0, 1, 2], dtype=jnp.float32)
    n2g, n3g, ncg = jnp.meshgrid(levels, levels, levels, indexing='ij')
    n_2_flat = n2g.reshape(-1)
    n_3_flat = n3g.reshape(-1)
    n_c_flat = ncg.reshape(-1)

    h = J_C * n_2_flat + J_M * n_3_flat - 0.5 * (J_C + J_M) * n_c_flat
    forward_boost = jnp.exp(0.5 * h)    # (27,)
    backward_boost = jnp.exp(-0.5 * h)  # (27,)

    # Boost ALL three forward TM transitions (0→1 Ca-bind, 1→2 intermediate shift,
    # 2→3 TM-to-M-position) and their backwards symmetrically. State 3 = M-position
    # (TM open, available for XB binding). The cycle-close step k_30 stays at base
    # — slowing it produces anti-cooperative cascade behavior (verified empirically).
    # This is the original Tanner 2012 prescription (Ψ on r_{t,12} and r_{t,23}),
    # plus the Ca-binding step, applied Glauber-symmetrically.
    k_01 = (k_01_base * ca_concentration) * forward_boost
    k_12 = k_12_base * forward_boost
    k_23 = k_23_base * forward_boost
    k_30 = jnp.broadcast_to(k_30_base, (27,))

    k_10 = (k_01_base / Keq_01) * backward_boost
    k_21 = (k_12_base / Keq_12) * backward_boost
    k_32 = (k_23_base / Keq_23) * backward_boost

    # Diagonals
    k_00 = -k_01
    k_11 = -(k_10 + k_12)
    k_22 = -(k_21 + k_23)
    k_33 = -(k_30 + k_32)

    Q_flat = _build_tm_Q_matrix_optimized(
        k_00, k_01, k_10, k_11, k_12,
        k_21, k_22, k_23, k_30, k_32, k_33,
    )  # (27, 4, 4)

    return Q_flat.reshape(3, 3, 3, 4, 4)


def thin_transitions_ising(state: 'State',
                           constants: 'DynamicParams',
                           topology: 'SarcTopology',
                           rng_key: jax.random.PRNGKey,
                           dt: float,
                           random_values: Optional[jnp.ndarray] = None) -> Tuple['State', jnp.ndarray]:
    """Advance every tropomyosin site one timestep — the DEFAULT path.

    Uses the symmetric Ising cooperativity described in the section header
    above. Unlike the legacy thin_transitions(), this does NOT read
    state.thin.subject_to_coop: it counts each site's neighbour states itself
    from the current tm_states and the structural chain adjacency. That is why
    the default kinetics phase can skip the thin-filament force calculation and
    the cooperativity update entirely — there is nothing to precompute.

    Sequence: build 27 rate matrices, exponentiate them in one batch, count each
    site's neighbours to get its configuration index, gather its probability
    vector, override locked sites, then sample.

    LOCKED SITES. A site in state 3 with a crossbridge attached is forced to
    stay there ([0, 0, 0, 1]) — the bound head physically blocks tropomyosin's
    return. Applied after the exponential, for the same reason as in the legacy
    path.

    Args:
        state: Current State (reads tm_states and bound_to)
        constants: DynamicParams with pCa, tm_J_C, tm_J_M and the tm_* rates
        topology: SarcTopology, for chain neighbours and the eye_4 identity
        rng_key: JAX random key for sampling
        dt: Timestep length (ms)
        random_values: Optional pre-drawn uniforms, for deterministic testing

    Returns:
        new_state: State with updated tm_states
        P_flat: (27, 4, 4) the distinct probability matrices used, for validation

    Note:
        Subpopulations are not supported on this path — scaling tm_* rates for a
        subset of sites requires legacy_coop=True. run() raises rather than
        silently ignoring it.
    """
    # Local import to avoid circular ref at module load
    from multifil_jax.kernels.cooperativity import count_neighbor_states_split

    tm_states = state.thin.tm_states                    # (n_thin, n_sites) int8
    tm_prev_neighbor = topology.tm_prev_neighbor        # (n_thin, n_sites)
    tm_next_neighbor = topology.tm_next_neighbor        # (n_thin, n_sites)
    bound_to = state.thin.bound_to
    eye_4 = topology.eye_4

    is_bound = bound_to >= 0
    n_thin, n_sites = tm_states.shape
    n_sites_total = n_thin * n_sites

    ca_conc = 10.0 ** (-constants.pCa)
    J_C = constants.tm_J_C
    J_M = constants.tm_J_M

    # Build 27 unique Q matrices, then expm in one batch
    Q_unique = _compute_unique_tm_Q_matrices_ising(ca_conc, J_C, J_M, constants)
    Q_flat = Q_unique.reshape(27, 4, 4)
    P_flat = expm_pade6_batch(Q_flat * dt, identity=eye_4)  # (27, 4, 4)

    # Per-filament neighbor counts (each function call processes one strand)
    n_2, n_3, n_c = jax.vmap(count_neighbor_states_split)(tm_states, tm_prev_neighbor, tm_next_neighbor)
    # all shape (n_thin, n_sites), int32 capped at 2

    config_idx = (n_2 * 9 + n_3 * 3 + n_c).reshape(-1)  # (n_sites_total,) int32
    P_indexed = P_flat[config_idx]                       # (n_sites_total, 4, 4)

    tm_states_flat = tm_states.reshape(-1).astype(jnp.int32)
    is_bound_flat = is_bound.reshape(-1)

    prob_vectors = jax.vmap(lambda P, s: P[s])(P_indexed, tm_states_flat)

    # Locked: in state 3 AND bound to XB → stay in state 3
    locked_mask = (tm_states_flat == 3) & is_bound_flat
    locked_prob = jnp.array([0.0, 0.0, 0.0, 1.0])
    prob_vectors = jnp.where(locked_mask[:, None], locked_prob, prob_vectors)

    if random_values is None:
        rng_key, subkey = jax.random.split(rng_key)
        random_values = jax.random.uniform(subkey, shape=(n_sites_total,))

    cum_probs = jnp.cumsum(prob_vectors, axis=1)
    new_states = jnp.argmax(random_values[:, None] < cum_probs, axis=1)

    new_tm_states = new_states.reshape(n_thin, n_sites).astype(jnp.int8)
    new_thin = state.thin._replace(tm_states=new_tm_states)
    new_state = state._replace(thin=new_thin)

    return new_state, P_flat


# ============================================================================
# CROSSBRIDGE TRANSITIONS
# ============================================================================

def xb_rate_matrix(xb_distances: jnp.ndarray,
                   lattice_spacing: float,
                   spring_constants: jnp.ndarray,
                   permissiveness: jnp.ndarray,
                   ca_concentration: float,
                   temp_celsius: float,
                   params: 'DynamicParams') -> jnp.ndarray:
    """Build 6x6 crossbridge rate matrices from geometry and calcium.

    This is where mechanics enters chemistry. Each input row describes a head's
    position relative to its target site; the output is that head's full rate
    matrix over the six states (0 DRX, 1 Loose, 2 Tight_1, 3 Tight_2, 4 Free_2,
    5 SRX — see the rate_functions module docstring for what each means).

    The chain of reasoning per head:

      1. Convert (axial, radial) offset into the head's polar geometry,
         r = sqrt(x^2 + y^2) and theta = atan2(y, x).
      2. Evaluate the two-spring elastic energy at that geometry, once for the
         weak configuration and once for the strong one.
      3. Add those elastic energies to the chemical free energies of the
         corresponding states. A strained head sits higher in free energy, and
         because reverse rates are derived from these totals, strain
         automatically makes unfavourable bonds break faster.
      4. Compute the force the head would carry in the strong state, and feed it
         to the load-dependent (Bell) rates for the working stroke and
         detachment.
      5. Assemble everything into the generator, with diagonals set so rows sum
         to zero.

    Despite the name, this function does not need to be called once per head:
    the caller evaluates it on a small grid of representative geometries and
    gathers per head. See _build_xb_Q_bins.

    Args:
        xb_distances: (n_xb, 2) (axial, radial) offset from each head to its
            target binding site (nm)
        lattice_spacing: Lattice spacing (nm). Redundant with the radial column
            of xb_distances in current callers, but kept explicit so callers can
            evaluate hypothetical geometries.
        spring_constants: (n_xb, 8) two-spring parameters per head —
            [:, 0:4] g_k_weak, g_rest_weak, c_k_weak, c_rest_weak
            [:, 4:8] g_k_strong, g_rest_strong, c_k_strong, c_rest_strong
        permissiveness: (n_xb,) 1 if the target site's tropomyosin is open,
            else 0. Gates attachment entirely.
        ca_concentration: Calcium concentration (M), for SRX recruitment
        temp_celsius: Temperature (C), which sets kT for every Boltzmann and
            Bell term
        params: DynamicParams with the xb_* rate coefficients and free energies

    Returns:
        Q: (n_xb, 6, 6) rate matrices (ms^-1), each row summing to zero
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

    # Thermal energy kT, converted from J to pN*nm (1 J = 1e21 pN*nm).
    # Boltzmann's constant is written to 5 significant figures here (the CODATA
    # value is 1.380649e-23 J/K); the 4e-5 relative difference is far below the
    # uncertainty in any rate constant it multiplies.
    k_t = 1.3810e-23 * (temp_celsius + 273.15) * 1e21  # pN*nm

    # Compute energies using helper function (vectorized)
    E_weak = (0.5 * g_k_weak * (r - g_r_weak)**2 +
              0.5 * c_k_weak * (theta - c_r_weak)**2) / k_t
    E_strong = (0.5 * g_k_strong * (r - g_r_strong)**2 +
                0.5 * c_k_strong * (theta - c_r_strong)**2) / k_t

    # Energy difference driving the weak->strong isomerization. Note this is a
    # plain subtraction, unlike compute_xb_energies() in rate_functions.py which
    # accumulates the same quantity term-wise to preserve float32 precision.
    # Here the inputs are bin-grid geometries rather than per-head positions, so
    # the cancellation is bounded and the simpler form is adequate.
    E_diff = E_weak - E_strong

    # Chemical free energy of each state (kT). Adding the elastic energy to the
    # bound states below is what couples mechanics to chemistry: a strained head
    # sits higher in free energy, so its reverse rates -- which are derived from
    # these totals -- rise accordingly.
    U_DRX = params.xb_U_DRX
    U_loose_base = params.xb_U_loose
    U_tight_1_base = params.xb_U_tight_1
    U_tight_2_base = params.xb_U_tight_2

    U_loose = U_loose_base + E_weak
    U_tight_1 = U_tight_1_base + E_strong
    U_tight_2 = U_tight_2_base + E_strong

    # Calculate forces for force-dependent rates
    f_strong = g_k_strong * (r - g_r_strong) + (1.0/r) * c_k_strong * (theta - c_r_strong)

    # ========================================================================
    # RATE DEFINITIONS using consolidated params and imported rate functions
    # ========================================================================

    ones = jnp.ones(n_xb)

    # Get consolidated rate coefficients from params (attribute access)
    # Pre-exponential coefficients: each is multiplied by a strain- or
    # load-dependent exponential in the rate functions below, so none of them is
    # the actual rate except at zero strain and zero load.
    r01_coeff = params.xb_r01_coeff
    r12_coeff = params.xb_r12_coeff
    r23_coeff = params.xb_r23_coeff
    r34_coeff = params.xb_r34_coeff
    r40_rate = params.xb_r40
    r04_rate = params.xb_r04
    r05_rate = params.xb_r05

    # SRX parameters
    srx_k0 = params.xb_srx_k0
    srx_kmax = params.xb_srx_kmax
    srx_b = params.xb_srx_b
    srx_ca50 = params.xb_srx_ca50

    # 0 DRX <-> 1 Loose : attachment, gated by tropomyosin permissiveness
    r01 = xb_rate_01(permissiveness, r01_coeff, E_weak)
    r10 = xb_rate_10(r01, U_DRX, U_loose)

    # 1 Loose <-> 2 Tight_1 : weak-to-strong isomerization (Pi release)
    r12 = xb_rate_12(r12_coeff, E_diff)
    r21 = xb_rate_21(r12, U_loose, U_tight_1)

    # 2 Tight_1 <-> 3 Tight_2 : the working stroke (load-dependent)
    r23 = xb_rate_23(r23_coeff, f_strong, params.xb_delta_23, k_t)
    r32 = xb_rate_32(r23, U_tight_1, U_tight_2)

    # 3 Tight_2 -> 4 Free_2 : ADP release and detachment; r43 is structurally 0
    r34 = xb_rate_34(r34_coeff, f_strong, params.xb_delta_34, k_t)
    r43 = xb_rate_43() * ones  # Always 0

    # 4 Free_2 <-> 0 DRX : recovery stroke, re-priming the lever arm
    r40 = xb_rate_40(r40_rate) * ones
    r04 = xb_rate_04(r04_rate) * ones

    # 5 SRX <-> 0 DRX : thick-filament regulation; r50 is the Ca-dependent one
    r50 = xb_rate_50(ca_concentration, srx_k0, srx_kmax, srx_b, srx_ca50) * ones
    r05 = xb_rate_05(r05_rate) * ones

    # Diagonal rates: row sums must be zero for valid rate matrices
    # Direct arithmetic — avoids vmap+stack overhead (ordered_sum was just jnp.sum)
    r00 = -(r01 + r04 + r05)
    r11 = -(r10 + r12)
    r22 = -(r21 + r23)
    r33 = -(r32 + r34)
    r44 = -(r43 + r40)
    r55 = -r50

    # ========================================================================
    # CONSTRUCT Q MATRICES (optimized - single construction instead of .at[].set())
    # ========================================================================

    Q = _build_xb_Q_matrix_optimized(r00, r01, r04, r05, r10, r11, r12,
                                      r21, r22, r23, r32, r33, r34,
                                      r40, r43, r44, r50, r55)

    return Q


def _build_xb_Q_bins(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate crossbridge rate matrices on a distance grid, and index each head.

    The cost-saving step behind thick_transitions. A head's rates depend on two
    things: its axial distance to its target site (continuous) and whether that
    target's tropomyosin is open (binary). Rather than build a rate matrix per
    head, build them on a grid:

        n_xb_bins axial positions x 2 permissiveness levels

    laid out as one block at permissiveness 0 followed by one block at
    permissiveness 1. Each head is then assigned the index of the cell it falls
    in, and gathers from there.

    Bin assignment uses jnp.digitize against topology.xb_bin_edges, clipped at
    both ends — heads outside the grid range are treated as if at the nearest
    edge. Since the bin range is chosen to bracket reachable distances, and
    binding probability decays sharply outside it, that clipping affects only
    heads that were never going to bind.

    THE xb_valid GATE. Heads with no real geometric partner carry a placeholder
    target index, so the tropomyosin state they read is meaningless. They are
    forced to permissiveness 0, which routes them into the block where the
    attachment rate r01 is exactly zero at every bin position. This is a hard
    gate, not a distance penalty: such a head can never bind, at any strain.

    RETURNING Q RATHER THAN P is deliberate. The gather key depends only on
    geometry and permissiveness, never on the rate constants, so it is shared
    across every population in a subpopulation run. Keeping the exponential
    separate lets mean-field blending average the GENERATORS before a single
    exponential — the mathematically correct blend — while the explicit modes
    exponentiate each population and select per head afterwards.

    Args:
        state: Current State (reads xb_distances, xb_nearest_bs, tm_states)
        constants: DynamicParams with rates and lattice_spacing
        topology: SarcTopology with bin edges/centres, xb_to_thin_id, xb_valid

    Returns:
        Q_bins: (2 * n_bins, 6, 6) rate matrices, permissiveness-0 block first
        key: (n_xb_total,) index into Q_bins for each crossbridge
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
        # xb_valid gate: XBs with no real geometric thin-filament partner this
        # crown (continuous-formula miss) must never see permissiveness>0 —
        # their nearest_tm_states was read from an arbitrary remapped site
        # (thin_idx/thin_face forced to (0,0)), not a real target. Forcing
        # permissiveness to 0 routes them through the ap=0 Q-bin block, where
        # r01 (the only entry rate into a bound state) is exactly 0 for every
        # bin position — a hard gate, not a distance-decay approximation.
        permissiveness = (nearest_tm_states == 3).astype(jnp.float32) * topology.xb_valid.astype(jnp.float32)
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

    # Assign each XB to a bin via digitize + clip
    x_axial = xb_distances_flat[:, 0]                              # (n_xb_total,)
    bin_idx = jnp.digitize(x_axial, topology.xb_bin_edges) - 1    # in [-1, n_bins]
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    ap  = permissiveness.astype(jnp.int32)                         # 0 or 1
    key = ap * n_bins + bin_idx                                    # in [0, 2*n_bins)

    return Q_bins, key


def _xb_P_from_Q_bins(
    Q_bins: jnp.ndarray,
    key: jnp.ndarray,
    dt: float,
    eye_6: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exponentiate the binned rate matrices and gather per crossbridge.

    Returns three matrices per head:

      Q     the rate matrix it was assigned (returned for metrics that need
            instantaneous rates rather than step probabilities)
      P     its transition probability matrix over dt
      P_abs a second probability matrix computed from a MODIFIED generator with
            row 4 zeroed, making Free_2 absorbing

    WHY THE ABSORBING VARIANT EXISTS. Counting ATP consumption by comparing
    states before and after a step undercounts: a head can pass 3 -> 4 -> 0
    within one timestep, consuming an ATP but appearing to have gone 3 -> 0.
    Zeroing row 4 traps any head that reaches Free_2, so P_abs[i, 3, 4] is the
    probability of VISITING state 4 at any point during the step, not just of
    ending there. That is the quantity the atp_expected_p metric needs.

    Args:
        Q_bins: (n_cells, 6, 6) rate matrices on the bin grid
        key: (n_xb_total,) each head's index into the grid
        dt: Timestep length (ms)
        eye_6: (6, 6) identity, passed through to the exponential

    Returns:
        (Q_all, P_all, P_abs_all), each (n_xb_total, 6, 6)
    """
    P_bins = matrix_exponential_batch(Q_bins, dt, identity=eye_6)

    # Absorbing variant: zero row 4 so a head reaching Free_2 cannot leave it
    # within the step. See docstring — this is what makes the ATP metric count
    # 3 -> 4 -> 0 traversals rather than missing them.
    Q_abs_bins = Q_bins.at[:, 4, :].set(0.0)
    P_abs_bins = matrix_exponential_batch(Q_abs_bins, dt, identity=eye_6)

    return Q_bins[key], P_bins[key], P_abs_bins[key]


def compute_xb_transition_matrices(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    dt: float,
    xb_subpop=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-crossbridge rate and probability matrices, via the binned grid.

    The single entry point used by both thick_transitions() (which samples new
    states from P) and compute_all_metrics() (which needs Q and P_abs for ATP
    accounting). Sharing it means those two never disagree about what the rates
    were on a given step.

    Evaluates 2 * n_xb_bins matrix exponentials instead of one per head — at a
    4x4 lattice that is roughly a sixfold reduction in exponentials, in a step
    that otherwise dominates the timestep cost.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        dt: Timestep length (ms)
        xb_subpop: None for the standard single-population path, or a tuple
            (mode, constants_k, extra) for subpopulations. constants_k is a
            length-K list of DynamicParams (population 0 = WT). For
            mode=='mean_field', extra is a (K,) fractions vector and the K
            binned generators are weight-summed before a single exp
            (Q_eff = Σ f_k Q_k). For mode=='explicit', extra is a (n_xb_total,)
            INT label array and each population is exponentiated then selected
            per-XB by (label, bin). The gather key is shared across populations
            (it depends on geometry/permissiveness, not rates).

    Returns:
        Q_all:    (n_xb_total, 6, 6) rate matrices per crossbridge
        P_all:    (n_xb_total, 6, 6) transition probability matrices per crossbridge
        P_abs_all:(n_xb_total, 6, 6) absorbing-state P (row 4 zeroed) for ATP metrics
    """
    if xb_subpop is None:
        Q_bins, key = _build_xb_Q_bins(state, constants, topology)
        return _xb_P_from_Q_bins(Q_bins, key, dt, topology.eye_6)

    mode, constants_k, extra = xb_subpop
    eye_6 = topology.eye_6
    built = [_build_xb_Q_bins(state, ck, topology) for ck in constants_k]
    key = built[0][1]  # shared across populations (geometry/permissiveness only)

    if mode == 'mean_field':
        fractions = extra  # (K,)
        Q_eff = sum(fractions[k] * built[k][0] for k in range(len(constants_k)))
        return _xb_P_from_Q_bins(Q_eff, key, dt, eye_6)

    # explicit mixture: exponentiate each population, select per-XB by label
    labels = extra  # (n_xb_total,) INT in [0, K)
    Q_stack = jnp.stack([b[0] for b in built])          # (K, 2*n_bins, 6, 6)
    K, nb2 = Q_stack.shape[0], Q_stack.shape[1]
    P_stack = matrix_exponential_batch(
        Q_stack.reshape(K * nb2, 6, 6), dt, identity=eye_6).reshape(K, nb2, 6, 6)
    Q_abs = Q_stack.at[:, :, 4, :].set(0.0)
    P_abs_stack = matrix_exponential_batch(
        Q_abs.reshape(K * nb2, 6, 6), dt, identity=eye_6).reshape(K, nb2, 6, 6)
    return Q_stack[labels, key], P_stack[labels, key], P_abs_stack[labels, key]


def thick_transitions(state: 'State',
                     constants: 'DynamicParams',
                     topology: 'SarcTopology',
                     rng_key: jax.random.PRNGKey,
                     dt: float,
                     random_values: Optional[jnp.ndarray] = None,
                     xb_subpop=None):
    """Advance every crossbridge one timestep, and update what it is bound to.

    Two things happen here, and the second is easy to overlook: heads sample new
    states from their transition probability matrices, AND the binding
    bookkeeping is updated on both filaments. A head entering a bound state must
    record which site it took, and that site must record which head took it —
    the thin filament's bound_to array is what locks a tropomyosin site open
    (see thin_transitions_ising) and what lets forces.py know where to apply
    crossbridge force.

    Rates come from compute_xb_transition_matrices(), which evaluates them on a
    distance grid rather than per head; see there and in _build_xb_Q_bins for
    how that works and what it costs in accuracy.

    Heads flagged invalid by topology.xb_valid are held at permissiveness 0
    throughout, so they can never enter a bound state and never claim a site.

    Args:
        state: Current State
        constants: DynamicParams with pCa, lattice_spacing and the xb_* rates
        topology: SarcTopology with xb_to_thin_id, xb_valid, eye_6
        rng_key: JAX random key for sampling
        dt: Timestep length (ms)
        random_values: Optional pre-drawn uniforms, for deterministic testing
        xb_subpop: Optional (mode, constants_k, extra) for mixed populations;
            None runs the single-population path verbatim, at zero cost. See
            compute_xb_transition_matrices() for the tuple contract and
            core/subpopulation.py for how these are built.

    Returns:
        new_state: State with updated xb_states, xb_bound_to, and thin bound_to
    """
    # Get current xb states
    xb_states = state.thick.xb_states  # (n_thick, n_crowns, n_xb_per_crown)
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape

    # Flatten for processing
    xb_states_flat = xb_states.reshape(-1)  # (n_thick * n_crowns * n_xb_per_crown,)
    n_xb_total = xb_states_flat.shape[0]

    # Compute per-XB Q/P matrices via shared helper (subpop-aware)
    _Q_all, P_all, _P_abs = compute_xb_transition_matrices(
        state, constants, topology, dt, xb_subpop=xb_subpop)

    # Sample new states (same logic as thin_transitions)
    current_states = xb_states_flat.astype(jnp.int32)

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
        # xb_valid gate (see matching comment in _build_xb_Q_bins): geometrically
        # invalid XBs must never be treated as permissive, or they could bind at
        # an arbitrary remapped site below.
        permissiveness = (nearest_tm_states == 3).astype(jnp.float32) * topology.xb_valid.astype(jnp.float32)
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
    new_states = new_states_indices

    # Reshape back — cast to int8 to match ThickState.xb_states dtype
    new_xb_states = new_states.reshape(n_thick, n_crowns, n_xb_per_crown).astype(jnp.int8)

    # ========================================================================
    # BINDING/UNBINDING LOGIC
    # ========================================================================
    old_states_flat = xb_states_flat
    new_states_flat = new_states

    old_is_bound = (old_states_flat >= 1) & (old_states_flat <= 3)
    new_is_bound = (new_states_flat >= 1) & (new_states_flat <= 3)

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

        # If binding failed (site occupied), revert to DRX state (state 0)
        new_states_flat = jnp.where(
            is_binding & (~can_bind),
            0,
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

