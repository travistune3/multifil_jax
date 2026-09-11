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
single timestep. Metrics that count ATP consumption depend on exactly that, and
go one step further: the same scaling-and-squaring also yields the occupancy-time
integral INT_0^dt exp(Qt)dt, from which the EXACT expected number of crossings of
any edge follows (see expm_pade6_batch's with_integral and
xb_expected_crossings).

Each unit then samples its next state from its own row of P.

WHY THIS IS NOT ONE MATRIX EXPONENTIAL PER UNIT
-----------------------------------------------
A lattice can hold hundreds of thousands of heads and sites, and a 6x6 matrix
exponential each is far too expensive. Both state machines exploit the same
observation: although every unit has its own rate matrix in principle, the rates
depend on only a few DISCRETE quantities, so there are far fewer distinct
matrices than there are units.

  Tropomyosin (thin_transitions): a site's rates depend only on how many of its
    two chain neighbours are in each state, and on whether a crossbridge is
    bound to it. That is 27 x 2 combinations, so 54 matrices serve every site on
    every filament.

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


REFERENCE
---------
Tanner BCW, Regnier M, Daniel TL (2012), "Filament compliance influences
    cooperative activation of thin filaments and the dynamics of force production
    in skeletal muscle", PLoS Comput Biol 8:e1002506,
    doi:10.1371/journal.pcbi.1002506.
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
)
# The two-spring potential and its derivatives. forces.py owns the crossbridge
# force law outright and imports nothing from the package, so the rate path can
# depend on it with no cycle — and the chemistry and the mechanics then read the
# same potential by construction rather than by review.
from .forces import (xb_elastic_energy, xb_geometry, xb_polar_forces,
                     polar_to_filament)


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
    with_integral: bool = False,
):
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

    THE OCCUPANCY-TIME INTEGRAL (with_integral=True). Alongside exp(A) this can
    return

        phi1(A) = INT_0^1 exp(A u) du = (e^A - 1) / A

    which is what turns a generator into EXPECTED CROSSING COUNTS: for Q held
    constant over dt, with A = Q*dt,

        E[# of i -> j crossings in dt | started in s] = q_ij * dt * phi1[s, i]

    Counting the EDGE, rather than asking whether a state was reached, is what
    makes this exact for multi-hop traversals AND for repeat crossings. See
    matrix_exponential_batch() and xb_expected_crossings().

    It is computed by the same scaling-and-squaring, one level at a time:

        J <- 0.5 * (J + E @ J)      because with X = A/2^k,
        E <- E @ E                  INT_0^2 exp(Xu)du = INT_0^1 + INT_1^2
                                                      = J + E @ J,
    the 0.5 renormalising back to the unit interval so J always means phi1 of
    the CURRENT level's matrix. The base case reuses the powers step 4 already
    computed:

        phi1 = (I + X^2/6 + X^4/120 + X^6/5040)
             + X @ (I/2 + X^2/24 + X^4/720)

    whose leading truncation X^7/40320 is 1.9e-7 at ||X|| = 0.5, at the float32
    epsilon floor. (A^-1 (E - I) is NOT available: a generator is singular by
    construction, since its rows sum to zero.)

    Rows of phi1 are renormalized for the same reason rows of exp(A) are, and
    the justification is NOT by analogy: for a generator every row of exp(Qt)
    sums to 1 at every t, so every row of INT_0^1 exp(Qu)du sums to 1 as well —
    it is a mixture of stochastic matrices over the unit interval.

    WHY THE INTEGRAL GETS ITS OWN fori_loop, AND WHY ITS MATMULS ARE PINNED TO
    precision=HIGHEST. Both were forced by measurement (2026-09-11), and the
    first version of this code had neither. Carrying J in the SAME loop carry as
    `result` changes `result` by max |d| = 0.90 — not one ulp. ROOT CAUSE: XLA
    lowers a loop body holding TWO einsums to the TF32 tensor-core GEMM on this
    GPU while the one-einsum body here is not so lowered, and TF32's 10-bit
    mantissa compounds over the 16 squarings the worst bins need (r12 reaches
    1e4 /ms, so ||Q*dt|| reaches 2e4 on a matrix whose entries span 1e-20 to 1).
    A separate loop keeps the probability path byte-for-byte identical to the
    with_integral=False path — which is the whole premise of the metric work
    this exists for — at the cost of 18 extra matmuls for a second copy of E.

    ACCURACY, scored against float64 (scipy) on real settled Q_bins, 8x8,
    dt = 1 ms, by local_projects/tension_cost/expm_integral_proto.py:
        this kernel's phi1*dt   3.2e-04     (cardiac)  2.6e-04 (skeletal)
        a float32 12x12 expm    2.3e-04                2.3e-04
        this kernel's own P     6.3e-04                7.0e-04
    i.e. the integral is MORE accurate than the probabilities it ships beside,
    and every exported flux is good to < 2e-5 of its own total. 1e-5 absolute is
    not reachable in float32 on these bins and never was.

    DO NOT REPLACE THIS WITH expm([[A, I], [0, 0]]). Two reasons, both measured:
    that block matrix's top rows sum to 1 + dt, so step 8's row normalization
    silently wrecks it (max |d| 0.409 on entries <= 0.831); and it costs 3.5x a
    6x6 expm and ~+20% of a BATCHED timestep (S138 [H]). It is cheaper than a
    6x6 expm at batch 1 — 0.65x here — but that is the launch-bound regime, and
    a sweep is not run at batch 1.

    Args:
        A_batch: (batch, n, n) matrices to exponentiate (typically Q * dt)
        identity: (n, n) identity from SarcTopology (eye_4 or eye_6). Passed in
            rather than built with jnp.eye(n) inside, which would make XLA
            materialize a fresh copy per batch element under vmap.
        with_integral: also return phi1(A). Static — it changes the returned
            pytree, so flipping it recompiles. The 4x4 tropomyosin call sites
            leave it off and pay nothing.

    Returns:
        (batch, n, n) matrix exponentials, or (exp(A), phi1(A)) if with_integral
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
    # `pade` is exp(A / 2^s), kept because the integral's loop (step 9) needs
    # its own copy of E to double alongside J.
    pade = jnp.linalg.solve(V - U, V + U)
    result = pade

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

    if not with_integral:
        return result

    # Step 9 (optional): the occupancy-time integral phi1(A), by the same
    # scaling-and-squaring one level at a time. Its own loop, at HIGHEST
    # precision — see the docstring for why neither is negotiable.
    HI = jax.lax.Precision.HIGHEST
    phi_even = I + A2/6.0 + A4/120.0 + A6/5040.0
    phi_odd_inner = I/2.0 + A2/24.0 + A4/720.0
    J = phi_even + jnp.einsum('...ij,...jk->...ik', A_scaled, phi_odd_inner,
                              precision=HI)

    def _double_step(i, carry):
        E, Jc = carry
        should = i < s
        J_dbl = 0.5 * (Jc + jnp.einsum('...ij,...jk->...ik', E, Jc, precision=HI))
        E_sq = jnp.einsum('...ij,...jk->...ik', E, E, precision=HI)
        return (jnp.where(should[:, None, None], E_sq, E),
                jnp.where(should[:, None, None], J_dbl, Jc))

    _, J = jax.lax.fori_loop(0, 18, _double_step, (pade, J))

    J = J / jnp.sum(J, axis=2, keepdims=True)
    J = jnp.where(jnp.isnan(J), 1.0/n, J)

    return result, J


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
    before it can bind strongly (no 0->2), it cannot detach from Tight_1 without
    first isomerizing to Tight_2 (no 2->4), and SRX connects only to DRX — a parked head has to rejoin the
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
    identity: Optional[jnp.ndarray] = None,
    with_integral: bool = False,
):
    """Convert rate matrices into transition probability matrices over one step.

    P = expm(Q * dt). Entry P[i,j] is the probability that a unit currently in
    state i is in state j after dt has elapsed — including via intermediate
    states, which is what distinguishes this from a first-order Euler step.

    Thin wrapper over expm_pade6_batch(); see there for the algorithm.

    WITH with_integral, also returns the occupancy-time integral

        G = INT_0^dt exp(Qt) dt = dt * phi1(Q*dt)

    whose entry G[s, i] is the expected TIME a unit starting in state s spends in
    state i during the step. Multiply by q_ij and you have the expected number of
    i -> j crossings — the exact flux, including repeat crossings, which is what
    the ATP metrics read. See expm_pade6_batch() for the algorithm and its cost.

    Args:
        Q: (n_matrices, n_states, n_states) rate matrices, rows summing to zero
        dt: Timestep length (ms)
        identity: (n_states, n_states) identity from the topology — topology.eye_4
            for tropomyosin, topology.eye_6 for crossbridges. Passing it in
            avoids XLA materializing a fresh identity per batch element.
        with_integral: also return G. Static.

    Returns:
        P: (n_matrices, n_states, n_states) row-stochastic probability matrices,
        or (P, G) if with_integral — G in units of dt (ms), P dimensionless.
    """
    # Scale Q by dt and compute exp using Pade6
    # Row normalization is done inside expm_pade6_batch
    if not with_integral:
        return expm_pade6_batch(Q * dt, identity=identity)
    P, phi1 = expm_pade6_batch(Q * dt, identity=identity, with_integral=True)
    return P, phi1 * dt


# ============================================================================
# TROPOMYOSIN TRANSITIONS — symmetric Ising cooperativity
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
# reversible leg. Multiplying only the forward rates — as the retired
# tension-span cooperativity model did — would not have that property.
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
# (n_2, n_3, n_closed), there are 3^3 = 27 distinct rate matrices per lock state
# for the entire system, however large the lattice. The crossbridge lock adds a
# second binary axis (see _compute_unique_tm_Q_matrices), giving 54. Build them
# all, exponentiate in one batch, then gather per site.


def count_neighbor_states_split(tm_states: jnp.ndarray,
                                tm_prev_neighbor: jnp.ndarray,
                                tm_next_neighbor: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Count same-chain neighbors in state 2, state 3, and closed (state 0/1),
    using the fixed (<=2) topological same-chain neighbor set per site.

    NEIGHBOURS ARE STRUCTURAL, NOT SPATIAL. A site couples to its immediate
    predecessor and successor in its own chain's axial ordering, precomputed once
    when the topology is built. There is no distance threshold and no length
    scale to tune, which is what makes the coupling a genuine nearest-neighbour
    Ising chain rather than a windowed approximation to one.

    Endpoints self-reference rather than carrying a sentinel index, so every
    gather is valid and no masking is needed. A site therefore has at most two
    real neighbours by construction, and the counts fall in {0, 1, 2} naturally —
    the jnp.minimum(..., 2) below is defensive, kept because the downstream
    Q-matrix gather assumes that range structurally.

    Nothing is written to state: the counts are consumed inline by
    thin_transitions().

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


def _compute_unique_tm_Q_matrices(ca_concentration: float,
                                  J_C: float,
                                  J_M: float,
                                  params) -> jnp.ndarray:
    """Build all 54 rate matrices of the Ising cooperativity model.

    One matrix per possible neighbour composition (n_2, n_3, n_closed), each
    count running over {0, 1, 2}, TIMES the two values of "is a crossbridge
    bound here". Because that is the only thing a site's rates depend on, these
    54 matrices cover every site in the system regardless of lattice size.

    THE CROSSBRIDGE LOCK IS A RATE, and it lives here. A site with a head bound
    has its two exits from the open state 3 divided by (1 + xb_tm_K2):

        k_30 /= (1 + K2),  k_32 /= (1 + K2),  k_33 = -(k_30 + k_32)

    McKillop & Geeves 1993 Fig. 1: one bound S1 multiplies the open/closed ratio
    by (1 + K2). That is a statement about an EQUILIBRIUM RATIO, and this is the
    form that reproduces it at any dt — steady-state open/closed becomes
    k_23/(k_32/(1+K2)) = K_T(1+K2) exactly. See the xb_tm_K2 block in
    core/params.py for the measured values.

    IT USED TO BE APPLIED AFTER THE MATRIX EXPONENTIAL, by dividing the sampled
    row's total exit PROBABILITY by (1 + K2). That equals this only to first
    order in dt, and dt = 1 ms is nowhere near that limit: the unlocked exit
    probability from state 3 over 1 ms spans 0.058-0.974 across the 27
    configurations, deep in the saturation of 1 - exp(-L*dt). Measured (S135,
    both presets, pCa 4.5 and 6.2, all 27 configurations, float64), a nominal
    xb_tm_K2 = 79 ran as an effective 174.2 at dt = 1 ms, falling to 79.9 at
    dt = 0.01 ms. The stated reason for the post-hoc form — that folding it into
    Q "would defeat the 27-matrix reduction" — was also wrong: the reduction is
    n_sites -> a constant, and 54 is as constant as 27.

    THE CONTROL PATH IS EXACT, not approximate. At K2 = jnp.inf the locked half
    gets k_30 = k_32 = k_33 = 0, so state 3 is absorbing and expm returns
    row 3 = [0, 0, 0, 1] — the old hard lock, bit for bit. Rows 0-2 of the
    locked half are never read there, because a head can only bind an open
    (state 3) site and a hard-locked site can never leave it.

    The local field h is applied as exp(+h/2) on the three forward rates and
    exp(-h/2) on their reverses, leaving the one-way cycle-closing rate k_30 at
    its base value — see the section header above for why.

    The rate laws themselves are NOT written here: each comes from its
    tm_rate_XY function in rate_functions.py, evaluated on all 27 configurations
    at once. This function only decides which modifier each rate receives and
    assembles the results, exactly as xb_rate_matrix does for the crossbridge
    cycle.

    Args:
        ca_concentration: Calcium concentration (M)
        J_C: Coupling to Ca-open neighbours (kT)
        J_M: Coupling to crossbridge-bound neighbours (kT)
        params: DynamicParams with the tm_* rates and equilibrium constants

    Returns:
        Q_unique: (54, 4, 4), indexed by
            (n_2*9 + n_3*3 + n_closed) + 27*is_bound
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
    # — slowing it produces anti-cooperative cascade behavior (verified empirically),
    # which is why tm_rate_30 takes no modifier argument at all.
    # This is the original Tanner 2012 prescription (Ψ on r_{t,12} and r_{t,23}),
    # plus the Ca-binding step, applied Glauber-symmetrically.
    #
    # k_30 is broadcast to (27,) so every rate entering the Q builder has the
    # same shape, even though this one does not vary across configurations.
    k_01 = tm_rate_01(ca_concentration, k_01_base, forward_boost)
    k_12 = tm_rate_12(k_12_base, forward_boost)
    k_23 = tm_rate_23(k_23_base, forward_boost)
    k_30 = tm_rate_30(jnp.broadcast_to(k_30_base, (27,)))

    k_10 = tm_rate_10(k_01_base, Keq_01, backward_boost)
    k_21 = tm_rate_21(k_12_base, Keq_12, backward_boost)
    k_32 = tm_rate_32(k_23_base, Keq_23, backward_boost)

    # Stack the unlocked half (entries 0-26) on the locked one (27-53). Only the
    # two exits from state 3 differ; every other rate is shared, so this is one
    # concatenate rather than a second pass over the rate laws.
    #
    # 1/(1 + K2) is formed once and multiplied, not divided by per rate: at
    # K2 = jnp.inf that is a multiply by exactly 0.0, whereas dividing by inf
    # would be too but only by IEEE luck, and 0.0 is what makes row 3 of the
    # locked generator identically zero and the control path exact.
    lock = 1.0 / (1.0 + params.xb_tm_K2)
    k_30_both = jnp.concatenate([k_30, k_30 * lock])
    k_32_both = jnp.concatenate([k_32, k_32 * lock])

    def _tile(a):
        return jnp.concatenate([a, a])

    k_01, k_12, k_23 = _tile(k_01), _tile(k_12), _tile(k_23)
    k_10, k_21 = _tile(k_10), _tile(k_21)
    k_30, k_32 = k_30_both, k_32_both

    # Diagonals
    k_00 = -k_01
    k_11 = -(k_10 + k_12)
    k_22 = -(k_21 + k_23)
    k_33 = -(k_30 + k_32)

    Q_flat = _build_tm_Q_matrix_optimized(
        k_00, k_01, k_10, k_11, k_12,
        k_21, k_22, k_23, k_30, k_32, k_33,
    )  # (54, 4, 4)

    return Q_flat


def thin_transitions(state: 'State',
                     constants: 'DynamicParams',
                     topology: 'SarcTopology',
                     rng_key: jax.random.PRNGKey,
                     dt: float,
                     random_values: Optional[jnp.ndarray] = None,
                     tm_subpop=None) -> Tuple['State', jnp.ndarray, jnp.ndarray]:
    """Advance every tropomyosin site one timestep.

    Uses the symmetric Ising cooperativity described in the section header
    above. Nothing about a site's cooperative status is precomputed or stored:
    this counts each site's neighbour states itself from the current tm_states
    and the structural chain adjacency. That is why the kinetics phase needs no
    thin-filament force calculation ahead of it.

    Sequence: build 54 rate matrices, exponentiate them in one batch, count each
    site's neighbours and read whether a head is bound to get its configuration
    index, gather its probability vector, sample, then detach any head whose
    tropomyosin closed underneath it.

    LOCKED SITES — A FINITE LOCK, APPLIED TO RATES. A site with a crossbridge
    attached has its two exits from the open state 3 divided by (1 + xb_tm_K2)
    inside the generator: a bound head biases tropomyosin towards open, it does
    not pin it. McKillop & Geeves 1993 Fig. 1 — one bound S1 multiplies the
    open/closed ratio by (1 + K2) — is a statement about an EQUILIBRIUM RATIO,
    and the rate-side form is the one that reproduces it at any dt. See
    _compute_unique_tm_Q_matrices for the derivation, for what the previous
    post-hoc probability form got wrong (a nominal K2 = 79 ran as ~174 at
    dt = 1 ms), and for why the K2 = jnp.inf hard lock is still exact. The
    xb_tm_K2 block in core/params.py has the measured values and why some lock
    is necessary.

    "Whether a site is locked changes every timestep while the rate matrices do
    not" was the old justification for doing this after the exponential. It is
    not a reason: the bound/unbound axis is a second constant-size axis on the
    matrix set, not a per-site one, and 27 extra 4x4 expms against the 200-300
    already run on the thick-filament side is noise.

    DETACHMENT ON CLOSURE. Once the lock is finite, tropomyosin CAN close over
    a bound head, and that configuration has to mean something. Here the head
    detaches and its bound_to is cleared on both sides. xb_rate_01 gates
    attachment on permissiveness, so this model does not permit a head to bind
    unless tropomyosin is fully open; leaving ANY bound head — weak included —
    behind after closure would contradict the model's own binding rule.

    WHERE IT LANDS DEPENDS ON WHAT IT HAD ALREADY SPENT. A Loose head returns to
    state 0 DRX and owes nothing; a Tight_1 or Tight_2 head goes to state 4
    Free_2 and is charged one ATP. See the closure block itself for why, and for
    why the destination is written as a test on states 2 and 3 rather than as an
    "else".

    >>> THAT SELF-CONSISTENCY ARGUMENT IS THE WHOLE JUSTIFICATION. Detachment
    on closure is NOT a measured mechanism, and this docstring used to imply it
    was. The nearest support is Smith & Geeves 2003 Biophys J 84:3168, Abstract,
    verbatim: "Myosin is detached by the actin binding of TnI" — but that is
    (a) about TnI competing with myosin for actin through oppositely-directed
    kinks in a flexible chain, NOT about tropomyosin's azimuthal position
    closing over a bound head, and (b) a PREDICTION OF THEIR MODEL, whose own
    parameters are fitted, not an observation. It is a theoretical precedent for
    a regulatory protein displacing myosin from actin. Cite it as that or not at
    all. Nothing in .claude/papers/ measures this; part I of the same pair
    records no detachment rate of any kind.

    AND THE STRUCTURAL EVIDENCE OPPOSES THE WEAK-HEAD HALF — see the rejected
    alternative below, which is not merely an option not taken but a direct
    measurement pointing the other way.

    >>> VIBERT DOES NOT DISCRIMINATE BETWEEN THIS MODEL AND THE OLD ONE, so do
    not cite it as support for the change. It establishes that a closed
    tropomyosin and an attached head are MUTUALLY EXCLUSIVE — a symmetric
    geometric fact, satisfied equally by (a) the hard lock, where the excluded
    configuration never arises, and (b) this code, where it arises and is
    resolved by detaching. Three static structures cannot say which mechanism
    prevents a state, only that the state does not exist.
    THE ENTIRE CASE FOR THE FINITE LOCK IS McKILLOP'S TABLE 1: K2 is measured
    at 241 / 79 / 18, all finite. The hard lock is the unmeasured K2 -> inf
    limit. That one fact is what makes the old model wrong; Vibert then applies
    only CONDITIONALLY — given that closure happens, the head cannot stay.
    Ordering matters when this is written up: lead with K2, not with sterics.

    REJECTED ALTERNATIVE, recorded so it is not mistaken for what the code does.
    Vibert et al. 1997 J Mol Biol 266:8 (negative-stain 3D EM), Results p.13,
    verbatim: in the off-state the residues taking part in "strong,
    stereospecific myosin-binding ... are seen to lie beneath tropomyosin and
    are therefore completely blocked ..., while residues 1 to 4 and 92 to 95,
    thought to be involved in weak binding, remain exposed" — which would spare
    weak heads and demote strong ones instead of detaching them. Taken seriously that implies
    weak binding is tropomyosin-INDEPENDENT: r01 should not be gated either,
    tropomyosin should gate 1 -> 2, and closure should demote. That is a
    different model and is NOT what this function implements.

    ATP: A WEAK TEAR IS FREE, A STRONG TEAR COSTS ONE, AND IT IS BOOKED IN
    metrics_fn FROM THE `torn` MASK THIS FUNCTION RETURNS — as
    `closure_detach_free` and `closure_detach_atp`, and added into both
    `atp_consumed` and `atp_expected`. It needs no expectation: the tear is
    fully observed and the charge is deterministic given it, unlike the 3 -> 4
    term, where the sampler gives only endpoints and multi-hop traversals are
    unobservable. Neither is `xb_give_up` this route: that key is the 2 -> 1
    crossing count, i.e. the non-ATP strain-gated reversal, and a head it counts
    is still weakly bound afterwards.

    This paragraph used to claim "atp_expected is computed from the Q matrix
    (P_abs_all[:, 3, 4]) so nothing is miscounted". That was wrong three ways:
    the index named a metric that no longer exists, the metric was itself off by
    0.06-0.46% (it was read from a stale state), and under the rule above a
    strong tear is no longer a non-ATP detachment at all.

    NOT QUITE EQUIVALENT TO THE S129 PROTOTYPE, and the residue is measured. The
    patch stack that produced the validated cohort detached only STRONG heads
    here and compensated by ungating r10, whose value at a covered site is then
    41.4 /ms — argued to be the same thing at dt = 1 ms, since P(detach within
    one step) rounds to 1 and this function runs before thick_transitions. It is
    nearly the same thing, not exactly. Measured over 200 steps, cardiac 2x2,
    pCa 6.2 (2026-09-01): of 24 weak heads caught by closure, the patch leaves 1
    still bound after thick_transitions (r12 competes with r10 for the same
    head), while detaching here returns the head to DRX one sub-step earlier, so
    1 of them re-binds a different open site within the same step. Over a
    150-step trace the two arms are bit-identical for 107 steps and then differ
    by one bound head for three steps: 0.072% of peak axial force. Both are
    ~1-in-24 events on a population that is itself rare. This form is the root
    cause and the patch's was the workaround; the difference is recorded so that
    nobody re-derives "exactly equivalent" from the algebra alone.

    >>> 0.072% IS SCOPED TO THOSE CONDITIONS AND IS NOT A BOUND. At 8x8, pCa 4.0,
    5000 ms the WT force-pCa plateau differs by **+11.2%** between the patch
    stack and this code (53.28 -> 59.24 kPa, measured 2026-09-08). frac
    0.75-vs-0.78 accounts for -2.1% of that, wrong sign; the lock formula and the
    rate matrix are byte-identical; the remaining candidates are this detach rule
    and the patch's axial/radial mismatch. UNRESOLVED, deliberately deferred. Do
    not quote 0.072% as the patch-core difference at other conditions.

    Args:
        state: Current State (reads tm_states and bound_to)
        constants: DynamicParams with pCa, tm_J_C, tm_J_M and the tm_* rates
        topology: SarcTopology, for chain neighbours and the eye_4 identity
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
        P_flat: the distinct probability matrices used, for validation —
            (54, 4, 4), or (K, 54, 4, 4) for an explicit mixture. The first 27
            are the unlocked half, the last 27 the crossbridge-locked one.
        closure_detached: (n_thick, n_crowns, n_xb_per_crown) bool, True for
            each head this call tore off because its tropomyosin closed. Not
            recoverable downstream — a torn head lands in state 0 or state 4,
            both of which an ordinary head also reaches — so it is returned
            rather than re-derived, and travels to metrics_fn inside the
            KineticsTrace. metrics_fn splits it into `closure_detach_free` (free)
            and `closure_detach_atp` (one ATP each) by the state the head is in
            after this call.
    """
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

    # Per-filament neighbor counts (each function call processes one strand)
    n_2, n_3, n_c = jax.vmap(count_neighbor_states_split)(tm_states, tm_prev_neighbor, tm_next_neighbor)
    # all shape (n_thin, n_sites), int32 capped at 2

    # The bound/unbound axis selects the locked half of the matrix set. It is
    # read here, once, from the state at the START of the step — the same
    # operator-splitting approximation the rest of the kinetics phase makes.
    config_idx = ((n_2 * 9 + n_3 * 3 + n_c).reshape(-1)
                  + 27 * is_bound.reshape(-1).astype(jnp.int32))  # (n_sites_total,)

    if tm_subpop is None:
        # Build the 54 unique Q matrices, then expm in one batch
        Q_flat = _compute_unique_tm_Q_matrices(ca_conc, J_C, J_M, constants)  # (54, 4, 4)
        P_flat = expm_pade6_batch(Q_flat * dt, identity=eye_4)  # (54, 4, 4)
        P_indexed = P_flat[config_idx]                          # (n_sites_total, 4, 4)
    else:
        mode, constants_k, extra = tm_subpop
        # Per-population 54-matrix sets. The couplings are per-population too,
        # so a subpopulation may scale tm_J_C / tm_J_M as well as the rates —
        # and, now that the lock is a rate, its own xb_tm_K2. The post-hoc form
        # could not do that: it read the base constants' K2 for every
        # population.
        Q_k = [_compute_unique_tm_Q_matrices(ca_conc, ck.tm_J_C, ck.tm_J_M, ck)
               for ck in constants_k]  # each (54, 4, 4)
        if mode == 'mean_field':
            fractions = extra  # (K,)
            Q_eff = sum(fractions[k] * Q_k[k] for k in range(len(constants_k)))
            P_flat = expm_pade6_batch(Q_eff * dt, identity=eye_4)  # (54, 4, 4)
            P_indexed = P_flat[config_idx]
        else:  # explicit mixture: per-site label select
            labels = extra  # (n_sites_total,) INT in [0, K)
            Q_stack = jnp.stack(Q_k)  # (K, 54, 4, 4)
            Kp = Q_stack.shape[0]
            P_flat = expm_pade6_batch(
                Q_stack.reshape(Kp * 54, 4, 4) * dt, identity=eye_4).reshape(Kp, 54, 4, 4)
            P_indexed = P_flat[labels, config_idx]  # (n_sites_total, 4, 4)

    tm_states_flat = tm_states.reshape(-1).astype(jnp.int32)

    # The lock is already in these rows: config_idx selected the locked half of
    # the matrix set for every bound site. There is no post-hoc rescaling step.
    prob_vectors = jax.vmap(lambda P, s: P[s])(P_indexed, tm_states_flat)

    if random_values is None:
        rng_key, subkey = jax.random.split(rng_key)
        random_values = jax.random.uniform(subkey, shape=(n_sites_total,))

    cum_probs = jnp.cumsum(prob_vectors, axis=1)
    new_states = jnp.argmax(random_values[:, None] < cum_probs, axis=1)

    new_tm_states = new_states.reshape(n_thin, n_sites).astype(jnp.int8)

    # ---------------------------------------------------------------- closure
    # Tropomyosin that closed over a bound head detaches it. See DETACHMENT ON
    # CLOSURE in the docstring: every bound head, weak included, because
    # xb_rate_01 lets nothing bind unless the site is fully open.
    # Unreachable while xb_tm_K2 is jnp.inf — the lock and this trigger read the
    # same bound_to, so a locked site can never leave state 3 — which is what
    # makes the control path those six parameter values and no flag.
    left = ((tm_states == 3) & (new_tm_states != 3) & (bound_to >= 0)).reshape(-1)

    xb_shape = state.thick.xb_states.shape
    xb_flat = state.thick.xb_states.reshape(-1)
    n_xb = xb_flat.shape[0]

    # thin.bound_to holds a FLAT crossbridge index (assigned from
    # jnp.arange(n_xb_total)), so scatter the per-site flag onto heads by that
    # index. Scatter-MAX, not add: unbound and unaffected sites clip to head 0
    # and contribute 0, which cannot raise the flag of a head that did not move.
    idx = jnp.clip(bound_to.reshape(-1), 0, n_xb - 1)
    hit = jnp.zeros(n_xb, jnp.int32).at[idx].max(left.astype(jnp.int32)) == 1

    # WHERE A TORN HEAD LANDS. A Loose (state 1) head is still primed —
    # A.M.ADP.Pi, nothing spent — so it goes to state 0 DRX and owes nothing. A
    # Tight_1 or Tight_2 head has already released phosphate and swung its
    # lever; putting it in DRX (= M.ADP.Pi) would regenerate that phosphate and
    # re-cock the lever for free, which is the same leak xb_rate_43 is held at
    # exactly zero to prevent. Those go to state 4 Free_2 — detached,
    # post-stroke, ATP bound — and are charged one ATP in metrics_fn. Measured:
    # this books 7.0 of the 8.0 percentage points of ATP the model spent and
    # never counted (closure_spy.py); the remaining ~1.0% is the give-up route,
    # a genuine refund.
    #
    # >>> WRITTEN AS `(s == 2) | (s == 3) -> 4, else 0`, NEVER AS
    #     `weak -> 0, else -> 4`. `hit` is raised from thin.bound_to, and if
    #     that record were ever stale — a site naming a head whose own
    #     xb_bound_to is already -1 — the "else" form would push an already
    #     detached head into Free_2 and charge it an ATP it never spent. Testing
    #     the states that actually owe one cannot do that. The two-sided binding
    #     invariant that makes such a record impossible is asserted by
    #     local_projects/regression/binding_invariant.py.
    #
    # >>> THIS IS A DYNAMICS CHANGE, NOT ONLY ACCOUNTING. A head in Free_2 must
    #     wait for xb_rate_40 before it can rebind, and xb_r40 = 0.1 /ms is a
    #     ~10 ms dwell — so `n_xb_free_2` rises and relaxation slows relative to
    #     landing in DRX. params.py flags xb_r40 [G]: unsourced, inherited.
    torn_dest = jnp.where((xb_flat == 2) | (xb_flat == 3), jnp.int8(4), jnp.int8(0))
    new_xb_states = jnp.where(hit, torn_dest, xb_flat).reshape(xb_shape)
    # Clear bound_to on BOTH sides. If the site kept its record it would stay
    # flagged occupied and nothing could ever rebind it.
    new_xb_bound_to = jnp.where(
        hit, jnp.int32(-1), state.thick.xb_bound_to.reshape(-1)).reshape(xb_shape)
    new_bound_to = jnp.where(left.reshape(bound_to.shape), jnp.int32(-1), bound_to)

    new_thin = state.thin._replace(tm_states=new_tm_states,
                                   bound_to=new_bound_to)
    new_thick = state.thick._replace(xb_states=new_xb_states,
                                     xb_bound_to=new_xb_bound_to)
    new_state = state._replace(thin=new_thin, thick=new_thick)

    return new_state, P_flat, hit.reshape(xb_shape)


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
        spring_constants: (n_xb, 12) two-spring parameters per head, one block
            per bound configuration —
            [:, 0:4]  g_k_weak, g_rest_weak, c_k_weak, c_rest_weak
            [:, 4:8]  g_k_strong, g_rest_strong, c_k_strong, c_rest_strong
            [:, 8:12] g_k_tight_1, g_rest_tight_1, c_k_tight_1, c_rest_tight_1
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

    # Convert to polar coordinates — the same primitive the force path uses, so
    # the two cannot disagree about what r and theta mean.
    x = xb_distances[:, 0]
    y = xb_distances[:, 1]
    r, theta, cos_theta, sin_theta = xb_geometry(x, y)

    # Get spring constants
    g_k_weak = spring_constants[:, 0]
    g_r_weak = spring_constants[:, 1]
    c_k_weak = spring_constants[:, 2]
    c_r_weak = spring_constants[:, 3]
    g_k_strong = spring_constants[:, 4]
    g_r_strong = spring_constants[:, 5]
    c_k_strong = spring_constants[:, 6]
    c_r_strong = spring_constants[:, 7]
    # Tight_1's own spring configuration (the split stroke, S129). Equal to the
    # Tight_2 block in stiffness and distinct only in rest position at the
    # shipped defaults, but read separately so the chemistry can never silently
    # fall back to Tight_2's geometry.
    g_k_t1 = spring_constants[:, 8]
    g_r_t1 = spring_constants[:, 9]
    c_k_t1 = spring_constants[:, 10]
    c_r_t1 = spring_constants[:, 11]

    # Thermal energy kT, converted from J to pN*nm (1 J = 1e21 pN*nm).
    # Boltzmann's constant is written to 5 significant figures here (the CODATA
    # value is 1.380649e-23 J/K); the 4e-5 relative difference is far below the
    # uncertainty in any rate constant it multiplies.
    k_t = 1.3810e-23 * (temp_celsius + 273.15) * 1e21  # pN*nm

    # Elastic energy in each of the three bound configurations, from the SAME
    # potential the force law integrates (forces.xb_elastic_energy). kT is
    # applied here rather than inside it, so mechanics keeps pN*nm and
    # chemistry gets kT without either owning the conversion.
    #
    # THE FORCE LAW AND THE ENERGETICS MUST MOVE TOGETHER. U_tight_1, r21, r32,
    # r12's E_diff and now r23's barrier all derive from state 2's elastic
    # energy, and the load-dependent rates must feel the force in the
    # configuration of the state they leave. Splitting only forces.py would
    # leave the chemistry describing a different molecule from the mechanics —
    # which is exactly how the rate path came to sum two orthogonal force
    # components; see the primitives header in forces.py.
    E_weak = xb_elastic_energy(r, theta, g_k_weak, g_r_weak, c_k_weak, c_r_weak) / k_t
    E_strong = xb_elastic_energy(r, theta, g_k_strong, g_r_strong, c_k_strong, c_r_strong) / k_t
    E_tight1 = xb_elastic_energy(r, theta, g_k_t1, g_r_t1, c_k_t1, c_r_t1) / k_t

    # Energy difference driving the weak->strong isomerization.
    E_diff = E_weak - E_tight1

    # Chemical free energy of each state (kT). Adding the elastic energy to the
    # bound states below is what couples mechanics to chemistry: a strained head
    # sits higher in free energy, so its reverse rates -- which are derived from
    # these totals -- rise accordingly.
    U_DRX = params.xb_U_DRX
    U_loose_base = params.xb_U_loose
    U_tight_1_base = params.xb_U_tight_1
    U_tight_2_base = params.xb_U_tight_2

    U_loose = U_loose_base + E_weak
    U_tight_1 = U_tight_1_base + E_tight1
    U_tight_2 = U_tight_2_base + E_strong

    # THE LOAD FED TO THE BELL RATES IS THE AXIAL FORCE, obtained by rotating the
    # head's two polar spring forces into the filament frame exactly as the force
    # law does. It is not F_r + F_theta.
    #
    # It read `F_r + F_theta` from the initial commit until 2026-09-09 — both
    # trig projections dropped and the converter term's sign flipped relative to
    # the axial law (the S50 sign, fixed in forces.py and never here). That sum
    # is not a force in any frame: F_r and F_theta are orthogonal components.
    # The error tracked LATTICE SPACING rather than strain, because
    # xb_g_rest_tight_1 = 17.231 nm exceeds d and r = sqrt(x^2 + d^2) >= d, so a
    # head splayed straight across the lattice at x ~ 0 carried F_r ~ -16 pN
    # while doing exactly zero axial work — and fed that phantom compression to
    # an exponential. See the primitives header in forces.py for the invariant
    # that now makes this class of error unwriteable.
    F_r_strong, F_theta_strong = xb_polar_forces(
        r, theta, g_k_strong, g_r_strong, c_k_strong, c_r_strong)
    f_strong = polar_to_filament(F_r_strong, F_theta_strong, cos_theta, sin_theta)[0]

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

    # 2 Tight_1 <-> 3 Tight_2 : the AM.ADP isomerization (strain-dependent).
    # LABEL CORRECTED 2026-09-01: this line used to read "the working stroke",
    # a survivor of the 1-indexed scheme that b63a724 missed. The stroke is
    # split across 1->2 (the larger part) and this step; see
    # rate_functions.xb_rate_23.
    #
    # Smith & Geeves' elastic-energy form, not a Bell distance (2026-09-09):
    # the barrier is the elastic energy the step must climb, floored at zero, so
    # the uphill direction is slowed and the downhill one is strain-FREE rather
    # than strain-accelerated. r32 is left as the plain detailed-balance line
    # below, which is not an oversight: substituting r23 into it gives
    # A23*exp(dU_base) when 2->3 is uphill and A23*exp(dU_base)*exp(dE) when it
    # is downhill — i.e. the strain factor lands on whichever direction climbs,
    # in both directions, reproducing S&G Eq. 2b exactly and preserving K_23.
    r23 = xb_rate_23(r23_coeff, E_tight1, E_strong)
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
        constants.xb_g_k_weak,    constants.xb_g_rest_weak,
        constants.xb_c_k_weak,    constants.xb_c_rest_weak,
        constants.xb_g_k_strong,  constants.xb_g_rest_strong,
        constants.xb_c_k_strong,  constants.xb_c_rest_strong,
        constants.xb_g_k_tight_1, constants.xb_g_rest_tight_1,
        constants.xb_c_k_tight_1, constants.xb_c_rest_tight_1,
    ])
    springs_grid = jnp.broadcast_to(spring_vec, (n_bins, 12))  # (n_bins, 12)

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


def _xb_Q_resolved(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    xb_subpop=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Effective binned rate matrices, and how to gather them per head.

    Everything subpopulation-related lives here, and nothing else does. No
    matrix exponentials are taken — that is deliberate, because it is the part
    every caller must share. Rates come from _build_xb_Q_bins in every branch,
    so the sampling path and the metrics path cannot disagree about the physics
    of a given step no matter how they diverge afterwards.

    UNTIL 2026-09-11 THIS DOCSTRING SAID the two callers "need exponentials of
    DIFFERENT generators" — the sampler wanted plain Q, the metrics wanted Q with
    rows 0 and 4 zeroed (the absorbing construction, deleted with
    xb_exit_probabilities). They now need the SAME generator: exp(Q*dt) and its
    occupancy-time integral come out of one call. That is what makes a single
    exponential per step possible.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        xb_subpop: None for the standard single-population path, or a tuple
            (mode, constants_k, extra) for subpopulations. constants_k is a
            length-K list of DynamicParams (population 0 = WT). For
            mode=='mean_field', extra is a (K,) fractions vector and the K
            binned generators are weight-summed into one effective generator
            (Q_eff = Σ f_k Q_k) — averaging the GENERATORS before exponentiating
            is the mathematically correct blend. For mode=='explicit', extra is
            a (n_xb_total,) INT label array and the populations stay stacked so
            each head can select its own afterwards. The gather key is shared
            across populations (it depends on geometry/permissiveness, not
            rates).

    Returns:
        Q_bins: (n_cells, 6, 6), or (K, n_cells, 6, 6) for mode=='explicit'
        key:    (n_xb_total,) each head's index into the bin grid
        labels: None, or (n_xb_total,) population index for mode=='explicit'
    """
    if xb_subpop is None:
        Q_bins, key = _build_xb_Q_bins(state, constants, topology)
        return Q_bins, key, None

    mode, constants_k, extra = xb_subpop
    built = [_build_xb_Q_bins(state, ck, topology) for ck in constants_k]
    key = built[0][1]  # shared across populations (geometry/permissiveness only)

    if mode == 'mean_field':
        fractions = extra  # (K,)
        Q_eff = sum(fractions[k] * built[k][0] for k in range(len(constants_k)))
        return Q_eff, key, None

    # explicit mixture: keep populations stacked, select per head after the exp
    return jnp.stack([b[0] for b in built]), key, extra


def _gather_per_xb(X: jnp.ndarray, key: jnp.ndarray,
                   labels: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Bin-grid matrices -> per-head matrices, with or without subpopulations."""
    return X[key] if labels is None else X[labels, key]


def _expm_bins(Q: jnp.ndarray, dt: float, eye_6: jnp.ndarray,
               with_integral: bool = False):
    """One batched matrix exponential, shape-agnostic in the leading dims.

    Accepts either (n_cells, 6, 6) or the stacked (K, n_cells, 6, 6) of an
    explicit subpopulation run, and returns the same shape. Flattening here
    rather than at each call site is what keeps the subpopulation modes from
    each needing their own reshape bookkeeping.

    With with_integral, returns (P, G) in that same shape; see
    matrix_exponential_batch().
    """
    shape = Q.shape
    flat = Q.reshape(-1, 6, 6)
    if not with_integral:
        return matrix_exponential_batch(flat, dt, identity=eye_6).reshape(shape)
    P, G = matrix_exponential_batch(flat, dt, identity=eye_6, with_integral=True)
    return P.reshape(shape), G.reshape(shape)


def xb_step_probabilities(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    dt: float,
    xb_subpop=None,
) -> jnp.ndarray:
    """Per-crossbridge transition probabilities over dt — what the sampler draws from.

    P[i, j] is the probability that a head in state i is in state j after dt.
    This is the ONLY thing thick_transitions needs, and taking a single
    exponential of the plain generator is the whole job.

    Evaluates 2 * n_xb_bins matrix exponentials instead of one per head — at a
    4x4 lattice roughly a sixfold reduction, on the grid rather than per head,
    so the cost does not grow with lattice size.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        dt: Timestep length (ms)
        xb_subpop: see _xb_Q_resolved

    Returns:
        P_all: (n_xb_total, 6, 6) transition probability matrices per crossbridge
    """
    Q_bins, key, labels = _xb_Q_resolved(state, constants, topology, xb_subpop)
    P_bins = _expm_bins(Q_bins, dt, topology.eye_6)
    return _gather_per_xb(P_bins, key, labels)


#: Ordered pairs whose expected crossing counts the ATP metrics read.
#: (1,0) free detachment, (1,2) Pi release, (2,1) the strain give-up reversal,
#: (3,4) ATP-consuming detachment, (0,4)/(4,0) the reversible recovery stroke.
XB_METRIC_PAIRS = ((1, 0), (1, 2), (2, 1), (3, 4), (0, 4), (4, 0))


def xb_expected_crossings(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    dt: float,
    xb_subpop=None,
    pairs=XB_METRIC_PAIRS,
) -> jnp.ndarray:
    """How many times each head crosses each edge during the step. Metrics only.

    The companion to xb_step_probabilities(): that one answers "what state is
    this head in after dt", this one answers "how many times did it make the
    i -> j transition on the way". For a generator held constant over the step —
    the same assumption the sampler makes —

        E[# of i -> j crossings in dt | started in s] = q_ij * G[s, i]
        G = INT_0^dt exp(Qt) dt

    and G comes free alongside P from the same scaling-and-squaring (see
    expm_pade6_batch, with_integral).

    WHY AN EDGE COUNT AND NOT A REACHABILITY PROBABILITY. Until 2026-09-11 the
    ATP metrics asked "did this head VISIT Free_2 during the step", read from a
    generator with rows 0 and 4 made absorbing. That is P(visit >= 1), which is a
    LOWER BOUND on E(visits), so it was biased low whenever a head could cycle
    twice inside one step: -0.07% cardiac, -1.2% skeletal at dt = 1 ms, measured
    by two independent routes (S137 [F], S138 [E]). It also needed the absorbing
    construction and its row mask purely to stop "reached state 4 from state 0"
    counting r04, the reverse recovery stroke, as an ATP. Counting the EDGE
    3 -> 4 cannot make that mistake, so the mask becomes unnecessary rather than
    merely wrong, and every head can be read — including one that runs
    0 -> 1 -> 2 -> 3 -> 4 inside a single step, which really does spend an ATP.

    Reads each head's OWN start state from `state`, so pass the mid state from
    KineticsTrace (the sarcomere as thick_transitions found it), not the state at
    the top of the step. Built on the same _xb_Q_resolved / _gather_per_xb pair
    as xb_step_probabilities, so subpopulation handling stays in the one place
    that owns it.

    Args:
        state: State whose thick.xb_states give each head's starting state
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        dt: Timestep length (ms)
        xb_subpop: see _xb_Q_resolved
        pairs: ordered (i, j) edges to count; defaults to XB_METRIC_PAIRS

    Returns:
        N: (len(pairs), n_xb_total) expected crossing counts, per head, over dt
    """
    Q_bins, key, labels = _xb_Q_resolved(state, constants, topology, xb_subpop)
    _P_bins, G_bins = _expm_bins(Q_bins, dt, topology.eye_6, with_integral=True)

    s = state.thick.xb_states.reshape(-1).astype(jnp.int32)
    counts = []
    for (i, j) in pairs:
        # Expected time in state i, by starting state, times the i -> j rate.
        # Ellipsis indexing covers the plain and the stacked (subpopulation)
        # layouts alike, so neither mode needs its own branch.
        per_cell = G_bins[..., :, i] * Q_bins[..., i, j][..., None]
        per_xb = _gather_per_xb(per_cell, key, labels)          # (n_xb_total, 6)
        counts.append(jnp.take_along_axis(per_xb, s[:, None], axis=1)[:, 0])
    return jnp.stack(counts)


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
    (see thin_transitions) and what lets forces.py know where to apply
    crossbridge force.

    Rates come from xb_step_probabilities(), which evaluates them on a distance
    grid rather than per head; see there and in _build_xb_Q_bins for how that
    works and what it costs in accuracy.

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
            _xb_Q_resolved() for the tuple contract and core/subpopulation.py
            for how these are built.

    Returns:
        new_state: State with updated xb_states, xb_bound_to, and thin bound_to
    """
    # Get current xb states
    xb_states = state.thick.xb_states  # (n_thick, n_crowns, n_xb_per_crown)
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape

    # Flatten for processing
    xb_states_flat = xb_states.reshape(-1)  # (n_thick * n_crowns * n_xb_per_crown,)
    n_xb_total = xb_states_flat.shape[0]

    # Per-XB transition probabilities via shared helper (subpop-aware)
    P_all = xb_step_probabilities(
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
        # ====================================================================
        # THE TWO BINDING RECORDS MUST AGREE, AND SCATTERS ARE WHERE THEY STOP
        # ====================================================================
        # An attachment is recorded twice — `xb_bound_to[h]` names the site,
        # `thin.bound_to[t, i]` names the head — and nothing reconciles them
        # afterwards. Both writes below therefore have to be collision-free by
        # construction, because `.at[].set()` with duplicate indices picks a
        # winner nondeterministically. Until 2026-09-10 neither was, and both
        # failure modes were live: see local_projects/regression/
        # binding_invariant.py, which asserts the two-sided property and
        # measured 115,846 violations over 400 steps before this rewrite.
        n_sites_total = n_thin * n_sites
        xb_indices_arr = jnp.arange(n_xb_total)
        site_flat = thin_indices * n_sites + site_indices

        nearest_site_occupied = thin_bound_to_flat[site_flat] >= 0
        can_bind = is_binding & (permissiveness > 0.5) & (~nearest_site_occupied)

        # ARBITRATION. `can_bind` is read against the occupancy at the START of
        # the step, so two heads whose nearest site is the same free site BOTH
        # pass it — and exactly one of them may have it. Scatter-MIN over head
        # index picks the lowest, deterministically. The old code scattered the
        # head index with `.at[].set()` and reverted only heads whose site was
        # occupied at the start of the step, so the loser of a same-step
        # collision kept a bound state and an `xb_bound_to` pointing at a site
        # another head owned: two heads on one site, both exerting force.
        # Non-binders contribute the `n_xb_total` sentinel, which no real head
        # index can reach, so they can never win and never touch the array —
        # which is why the write-back of the existing value is gone.
        winner = jnp.full(n_sites_total, n_xb_total, jnp.int32).at[site_flat].min(
            jnp.where(can_bind, xb_indices_arr, n_xb_total))
        won = can_bind & (winner[site_flat] == xb_indices_arr)

        new_xb_bound_to_flat = jnp.where(
            won,
            xb_nearest_bs_flat,
            jnp.where(
                is_unbinding,
                -1,
                xb_bound_to_flat
            )
        )

        # STEP 1: clear the sites of heads that unbound.
        # Scatter-MAX of a FLAG, not gather-modify-scatter. Every UNBOUND head
        # clips to site 0 of its thin filament, so under the old
        # `.at[].set()` all of them wrote the stale value back at that one
        # index and could clobber the -1 written by a head genuinely unbinding
        # from site 0 — leaving the site naming a head whose own record was
        # already cleared. A flag cannot be clobbered that way: a head
        # contributing 0 cannot lower a raised flag. Same idiom as the closure
        # tear in thin_transitions, and one scatter rather than a gather and a
        # scatter.
        really_unbinding = is_unbinding & (xb_bound_to_flat >= 0)
        old_thin_indices = topology.xb_to_thin_id
        old_site_flat = (old_thin_indices * n_sites
                         + jnp.clip(xb_bound_to_flat, 0, n_sites - 1))
        clear_site = jnp.zeros(n_sites_total, jnp.int32).at[old_site_flat].max(
            really_unbinding.astype(jnp.int32)) == 1
        new_thin_bound_to_flat = jnp.where(clear_site, -1, thin_bound_to_flat)

        # STEP 2: record the winners. `winner` is already a per-SITE array, so
        # this is an elementwise select rather than a second scatter, and the
        # two writes cannot conflict: a site occupied at the start of the step
        # fails `can_bind` for every head, so no site is both cleared and won.
        new_thin_bound_to_flat = jnp.where(winner < n_xb_total, winner,
                                           new_thin_bound_to_flat)

        new_xb_bound_to = new_xb_bound_to_flat.reshape(n_thick, n_crowns, n_xb_per_crown)
        new_thin_bound_to = new_thin_bound_to_flat.reshape(n_thin, n_sites)

        # Binding failed — the site was taken before the step, or another head
        # won it during the step. One line now covers both; the old form saw
        # only the first.
        #
        # >>> THIS IS A FLUX THAT IS NOT IN Q, AND IT IS NOT SMALL. A head sent
        #     back here did not follow its own generator: it is placed in DRX
        #     regardless of which bound state it sampled. Measured 2026-09-11,
        #     8x8, z 1100, dt = 1 ms, as a fraction of all heads that sampled a
        #     bound state from 0/4/5: cardiac 4.8% (pCa 4.5) / 1.2% (pCa 6.2),
        #     skeletal 7.1% / 5.3%. The realised endpoint histogram therefore
        #     departs from xb_step_probabilities by ~45 heads per step at
        #     cardiac pCa 4.5 — all of it moved from Loose into DRX.
        #     ANY METRIC THAT READS P OR Q AS THE TRUTH ABOUT THE STEP must say
        #     why that gap does not reach it.
        #     >>> UNTIL 2026-09-11 THE ANSWER HERE WAS STRUCTURAL AND IT NO
        #         LONGER IS. atp_expected masked on mid states 1-3, which are
        #         ALREADY BOUND, so `is_binding` was false for every head it
        #         read. The exact estimator drops that mask and reads every
        #         head, states 0/4/5 included — which is correct for the ATP
        #         (a head really can run 0 -> 1 -> 2 -> 3 -> 4 in one step) but
        #         removes the immunity along with it. The answer is now a
        #         measured BOUND, not a structural argument.
        #         8x8, dt = 1 ms: heads in mid states 0/4/5 contribute
        #         0.028% (cardiac 4.5) / 0.024% (6.2) and 0.567% (skeletal 4.5)
        #         / 0.593% (6.2) of total N34. Only the reverted subset of those
        #         is misattributed, and reverts are 4.8-7.1% of binding
        #         attempts, so the exposure is at most ~0.002% cardiac and
        #         ~0.04% skeletal — an order below the noise floor on the
        #         ledger, and an order below the bias the exact estimator
        #         removed. It is a bound, not a measurement of the effect
        #         itself; a tighter figure needs `won` exported.
        #     The residue this leaves in the ATP ledger is bounded and measured
        #     in local_projects/tension_cost/atp_balance_spy.py.
        new_states_flat = jnp.where(is_binding & (~won), 0, new_states_flat)
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

