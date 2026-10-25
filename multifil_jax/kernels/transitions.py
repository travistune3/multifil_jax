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

  Tropomyosin (thin_transitions): a unit's rates depend only on how many of its
    two chain neighbours are in each state, on how many STRONG crossbridges are
    bound on it (0..tm_max_heads). That is 27 x (tm_max_heads + 1)
    combinations, so that many
    matrices serve every unit on every filament.

  Crossbridges (thick_transitions): rates depend on the head's axial distance to
    its target (continuous) and on the target unit's block: blocked, closed,
    closed but screened by thin-thin overlap, open, open but screened. The
    continuous axis is discretized into n_xb_bins bins, giving 1 + 4 * n_xb_bins
    matrices regardless of lattice size (blocked needs only one).
    Heads then gather the row for their own bin. Bin resolution is a genuine
    accuracy/cost tradeoff, controlled by StaticParams.n_xb_bins / xb_bin_lo /
    xb_bin_hi.

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
from typing import Tuple, Dict, NamedTuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams

# Import rate functions and energy calculations
from .rate_functions import (
    tm_ca_affinity_ratios, tm_rate_ca_on, tm_rate_ca_off, tm_rate_forward, tm_rate_reverse,
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
from multifil_jax.core.state import monomer_axial


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

def _matmul(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Batched X @ Y over the last two axes, as an elementwise multiply-sum.

    For 6x6 matrices this is what XLA fuses into one elementwise loop, in plain
    float32. An einsum is lowered to a batched GEMM instead, whose 128x128 tile
    is almost entirely padding at n = 6, and which may run in TF32.
    """
    return jnp.sum(X[..., :, :, None] * Y[..., None, :, :], axis=-2)


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
    buy full fusion across the batch. (The observed max s is 16 at dt = 1 ms and
    12 at dt = 0.1; a loop to the batch's max s instead runs slower, not faster.)

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

    EVERY PRODUCT IS AN ELEMENTWISE MULTIPLY-SUM (_matmul), NOT AN einsum. The
    batch is thousands of 6x6 matrices. As an einsum, XLA may hand a product to a
    batched cuBLAS GEMM, padding each 6x6 into a 128x128 tile, and in TF32. TF32's
    10-bit mantissa compounds over the 16 squarings the stiffest bins need (r12
    reaches 1e4 /ms, so ||Q*dt|| reaches 2e4 on a matrix whose entries span 1e-20
    to 1). The multiply-sum fuses into elementwise loops in plain float32, so
    with_integral doubles E and J together in ONE loop.

    ACCURACY against float64 on settled Q_bins (8x8, dt = 1 ms): phi1*dt is good
    to ~3e-4, slightly better than the probabilities it ships beside (~7e-4), and
    every exported flux is good to < 2e-5 of its own total.

    DO NOT REPLACE THIS WITH expm([[A, I], [0, 0]]). That block matrix's top rows
    sum to 1 + dt, so step 8's row normalization wrecks it, and it costs 3.5x a
    6x6 expm (~+20% of a batched timestep).

    Args:
        A_batch: (batch, n, n) matrices to exponentiate (typically Q * dt)
        identity: (n, n) identity from SarcTopology (eye_6). Passed in
            rather than built with jnp.eye(n) inside, which would make XLA
            materialize a fresh copy per batch element under vmap.
        with_integral: also return phi1(A). Static — it changes the returned
            pytree, so flipping it recompiles. The tropomyosin call sites
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
    A2 = _matmul(A_scaled, A_scaled)
    A4 = _matmul(A2, A2)
    A6 = _matmul(A4, A2)

    # Step 5: Compute U and V for Pade approximant
    b = PADE6_B

    # U = A * (b1*I + b3*A2 + b5*A4)
    inner = b[1]*I + b[3]*A2 + b[5]*A4
    U = _matmul(A_scaled, inner)

    # V = b0*I + b2*A2 + b4*A4 + b6*A6
    V = b[0]*I + b[2]*A2 + b[4]*A4 + b[6]*A6

    # Step 6: Solve (V - U) @ R = (V + U); `pade` is exp(A / 2^s)
    pade = jnp.linalg.solve(V - U, V + U)

    def _normalize(M):
        # Step 8: Row normalization (fix float32 drift), and guard against NaN
        M = M / jnp.sum(M, axis=2, keepdims=True)
        return jnp.where(jnp.isnan(M), 1.0/n, M)

    if not with_integral:
        # Step 7: Square s times using fori_loop for XLA fusion
        # Handles ||A|| up to 2^18 = 262144; 2 extra no-op iters for typical norms
        def _square_step(i, result):
            return jnp.where((i < s)[:, None, None], _matmul(result, result), result)

        return _normalize(jax.lax.fori_loop(0, 18, _square_step, pade))

    # Step 7 with the occupancy-time integral phi1(A): E and J doubled together,
    # one level at a time (see the docstring).
    phi_even = I + A2/6.0 + A4/120.0 + A6/5040.0
    phi_odd_inner = I/2.0 + A2/24.0 + A4/720.0
    J = phi_even + _matmul(A_scaled, phi_odd_inner)

    def _double_step(i, carry):
        E, Jc = carry
        should = (i < s)[:, None, None]
        return (jnp.where(should, _matmul(E, E), E),
                jnp.where(should, 0.5 * (Jc + _matmul(E, Jc)), Jc))

    E, J = jax.lax.fori_loop(0, 18, _double_step, (pade, J))
    return _normalize(E), _normalize(J)


# ============================================================================
# OPTIMIZED RATE MATRIX CONSTRUCTION
# ============================================================================

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
        identity: (n_states, n_states) identity from the topology — topology.eye_6
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
# THE FIELD IS A STATE ENERGY ON POSITION. Each position carries a weight
# W = (-1, 0, +1) for (blocked, closed, open), and a unit's energy in its
# neighbours' field is -J * W(own) * sum W(neighbours). A transition s -> t is
# multiplied by
#
#     exp((W(pos t) - W(pos s)) * h / 2),   h = J * (W . (n_blocked, n_closed, n_open))
#
# so every equilibrium shifts by exactly the Boltzmann factor and detailed
# balance holds around every cycle, for every neighbour configuration. A
# CALCIUM edge joins two states of the same position, so its multiplier is
# exp(0) = 1: the field cannot touch calcium affinity at all. Counts are in
# {0, 1, 2}, since a unit has at most two same-chain neighbours by construction.
#
# The weights are symmetric, so at J = 0 the chain is uncoupled and J > 0 only
# steepens (nH) without a built-in shift of the midpoint.
#
# IMPLEMENTATION. The field depends only on (n_blocked, n_closed, n_open), so
# there are 27 index slots per block for the entire system, however large the
# lattice. The strong-head count adds one axis (see
# _compute_unique_tm_Q_matrices), giving 27 x (tm_max_heads + 1). Build them
# all, exponentiate in one batch, then gather per unit.


#: Number of tropomyosin unit states: 2 * position + has_calcium.
N_TM_STATES = 6

#: Ising weight of each position (blocked, closed, open).
TM_POSITION_WEIGHT = (-1.0, 0.0, 1.0)


def tm_position(tm_states):
    """Tropomyosin POSITION of each unit: 0 blocked, 1 closed, 2 open.

    state = 2 * position + has_calcium, so position = state >> 1. Every rule that reads a unit's position — the crossbridge gates and
    the closure tears — reads it through this, never through a state literal.
    Works on NumPy and JAX integer arrays alike.
    """
    return tm_states >> 1


def tm_has_calcium(tm_states):
    """1 where troponin has calcium bound (odd state), else 0."""
    return tm_states & 1


def xb_gate_position(state: 'State', topology: 'SarcTopology') -> jnp.ndarray:
    """Position of the unit whose gates each head reads, (n_xb_total,) in {0, 1, 2}.

    A BOUND head reads the unit under the monomer it is bound to; an unbound
    head reads the unit under its nearest monomer, the one it would bind. The
    nearest monomer of a bound head can sit on a neighbouring unit, and that
    unit's gate says nothing about the site the head actually holds.

    Heads with no geometric partner (xb_valid False) carry a placeholder target
    and are forced to blocked, the block where attachment is exactly zero.
    """
    thin = topology.xb_to_thin_id
    bound = state.thick.xb_bound_to.reshape(-1)
    mono = jnp.where(bound >= 0, bound, state.thick.xb_nearest_bs.reshape(-1))
    mono = jnp.clip(mono, 0, topology.n_mono - 1)
    pos = tm_position(state.thin.tm_states[thin, topology.mono_tm[thin, mono]])
    return jnp.where(topology.xb_valid, pos, 0).astype(jnp.int32)


def count_neighbor_positions(tm_states: jnp.ndarray,
                             tm_prev_neighbor: jnp.ndarray,
                             tm_next_neighbor: jnp.ndarray) -> jnp.ndarray:
    """Field-slot index of each unit: n_blocked*9 + n_closed*3 + n_open over its
    real same-chain neighbours, one strand.

    NEIGHBOURS ARE STRUCTURAL, NOT SPATIAL. A unit couples to its immediate
    predecessor and successor in its own chain's axial ordering, precomputed once
    when the topology is built. Endpoints self-reference rather than carrying a
    sentinel index, so every gather is valid; a self-reference is not counted.
    Each count is in {0, 1, 2} by construction.

    Args:
        tm_states:        (n_tm,) unit states
        tm_prev_neighbor: (n_tm,) predecessor index (self at a chain end)
        tm_next_neighbor: (n_tm,) successor index (self at a chain end)

    Returns:
        (n_tm,) int32 in [0, 27)
    """
    idx = jnp.arange(tm_states.shape[0])
    pos = tm_position(tm_states.astype(jnp.int32))

    def count(p):
        hit = pos == p
        return ((hit[tm_prev_neighbor] & (tm_prev_neighbor != idx)).astype(jnp.int32)
                + (hit[tm_next_neighbor] & (tm_next_neighbor != idx)).astype(jnp.int32))

    return count(0) * 9 + count(1) * 3 + count(2)


def _compute_unique_tm_Q_matrices(ca_concentration: float,
                                  params,
                                  n_lock: int) -> jnp.ndarray:
    """Build the 27 x (n_lock + 1) six-state generators of the tropomyosin unit.

    One matrix per neighbour composition (n_blocked, n_closed, n_open), each
    count in {0, 1, 2}, TIMES the number of STRONG crossbridges (Tight_1,
    Tight_2) bound on the unit, 0..n_lock. That is everything a unit's rates
    depend on, so these matrices cover every unit regardless of lattice size.

    THE UNIT (rate_functions module docstring): state = 2*position + has_Ca.
    Position edges blocked <-> closed <-> open at each calcium level, with that
    level's equilibrium (tm_Keq_unblock_*, tm_Keq_swing_*); calcium edges at
    every position, with the affinity there forced by detailed balance
    (tm_ca_affinity_ratios). No blocked <-> open edge.

    THE FIELD is a state energy on position (section header above): detailed
    balance holds on every cycle of every unlocked block, and calcium
    edges are untouched.

    THE CROSSBRIDGE LOCK IS A RATE. A unit with n STRONG heads has its exits
    from the open set (4 -> 2, 5 -> 3) divided by (1 + xb_tm_K2)^n. Both exits
    scale together, so the lock shifts the open/closed equilibrium by
    (1 + K2)^n at either calcium level and detailed balance survives. Weak
    heads do not lock: in McKillop & Geeves' scheme the A state binds closed
    and open units identically, and only the R state shifts the equilibrium.
    THE LOCK IS PER HEAD: their Eq. 4 (p. 695) gives each actin
    P = 1 + K1[M](1 + K2), so n S1 in one unit multiply the ratio by (1 + K2)^n.
    n_lock = SarcTopology.tm_max_heads, the most candidates any unit covers. A
    rate inside Q reproduces the equilibrium ratio at any dt; scaling a sampled
    exit PROBABILITY would match only to first order in dt.

    Weak heads do not shift blocked <-> closed: a unit carrying weak heads
    blocks at the same rate as a bare one, and thin_transitions tears them off
    when it does.

    THE CONTROL PATH IS EXACT. At K2 = jnp.inf every block with a strong head
    has zero exits from the open set, so open is absorbing (up to its calcium
    edge) and a strong head can never be torn.

    The rate laws come from rate_functions.py (tm_rate_forward/_reverse,
    tm_rate_ca_on/_off, tm_ca_affinity_ratios); this function only decides which
    modifier each rate receives and assembles the blocks.

    Args:
        ca_concentration: Calcium concentration (M)
        params: DynamicParams with the tm_* rates and equilibria, tm_J
            (kT) and xb_tm_K2
        n_lock: the most heads one unit can hold (SarcTopology.tm_max_heads)

    Returns:
        Q_unique: (27 * (n_lock + 1), 6, 6), indexed by
            (n_blocked*9 + n_closed*3 + n_open) + 27*n_strong
    """
    W = TM_POSITION_WEIGHT
    lv = jnp.arange(3, dtype=jnp.float32)
    n_b, n_c, n_o = (g.reshape(-1) for g in jnp.meshgrid(lv, lv, lv, indexing='ij'))
    h = params.tm_J * (W[0] * n_b + W[1] * n_c + W[2] * n_o)        # (27,) field, kT

    def field(s_, t_):
        """Multiplier on s -> t: exp((W(pos t) - W(pos s)) * h / 2)."""
        return jnp.exp((W[t_ >> 1] - W[s_ >> 1]) * 0.5 * h)

    ones = jnp.ones(27, jnp.float32)
    rate = {}
    for ca, sfx in ((0, 'noca'), (1, 'ca')):
        b, c, o = 0 + ca, 2 + ca, 4 + ca
        K_B = getattr(params, f'tm_Keq_unblock_{sfx}')
        K_T = getattr(params, f'tm_Keq_swing_{sfx}')
        rate[b, c] = tm_rate_forward(params.tm_k_unblock) * field(b, c)
        rate[c, b] = tm_rate_reverse(params.tm_k_unblock, K_B) * field(c, b)
        rate[c, o] = tm_rate_forward(params.tm_k_swing) * field(c, o)
        rate[o, c] = tm_rate_reverse(params.tm_k_swing, K_T) * field(o, c)
    ratio = tm_ca_affinity_ratios(params.tm_Keq_unblock_noca, params.tm_Keq_unblock_ca,
                                  params.tm_Keq_swing_noca, params.tm_Keq_swing_ca)
    for p_ in range(3):
        s_ = 2 * p_
        # Calcium edges: same position, so the field multiplier is exactly 1.
        rate[s_, s_ + 1] = tm_rate_ca_on(ca_concentration, params.tm_k_ca_on, ratio[p_],
                                         params.tm_alpha_ca) * ones
        rate[s_ + 1, s_] = tm_rate_ca_off(params.tm_k_ca_on, ratio[p_], params.tm_alpha_ca,
                                          params.tm_Keq_ca) * ones

    # One block of 27 per n_strong. The lock scales the two exits from the open
    # SET (4 -> 2, 5 -> 3), never the 4 <-> 5 calcium edge.
    #
    # 1/(1 + K2) is formed once and multiplied, not divided by per rate: at
    # K2 = jnp.inf that is a multiply by exactly 0.0, which makes the open rows
    # of every locked generator absorbing and the control path exact. Powers by
    # repeated multiplication, so the n = 1 block is bit-identical to a
    # single-head lock.
    lock = 1.0 / (1.0 + params.xb_tm_K2)
    lock_n = [1.0]
    for _ in range(n_lock):
        lock_n.append(lock_n[-1] * lock)
    full = {e: jnp.concatenate([r * (lock_n[n] if e in ((4, 2), (5, 3)) else 1.0)
                                for n in range(n_lock + 1)])
            for e, r in rate.items()}
    zero = jnp.zeros(27 * (n_lock + 1), jnp.float32)
    rows = [[full.get((i, j), zero) for j in range(N_TM_STATES)] for i in range(N_TM_STATES)]
    for i in range(N_TM_STATES):
        rows[i][i] = -sum(full[e] for e in full if e[0] == i)
    # (27 * (n_lock + 1), 6, 6), built by stacking rather than scattering.
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def _populations(subpop, constants):
    """(mode, constants_k, extra) for a transition kernel. None is the single
    wild-type population: mean-field, K = 1, fraction 1.0. 1.0 * Q is exact,
    so there is one code path whether or not a subpopulation is present."""
    if subpop is None:
        return 'mean_field', [constants], jnp.ones((1,), jnp.float32)
    return subpop


def _blend(fractions, mats):
    """Mean-field generator: sum_k f_k Q_k. Averaging GENERATORS before the
    exponential is the correct blend (see _xb_Q_resolved)."""
    return sum(fractions[k] * mats[k] for k in range(len(mats)))


def thin_transitions(state: 'State',
                     constants: 'DynamicParams',
                     topology: 'SarcTopology',
                     pCa,
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

    Sequence: build 27 x (tm_max_heads + 1) rate matrices, exponentiate them
    in one batch, count each unit's neighbours and its strong heads to get its
    configuration index, gather its probability vector, sample, then tear the
    heads the new position does not support.

    LOCKED UNITS — A FINITE LOCK, PER STRONG HEAD, APPLIED TO RATES. A unit with
    n Tight_1/Tight_2 heads has its two exits from the open set divided by
    (1 + xb_tm_K2)^n inside the generator. Weak heads do not lock. See
    _compute_unique_tm_Q_matrices for the derivation and for why the
    K2 = jnp.inf hard lock is exact.

    CLOSURE. A head is torn when its unit moves to a position that does not
    support it. Only the open position permits the weak -> strong step, so a
    STRONG head on a unit that leaves open is torn to state 4 Free_2 and charged
    one ATP (it has already released phosphate and swung its lever; returning
    it to DRX = M.ADP.Pi would hand that phosphate back for free). Closed and
    open units bind weak heads (xb_rate_01 is attach-gated) and blocked units
    bind nothing, so a WEAK head stays through open -> closed and is torn to
    state 0 DRX, owing nothing, when its unit enters blocked.

    ATP: metrics_fn books every tear from the `torn` mask this function returns,
    split by destination into `closure_detach_free` and `closure_detach_atp`,
    and adds the second into `atp_consumed` and `atp_expected`.
    No expectation is needed: the tear is fully observed and the charge is
    deterministic given it. `xb_give_up` is a different route — the 2 -> 1
    crossing count, after which the head is still weakly bound.

    Args:
        state: Current State (reads tm_states and bound_to)
        constants: DynamicParams with tm_J and the tm_* rates
        topology: SarcTopology, for chain neighbours and the eye_6 identity
        pCa: this step's calcium level, -log10([Ca2+] in M)
        rng_key: JAX random key for sampling
        dt: Timestep length (ms)
        random_values: Optional pre-drawn uniforms, for deterministic testing
        tm_subpop: Optional (mode, constants_k, extra) for mixed populations;
            None is the wild type alone (mean-field, K = 1). 'mean_field'
            weight-sums the per-population rate matrices before one exponential;
            'explicit' exponentiates each population and selects per site by
            integer label. See core/subpopulation.py.

    Returns:
        new_state: State with updated tm_states
        P_flat: the distinct probability matrices used, for validation —
            (27 * (tm_max_heads + 1), 6, 6), or (K, ..., 6, 6) for an
            explicit mixture, indexed as _compute_unique_tm_Q_matrices.
        closure_detached: (n_thick, n_crowns, n_xb_per_crown) bool, True for
            each head this call tore off. Not recoverable downstream — a torn
            head lands in state 0 or 4, which an ordinary head also reaches —
            so it is returned rather than re-derived, and travels to metrics_fn
            inside the KineticsTrace.
    """
    tm_states = state.thin.tm_states                    # (n_thin, n_tm) int8
    tm_prev_neighbor = topology.tm_prev_neighbor        # (n_thin, n_tm)
    tm_next_neighbor = topology.tm_next_neighbor        # (n_thin, n_tm)
    bound_to = state.thin.bound_to                      # (n_thin, n_cand)
    eye = topology.eye_6

    n_thin, n_tm = tm_states.shape
    n_tm_total = n_thin * n_tm

    # A UNIT's lock level is the number of STRONG heads bound on the monomers it
    # covers. Read from the HEADS,
    # not from the monomers: scatter-ADD of one per head onto its unit, n_xb
    # updates rather than one per monomer. A candidate holds at most one head,
    # so the count never exceeds topology.tm_max_heads. The two binding records
    # agree (thick_transitions keeps them so), which is what makes the head side
    # a complete record of what is bound.
    # Heads that do not count are sent past the end and dropped: left in place
    # they all land on one unit (monomer 0, and the (0, 0) placeholder of
    # invalid heads), and a GPU scatter that hammers one slot is the slow path.
    xb_mono = state.thick.xb_bound_to.reshape(-1)       # (n_xb,) monomer, -1 unbound
    xb_flat = state.thick.xb_states.reshape(-1)
    head_bound = xb_mono >= 0
    head_strong = head_bound & ((xb_flat == 2) | (xb_flat == 3))
    head_unit = (topology.xb_to_thin_id * n_tm
                 + topology.mono_tm[topology.xb_to_thin_id, jnp.maximum(xb_mono, 0)])

    def _per_unit(mask):
        return jnp.zeros(n_tm_total, jnp.int32).at[
            jnp.where(mask, head_unit, n_tm_total)].add(1).reshape(n_thin, n_tm)

    n_strong = _per_unit(head_strong)
    n_lock = topology.tm_max_heads
    n_mat = 27 * (n_lock + 1)

    ca_conc = 10.0 ** (-pCa)

    # Per-filament neighbor counts (each function call processes one strand)
    nb_idx = jax.vmap(count_neighbor_positions)(tm_states, tm_prev_neighbor, tm_next_neighbor)

    # The strong-head count selects the unit's block. Read here, once, from the
    # state at the START of the step — the same operator-splitting
    # approximation the rest of the kinetics phase makes.
    config_idx = (nb_idx + 27 * n_strong).reshape(-1)  # (n_tm_total,)

    mode, constants_k, extra = _populations(tm_subpop, constants)
    # Per-population n_mat-matrix sets. The coupling is per-population too,
    # so a subpopulation may scale tm_J as well as the rates, and its own xb_tm_K2.
    Q_k = [_compute_unique_tm_Q_matrices(ca_conc, ck, n_lock)
           for ck in constants_k]  # each (n_mat, 6, 6)
    if mode == 'mean_field':
        P_flat = expm_pade6_batch(_blend(extra, Q_k) * dt, identity=eye)  # (n_mat, 6, 6)
        P_indexed = P_flat[config_idx]                          # (n_tm_total, 6, 6)
    else:  # explicit mixture: per-unit label select
        labels = extra  # (n_tm_total,) INT in [0, K)
        Q_stack = jnp.stack(Q_k)  # (K, n_mat, 6, 6)
        Kp = Q_stack.shape[0]
        P_flat = expm_pade6_batch(
            Q_stack.reshape(Kp * n_mat, 6, 6) * dt, identity=eye).reshape(Kp, n_mat, 6, 6)
        P_indexed = P_flat[labels, config_idx]  # (n_tm_total, 6, 6)

    tm_states_flat = tm_states.reshape(-1).astype(jnp.int32)

    # The lock is already in these rows: config_idx selected each unit's block
    # by its strong-head count. No post-hoc step.
    prob_vectors = jax.vmap(lambda P, s: P[s])(P_indexed, tm_states_flat)

    if random_values is None:
        rng_key, subkey = jax.random.split(rng_key)
        random_values = jax.random.uniform(subkey, shape=(n_tm_total,))

    cum_probs = jnp.cumsum(prob_vectors, axis=1)
    new_states = jnp.argmax(random_values[:, None] < cum_probs, axis=1)

    new_tm_states = new_states.reshape(n_thin, n_tm).astype(jnp.int8)

    # ---------------------------------------------------------------- closure
    # A strong head whose unit left the open position is torn to Free_2; a weak
    # head whose unit entered blocked is torn to DRX (see CLOSURE in the
    # docstring). The strong tear is unreachable while xb_tm_K2 is jnp.inf — the
    # lock and this trigger read the same head records, so a unit with a strong
    # head can never leave the open set.
    old_pos, new_pos = tm_position(tm_states), tm_position(new_tm_states)
    left_open = ((old_pos == 2) & (new_pos != 2)).reshape(-1)
    blocked = ((old_pos != 0) & (new_pos == 0)).reshape(-1)
    hit_strong = head_strong & left_open[head_unit]
    hit_weak = head_bound & (xb_flat == 1) & blocked[head_unit]
    hit = hit_strong | hit_weak

    xb_shape = state.thick.xb_states.shape
    new_xb_states = jnp.where(hit_strong, jnp.int8(4),
                              jnp.where(hit_weak, jnp.int8(0), xb_flat)).reshape(xb_shape)
    # Clear bound_to on BOTH sides, from the same per-head `hit`, so the two
    # records cannot disagree. If the site kept its record it would stay
    # flagged occupied and nothing could ever rebind it. The site side is a
    # scatter-MAX of a flag (as in thick_transitions): a head contributing 0
    # cannot lower a raised flag, and non-hit heads are sent past the end.
    new_xb_bound_to = jnp.where(hit, jnp.int32(-1), xb_mono).reshape(xb_shape)
    n_cand = topology.n_cand
    thin_id = topology.xb_to_thin_id
    cand_flat = thin_id * n_cand + jnp.maximum(
        topology.mono_cand[thin_id, jnp.maximum(xb_mono, 0)], 0)
    clear = jnp.zeros(n_thin * n_cand, jnp.int32).at[
        jnp.where(hit, cand_flat, n_thin * n_cand)].max(1) == 1
    new_bound_to = jnp.where(clear.reshape(n_thin, n_cand), jnp.int32(-1), bound_to)

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
                   attach: jnp.ndarray,
                   stroke: jnp.ndarray,
                   site_access: jnp.ndarray,
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
        attach: (n_xb,) 1 if the target unit is closed or open, 0 if blocked.
            Gates weak attachment, r01, only.
        stroke: (n_xb,) 1 if the target unit is open, else 0. Gates the
            weak -> strong step, r12, only.
        site_access: (n_xb,) fraction of the attachment rate that survives
            thin-thin overlap screening: 1 - thin_thin_overlap_screening for a
            target past the hiding line, 1 otherwise. Scales r01 only.
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
    # Tight_1's own spring configuration (the split stroke). Equal to the
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
    # law does. It is not F_r + F_theta: those are orthogonal components, and
    # their sum tracks lattice spacing rather than strain (a head splayed across
    # the lattice at x ~ 0 carries a large F_r while doing zero axial work). See
    # the primitives header in forces.py.
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

    # TWO GATES, AND EACH STOPS ONLY ARRIVAL. A blocked unit stops weak
    # attachment (r01); a unit that is not open stops the weak -> strong step
    # (r12). Each reverse rate comes from its UNGATED forward rate: a steric
    # block stops a head arriving, not one leaving, so a weak head on a closed
    # unit detaches at the ordinary r10 and a strong head reverses at the
    # ordinary r21.
    # site_access (thin-thin overlap screening) scales the FORWARD rate only,
    # for the same reason: a screened site is bound less often rather than
    # just more slowly.
    r01_free = xb_rate_01(1.0, r01_coeff, E_weak)
    r10 = xb_rate_10(r01_free, U_DRX, U_loose)
    r01 = attach * site_access * r01_free

    # 1 Loose <-> 2 Tight_1 : weak-to-strong isomerization (Pi release)
    r12_free = xb_rate_12(r12_coeff, E_diff)
    r21 = xb_rate_21(r12_coeff, E_diff, U_loose, U_tight_1)
    r12 = stroke * r12_free

    # 2 Tight_1 <-> 3 Tight_2 : the AM.ADP isomerization (strain-dependent).
    # The stroke is split across 1->2 (the larger part) and this step; see
    # rate_functions.xb_rate_23.
    #
    # Smith & Geeves' elastic-energy form, not a Bell distance:
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
    r34 = xb_rate_34(r34_coeff, f_strong, params.xb_delta_34_catch,
                     params.xb_delta_34_slip, params.xb_r34_slip_frac, k_t)
    r43 = xb_rate_43() * ones  # Always 0

    # 4 Free_2 <-> 0 DRX : recovery stroke, re-priming the lever arm
    r40 = xb_rate_40(r40_rate) * ones
    r04 = xb_rate_04(r04_rate) * ones

    # 5 SRX <-> 0 DRX : thick-filament regulation; r50 is the Ca-dependent one
    r50 = xb_rate_50(ca_concentration, srx_k0, srx_kmax, srx_b, srx_ca50) * ones
    r05 = xb_rate_05(r05_rate) * ones

    # Diagonal rates: row sums must be zero for valid rate matrices
    # Direct arithmetic — avoids vmap+stack overhead
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
    pCa,
    z_line,
    lattice_spacing,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate crossbridge rate matrices on a distance grid, and index each head.

    The cost-saving step behind thick_transitions. A head's rates depend on three
    things: its axial distance to its target site (continuous), the position of
    its target unit (blocked / closed / open, see xb_gate_position), and
    whether the target lies in the thin-thin double-overlap zone (binary).
    Rather than build a rate matrix per head, build them on a grid:

        1 blocked matrix + n_xb_bins axial positions x 4 blocks

    laid out as blocked, then closed, closed-screened, open, open-screened.
    Each head is then assigned the index of the cell it falls in, and gathers
    from there.

    WHY BLOCKED IS ONE MATRIX. A blocked site binds at no rate, screened or
    not, and the rates out of the unbound states (r04, r40, r05, r50) do not
    depend on distance, so rows {0, 4, 5} of a blocked matrix are the same at
    every bin. No bound head ever reads a blocked unit: thin_transitions tears
    a weak head when its unit enters blocked, and a strong head when its unit
    leaves open. So rows 1-3 of the blocked matrix are never used, and one
    matrix serves every blocked head.

    THE HIDING LINE. Every thin filament has the same length, so when the
    half-sarcomere is shorter than the thin filament, the thin filaments pass
    the M-line by the same amount. By symmetry the opposite half's thin
    filaments reach the same distance into THIS half: hiding_line = how far our
    filaments pass the M-line, measured at their TIP (the farthest monomer, not
    the farthest binding candidate). A target site between the M-line and the hiding
    line sits in the double-overlap zone, where the opposing filament gets in
    the way, and its attachment rate is scaled by
    1 - constants.thin_thin_overlap_screening. Sites past the M-line on our
    own filament are excluded earlier, by the binding-site search.

    Bin assignment is arithmetic on the uniform topology.xb_bin_edges, clipped at
    both ends — heads outside the grid range are treated as if at the nearest
    edge. Since the bin range is chosen to bracket reachable distances, and
    binding probability decays sharply outside it, that clipping affects only
    heads that were never going to bind.

    THE xb_valid GATE. Heads with no real geometric partner carry a placeholder
    target index, so the tropomyosin state they read is meaningless. They are
    forced to blocked, where the attachment rate r01 is exactly zero. This is
    a hard gate, not a distance penalty: such a head can never bind, at any
    strain.

    RETURNING Q RATHER THAN P is deliberate. The gather key depends only on
    geometry and the gate, never on the rate constants, so it is shared
    across every population in a subpopulation run. Keeping the exponential
    separate lets mean-field blending average the GENERATORS before a single
    exponential — the mathematically correct blend — while the explicit modes
    exponentiate each population and select per head afterwards.

    Args:
        state: Current State (reads xb_distances, xb_nearest_bs, tm_states)
        constants: DynamicParams with the xb_* rates and spring constants
        topology: SarcTopology with bin edges/centres, xb_to_thin_id, xb_valid
        pCa: this step's calcium level
        z_line: this step's Z-line position (nm), which places the hiding line
        lattice_spacing: this step's (pre-solve) lattice spacing, nm

    Returns:
        Q_bins: (1 + 4 * n_bins, 6, 6) rate matrices: blocked, then closed,
            closed-screened, open, open-screened
        key: (n_xb_total,) index into Q_bins for each crossbridge
    """
    xb_states = state.thick.xb_states
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape
    n_xb_total = n_thick * n_crowns * n_xb_per_crown

    # Get axial distances for bin assignment
    xb_distances = state.thick.xb_distances

    if xb_distances is not None:
        xb_distances_flat = xb_distances.reshape(-1, 2)
    else:
        xb_distances_flat = jnp.zeros((n_xb_total, 2))
        xb_distances_flat = xb_distances_flat.at[:, 0].set(5.0)
        xb_distances_flat = xb_distances_flat.at[:, 1].set(lattice_spacing)

    # The gate: position of the unit under the bound (or nearest) monomer.
    position = xb_gate_position(state, topology)
    thin_indices = topology.xb_to_thin_id
    mono_indices = jnp.clip(state.thick.xb_nearest_bs.reshape(-1), 0, topology.n_mono - 1)
    thin_pos = monomer_axial(state.thin.displacement, topology, z_line)
    # The hiding line is the TIP of the thin filaments: the farthest any
    # monomer reaches past the M-line. Screening scales r01 only, so it is read
    # at the nearest monomer, the one an unbound head would bind.
    hiding_line = jnp.maximum(0.0, -jnp.min(thin_pos))
    screened = thin_pos[thin_indices, mono_indices] < hiding_line

    ca_conc = 10.0 ** (-pCa)
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

    # Q matrices at each bin position, one block per (attach, stroke, access):
    # closed, closed-screened, open, open-screened. Blocked is ONE matrix, read
    # at the middle bin: with nothing able to attach, its unbound rows {0, 4, 5}
    # are distance-free, and no bound head ever reads a blocked unit (see the
    # docstring), so its rows 1-3 are never used.
    ones = jnp.ones(n_bins)
    scr = 1.0 - constants.thin_thin_overlap_screening
    mid = slice(n_bins // 2, n_bins // 2 + 1)
    Q_blocked = xb_rate_matrix(dist_grid[mid], d, springs_grid[mid], jnp.zeros(1),
                               jnp.zeros(1), jnp.ones(1), ca_conc, constants.temp_celsius,
                               constants)
    Q_bins = jnp.concatenate([Q_blocked] + [
        xb_rate_matrix(dist_grid, d, springs_grid, a * ones, st * ones, acc * ones,
                       ca_conc, constants.temp_celsius, constants)
        for a, st, acc in ((1.0, 0.0, 1.0), (1.0, 0.0, scr),
                           (1.0, 1.0, 1.0), (1.0, 1.0, scr))], axis=0)  # (1 + 4*n_bins, 6, 6)

    # Assign each XB to a bin. The edges are uniform (a linspace), so the bin is
    # arithmetic, not a binary search: ~30x cheaper on GPU at 25,600 heads.
    x_axial = xb_distances_flat[:, 0]                              # (n_xb_total,)
    edges = topology.xb_bin_edges
    bin_idx = jnp.floor((x_axial - edges[0]) * (n_bins / (edges[-1] - edges[0]))).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

    block = 2 * position - 2 + screened.astype(jnp.int32)         # 0..3 when not blocked
    key   = jnp.where(position == 0, 0, 1 + block * n_bins + bin_idx)  # in [0, 1 + 4*n_bins)

    return Q_bins, key


def _xb_Q_resolved(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    pCa,
    z_line,
    lattice_spacing,
    xb_subpop=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Effective binned rate matrices, and how to gather them per head.

    Everything subpopulation-related lives here, and nothing else does. No
    matrix exponentials are taken — that is deliberate, because it is the part
    every caller must share. Rates come from _build_xb_Q_bins in every branch,
    so the sampling path and the metrics path cannot disagree about the physics
    of a given step no matter how they diverge afterwards.

    Both callers need the SAME generator: exp(Q*dt) and its occupancy-time
    integral come out of one call, so there is a single exponential per step.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        pCa, z_line, lattice_spacing: this step's drivers, shared by every
            population
        xb_subpop: None for the wild type alone (mean-field, K = 1), or a tuple
            (mode, constants_k, extra) for subpopulations. constants_k is a
            length-K list of DynamicParams (population 0 = WT). For
            mode=='mean_field', extra is a (K,) fractions vector and the K
            binned generators are weight-summed into one effective generator
            (Q_eff = Σ f_k Q_k) — averaging the GENERATORS before exponentiating
            is the mathematically correct blend. For mode=='explicit', extra is
            a (n_xb_total,) INT label array and the populations stay stacked so
            each head can select its own afterwards. The gather key is shared
            across populations (it depends on geometry and the gate, not
            rates).

    Returns:
        Q_bins: (n_cells, 6, 6), or (K, n_cells, 6, 6) for mode=='explicit'
        key:    (n_xb_total,) each head's index into the bin grid
        labels: None, or (n_xb_total,) population index for mode=='explicit'
    """
    mode, constants_k, extra = _populations(xb_subpop, constants)
    built = [_build_xb_Q_bins(state, ck, topology, pCa, z_line, lattice_spacing)
             for ck in constants_k]
    key = built[0][1]  # shared across populations (geometry and gate only)

    if mode == 'mean_field':
        return _blend(extra, [b[0] for b in built]), key, None

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


class XBBins(NamedTuple):
    """The one binned crossbridge generator a step is built on, and both
    exponentials of it.

    THERE IS EXACTLY ONE PER STEP. The sampler and the ATP estimator both use
    exp(Q*dt) of the same matrix, and its occupancy-time integral comes out of
    the same scaling-and-squaring. `kinetics_step` builds this once and hands it
    to both.

    Fields:
        Q: (n_cells, 6, 6) rate matrices on the bin grid, or the stacked
            (K, n_cells, 6, 6) of an explicit subpopulation run
        P: exp(Q*dt), same shape — what the sampler draws from
        G: INT_0^dt exp(Qt)dt, same shape — expected time in each state, from
            which q_ij * G[s, i] gives exact expected crossing counts
        key: (n_xb_total,) each head's index into the bin grid
        labels: (n_xb_total,) population index for mode=='explicit', else None
    """
    Q: jnp.ndarray
    P: jnp.ndarray
    G: jnp.ndarray
    key: jnp.ndarray
    labels: object


def xb_binned_generator(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    pCa,
    z_line,
    lattice_spacing,
    dt: float,
    xb_subpop=None,
) -> XBBins:
    """Build the step's crossbridge generator and exponentiate it, once.

    Evaluates 1 + 4 * n_xb_bins matrix exponentials instead of one per head — on the
    distance grid rather than per head, so the cost does not grow with lattice
    size. Bin resolution is a genuine accuracy/cost tradeoff; see
    _build_xb_Q_bins and StaticParams.n_xb_bins / xb_bin_lo / xb_bin_hi.

    The occupancy-time integral G comes out of the same squaring loop as P,
    which is what lets the metrics path drop its own second exponential
    entirely.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with xb_bin_edges, xb_bin_centers, eye_6
        pCa: this step's calcium level
        z_line: this step's Z-line position (nm)
        lattice_spacing: this step's (pre-solve) lattice spacing, nm
        dt: Timestep length (ms)
        xb_subpop: see _xb_Q_resolved

    Returns:
        XBBins
    """
    Q_bins, key, labels = _xb_Q_resolved(state, constants, topology, pCa,
                                         z_line, lattice_spacing, xb_subpop)
    P_bins, G_bins = _expm_bins(Q_bins, dt, topology.eye_6, with_integral=True)
    return XBBins(Q=Q_bins, P=P_bins, G=G_bins, key=key, labels=labels)


#: Ordered pairs whose expected crossing counts the ATP metrics read.
#: (1,0) free detachment, (1,2) Pi release, (2,1) the strain give-up reversal,
#: (3,4) ATP-consuming detachment, (0,4)/(4,0) the reversible recovery stroke.
XB_METRIC_PAIRS = ((1, 0), (1, 2), (2, 1), (3, 4), (0, 4), (4, 0))


def xb_expected_crossings(
    bins: XBBins,
    states: jnp.ndarray,
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

    WHY AN EDGE COUNT AND NOT A REACHABILITY PROBABILITY. "Did this head visit
    Free_2 during the step" is P(visit >= 1), a LOWER BOUND on E(visits), biased
    low whenever a head can cycle twice inside one step (about -1% skeletal at
    dt = 1 ms). Counting the EDGE 3 -> 4 is exact in expectation, never counts
    r04 (the reverse recovery stroke) as an ATP, and needs no mask, so every head
    can be read — including one that runs 0 -> 1 -> 2 -> 3 -> 4 inside a single
    step, which really does spend an ATP.

    Reads each head's OWN start state, so pass the MID state — the sarcomere as
    thick_transitions found it, carried on KineticsTrace — not the state at the
    top of the step. Takes the already-built XBBins rather than rebuilding them,
    so the sampler and the metrics cannot disagree about the step and only one
    matrix exponential is ever taken.

    Args:
        bins: XBBins from xb_binned_generator(), for THIS step
        states: (n_xb_total,) or any shape reshapeable to it — each head's
            starting state
        pairs: ordered (i, j) edges to count; defaults to XB_METRIC_PAIRS

    Returns:
        N: (len(pairs), n_xb_total) expected crossing counts, per head, over dt
    """
    s = states.reshape(-1).astype(jnp.int32)
    counts = []
    for (i, j) in pairs:
        # Expected time in state i, by starting state, times the i -> j rate.
        # Ellipsis indexing covers the plain and the stacked (subpopulation)
        # layouts alike, so neither mode needs its own branch.
        per_cell = bins.G[..., :, i] * bins.Q[..., i, j][..., None]
        per_xb = _gather_per_xb(per_cell, bins.key, bins.labels)  # (n_xb_total, 6)
        counts.append(jnp.take_along_axis(per_xb, s[:, None], axis=1)[:, 0])
    return jnp.stack(counts)


#: The calcium level a simulation starts relaxed at.
REST_PCA = 9.0


def xb_rest_states(state: 'State',
                   constants: 'DynamicParams',
                   topology: 'SarcTopology',
                   z_line,
                   lattice_spacing,
                   rng_key: jax.random.PRNGKey,
                   xb_subpop=None) -> 'State':
    """Draw every head from the stationary distribution of its own rate matrix
    at rest: pCa 9, tropomyosin blocked.

    run() starts from relaxed muscle. Every head gathers from the BLOCKED block
    of the same binned generator the scan uses (rest is evaluated with every
    unit blocked), where attachment (r01) is exactly 0 and r43 is structurally 0. The detached states {0 DRX, 4 Free_2, 5 SRX} are then
    a closed class and the bound states get no inflow, so the resting occupancy
    is pi Q_c = 0, sum(pi) = 1 on that 3x3 block, and exactly 0 bound. Solving
    the block rather than the full 6x6 matters: at strained bins Tight_1's exit
    rate is ~1e-10 /ms, which makes the full system singular to float32 and
    leaks spurious mass onto a bound state. A head in a subpopulation draws
    from its own population's generator.

    Args:
        state: a freshly realised State (all tropomyosin 0, no head bound)
        constants: DynamicParams with physics values
        topology: SarcTopology
        z_line, lattice_spacing: the drivers the simulation starts at
        rng_key: key for this draw
        xb_subpop: see _xb_Q_resolved

    Returns:
        state with thick.xb_states drawn from rest
    """
    Q, key, labels = _xb_Q_resolved(state, constants, topology, REST_PCA,
                                    z_line, lattice_spacing, xb_subpop)
    detached = jnp.array([0, 4, 5])
    Q_c = Q[..., detached, :][..., detached]
    # pi Q_c = 0 with the last balance equation replaced by the normalisation
    A = jnp.swapaxes(Q_c, -1, -2).at[..., -1, :].set(1.0)
    b = jnp.zeros(Q_c.shape[:-1]).at[..., -1].set(1.0)
    pi_c = jnp.linalg.solve(A, b[..., None])[..., 0]
    pi = jnp.zeros(Q.shape[:-1]).at[..., detached].set(pi_c)
    pi_xb = _gather_per_xb(pi, key, labels)                        # (n_xb_total, 6)

    xb_states = state.thick.xb_states
    u = jax.random.uniform(rng_key, shape=(pi_xb.shape[0],))
    drawn = jnp.argmax(u[:, None] < jnp.cumsum(pi_xb, axis=1), axis=1)
    return state._replace(thick=state.thick._replace(
        xb_states=drawn.astype(jnp.int8).reshape(xb_states.shape)))


def tm_rest_states(state: 'State',
                   constants: 'DynamicParams',
                   topology: 'SarcTopology',
                   rng_key: jax.random.PRNGKey,
                   tm_subpop=None) -> 'State':
    """Draw every tropomyosin unit from the stationary distribution of its own
    uncoupled generator at rest: pCa 9, J = 0, no heads.

    The six-state unit is not all-blocked at rest: the Ca-free equilibria put a
    measurable share closed and open (cardiac more than skeletal). Drawing that
    share directly removes a settling transient for the units themselves; the
    chain coupling and the heads then settle in the scan. Call AFTER
    xb_rest_states, which is evaluated on all-blocked units. A unit in a
    subpopulation draws from its own population's generator.

    Args:
        state: State with heads at rest and no head bound
        constants: DynamicParams
        topology: SarcTopology
        rng_key: key for this draw
        tm_subpop: see thin_transitions

    Returns:
        state with thin.tm_states drawn from rest
    """
    mode, constants_k, extra = _populations(tm_subpop, constants)
    # n_lock = 0: one block; slot 0, where the field is exp(0) at J = 0 for
    # every slot anyway.
    Q_k = [_compute_unique_tm_Q_matrices(10.0 ** (-REST_PCA), ck.copy(tm_J=0.0), 0)[0]
           for ck in constants_k]
    Q = _blend(extra, Q_k)[None] if mode == 'mean_field' else jnp.stack(Q_k)  # (K, 6, 6)
    A = jnp.swapaxes(Q, -1, -2).at[..., -1, :].set(1.0)
    b = jnp.zeros(Q.shape[:-1]).at[..., -1].set(1.0)
    pi = jnp.linalg.solve(A, b[..., None])[..., 0]                       # (K, 6)
    n = topology.n_thin * topology.n_tm
    pi_unit = pi[jnp.zeros(n, jnp.int32) if mode == 'mean_field' else extra]
    u = jax.random.uniform(rng_key, shape=(n,))
    drawn = jnp.argmax(u[:, None] < jnp.cumsum(pi_unit, axis=1), axis=1)
    return state._replace(thin=state.thin._replace(
        tm_states=drawn.astype(jnp.int8).reshape(topology.n_thin, topology.n_tm)))


def thick_transitions(state: 'State',
                     bins: XBBins,
                     topology: 'SarcTopology',
                     rng_key: jax.random.PRNGKey,
                     random_values: Optional[jnp.ndarray] = None):
    """Advance every crossbridge one timestep, and update what it is bound to.

    Two things happen here, and the second is easy to overlook: heads sample new
    states from their transition probability matrices, AND the binding
    bookkeeping is updated on both filaments. A head entering a bound state must
    record which site it took, and that site must record which head took it —
    the thin filament's bound_to array is what locks a tropomyosin site open
    (see thin_transitions) and what lets forces.py know where to apply
    crossbridge force.

    Rates come from xb_binned_generator(), which evaluates them on a distance
    grid rather than per head; see there and in _build_xb_Q_bins for how that
    works and what it costs in accuracy. The bins are built by `kinetics_step`
    and passed in, because the metrics path reads the same ones.

    Heads flagged invalid by topology.xb_valid read a blocked gate throughout,
    so they can never enter a bound state and never claim a site.

    Args:
        state: Current State
        bins: XBBins for this step, from xb_binned_generator(). Carries the
            subpopulation resolution already applied; see _xb_Q_resolved().
        topology: SarcTopology with xb_to_thin_id, xb_valid, eye_6
        rng_key: JAX random key for sampling
        random_values: Optional pre-drawn uniforms, for deterministic testing

    Returns:
        new_state: State with updated xb_states, xb_bound_to, and thin bound_to
    """
    # Get current xb states
    xb_states = state.thick.xb_states  # (n_thick, n_crowns, n_xb_per_crown)
    n_thick, n_crowns, n_xb_per_crown = xb_states.shape

    # Flatten for processing
    xb_states_flat = xb_states.reshape(-1)  # (n_thick * n_crowns * n_xb_per_crown,)
    n_xb_total = xb_states_flat.shape[0]

    # Per-XB transition probabilities, gathered from the step's bin grid
    P_all = _gather_per_xb(bins.P, bins.key, bins.labels)

    # Sample new states (same logic as thin_transitions)
    current_states = xb_states_flat.astype(jnp.int32)

    # Get probability vectors — index directly into P_all using current state
    prob_vectors = jax.vmap(lambda P, s: P[s])(P_all, current_states)  # (n_xb_total, 6)

    # Binding info (needed for binding logic below). A binding head is unbound,
    # so its gate is the unit under its nearest monomer; weak attachment needs
    # that unit closed or open. Invalid heads read blocked (xb_gate_position).
    xb_nearest_bs = state.thick.xb_nearest_bs
    n_thin, n_mono = topology.n_thin, topology.n_mono
    xb_nearest_bs_flat = xb_nearest_bs.reshape(-1)
    thin_indices = topology.xb_to_thin_id
    mono_indices = jnp.clip(xb_nearest_bs_flat, 0, n_mono - 1)
    attach = xb_gate_position(state, topology) >= 1

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

    # ====================================================================
    # THE TWO BINDING RECORDS MUST AGREE, AND SCATTERS ARE WHERE THEY STOP
    # ====================================================================
    # An attachment is recorded twice — `xb_bound_to[h]` names the site,
    # `thin.bound_to[t, i]` names the head — and nothing reconciles them
    # afterwards. Both writes below therefore have to be collision-free by
    # construction, because `.at[].set()` with duplicate indices picks a
    # winner nondeterministically.
    # thin.bound_to is per CANDIDATE; every nearest site is a candidate.
    n_cand = topology.n_cand
    n_cand_total = n_thin * n_cand
    xb_indices_arr = jnp.arange(n_xb_total)
    site_flat = thin_indices * n_cand + topology.mono_cand[thin_indices, mono_indices]

    nearest_site_occupied = thin_bound_to_flat[site_flat] >= 0
    can_bind = is_binding & attach & (~nearest_site_occupied)

    # ARBITRATION. `can_bind` is read against the occupancy at the START of
    # the step, so two heads whose nearest site is the same free site BOTH
    # pass it — and exactly one of them may have it. Scatter-MIN over head
    # index picks the lowest, deterministically; the loser reverts below, so
    # two heads can never share a site. Non-binders are sent past the end and dropped, so they can never win
    # and never touch the array. Dropping rather than writing a sentinel
    # also keeps them off the few sites they would otherwise pile onto
    # (invalid heads all name thin 0), which is the slow path for a GPU scatter.
    winner = jnp.full(n_cand_total, n_xb_total, jnp.int32).at[
        jnp.where(can_bind, site_flat, n_cand_total)].min(xb_indices_arr)
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
    # Only heads really unbinding scatter; the rest are sent past the end and
    # dropped (left in place, every unbound head lands on monomer 0 of its thin).
    old_site_flat = (old_thin_indices * n_cand + jnp.maximum(
        topology.mono_cand[old_thin_indices, jnp.clip(xb_bound_to_flat, 0, n_mono - 1)], 0))
    clear_site = jnp.zeros(n_cand_total, jnp.int32).at[
        jnp.where(really_unbinding, old_site_flat, n_cand_total)].max(1) == 1
    new_thin_bound_to_flat = jnp.where(clear_site, -1, thin_bound_to_flat)

    # STEP 2: record the winners. `winner` is already a per-SITE array, so
    # this is an elementwise select rather than a second scatter, and the
    # two writes cannot conflict: a site occupied at the start of the step
    # fails `can_bind` for every head, so no site is both cleared and won.
    new_thin_bound_to_flat = jnp.where(winner < n_xb_total, winner,
                                       new_thin_bound_to_flat)

    new_xb_bound_to = new_xb_bound_to_flat.reshape(n_thick, n_crowns, n_xb_per_crown)
    new_thin_bound_to = new_thin_bound_to_flat.reshape(n_thin, n_cand)

    # Binding failed — the site was taken before the step, or another head
    # won it during the step.
    #
    # >>> THIS IS A FLUX THAT IS NOT IN Q, AND IT IS NOT SMALL. A head sent
    #     back here did not follow its own generator: it is placed in DRX
    #     regardless of which bound state it sampled (8x8, dt = 1 ms: 1-5% of
    #     heads that sampled a bound state from 0/4/5 cardiac, 5-7%
    #     skeletal). ANY METRIC THAT READS P OR Q AS THE TRUTH ABOUT THE STEP
    #     must say why that gap does not reach it. For atp_expected the
    #     exposure is bounded: heads in mid states 0/4/5 contribute < 0.6% of
    #     the 3 -> 4 count, and only the reverted subset of those is
    #     misattributed (~0.002% cardiac, ~0.04% skeletal).
    new_states_flat = jnp.where(is_binding & (~won), 0, new_states_flat)
    new_xb_states = new_states_flat.reshape(n_thick, n_crowns, n_xb_per_crown).astype(jnp.int8)

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

