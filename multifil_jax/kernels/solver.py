"""
Mechanical equilibrium solver.

THE PROBLEM
-----------
After the kinetics phase, crossbridges have attached and detached, so the
lattice is no longer in force balance. This module finds the filament node
positions x at which every node's net axial force vanishes:

    F(x) = 0

This is solved to convergence at every timestep. There is no time integration
of the mechanics — see kernels/forces.py for why an instantaneously equilibrated
treatment is appropriate at this scale. It is also, by a wide margin, the most
expensive part of a timestep.

WHY NEWTON, AND WHY IT CONVERGES SO FAST
----------------------------------------
F is nonlinear, mostly because crossbridge force depends on head geometry
through square roots and arctangents. But it is very nearly linear: the
filament backbones are exactly linear springs, and they dominate the Jacobian.
So Newton's method converges in one to a few iterations from a warm start, and
the previous timestep's positions are always an excellent warm start — the
lattice moves only slightly per millisecond.

The outer loop is a lax.while_loop that exits as soon as the residual falls
below tolerance, rather than always running a fixed count.

WHY THE LINEAR SOLVE IS ITERATIVE
---------------------------------
Each Newton step needs to solve J dx = F. Forming J explicitly is out of the
question: it is (n_nodes x n_nodes) for a system with tens of thousands of
nodes. Instead conjugate gradient is used, which needs only the ACTION of J on a
vector — and that comes free from JAX's forward-mode autodiff (jax.jvp) applied
to the residual function. The Jacobian is never assembled at all.

CG rather than a general-purpose Krylov method because J is symmetric positive
definite here (it is the Hessian of an elastic energy), and CG needs one
Jacobian-vector product per iteration where BiCGStab would need two.

THE PRECONDITIONER
------------------
CG converges quickly only if the system is well conditioned, so it is
preconditioned with an approximate inverse. The approximation used is the
backbone springs alone — no crossbridges, no titin. That makes each filament's
block a TRIDIAGONAL matrix (each node couples only to its two neighbours), which
can be factored exactly by the Thomas algorithm.

Two properties make this a good choice rather than a lazy one:

  It is constant. The backbone stiffness never changes during a simulation, so
  the factorization is computed ONCE before the time loop and reused every step
  thereafter, rather than being rebuilt as crossbridges come and go.

  It is robust. Including titin would be more accurate in principle, but titin's
  exponential stiffness overflows at extreme stretch and would destroy the
  preconditioner exactly when the solve is hardest. The backbone-only
  approximation degrades gracefully instead.

LATTICE SPACING AS AN UNKNOWN
-----------------------------
In dynamic lattice spacing mode the filament separation d is not prescribed but
determined by radial force balance. Rather than alternating between an axial and
a radial solve, d is appended to the position vector as one extra degree of
freedom and the whole augmented system is solved at once. Autodiff then supplies
the cross-coupling terms — how d affects axial forces, and how the node
positions affect the radial balance — automatically and exactly.

Usage:
    from multifil_jax.kernels.solver import solve_equilibrium
    state, residual, new_ls, n_iters = solve_equilibrium(state, constants, topology)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Union, NamedTuple, TYPE_CHECKING
from functools import partial

from multifil_jax.kernels.forces import (
    compute_forces_vectorized,
    _xb_radial_force_total,
    _titin_radial_force_total,
)
from multifil_jax.core.state import PreconditionerParams

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams

# Absolute floor on the convergence tolerance, in pN.
#
# There is a limit to how small a residual float32 can even represent here. Node
# positions are ~1000 nm, where consecutive float32 values differ by ~1e-4 nm, so
# a position cannot be resolved more finely than that. Multiplying by the
# backbone stiffness turns that position quantum into a force quantum:
# thick_k * 1e-4 pN. Asking the solver for a residual below it makes the
# while_loop iterate until its cap, burning time to chase a target that
# arithmetic cannot express.
#
# At the default thick_k = 7500 pN/nm that floor is ~0.75 pN, so this constant
# is not what binds at default parameters — solve_equilibrium() takes the
# maximum of this and the stiffness-scaled floor. It matters at soft parameters.
MIN_FLOAT32_TOLERANCE = 0.25


# ============================================================================
# THOMAS ALGORITHM (Pure-JAX tridiagonal solver — replaces Lineax cusparse)
# ============================================================================

class ThomasFactors(NamedTuple):
    """Pre-factored Thomas algorithm data for tridiagonal solves.

    The forward sweep depends only on the matrix (lower, diag, upper),
    which is constant across all timesteps and CG iterations. Factor once
    before the scan loop, then reuse for every back-substitution.

    Attributes:
        inv_diag: (n,) reciprocal of modified diagonal after forward elimination
        upper: (n-1,) original upper diagonal (stored for back-sub convenience)
        multipliers: (n-1,) L[i] = lower[i] / diag'[i-1], the elimination multipliers
    """
    inv_diag: jnp.ndarray   # (n,) — 1/d'[i] after forward elimination
    upper: jnp.ndarray      # (n-1,) — original upper diagonal
    multipliers: jnp.ndarray # (n-1,) — forward sweep multipliers


def thomas_factor(lower: jnp.ndarray, diag: jnp.ndarray, upper: jnp.ndarray) -> ThomasFactors:
    """Factor a tridiagonal matrix — the forward sweep of the Thomas algorithm.

    Thomas is Gaussian elimination specialized to tridiagonal systems: it runs
    in O(n) rather than O(n^3) because eliminating each row touches only the one
    below it. This function does the elimination and stores what
    back-substitution will need.

    Called ONCE per simulation, before the time loop, since the preconditioner
    matrix is constant. That is why a Python for loop is acceptable here despite
    being sequential — it is unrolled at trace time and never appears in the hot
    path. (thomas_solve, which DOES run every CG iteration, uses a parallel scan
    instead.)

    Args:
        lower: (n-1,) sub-diagonal
        diag: (n,) main diagonal
        upper: (n-1,) super-diagonal

    Returns:
        ThomasFactors for use with thomas_solve()
    """
    n = diag.shape[0]
    modified_diag = [diag[0]]
    multipliers = []
    for i in range(1, n):
        m = lower[i - 1] / modified_diag[i - 1]
        multipliers.append(m)
        modified_diag.append(diag[i] - m * upper[i - 1])

    return ThomasFactors(
        inv_diag=1.0 / jnp.stack(modified_diag),
        upper=upper,
        multipliers=jnp.stack(multipliers),
    )


def thomas_solve(factors: ThomasFactors, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve a tridiagonal system using pre-computed factors.

    Runs on every CG iteration of every Newton step of every timestep, so it is
    genuinely hot, and a sequential sweep would serialize a GPU that has
    thousands of idle lanes.

    Both substitution passes are linear recurrences of the form

        state[i] = a[i] * state[i-1] + b[i]

    which look inherently sequential but are not. Such a recurrence is an affine
    map, and affine maps COMPOSE associatively:

        (a2, b2) o (a1, b1) = (a2*a1, a2*b1 + b2)

    so the whole recurrence can be evaluated by a parallel prefix scan in
    O(log n) depth instead of O(n) steps. jax.lax.associative_scan does exactly
    that. Back substitution runs the same way on reversed arrays.

    The parallel form is also far cheaper to compile: it emits roughly a fifth
    as many jaxpr equations as an unrolled loop of the same length.

    Accuracy: the reassociation costs about 1e-4 relative error compared to the
    sequential sweep. Irrelevant here — this matrix is only a preconditioner, an
    approximation to the true Jacobian by construction, and CG corrects for its
    inexactness. Do not reuse this routine where an exact tridiagonal solve is
    required.

    Args:
        factors: Pre-computed ThomasFactors from thomas_factor()
        rhs: (n,) right-hand side vector

    Returns:
        x: (n,) solution vector
    """
    inv_diag, upper, multipliers = factors.inv_diag, factors.upper, factors.multipliers

    def compose(left, right):
        """Compose two affine-map segments: f_right ∘ f_left."""
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    # Forward substitution: y[i] = rhs[i] - multipliers[i-1] * y[i-1]
    # Linear recurrence y[i] = a[i]*y[i-1] + b[i] with a[0]=1, b[0]=rhs[0]
    a_fwd = jnp.concatenate([jnp.ones(1), -multipliers])
    _, y = jax.lax.associative_scan(compose, (a_fwd, rhs))

    # Back substitution: x[i] = (y[i] - upper[i]*x[i+1]) * inv_diag[i]
    # Backward linear recurrence — flip, scan forward, flip back.
    # a_back[n-1]=0 (boundary: no x[n] term).
    a_back = jnp.concatenate([-upper * inv_diag[:-1], jnp.zeros(1)])
    _, x_rev = jax.lax.associative_scan(compose, (a_back[::-1], (y * inv_diag)[::-1]))
    return x_rev[::-1]


class PreFactoredPreconditioner(NamedTuple):
    """Pre-factored block-diagonal preconditioner.

    Built once per batch element before the scan loop. Reused across
    all CG iterations/timestep x all timesteps.

    Factors can be either:
    - Shared (single-filament): fields have shape (n_crowns,) / (n_sites,)
      All filaments of the same type share the same factors. Broadcast via
      vmap in_axes=(None, 0) at apply time.
    - Per-filament: fields have shape (n_thick, n_crowns) / (n_thin, n_sites)
      Each filament has its own factors (e.g. with XB binding corrections).

    Attributes:
        thick_factors: ThomasFactors for thick filament type(s)
        thin_factors: ThomasFactors for thin filament type(s)
    """
    thick_factors: ThomasFactors
    thin_factors: ThomasFactors


def build_prefactored_preconditioner(
    params: PreconditionerParams,
    negate: bool = True,
    eps: float = 1e-9,
) -> PreFactoredPreconditioner:
    """Build and factor the block-diagonal preconditioner once.

    Factors ONE thick and ONE thin tridiagonal matrix. All filaments of
    the same type share the same base spring-constant structure, so a
    single factorization suffices. At apply time, factors are broadcast
    across filaments via vmap in_axes=(None, 0).

    Call ONCE before the scan loop; reuse across all timesteps.

    Args:
        params: PreconditionerParams with single-filament tridiagonal arrays
        negate: If True, negate arrays for positive definiteness (CG compatibility)
        eps: Regularization for numerical stability

    Returns:
        PreFactoredPreconditioner ready for apply_preconditioner()
    """
    sign = -1.0 if negate else 1.0

    # Scale and regularize single-filament arrays
    diag_thick_reg = sign * params.diag_thick + eps
    diag_thin_reg = sign * params.diag_thin + eps
    lower_thick_scaled = sign * params.lower_thick
    upper_thick_scaled = sign * params.upper_thick
    lower_thin_scaled = sign * params.lower_thin
    upper_thin_scaled = sign * params.upper_thin

    # Factor once per filament type (no vmap — arrays are single-filament)
    thick_factors = thomas_factor(lower_thick_scaled, diag_thick_reg, upper_thick_scaled)
    thin_factors = thomas_factor(lower_thin_scaled, diag_thin_reg, upper_thin_scaled)

    return PreFactoredPreconditioner(
        thick_factors=thick_factors,
        thin_factors=thin_factors,
    )


def apply_preconditioner(
    prefactored: PreFactoredPreconditioner,
    v: jnp.ndarray,
    n_thick: int,
    n_crowns: int,
    n_thin: int,
    n_sites: int,
) -> jnp.ndarray:
    """Apply pre-factored block-diagonal preconditioner: M^{-1} @ v.

    Uses the pre-computed Thomas factors for back-substitution only.
    Shared factors (single-filament) are broadcast across all filaments
    via vmap in_axes=(None, 0). Per-filament factors use in_axes=(0, 0).

    Args:
        prefactored: Pre-factored preconditioner from build_prefactored_preconditioner()
        v: Position/force vector to precondition
        n_thick, n_crowns, n_thin, n_sites: Static dimensions

    Returns:
        M^{-1} @ v
    """
    n_thick_nodes = n_thick * n_crowns

    v_thick = v[:n_thick_nodes].reshape(n_thick, n_crowns)
    v_thin = v[n_thick_nodes:].reshape(n_thin, n_sites)

    # Determine if factors are shared (1D) or per-filament (2D)
    shared = prefactored.thick_factors.inv_diag.ndim == 1
    factor_axes = None if shared else 0

    # Solve all filaments in parallel via vmap (back-sub only)
    x_thick = jax.vmap(thomas_solve, in_axes=(factor_axes, 0))(
        prefactored.thick_factors, v_thick)
    x_thin = jax.vmap(thomas_solve, in_axes=(factor_axes, 0))(
        prefactored.thin_factors, v_thin)

    return jnp.concatenate([x_thick.flatten(), x_thin.flatten()])


# ============================================================================
# PRECONDITIONED CG (Python loop unrolled at trace time)
# ============================================================================

def _preconditioned_cg(neg_jac_mv, precond_mv, b, x0, n_cg_steps):
    """Preconditioned CG solver using Python for loop (unrolled at trace time).

    Solves A @ x = b where A = -J (positive semi-definite).
    Uses block-tridiagonal preconditioner M^{-1} for faster convergence.

    Python loop unrolling eliminates WhileOp fusion barriers from fori_loop,
    allowing XLA to fuse all CG iterations into fewer GPU kernels.

    Args:
        neg_jac_mv: Function v -> (-J) @ v (Jacobian-vector product, negated)
        precond_mv: Function v -> M^{-1} @ v (preconditioner apply)
        b: Right-hand side vector (residual forces)
        x0: Initial guess (zeros)
        n_cg_steps: Fixed number of CG iterations

    Returns:
        x: Approximate solution
    """
    r = b - neg_jac_mv(x0)
    z = precond_mv(r)
    p = z
    rz = jnp.dot(r, z)
    x = x0

    for _ in range(n_cg_steps):
        Ap = neg_jac_mv(p)
        pAp = jnp.dot(p, Ap)
        alpha = rz / (pAp + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = precond_mv(r)
        rz_new = jnp.dot(r, z)
        beta = rz_new / (rz + 1e-30)
        p = z + beta * p
        rz = rz_new

    return x


def _run_newton(residual_fn, precond_mv, pos0, tol, n_newton_steps, n_cg_steps, post_step=None):
    """The Newton iteration itself, shared by the fixed- and dynamic-spacing solvers.

    Each step solves J dx = F for the update and applies it. The Jacobian-vector
    product is obtained by forward-mode autodiff of residual_fn — the Jacobian
    is never assembled. Note the sign is folded into neg_jac_mv rather than
    negating dx afterwards, saving a pass over the vector each CG iteration.

    Exits as soon as max|F| falls below tol, or after n_newton_steps. Both
    conditions live in the while_loop predicate, so the body is traced once
    regardless of how many iterations actually run.

    Non-finite updates are zeroed rather than propagated. This is a safety net
    for extreme parameter combinations where the Jacobian is near-singular:
    the step is skipped, the iteration continues, and a poor residual is
    reported at the end rather than NaN spreading through the whole batch.

    Args:
        residual_fn: pos -> force residual vector
        precond_mv: v -> M^{-1} @ v
        pos0: initial position vector
        tol: convergence tolerance (scalar JAX array)
        n_newton_steps: hard iteration cap
        n_cg_steps: CG iterations per Newton step
        post_step: optional callable applied to x_new after each step
                   (e.g. lattice spacing floor for dynamic LS)

    Returns:
        (x, n_iters, final_residual)
    """
    f0 = residual_fn(pos0)

    def body(carry):
        x, f, i = carry
        if n_cg_steps == 0:
            # Richardson iteration: take the preconditioner's own solve as the
            # step, with no CG refinement and no Jacobian-vector product at all.
            # This is only valid when the preconditioner is nearly the true
            # Jacobian, i.e. when the backbone springs are essentially the whole
            # system. That holds with no crossbridges attached and fails once
            # they are: attached heads add stiffness the preconditioner does not
            # model, and the iteration diverges. Not a usable default.
            dx = precond_mv(f)
        else:
            neg_jac_mv = lambda v: -jax.jvp(residual_fn, (x,), (v,))[1]
            dx = _preconditioned_cg(neg_jac_mv, precond_mv, f, jnp.zeros_like(x), n_cg_steps)
        dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
        x_new = x + dx
        if post_step is not None:
            x_new = post_step(x_new)
        f_new = residual_fn(x_new)
        return x_new, f_new, i + jnp.int32(1)

    def cond(carry):
        _, f, i = carry
        return (jnp.max(jnp.abs(f)) > tol) & (i < n_newton_steps)

    x, f, n_iters = jax.lax.while_loop(cond, body, (pos0, f0, jnp.int32(0)))
    return x, n_iters, jnp.max(jnp.abs(f))


# ============================================================================
# NEWTON-RAPHSON SOLVER (Python loop unrolled at trace time)
# ============================================================================

def _newton_solve(
    positions_init: jnp.ndarray,
    thick_k: float,
    thin_k: float,
    z_line: float,
    lattice_spacing: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params: 'DynamicParams',
    precond_params: PreconditionerParams,
    topology: 'SarcTopology',
    n_thick: int,
    n_crowns: int,
    n_thin: int,
    n_sites: int,
    n_newton_steps: int = 16,
    n_cg_steps: int = 6,
    tolerance: Optional[jnp.ndarray] = None,
    prefactored_precond: Optional[PreFactoredPreconditioner] = None,
) -> Tuple[jnp.ndarray, int, float]:
    """Newton-Raphson solver with while_loop Newton and unrolled CG.

    Uses jax.lax.while_loop for the outer Newton loop — body traced once,
    exits when max|f| < tolerance OR n_newton_steps cap is reached.
    Compile time ∝ n_cg_steps (not n_newton_steps × n_cg_steps).

    Inner CG uses Python for loop (unrolled at trace time), enabling
    full XLA fusion across CG iterations.

    Args:
        n_newton_steps: Hard cap on Newton iterations (default 16).
                        while_loop exits early when converged.
        n_cg_steps: Fixed number of CG iterations per Newton step (default 6; 0=Richardson).
        tolerance: Convergence target (pN). If None, uses MIN_FLOAT32_TOLERANCE.

    Optimizations:
    1. Unrolled CG with Python for loop (no fori_loop WhileOp)
    2. Block-tridiagonal preconditioner via vmap'd tridiagonal solves
    3. Baked-in negation for positive definiteness (CG compatibility)
    4. JVP-based Jacobian-vector products (1 JVP per CG iteration)
    """
    n_thick_nodes = n_thick * n_crowns

    def residual_fn(pos):
        """Compute force residual F(x) at given positions."""
        pos_thick = pos[:n_thick_nodes].reshape(n_thick, n_crowns)
        pos_thin = pos[n_thick_nodes:].reshape(n_thin, n_sites)
        return compute_forces_vectorized(
            pos_thick, pos_thin,
            thick_k, thin_k, z_line, lattice_spacing,
            titin_a, titin_b, titin_rest,
            xb_states, xb_bound_to, params, topology
        )

    prefactored = prefactored_precond if prefactored_precond is not None else \
        build_prefactored_preconditioner(precond_params, negate=True, eps=1e-9)
    precond_mv = lambda v: apply_preconditioner(prefactored, v, n_thick, n_crowns, n_thin, n_sites)
    tol = tolerance if tolerance is not None else jnp.asarray(MIN_FLOAT32_TOLERANCE)
    return _run_newton(residual_fn, precond_mv, positions_init, tol, n_newton_steps, n_cg_steps)


# ============================================================================
# DYNAMIC LATTICE SPACING — internal helpers
# ============================================================================

def _radial_residual(
    d: float,
    positions_thick: jnp.ndarray,
    positions_thin: jnp.ndarray,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    z_line: float,
    params,
    topology,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    K_lat: float,
    d_ref: float,
) -> float:
    """Radial force residual: zero at radial equilibrium.

    F_rad = -K_lat*(d - d_ref) - F_xb_code - F_titin_code = 0
    """
    f_lat = -K_lat * (d - d_ref)
    f_xb = -_xb_radial_force_total(
        xb_states, xb_bound_to, positions_thick, positions_thin, d, params, topology
    )
    f_titin = -_titin_radial_force_total(
        positions_thick, z_line, d, titin_a, titin_b, titin_rest
    )
    return f_lat + f_xb + f_titin


def _apply_augmented_preconditioner(
    prefactored: PreFactoredPreconditioner,
    d_block_inv: float,
    v: jnp.ndarray,
    n_thick: int,
    n_crowns: int,
    n_thin: int,
    n_sites: int,
) -> jnp.ndarray:
    """Block-diagonal preconditioner for the augmented (n+1)-dim system.

    v[:-1] -> Thomas solver for axial block
    v[-1]  -> d_block_inv * v[-1]  (exact Jacobian diagonal inverse for d block)
    """
    v_axial = v[:-1]
    v_d = v[-1]
    x_axial = apply_preconditioner(prefactored, v_axial, n_thick, n_crowns, n_thin, n_sites)
    x_d = d_block_inv * v_d
    return jnp.concatenate([x_axial, jnp.array([x_d])])


def _augmented_residual_fn(
    pos_aug: jnp.ndarray,
    thick_k: float,
    thin_k: float,
    z_line: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params,
    topology,
    K_lat: float,
    d_ref: float,
    n_thick: int,
    n_crowns: int,
    n_thin: int,
    n_sites: int,
) -> jnp.ndarray:
    """Augmented (n+1)-dim residual: [f_axial, f_radial].

    d = pos_aug[-1] is used as lattice_spacing in compute_forces_vectorized,
    so JAX JVP automatically captures df_axial/dd and df_radial/dpositions.
    """
    d = pos_aug[-1]
    pos = pos_aug[:-1]
    n_thick_nodes = n_thick * n_crowns
    pos_thick = pos[:n_thick_nodes].reshape(n_thick, n_crowns)
    pos_thin = pos[n_thick_nodes:].reshape(n_thin, n_sites)

    f_axial = compute_forces_vectorized(
        pos_thick, pos_thin,
        thick_k, thin_k, z_line, d,
        titin_a, titin_b, titin_rest,
        xb_states, xb_bound_to, params, topology
    )

    f_rad = _radial_residual(
        d, pos_thick, pos_thin, xb_states, xb_bound_to,
        z_line, params, topology,
        titin_a, titin_b, titin_rest,
        K_lat, d_ref,
    )

    return jnp.concatenate([f_axial, jnp.array([f_rad])])


def _newton_solve_dynamic_ls(
    positions_init: jnp.ndarray,
    d_init: float,
    thick_k: float,
    thin_k: float,
    z_line: float,
    titin_a: float,
    titin_b: float,
    titin_rest: float,
    xb_states: jnp.ndarray,
    xb_bound_to: jnp.ndarray,
    params,
    topology,
    K_lat: float,
    d_ref: float,
    prefactored_precond: PreFactoredPreconditioner,
    n_thick: int,
    n_crowns: int,
    n_thin: int,
    n_sites: int,
    n_newton_steps: int = 16,
    n_cg_steps: int = 6,
    tolerance: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int, float]:
    """Newton-Raphson solver for augmented system (positions + lattice spacing).

    Uses while_loop for Newton, Python for-loop for CG (unrolled by XLA).
    Includes d > 1.0 nm projection in the while_loop body.
    """
    n_thick_nodes = n_thick * n_crowns
    pos_thick_init = positions_init[:n_thick_nodes].reshape(n_thick, n_crowns)
    pos_thin_init = positions_init[n_thick_nodes:].reshape(n_thin, n_sites)

    # Exact d-block Jacobian diagonal via scalar autodiff
    J_dd = jax.grad(_radial_residual, argnums=0)(
        d_init, pos_thick_init, pos_thin_init,
        xb_states, xb_bound_to, z_line, params, topology,
        titin_a, titin_b, titin_rest, K_lat, d_ref
    )
    d_block_inv = -1.0 / J_dd

    pos_aug0 = jnp.concatenate([positions_init, jnp.array([d_init])])

    def residual_fn(pos_aug):
        return _augmented_residual_fn(
            pos_aug, thick_k, thin_k, z_line,
            titin_a, titin_b, titin_rest, xb_states, xb_bound_to,
            params, topology, K_lat, d_ref,
            n_thick, n_crowns, n_thin, n_sites,
        )

    precond_mv = lambda v: _apply_augmented_preconditioner(
        prefactored_precond, d_block_inv, v, n_thick, n_crowns, n_thin, n_sites
    )
    tol = tolerance if tolerance is not None else jnp.asarray(MIN_FLOAT32_TOLERANCE)
    _clamp_d = lambda x: x.at[-1].set(jnp.maximum(x[-1], 1.0))
    return _run_newton(residual_fn, precond_mv, pos_aug0, tol, n_newton_steps, n_cg_steps,
                       post_step=_clamp_d)


# ============================================================================
# PUBLIC API
# ============================================================================

def solve_equilibrium(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    K_lat: float = None,
    d_ref: float = None,
    tolerance: float = None,
    n_newton_steps: int = 16,
    n_cg_steps: int = 6,
    precond_params: Optional[PreconditionerParams] = None,
    prefactored_precond: Optional[PreFactoredPreconditioner] = None,
) -> Tuple['State', jnp.ndarray, float, int]:
    """Solve for equilibrium filament positions.

    When K_lat is None: standard n-DOF axial solve (fixed lattice spacing).
    When K_lat > 0: augmented (n+1)-DOF solve with lattice spacing d as an
    extra unknown, finding radial force balance alongside axial equilibrium.

    Uses jax.lax.while_loop — body traced once, exits at convergence or cap.
    Typically converges in 1-2 Newton iterations.

    Args:
        state: Current State NamedTuple
        constants: DynamicParams with physics values. constants.lattice_spacing
                   is used as the initial d guess in dynamic LS mode.
        topology: SarcTopology with structural index maps
        K_lat: Effective lattice stiffness (pN/nm), already scaled by n_thick.
               None = fixed LS mode.
        d_ref: Poisson-scaled reference spacing (nm). Required if K_lat is not None.
        tolerance: Convergence tolerance (pN). None -> constants.solver_tol,
                   floored at thick_k × 1e-4 (float32 precision limit).
        n_newton_steps: Hard cap on Newton iterations (default 16)
        n_cg_steps: CG iterations per Newton step. Default 6; 0 = Richardson
                   (no JVP — diverges with attached XBs, see CLAUDE.md DO NOT).
        precond_params: Pre-built PreconditionerParams (optional, avoids rebuild per step)
        prefactored_precond: Pre-factored Thomas data (optional, avoids re-factoring per step)

    Returns:
        (new_state, residual_scalar, new_lattice_spacing, n_iters)
        new_lattice_spacing = solved d (dynamic) or constants.lattice_spacing (fixed)
    """
    thick_axial = state.thick.axial
    thin_axial = state.thin.axial
    n_thick, n_crowns = thick_axial.shape
    n_thin, n_sites = thin_axial.shape
    n_thick_nodes = n_thick * n_crowns

    # Raise the requested tolerance to the float32 precision floor if needed.
    # See MIN_FLOAT32_TOLERANCE: the floor scales as thick_k * 1e-4, which is
    # ~0.75 pN at the default thick_k = 7500. Asking for less is unachievable
    # and merely makes the Newton loop run to its iteration cap.
    if tolerance is None:
        tolerance = constants.solver_tol
    float32_floor = constants.thick_k * jnp.asarray(1e-4)
    tolerance = jnp.maximum(jnp.asarray(tolerance),
                            jnp.maximum(float32_floor, jnp.asarray(MIN_FLOAT32_TOLERANCE)))

    if precond_params is None:
        from multifil_jax.core.state import build_preconditioner_params
        precond_params = build_preconditioner_params(
            n_thick, n_crowns, n_thin, n_sites,
            constants.thick_k, constants.thin_k,
        )

    positions_init = jnp.concatenate([thick_axial.flatten(), thin_axial.flatten()])

    if K_lat is None:
        # Fixed LS: standard n-DOF solve
        positions_final, n_iters, final_residual = _newton_solve(
            positions_init,
            constants.thick_k, constants.thin_k,
            constants.z_line, constants.lattice_spacing,
            constants.titin_a, constants.titin_b, constants.titin_rest,
            state.thick.xb_states, state.thick.xb_bound_to,
            constants, precond_params, topology,
            n_thick, n_crowns, n_thin, n_sites,
            n_newton_steps, n_cg_steps,
            tolerance=tolerance,
            prefactored_precond=prefactored_precond,
        )
        new_positions = positions_final
        new_lattice_spacing = constants.lattice_spacing
    else:
        # Dynamic LS: augmented (n+1)-DOF solve
        if prefactored_precond is None:
            prefactored_precond = build_prefactored_preconditioner(precond_params)
        pos_aug_final, n_iters, final_residual = _newton_solve_dynamic_ls(
            positions_init, constants.lattice_spacing,
            constants.thick_k, constants.thin_k,
            constants.z_line,
            constants.titin_a, constants.titin_b, constants.titin_rest,
            state.thick.xb_states, state.thick.xb_bound_to,
            constants, topology,
            K_lat, d_ref,
            prefactored_precond,
            n_thick, n_crowns, n_thin, n_sites,
            n_newton_steps, n_cg_steps,
            tolerance=tolerance,
        )
        new_positions = pos_aug_final[:-1]
        new_lattice_spacing = pos_aug_final[-1]

    new_thick_axial = new_positions[:n_thick_nodes].reshape(n_thick, n_crowns)
    new_thin_axial = new_positions[n_thick_nodes:].reshape(n_thin, n_sites)
    new_state = state._replace(
        thick=state.thick._replace(axial=new_thick_axial),
        thin=state.thin._replace(axial=new_thin_axial),
    )
    return new_state, final_residual, new_lattice_spacing, n_iters
