"""
Equilibrium Solver for multifil_jax.

Solves for filament positions where all forces balance (F(x) = 0) using
Newton-Raphson with CG and block-tridiagonal preconditioning.

Loop Architecture:
- Newton: jax.lax.while_loop (body traced once; exits at convergence or cap)
- CG and Thomas solve: Python for loops (XLA unrolls at trace time)
- thomas_factor stays as Python for loop (called once before scan, not in hot path)

Performance:
- CG uses 1 JVP per iteration vs BiCGStab's 2
- Baked-in negation avoids per-iteration sign flip overhead
- Python loop unrolling enables full XLA fusion (no WhileOp barriers)

Preconditioner Strategy:
-----------------------
Static preconditioner (passive springs only, no titin) is OPTIMAL because:
1. Titin stiffness is negligible at normal conditions (<0.01% of passive)
2. At extreme stretch, titin exp(b*stretch) overflows and breaks convergence
3. Static preconditioner handles all conditions including stiff titin

Usage:
    from multifil_jax.kernels.solver import solve_equilibrium
    state, info = solve_equilibrium(state, constants, topology)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, Union, NamedTuple, TYPE_CHECKING
from functools import partial

from multifil_jax.kernels.forces import compute_forces_vectorized
from multifil_jax.core.state import PreconditionerParams

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams

# Minimum tolerance achievable with float32 arithmetic at default spring constants.
# Float32 position epsilon at typical filament positions (~900 nm) ≈ 1e-4 nm.
# Force floor = thick_k × 1e-4 nm. At default thick_k=2020 pN/nm: floor ≈ 0.20 pN.
# For stiffer springs this floor scales proportionally — see solve_equilibrium().
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
    """Forward sweep of Thomas algorithm (LU factorization of tridiagonal matrix).

    Uses Python for loop (unrolled at trace time) so XLA can fuse the entire
    sweep — no WhileOp fusion barriers from lax.scan.

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
    """Back-substitution using pre-factored Thomas data.

    Uses Python for loops (unrolled at trace time) so XLA can fuse the entire
    solve — no WhileOp fusion barriers from fori_loop.

    Note: fori_loop was tried (Session 3→4) and reduced compile time but caused
    a ~20x runtime regression (5 ms/step → 105 ms/step) because XLA cannot
    fuse memory operations across fori_loop boundaries.

    Args:
        factors: Pre-computed ThomasFactors from thomas_factor()
        rhs: (n,) right-hand side vector

    Returns:
        x: (n,) solution vector
    """
    n = rhs.shape[0]
    inv_diag, upper, multipliers = factors.inv_diag, factors.upper, factors.multipliers

    # Forward substitution
    y = [rhs[0]]
    for i in range(1, n):
        y.append(rhs[i] - multipliers[i - 1] * y[i - 1])
    y = jnp.stack(y)

    # Back substitution
    x_rev = [y[n - 1] * inv_diag[n - 1]]
    for i in range(n - 2, -1, -1):
        x_rev.append((y[i] - upper[i] * x_rev[-1]) * inv_diag[i])
    x = jnp.stack(x_rev[::-1])

    return x


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


# ============================================================================
# NEWTON-RAPHSON SOLVER (Python loop unrolled at trace time)
# ============================================================================

@partial(jax.jit, static_argnames=['n_thick', 'n_crowns', 'n_thin', 'n_sites',
                                    'n_newton_steps', 'n_cg_steps'])
def _newton_solve(
    positions_init: jnp.ndarray,
    rests_thick: jnp.ndarray,
    rests_thin: jnp.ndarray,
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
        n_cg_steps: Fixed number of CG iterations per Newton step (default 6).
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
            pos_thick, pos_thin, rests_thick, rests_thin,
            thick_k, thin_k, z_line, lattice_spacing,
            titin_a, titin_b, titin_rest,
            xb_states, xb_bound_to, params, topology
        )

    def neg_jac_mv_at(x, v):
        """Compute (-J) @ v via JVP: -d(residual_fn)/dx @ v."""
        _, jv = jax.jvp(residual_fn, (x,), (v,))
        return -jv

    # --- SETUP PHASE ---
    if prefactored_precond is not None:
        prefactored = prefactored_precond
    else:
        prefactored = build_prefactored_preconditioner(precond_params, negate=True, eps=1e-9)

    precond_mv = lambda v: apply_preconditioner(
        prefactored, v, n_thick, n_crowns, n_thin, n_sites
    )

    # --- ADAPTIVE LOOP: while_loop traced once, runs until convergence or cap ---
    tol = tolerance if tolerance is not None else jnp.asarray(MIN_FLOAT32_TOLERANCE)

    f0 = residual_fn(positions_init)

    def body(carry):
        x, f, i = carry
        neg_jac_mv = lambda v: neg_jac_mv_at(x, v)
        dx = _preconditioned_cg(neg_jac_mv, precond_mv, f, jnp.zeros_like(x), n_cg_steps)
        dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
        x_new = x + dx
        f_new = residual_fn(x_new)
        return x_new, f_new, i + jnp.int32(1)

    def cond(carry):
        _, f, i = carry
        return (jnp.max(jnp.abs(f)) > tol) & (i < n_newton_steps)

    x, f, n_iters = jax.lax.while_loop(cond, body, (positions_init, f0, jnp.int32(0)))
    final_residual = jnp.max(jnp.abs(f))
    return x, n_iters, final_residual


# ============================================================================
# PUBLIC API
# ============================================================================

def solve_equilibrium(
    state: 'State',
    constants: 'DynamicParams',
    topology: 'SarcTopology',
    tolerance: float = None,
    n_newton_steps: int = 16,
    n_cg_steps: int = 6,
    precond_params: Optional[PreconditionerParams] = None,
    prefactored_precond: Optional[PreFactoredPreconditioner] = None,
) -> Tuple['State', jnp.ndarray]:
    """Solve for equilibrium filament positions.

    Uses jax.lax.while_loop for Newton iterations — body traced once, exits
    when max|f| < tolerance OR n_newton_steps cap is reached. Typically
    converges in 1-2 iterations, making the cap rarely active.

    The preconditioner is built from constants + topology dimensions.
    This function is JIT-compatible and can be called inside jax.lax.scan.

    Args:
        state: State NamedTuple (pure state, no embedded params)
        constants: DynamicParams with physics values (thick_k, thin_k, etc.)
        topology: SarcTopology with structural index maps
        tolerance: Convergence tolerance (pN). If None, uses constants.solver_tol.
                   Automatically floored at thick_k × 1e-4 (float32 precision limit).
        n_newton_steps: Hard cap on Newton iterations (default 16). while_loop
                        exits early when converged.
        n_cg_steps: Fixed number of CG iterations per Newton step (default 6)
        precond_params: Pre-built PreconditionerParams. If None, builds from
                        constants. Pass pre-built params to avoid rebuilding
                        every timestep inside scan loops.
        prefactored_precond: Pre-factored Thomas algorithm data. If provided,
                             skips both preconditioner build AND factorization.
                             Build once before scan loop for maximum reuse.

    Returns:
        final_state: State with equilibrium positions
        final_residual: Scalar max residual force (pN)
    """
    thick_axial = state.thick.axial
    thin_axial = state.thin.axial
    thick_rests = state.thick.rests
    thin_rests = state.thin.rests

    # Resolve tolerance. The float32 force floor scales with spring constants:
    # floor ≈ thick_k × 1e-4 nm (float32 position epsilon at ~900 nm filament positions).
    # At default thick_k=2020: floor ≈ 0.20 pN, dominated by MIN_FLOAT32_TOLERANCE=0.25.
    # At 5× stiff thick_k=10100: floor ≈ 1.01 pN — tolerance must be at least this.
    # Without this scaling the while-loop chases an unreachable target at stiff params.
    if tolerance is None:
        tolerance = constants.solver_tol
    float32_floor = constants.thick_k * jnp.asarray(1e-4)
    tolerance = jnp.maximum(tolerance, jnp.maximum(float32_floor, MIN_FLOAT32_TOLERANCE))

    n_thick, n_crowns = thick_axial.shape
    n_thin, n_sites = thin_axial.shape
    n_thick_nodes = n_thick * n_crowns

    # Build preconditioner from constants if not pre-built
    if precond_params is None:
        from multifil_jax.core.state import build_preconditioner_params
        precond_params = build_preconditioner_params(
            n_thick, n_crowns, n_thin, n_sites,
            constants.thick_k, constants.thin_k,
        )

    positions_init = jnp.concatenate([
        thick_axial.flatten(),
        thin_axial.flatten()
    ])

    positions_final, iterations, final_residual = _newton_solve(
        positions_init,
        thick_rests,
        thin_rests,
        constants.thick_k,
        constants.thin_k,
        constants.z_line,
        constants.lattice_spacing,
        constants.titin_a,
        constants.titin_b,
        constants.titin_rest,
        state.thick.xb_states,
        state.thick.xb_bound_to,
        constants,
        precond_params,
        topology,
        n_thick, n_crowns, n_thin, n_sites,
        n_newton_steps, n_cg_steps,
        tolerance=tolerance,
        prefactored_precond=prefactored_precond,
    )

    # Update positions
    new_thick_axial = positions_final[:n_thick_nodes].reshape(n_thick, n_crowns)
    new_thin_axial = positions_final[n_thick_nodes:].reshape(n_thin, n_sites)

    new_thick = state.thick._replace(axial=new_thick_axial)
    new_thin = state.thin._replace(axial=new_thin_axial)
    final_state = state._replace(thick=new_thick, thin=new_thin)

    return final_state, final_residual
