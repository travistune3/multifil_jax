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
    state, residual, iters = solve_equilibrium(state, constants, topology)
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

    Uses jax.lax.associative_scan for both forward and back substitution.
    Both passes are linear recurrences of the form  state[i] = a[i]*state[i-1] + b[i],
    which are associative under  (a2,b2) ∘ (a1,b1) = (a2*a1, a2*b1+b2).

    Compared to Python for-loops:
      - ~5× fewer jaxpr equations → ~23× faster JIT compile
      - ~20% faster runtime on GPU (parallel O(log n) vs sequential O(n))
      - Numerical error ~1e-4 relative, acceptable since preconditioner
        is already an approximation (κ=1.04 → M is ~4% off from J)

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
    """Newton while_loop kernel shared by fixed-LS and dynamic-LS solvers.

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
            # Richardson iteration: dx = M^{-1} f (no JVP).
            # Only converges well when preconditioner M ≈ J (detached XBs).
            # Use n_cg_steps>=1 for systems with many attached crossbridges.
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
    n_cg_steps: int = 1,
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
        n_cg_steps: Fixed number of CG iterations per Newton step (default 1; 0=Richardson).
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
    n_cg_steps: int = 1,
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
    n_cg_steps: int = 1,
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
        n_cg_steps: CG iterations per Newton step. 0 = Richardson (default, no JVP).
                   Set >0 to use CG with exact Jacobian-vector products.
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

    # Resolve tolerance with float32 precision floor.
    # floor ≈ thick_k × 1e-4 (float32 position epsilon at ~900 nm).
    # At default thick_k=2020: floor ≈ 0.20 pN. At 5× stiff: floor ≈ 1.01 pN.
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
