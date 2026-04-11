"""
Unified Metrics for JAX Half-Sarcomere Simulation

Single function that computes ALL metrics every timestep. Returns a fixed dict with identical keys every call so JAX sees consistent pytree structure.

Usage:
    from multifil_jax.metrics_fn import compute_all_metrics
    metrics = compute_all_metrics(old_state, new_state, constants, drivers,
                                  topology, pre_solve_thick_pos, force, solver_residual, dt)
"""

import jax
import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

from multifil_jax.kernels.forces import axial_force_at_mline
from multifil_jax.kernels.transitions import compute_xb_transition_matrices
from multifil_jax.core.state import Drivers, resolve_value, MetricsDict

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


def compute_all_metrics(
    old_state: 'State',
    new_state: 'State',
    constants: 'DynamicParams',
    drivers: Drivers,
    topology: 'SarcTopology',
    pre_solve_thick_pos: jnp.ndarray,
    force: jnp.ndarray,
    solver_residual: jnp.ndarray,
    newton_iters,
    dt: float,
) -> 'MetricsDict':
    """Compute all metrics for a single timestep.

    Returns a fixed MetricsDict (same keys every call) so JAX sees identical
    pytree structure — no recompilation from different metric selections.

    Args:
        old_state: State BEFORE timestep
        new_state: State AFTER timestep (equilibrium solved)
        constants: DynamicParams with resolved physics values
        drivers: Drivers NamedTuple with per-step pCa/z_line/ls
        topology: SarcTopology for structural lookups
        pre_solve_thick_pos: (n_thick, n_crowns) positions before equilibrium solve
        force: Scalar M-line force (already computed)
        solver_residual: Scalar equilibrium solver residual (pN)
        newton_iters: Number of Newton iterations used by solver
        dt: Timestep size (ms)

    Returns:
        MetricsDict with all metric values (supports both dict and attribute access)
    """
    old_xb = old_state.thick.xb_states
    new_xb = new_state.thick.xb_states
    old_tm = old_state.thin.tm_states
    new_tm = new_state.thin.tm_states

    n_total_xb = jnp.float32(jnp.size(new_xb))
    n_total_tm = jnp.float32(jnp.size(new_tm))

    # Resolve driver values
    z_line = resolve_value(drivers.z_line, constants.z_line)
    pCa_val = resolve_value(drivers.pCa, constants.pCa)
    lattice_spacing = resolve_value(drivers.lattice_spacing, constants.lattice_spacing)

    # ========================================================================
    # CROSSBRIDGE STATE COUNTS
    # ========================================================================
    n_drx = jnp.sum(new_xb == 1).astype(jnp.float32)
    n_loose = jnp.sum(new_xb == 2).astype(jnp.float32)
    n_tight_1 = jnp.sum(new_xb == 3).astype(jnp.float32)
    n_tight_2 = jnp.sum(new_xb == 4).astype(jnp.float32)
    n_free_2 = jnp.sum(new_xb == 5).astype(jnp.float32)
    n_srx = jnp.sum(new_xb == 6).astype(jnp.float32)
    n_bound = jnp.sum((new_xb >= 2) & (new_xb <= 4)).astype(jnp.float32)

    # ========================================================================
    # TROPOMYOSIN STATE COUNTS
    # ========================================================================
    n_tm_0 = jnp.sum(new_tm == 0).astype(jnp.float32)
    n_tm_1 = jnp.sum(new_tm == 1).astype(jnp.float32)
    n_tm_2 = jnp.sum(new_tm == 2).astype(jnp.float32)
    n_tm_3 = jnp.sum(new_tm == 3).astype(jnp.float32)
    actin_permissiveness = jnp.mean((new_state.thin.tm_states == 3).astype(jnp.float32))

    # ========================================================================
    # TRANSITION EVENT COUNTS
    # ========================================================================
    # Count XBs that visited state 5 this timestep, including those that continued
    # to state 1 (4→5→1) within the same timestep. State 4→3→2→1 reversal also
    # lands in state 1 but is negligibly rare compared to the 4→5→1 path.
    atp_consumed = jnp.sum((old_xb == 4) & ((new_xb == 5) | (new_xb == 1))).astype(jnp.float32)
    newly_bound = jnp.sum((old_xb == 1) & (new_xb == 2)).astype(jnp.float32)

    # ========================================================================
    # DISPLACEMENT STATISTICS
    # ========================================================================
    thick_axial = new_state.thick.axial
    thin_axial = new_state.thin.axial

    thick_rest_positions = topology.crown_offsets[None, :]
    thick_displacement = thick_axial - thick_rest_positions
    thick_displace_flat = thick_displacement.flatten()

    thin_rest_positions = jnp.cumsum(topology.binding_rests, axis=1)
    thin_displacement = thin_axial - thin_rest_positions
    thin_displace_flat = thin_displacement.flatten()

    # ========================================================================
    # ENERGY METRICS
    # ========================================================================
    k_thick = constants.thick_k
    L0_thick = topology.crown_offsets[0]
    x1 = new_state.thick.axial[:, 0]
    thick_energy_first = 0.5 * k_thick * (x1 - L0_thick)**2
    thick_energy_first_avg = jnp.mean(thick_energy_first)

    x1_old = old_state.thick.axial[:, 0]
    thick_energy_first_old = 0.5 * k_thick * (x1_old - L0_thick)**2
    thick_energy_first_delta_avg = jnp.mean(thick_energy_first - thick_energy_first_old)

    # Titin energy
    a_tit = constants.titin_a
    b_tit = constants.titin_b
    L0_tit = constants.titin_rest
    thick_tip_new = new_state.thick.axial[:, -1]
    axial_dist_new = z_line - thick_tip_new
    titin_length_new = jnp.sqrt(axial_dist_new**2 + lattice_spacing**2)
    extension_new = titin_length_new - L0_tit
    titin_energy_new = (a_tit / b_tit) * (jnp.exp(b_tit * extension_new) - 1.0)
    titin_energy_avg = jnp.mean(titin_energy_new)

    thick_tip_old = old_state.thick.axial[:, -1]
    axial_dist_old = z_line - thick_tip_old
    titin_length_old = jnp.sqrt(axial_dist_old**2 + lattice_spacing**2)
    extension_old = titin_length_old - L0_tit
    titin_energy_old = (a_tit / b_tit) * (jnp.exp(b_tit * extension_old) - 1.0)
    titin_energy_delta_avg = jnp.mean(titin_energy_new - titin_energy_old)

    # ========================================================================
    # WORK METRICS
    # ========================================================================
    post_pos = new_state.thick.axial
    dx = post_pos - pre_solve_thick_pos
    work_thick = force * jnp.mean(dx)
    n_thick, n_crowns = post_pos.shape
    work_thick_mean = work_thick / jnp.float32(n_thick * n_crowns)

    # ========================================================================
    # ATP EXPECTED (P-matrix method) — recompute per-XB P via shared helper
    # ========================================================================
    # Use resolved constants (same as timestep.py passed to thick_transitions)
    # so Q/P matrices match what actually drove the transitions this step.
    resolved_constants = constants.with_drivers(pCa_val, z_line, lattice_spacing)
    Q_all, P_all, P_abs_all = compute_xb_transition_matrices(
        old_state, resolved_constants, topology, dt
    )

    old_xb_flat = old_xb.reshape(-1)
    mask_state4 = (old_xb_flat == 4).astype(jnp.float32)

    # Absorbing-state trick: P_abs_all has row 4 zeroed so state 5 cannot exit.
    # P_abs_all[:,3,4] gives the probability of visiting state 5 at any point in
    # [0, dt], correctly counting 4→5→1 paths. Pre-computed in compute_xb_transition_matrices.
    atp_expected_p = jnp.sum(mask_state4 * P_abs_all[:, 3, 4])

    # Q-matrix branching ratio method
    k_total = Q_all[:, 3, 4]  # rate 4->5 per XB
    # Minimum r45 rate at rest geometry: f_3_4=0 → Bell exp(0)=1 → r45_min = A45.
    # Pure function of constants — recalculated each call, trivially fast.
    k_min = constants.xb_r45_coeff
    ratio = jnp.where(k_total > 1e-10, k_min / k_total, 1.0)
    ratio = jnp.clip(ratio, 0.0, 1.0)  # clip handles XBs doing positive work (k_total < k_min possible)
    atp_expected_q = jnp.sum(mask_state4 * k_total * dt * ratio)

    # Work per ATP
    work_per_atp = jnp.where(atp_expected_p > 0.01, work_thick / atp_expected_p, 0.0)

    # ========================================================================
    # ASSEMBLE RESULT DICT (fixed keys — same pytree every call)
    # ========================================================================
    return MetricsDict({
        # Driver / protocol values
        'axial_force': force,
        'solver_residual': solver_residual,
        'z_line': z_line,
        'pCa': pCa_val,
        'lattice_spacing': lattice_spacing,

        # Crossbridge state counts
        'n_bound': n_bound,
        'n_xb_drx': n_drx,
        'n_xb_loose': n_loose,
        'n_xb_tight_1': n_tight_1,
        'n_xb_tight_2': n_tight_2,
        'n_xb_free_2': n_free_2,
        'n_xb_srx': n_srx,

        # Crossbridge state fractions
        'frac_xb_bound': n_bound / n_total_xb,
        'frac_xb_drx': n_drx / n_total_xb,
        'frac_xb_loose': n_loose / n_total_xb,
        'frac_xb_tight_1': n_tight_1 / n_total_xb,
        'frac_xb_tight_2': n_tight_2 / n_total_xb,
        'frac_xb_free_2': n_free_2 / n_total_xb,
        'frac_xb_srx': n_srx / n_total_xb,

        # TM state counts
        'n_tm_state_0': n_tm_0,
        'n_tm_state_1': n_tm_1,
        'n_tm_state_2': n_tm_2,
        'n_tm_state_3': n_tm_3,

        # TM state fractions
        'frac_tm_state_0': n_tm_0 / n_total_tm,
        'frac_tm_state_1': n_tm_1 / n_total_tm,
        'frac_tm_state_2': n_tm_2 / n_total_tm,
        'frac_tm_state_3': n_tm_3 / n_total_tm,
        'actin_permissiveness': actin_permissiveness,

        # Transition events
        'atp_consumed': atp_consumed,
        'newly_bound': newly_bound,

        # Displacement statistics
        'thick_displace_mean': jnp.mean(thick_displace_flat),
        'thick_displace_max': jnp.max(thick_displace_flat),
        'thick_displace_min': jnp.min(thick_displace_flat),
        'thick_displace_std': jnp.std(thick_displace_flat),
        'thin_displace_mean': jnp.mean(thin_displace_flat),
        'thin_displace_max': jnp.max(thin_displace_flat),
        'thin_displace_min': jnp.min(thin_displace_flat),
        'thin_displace_std': jnp.std(thin_displace_flat),

        # Energy metrics
        'thick_energy_first_avg': thick_energy_first_avg,
        'thick_energy_first_delta_avg': thick_energy_first_delta_avg,
        'titin_energy_avg': titin_energy_avg,
        'titin_energy_delta_avg': titin_energy_delta_avg,

        # Work metrics
        'work_thick': work_thick,
        'work_thick_mean': work_thick_mean,

        # ATP expected metrics
        'atp_expected_p': atp_expected_p,
        'atp_expected_q': atp_expected_q,
        'work_per_atp': work_per_atp,

        # Solver diagnostics
        'newton_iters': newton_iters,
    })
