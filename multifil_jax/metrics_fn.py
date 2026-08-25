"""
Per-timestep measurements of the simulation.

Everything a run reports comes from here. compute_all_metrics() is called once
per timestep inside the scan, comparing the state before and after the step, and
returns a fixed set of scalars that are stacked over time into the arrays a user
finally sees.

WHY ALL METRICS, ALWAYS. The returned dict has the same keys on every call, no
matter the configuration. This is not laziness: JAX requires the scan body to
return an identical pytree structure each iteration, and making the metric set
configurable would put it in the JIT signature, so selecting different metrics
would trigger a recompilation. Computing all of them is much cheaper than that
would be — they are reductions over arrays already in registers, next to a
Newton solve that dominates the step.

WHAT THESE MEASURE, AND WHAT THEY DO NOT
----------------------------------------
Force (`axial_force`) is read from the strain in the first backbone spring, not
summed over crossbridges — see kernels/forces.axial_force_at_mline. It includes
titin, which at long sarcomere lengths can exceed the active contribution
entirely. Subtract a relaxed (pCa 9) baseline before interpreting active force.

Occupancy metrics come in two flavours. The plain `frac_tm_*` fractions average
over every site on the filament; the `*_overlap` variants average only over
sites a crossbridge could reach. Prefer the latter when comparing across
geometries — see compute_overlap_tm_fractions() for why the difference bites.

ATP consumption is reported two ways, which will not agree exactly, and should
not:

    atp_expected_p   the expected number, from transition probabilities. Smooth,
                     and correctly counts heads that passed through detachment
                     and out again within one timestep. PREFER THIS.
    atp_consumed     a stochastic count of heads observed to detach this step.
                     Noisy, and it undercounts multi-hop traversals (measured
                     ~5% low at dt = 1 ms against a dt = 0.1 ms reference).

Use atp_expected_p for rates and efficiencies, atp_consumed only when you
specifically want realised events.

NOT EVERY DETACHMENT COSTS AN ATP. A strongly bound head can back down the cycle,
3 -> 2 -> 1 -> 0, without ever reaching Free_2, and pay nothing. That route is
strain-gated at 2 -> 1 (see xb_rate_21) — it is how a badly-positioned head gives
up rather than completing a cycle it cannot afford. xb_tear_expected counts it.
atp_expected_p already excludes it, so the two are disjoint and their sum is
total detachment of strongly bound heads. Measured on the cardiac preset, tearing
is ~0.1% of detachments isometrically but 14-19% during imposed lengthening, so
it is negligible for isometric work and emphatically not for work loops.

(A third metric, atp_expected_q, was removed in Session 108. It capped each
head's detachment rate at its zero-load value, which silently encoded a DIFFERENT
model — "load-accelerated detachment is mechanical, not ATP-driven" — that this
model does not hold: xb_rate_34 is a slip bond where load accelerates the same
ATP-consuming step. Its documented use as a timestep-adequacy check was also
wrong; the p/q gap tracked load, not dt.)

Usage:
    from multifil_jax.metrics_fn import compute_all_metrics
    metrics = compute_all_metrics(old_state, new_state, constants, drivers,
                                  topology, pre_solve_thick_pos, force,
                                  solver_residual, newton_iters, dt)
"""

import jax
import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

from multifil_jax.kernels.forces import axial_force_at_mline, xb_axial_force_by_state
from multifil_jax.kernels.transitions import xb_exit_probabilities
from multifil_jax.core.state import Drivers, resolve_value, MetricsDict

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


def compute_overlap_tm_fractions(
    state: 'State',
    topology: 'SarcTopology',
) -> Dict[str, jnp.ndarray]:
    """Tropomyosin activation restricted to sites a crossbridge could reach.

    THE PROBLEM THIS SOLVES. The plain `frac_tm_state_3` metric averages over
    every tropomyosin site on every thin filament. Many of those sites can never
    host a crossbridge no matter how activated they are:

      - sites opposite the thick filament's bare zone, where there are no crowns
      - sites beyond the tip of the thick filament, past the end of the overlap
      - sites that have slid past the M-line into the other half-sarcomere

    Including them dilutes the average with a constant, and worse, the size of
    that dilution depends on filament lengths and sarcomere length. So the
    all-site metric changes when the geometry changes even if activation itself
    is unaltered. This is not hypothetical: a filament-length correction once
    moved the all-site fraction from 13.5% to 17.3% almost entirely through the
    denominator, while the genuinely reachable fraction barely moved
    (17.9% to 18.2%).

    PREFER THESE VARIANTS whenever comparing across geometries, filament
    lengths, or species. The all-site versions remain available and are fine for
    tracking a single configuration over time.

    OVERLAP ZONE DEFINITION. A site counts if it satisfies all of:
      - at or beyond `crown_offsets.min() - 13.0` (the M-line end of the crown
        span, extended by the myosin head's reach)
      - at or before `crown_offsets.max() + 13.0` (the tip end, same reach)
      - strictly past the M-line, `thin_axial > 0`

    The 13.0 nm is the same head reach used in kernels/geometry.py when
    searching for binding partners. The two must agree: if this bound were more
    generous than the search, the denominator would include sites no head can
    actually reach.

    Note the bounds are computed from `crown_offsets`, which is per-filament, so
    they automatically follow a myosin superlattice that shifts different
    filaments to different axial positions.

    Returns a dict with:
      - frac_tm_state_2_overlap: Ca-open fraction. Responds earlier than
        state 3 and is the more direct readout of cooperative propagation.
      - frac_tm_state_3_overlap: fully open, crossbridge-bindable fraction.
      - frac_tm_available_overlap: states 2 and 3 combined.
      - n_overlap_sites: the denominator, worth checking when a result surprises
        you — it should change with sarcomere length and filament geometry, and
        should NOT change with calcium.
    """
    tm_states = state.thin.tm_states
    thin_axial = state.thin.axial

    near_bound = topology.crown_offsets.min() - 13.0
    far_bound = topology.crown_offsets.max() + 13.0

    in_reach = (thin_axial >= near_bound) & (thin_axial <= far_bound)
    visible = thin_axial > 0.0
    overlap_mask = in_reach & visible

    n_overlap_sites = jnp.sum(overlap_mask).astype(jnp.float32)
    is_state_2 = (tm_states == 2) & overlap_mask
    is_state_3 = (tm_states == 3) & overlap_mask

    n_state_2 = jnp.sum(is_state_2).astype(jnp.float32)
    n_state_3 = jnp.sum(is_state_3).astype(jnp.float32)

    return {
        'frac_tm_state_2_overlap': n_state_2 / n_overlap_sites,
        'frac_tm_state_3_overlap': n_state_3 / n_overlap_sites,
        'frac_tm_available_overlap': (n_state_2 + n_state_3) / n_overlap_sites,
        'n_overlap_sites': n_overlap_sites,
    }


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
    xb_subpop=None,
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

    f_xb_loose, f_xb_tight_1, f_xb_tight_2 = xb_axial_force_by_state(
        new_state.thick.axial, new_state.thin.axial, new_xb,
        new_state.thick.xb_bound_to,
        resolve_value(drivers.lattice_spacing, constants.lattice_spacing),
        constants, topology)
    n_total_tm = jnp.float32(jnp.size(new_tm))

    # Resolve driver values
    z_line = resolve_value(drivers.z_line, constants.z_line)
    pCa_val = resolve_value(drivers.pCa, constants.pCa)
    lattice_spacing = resolve_value(drivers.lattice_spacing, constants.lattice_spacing)

    # ========================================================================
    # CROSSBRIDGE STATE COUNTS
    # ========================================================================
    n_drx = jnp.sum(new_xb == 0).astype(jnp.float32)
    n_loose = jnp.sum(new_xb == 1).astype(jnp.float32)
    n_tight_1 = jnp.sum(new_xb == 2).astype(jnp.float32)
    n_tight_2 = jnp.sum(new_xb == 3).astype(jnp.float32)
    n_free_2 = jnp.sum(new_xb == 4).astype(jnp.float32)
    n_srx = jnp.sum(new_xb == 5).astype(jnp.float32)
    n_bound = jnp.sum((new_xb >= 1) & (new_xb <= 3)).astype(jnp.float32)

    # ========================================================================
    # TROPOMYOSIN STATE COUNTS
    # ========================================================================
    n_tm_0 = jnp.sum(new_tm == 0).astype(jnp.float32)
    n_tm_1 = jnp.sum(new_tm == 1).astype(jnp.float32)
    n_tm_2 = jnp.sum(new_tm == 2).astype(jnp.float32)
    n_tm_3 = jnp.sum(new_tm == 3).astype(jnp.float32)
    actin_permissiveness = jnp.mean((new_state.thin.tm_states == 3).astype(jnp.float32))
    overlap_tm_fractions = compute_overlap_tm_fractions(new_state, topology)

    # ========================================================================
    # TRANSITION EVENT COUNTS
    # ========================================================================
    # Count XBs that visited state 4 (Free_2) this timestep, including those that continued
    # to state 0 (4→4→0) within the same timestep. State 3→2→1→0 reversal also
    # lands in state 0 but is negligibly rare compared to the 3→4→0 path.
    atp_consumed = jnp.sum((old_xb == 3) & ((new_xb == 4) | (new_xb == 0))).astype(jnp.float32)
    newly_bound = jnp.sum((old_xb == 0) & (new_xb == 1)).astype(jnp.float32)

    # ========================================================================
    # DISPLACEMENT STATISTICS
    # ========================================================================
    thick_axial = new_state.thick.axial
    thin_axial = new_state.thin.axial

    thick_rest_positions = topology.crown_offsets
    thick_displacement = thick_axial - thick_rest_positions
    thick_displace_flat = thick_displacement.flatten()

    thin_rest_positions = jnp.cumsum(topology.binding_rests, axis=1)
    thin_displacement = thin_axial - thin_rest_positions
    thin_displace_flat = thin_displacement.flatten()

    # ========================================================================
    # ENERGY METRICS
    # ========================================================================
    k_thick = constants.thick_k
    L0_thick = topology.crown_offsets[:, 0]
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
    if xb_subpop is None:
        xb_subpop_r = None
    else:
        _mode, _constants_k, _extra = xb_subpop
        xb_subpop_r = (_mode,
                       [ck.with_drivers(pCa_val, z_line, lattice_spacing) for ck in _constants_k],
                       _extra)
    P_abs_all = xb_exit_probabilities(
        old_state, resolved_constants, topology, dt, xb_subpop=xb_subpop_r
    )

    old_xb_flat = old_xb.reshape(-1)
    mask_state3 = (old_xb_flat == 3).astype(jnp.float32)

    # Expected ATP, from transition probabilities.
    #
    # The naive quantity, P[3,4], is the probability of ENDING the step in
    # Free_2, which undercounts: a head can go 3 -> 4 -> 0 within one timestep,
    # spending an ATP but ending where a before/after comparison sees no
    # detachment at all. P_abs comes from a generator with states 4 and 0 made
    # absorbing, so P_abs[3,4] is the probability of VISITING Free_2 at any
    # point during the step — the quantity actually wanted. The error grows with
    # dt and with detachment rate, so it is not negligible at the fast rates of
    # skeletal myosin.
    atp_expected_p = jnp.sum(mask_state3 * P_abs_all[:, 3, 4])

    # Expected NON-ATP detachment ("tearing"), from the same absorbing matrix.
    #
    # A strongly bound head has a second way out: back down the cycle,
    # 3 -> 2 -> 1 -> 0, without ever reaching Free_2 and so without spending an
    # ATP. That route is strain-gated at the 2 -> 1 step (see xb_rate_21), which
    # is how a badly-positioned head gives up rather than completing a cycle it
    # cannot afford. Because state 0 is absorbing in the same generator that
    # traps state 4, P_abs[i, 0] is the probability of reaching DRX WITHOUT
    # passing through Free_2 — mutually exclusive with the ATP route above, and
    # free of any extra matrix exponential.
    #
    # Restricted to the strongly bound states 2 and 3 on purpose: a state-1
    # (Loose) head falling off is an ordinary failed weak attachment, not a
    # load-driven tear, and pooling them would obscure both.
    mask_strong = ((old_xb_flat == 2) | (old_xb_flat == 3)).astype(jnp.float32)
    xb_tear_expected = jnp.sum(
        mask_strong * jnp.take_along_axis(
            P_abs_all[:, :, 0], old_xb_flat[:, None].astype(jnp.int32), axis=1
        )[:, 0]
    )

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

        # Axial XB force split by bound state (pN).  Divide by the matching
        # count above for mean force per head in that state: that is what
        # separates "more heads are strong" from "each strong head pulls
        # harder".  These sum to the total XB force on the thick filaments,
        # which is NOT 'axial_force' -- see the module docstring.
        'force_xb_loose': f_xb_loose,
        'force_xb_tight_1': f_xb_tight_1,
        'force_xb_tight_2': f_xb_tight_2,

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
        'frac_tm_state_2_overlap': overlap_tm_fractions['frac_tm_state_2_overlap'],
        'frac_tm_state_3_overlap': overlap_tm_fractions['frac_tm_state_3_overlap'],
        'frac_tm_available_overlap': overlap_tm_fractions['frac_tm_available_overlap'],
        'n_overlap_sites': overlap_tm_fractions['n_overlap_sites'],

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
        'xb_tear_expected': xb_tear_expected,
        'work_per_atp': work_per_atp,

        # Solver diagnostics
        'newton_iters': newton_iters,
    })
