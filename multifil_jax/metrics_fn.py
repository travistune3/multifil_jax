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

Occupancy is exported as COUNTS (`n_xb_*`, `n_tm_state_*`) plus the two
`*_overlap` FRACTIONS. The all-site fractions are not exported because they are
the counts over a constant — divide by `topology.total_xbs` or by
`topology.n_thin * topology.n_tm`. The `*_overlap` variants cannot be recovered that way: they
average only over sites a crossbridge could reach, and their denominator moves
with the geometry. Prefer them when comparing across geometries — see
compute_overlap_tm_fractions() for why the difference bites.

ATP consumption is reported two ways, which will not agree exactly, and should
not:

    atp_expected   the EXACT expected number of ATP-consuming crossings,
                     q34 * INT_0^dt exp(Qt)dt summed over every head, PLUS the
                     realised strong closure tears. Smooth, and it counts a head
                     that passed through detachment and out again — or twice —
                     within one timestep. PREFER THIS.
    atp_consumed     a stochastic count of realised events: heads observed to
                     leave Tight_2 via Free_2 or DRX, plus the same strong
                     closure tears. Noisy, and it undercounts multi-hop
                     traversals badly.

Use atp_expected for rates and efficiencies, atp_consumed only when you
specifically want realised events.

THE ONE APPROXIMATION IN atp_expected is that Q is held constant across the
step; only halving dt tests it. At dt = 1 ms (8x8) the ATP rate is within 0.5%
of dt = 0.5 ms at saturating calcium, but cardiac pCa 6.2 moves by -5.5%. The
3 -> 4 term and the closure-tear term move in opposite directions under dt, so
check them separately, not only their sum.

atp_consumed is low, and by how much depends on the preset (8x8, dt = 1 ms):
about -5% cardiac and -25 to -28% skeletal. Skeletal turns over faster, so more
heads clear two or three stages inside one step and land outside the
(mid == 3) -> {4, 0} window this counter reads. Both are read against the MID
state — the sarcomere as thick_transitions found it — not against the state at
the top of the step; see compute_all_metrics.

THE CYCLE FLUXES ARE EXPORTED TOO, so the ATP books self-verify on any run
on any run. All are exact expected crossing counts per step
(see kernels/transitions.xb_expected_crossings):

    xb_detach_free       N10, a Loose head falling off having never bound ATP
    xb_give_up           N21, and this is NOT A DETACHMENT — see below
    atp_net_pi_release   N12 - N21, which must equal atp_expected at steady
                         state; a gap is a real leak, not a counter artefact
    atp_net_hydrolysis   N40 - N04, a third route touching no state 1/2/3

The residual on either cross-check is a MARTINGALE with sd ~ sqrt(n_4)/sqrt(n_steps),
about 1.6 /ms at 8x8 over 300 ms. A within-one-sigma drift is not a bug.

TWO WAYS OUT OF THE CYCLE COST NOTHING, AND BOTH ARE REPORTED SEPARATELY.

  The give-up route, `xb_give_up`. A strongly bound head can back down the
  cycle, 3 -> 2 -> 1 -> 0, without ever reaching Free_2. The route is
  strain-gated at 2 -> 1 (see xb_rate_21) — it is how a badly-positioned head
  gives up rather than completing a cycle it cannot afford. It refunds a
  phosphate and costs no ATP: 3 -> 2 -> 1 -> 0 is the microscopic reverse of the
  forward path, so charging it would be the opposite error. NOTE WHAT THE METRIC
  IS: the 2 -> 1 crossing count, so the head is still weakly bound afterwards
  and may well climb straight back. It is NOT a detachment count. Measured on
  the cardiac preset the route is ~0.1% of detachments isometrically but 14-19%
  during imposed lengthening — negligible for isometric work and emphatically
  not for work loops, which also makes `atp_consumed` unusable as an ATP figure
  during lengthening.

  A weak closure tear. Tropomyosin closing over a Loose head returns it to DRX
  still primed, owing nothing. `closure_detach_free` counts those.

A STRONG CLOSURE TEAR DOES COST ONE, and is counted by `closure_detach_atp`.
Tropomyosin closing over a Tight_1 or Tight_2 head sends it to Free_2, because
it has already released its phosphate and swung its lever; returning it to DRX
(= M.ADP.Pi) would hand back that phosphate for free.

THE CLOSURE CHARGE IS A REALISED COUNT, not an expectation. The tear is fully
observed and its charge is deterministic given it; an expectation would bill
heads that are still bound. The 3 -> 4 term is an expectation for the opposite
reason: the sampler reports only endpoints, so a head that traverses
3 -> 4 -> 0 inside one step is invisible to any realised count.

Usage:
    from multifil_jax.metrics_fn import compute_all_metrics
    metrics = compute_all_metrics(old_state, new_state, constants, drivers,
                                  topology, force, solver_residual,
                                  newton_iters, dt, trace,
                                  solver_residual_norm, solver_tolerance)
"""

import jax
import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

from multifil_jax.kernels.forces import xb_axial_force_by_state, xb_axial_work
from multifil_jax.kernels.transitions import xb_expected_crossings
from multifil_jax.core.state import (Drivers, MetricsDict,
                                     thick_axial, monomer_axial)

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams


def compute_overlap_tm_fractions(
    state: 'State',
    topology: 'SarcTopology',
    z_line,
) -> Dict[str, jnp.ndarray]:
    """Tropomyosin activation restricted to sites a crossbridge could reach.

    THE PROBLEM THIS SOLVES. An all-site activated fraction — `n_tm_state_3`
    over every tropomyosin site on every thin filament — averages over sites
    that can never host a crossbridge no matter how activated they are:

      - sites opposite the thick filament's bare zone, where there are no crowns
      - sites beyond the tip of the thick filament, past the end of the overlap
      - sites that have slid past the M-line into the other half-sarcomere

    Including them dilutes the average with a constant, and worse, the size of
    that dilution depends on filament lengths and sarcomere length. So the
    all-site metric changes when the geometry changes even if activation itself
    is unaltered: a filament-length change can move the all-site fraction by a
    quarter while the reachable fraction barely moves.

    PREFER THESE VARIANTS whenever comparing across geometries, filament
    lengths, or species. The all-site versions remain available and are fine for
    tracking a single configuration over time.

    OVERLAP ZONE DEFINITION. A site counts if it satisfies all of:
      - at or beyond `crown_offsets.min() - reach` (the M-line end of the crown
        span, extended by the myosin head's reach)
      - at or before `crown_offsets.max() + reach` (the tip end, same reach)
      - strictly past the M-line, absolute position > 0

    `reach` is StaticParams.xb_head_reach, the same value the binding search in
    kernels/geometry.py uses, so the denominator holds only sites a head can reach.

    Note the bounds are computed from `crown_offsets`, which is per-filament, so
    they automatically follow a myosin superlattice that shifts different
    filaments to different axial positions.

    Returns a dict with:
      - frac_tm_state_2_overlap: Ca-open fraction. Responds earlier than
        state 3 and is the more direct readout of cooperative propagation.
      - frac_tm_state_3_overlap: fully open, crossbridge-bindable fraction.
      - n_overlap_sites: the denominator, worth checking when a result surprises
        you — it should change with sarcomere length and filament geometry, and
        should NOT change with calcium.
    """
    tm_states = state.thin.tm_states
    # TRUE positions: both the crown-span window and the M-line gate below are
    # absolute-coordinate tests, so the displacements have to be lifted first.
    # Each unit sits at its representative monomer.
    thin_pos = jnp.take_along_axis(
        monomer_axial(state.thin.displacement, topology, z_line),
        topology.tm_rep_mono, axis=1)

    reach = topology.static.xb_head_reach
    near_bound = topology.crown_offsets.min() - reach
    far_bound = topology.crown_offsets.max() + reach

    in_reach = (thin_pos >= near_bound) & (thin_pos <= far_bound)
    visible = thin_pos > 0.0
    overlap_mask = in_reach & visible

    n_overlap_sites = jnp.sum(overlap_mask).astype(jnp.float32)
    is_state_2 = (tm_states == 2) & overlap_mask
    is_state_3 = (tm_states == 3) & overlap_mask

    n_state_2 = jnp.sum(is_state_2).astype(jnp.float32)
    n_state_3 = jnp.sum(is_state_3).astype(jnp.float32)

    return {
        'frac_tm_state_2_overlap': n_state_2 / n_overlap_sites,
        'frac_tm_state_3_overlap': n_state_3 / n_overlap_sites,
        'n_overlap_sites': n_overlap_sites,
    }


def compute_all_metrics(
    old_state: 'State',
    new_state: 'State',
    constants: 'DynamicParams',
    drivers: Drivers,
    topology: 'SarcTopology',
    force: jnp.ndarray,
    solver_residual: jnp.ndarray,
    newton_iters,
    dt: float,
    trace: 'KineticsTrace',
    solver_residual_norm: jnp.ndarray,
    solver_tolerance: jnp.ndarray,
) -> 'MetricsDict':
    """Compute all metrics for a single timestep.

    Returns a fixed MetricsDict (same keys every call) so JAX sees identical
    pytree structure — no recompilation from different metric selections.

    Args:
        old_state: State BEFORE timestep
        new_state: State AFTER timestep (equilibrium solved)
        constants: DynamicParams with physics values
        drivers: Drivers NamedTuple with this step's pCa/z_line and the
            POST-solve lattice spacing
        topology: SarcTopology for structural lookups
        force: Scalar M-line force (already computed)
        solver_residual: Scalar equilibrium solver residual (pN)
        newton_iters: Number of Newton iterations used by solver
        dt: Timestep size (ms)
        trace: KineticsTrace from kinetics_step — the MID state (post
            thin_transitions, pre thick_transitions), the drivers at the
            PRE-solve lattice spacing, the subpopulation tuple, and the
            closure-tear mask. Required, not
            optional: every realised transition count and every Q-matrix metric
            below reads it, because those are the state and the generator that
            actually drove this step.
        solver_residual_norm: max(|F| / tol_vec) from the solve. Dimensionless,
            <= 1 when converged, and the only such number that is sufficient in
            both lattice-spacing modes.
        solver_tolerance: the axial convergence tolerance used (pN), so the raw
            residual can be read against a physical scale.

    THE TWO DRIVERS DIFFER ON PURPOSE. `drivers` carries the SOLVED lattice
    spacing, and the mechanics metrics must use it — `xb_axial_force_by_state`
    and the reported `lattice_spacing` are post-solve quantities.
    `trace.drivers` carries the PRE-solve spacing, the one the rates were
    evaluated at. Do not "unify" them; in fixed-LS mode they agree anyway, and
    in dynamic-LS mode each is right for its own question. `constants` is the
    one physics object both share.

    Returns:
        MetricsDict with all metric values (supports both dict and attribute access)
    """
    # `old_state` still means "before the timestep" and is used as such below,
    # for the energy and work deltas. It is NOT what the transitions were drawn
    # from: `trace.state` is. Every realised-event count and every Q-matrix
    # metric reads the mid state.
    new_xb = new_state.thick.xb_states
    mid_xb = trace.state.thick.xb_states
    new_tm = new_state.thin.tm_states

    z_line = drivers.z_line
    lattice_spacing = drivers.lattice_spacing
    f_xb_loose, f_xb_tight_1, f_xb_tight_2 = xb_axial_force_by_state(
        thick_axial(new_state.thick.displacement, topology),
        monomer_axial(new_state.thin.displacement, topology, z_line), new_xb,
        new_state.thick.xb_bound_to,
        lattice_spacing,
        constants, topology)

    # ========================================================================
    # CROSSBRIDGE STATE COUNTS
    # ========================================================================
    n_drx = jnp.sum(new_xb == 0).astype(jnp.float32)
    n_loose = jnp.sum(new_xb == 1).astype(jnp.float32)
    n_tight_1 = jnp.sum(new_xb == 2).astype(jnp.float32)
    n_tight_2 = jnp.sum(new_xb == 3).astype(jnp.float32)
    n_free_2 = jnp.sum(new_xb == 4).astype(jnp.float32)
    n_srx = jnp.sum(new_xb == 5).astype(jnp.float32)

    # ========================================================================
    # TROPOMYOSIN STATE COUNTS
    # ========================================================================
    n_tm_0 = jnp.sum(new_tm == 0).astype(jnp.float32)
    n_tm_1 = jnp.sum(new_tm == 1).astype(jnp.float32)
    n_tm_2 = jnp.sum(new_tm == 2).astype(jnp.float32)
    n_tm_3 = jnp.sum(new_tm == 3).astype(jnp.float32)
    overlap_tm_fractions = compute_overlap_tm_fractions(new_state, topology, z_line)

    # ========================================================================
    # TRANSITION EVENT COUNTS
    # ========================================================================
    # READ AGAINST THE MID STATE, NOT old_state. Every transition counted here
    # happens inside thick_transitions, so the "before" endpoint is the state
    # thick_transitions was handed — after update_nearest_neighbors and
    # thin_transitions have run. Reading old_state instead conflates the tear
    # (which happens in thin_transitions) with the cycle.
    #
    # No closure mask is needed and none is applied. A head torn off by
    # tropomyosin is already out of state 3 in `mid_xb`, because the tear
    # happened in thin_transitions, one phase earlier.
    #
    # CLOSURE TEARS, SPLIT BY WHAT THE HEAD HAD ALREADY SPENT. thin_transitions
    # sends a torn Loose head to state 0 (still primed, owes nothing) and a torn
    # Tight_1/Tight_2 head to state 4 Free_2 (post-stroke, owes one ATP), so the
    # state a torn head is in AFTER that call is exactly the split. Nothing else
    # can put a head in state 4 within thin_transitions, so `torn & (mid == 4)`
    # is unambiguous.
    n_tear_strong = jnp.sum(trace.torn & (mid_xb == 4)).astype(jnp.float32)
    n_tear_weak = jnp.sum(trace.torn).astype(jnp.float32) - n_tear_strong

    # Counts XBs that visited state 4 (Free_2) this timestep, including those
    # that continued to state 0 within the same timestep. The 3 -> 2 -> 1 -> 0
    # reversal also lands in state 0 but is negligibly rare next to 3 -> 4 -> 0.
    # Strong closure tears are added on: they are a real turnover booked in
    # thin_transitions, one phase before this comparison window opens.
    atp_consumed = n_tear_strong + jnp.sum(
        (mid_xb == 3) & ((new_xb == 4) | (new_xb == 0))
    ).astype(jnp.float32)
    # Binding only ever happens in thick_transitions, so this too reads the mid
    # state: a head torn off in thin_transitions (3 -> 0) and re-bound in
    # thick_transitions (0 -> 1) is a genuine new attachment.
    newly_bound = jnp.sum((mid_xb == 0) & (new_xb == 1)).astype(jnp.float32)

    # ========================================================================
    # DISPLACEMENT STATISTICS
    # ========================================================================
    # Read straight off the state, which IS a displacement from the frame each
    # filament is built in.
    thick_displace_flat = new_state.thick.displacement.flatten()
    thin_displace_flat = new_state.thin.displacement.flatten()

    # ========================================================================
    # ENERGY METRICS
    # ========================================================================
    k_thick = constants.thick_k
    # The first backbone spring's extension IS the first crown's displacement:
    # crown_offsets[:, 0] is the bare-zone rest length it is measured from.
    u1 = new_state.thick.displacement[:, 0]
    thick_energy_first = 0.5 * k_thick * u1**2
    thick_energy_first_avg = jnp.mean(thick_energy_first)

    # Titin energy
    a_tit = constants.titin_a
    b_tit = constants.titin_b
    L0_tit = constants.titin_rest
    thick_tip_new = thick_axial(new_state.thick.displacement, topology)[:, -1]
    axial_dist_new = z_line - thick_tip_new
    titin_length_new = jnp.sqrt(axial_dist_new**2 + lattice_spacing**2)
    extension_new = titin_length_new - L0_tit
    titin_energy_new = (a_tit / b_tit) * (jnp.exp(b_tit * extension_new) - 1.0)
    titin_energy_avg = jnp.mean(titin_energy_new)

    thick_tip_old = thick_axial(old_state.thick.displacement, topology)[:, -1]
    axial_dist_old = z_line - thick_tip_old
    titin_length_old = jnp.sqrt(axial_dist_old**2 + lattice_spacing**2)
    extension_old = titin_length_old - L0_tit
    titin_energy_old = (a_tit / b_tit) * (jnp.exp(b_tit * extension_old) - 1.0)
    titin_energy_delta_avg = jnp.mean(titin_energy_new - titin_energy_old)

    # ========================================================================
    # WORK METRICS — TWO DIFFERENT QUANTITIES, REPORTED SEPARATELY
    # ========================================================================
    # 1. xb_work_on_filaments: work done ON the lattice BY the crossbridges.
    #    Path-dependent and per-head, so it cannot be reconstructed afterwards
    #    from the returned traces — it has to be computed here. This is the
    #    numerator for an efficiency, because ATP is spent by crossbridges.
    #
    # 2. The EXTERNAL work done by the half-sarcomere at the driven z-line,
    #    -F*dz, positive when shortening against tension, is NOT exported. It
    #    is exactly reconstructible afterwards from the axial_force and z_line
    #    traces (docs/README.md, Metrics, gives the expression) and is
    #    identically zero under an isometric hold.
    #
    # `old_state` IS the pre-solve configuration: no kinetics call moves a
    # filament and the thin frame is Z-disc-anchored, so reconstructing the OLD
    # displacements at the NEW z_line gives exactly the pre-solve absolute
    # positions.
    ls_old = trace.drivers.lattice_spacing
    work_xb = xb_axial_work(
        thick_axial(old_state.thick.displacement, topology),
        monomer_axial(old_state.thin.displacement, topology, z_line), ls_old,
        thick_axial(new_state.thick.displacement, topology),
        monomer_axial(new_state.thin.displacement, topology, z_line), lattice_spacing,
        new_xb, new_state.thick.xb_bound_to, constants, topology)

    # ========================================================================
    # ATP AND THE CYCLE FLUXES — exact expected crossing counts
    # ========================================================================
    # READ STRAIGHT OFF THE TRACE, which is the whole point of the trace. The
    # generator and its two exponentials were built once in kinetics_step, from
    # the mid state, at the driver-resolved constants and the PRE-solve lattice
    # spacing, with the subpopulation tuple already resolved — so this IS the
    # step thick_transitions took, not a reconstruction of it. Rebuilding it
    # here would read the wrong state and, under dynamic LS, the SOLVED spacing
    # the rates were never evaluated at.
    #
    # WHAT THIS RETURNS. For each head, and for each ordered pair (i, j), the
    # EXPECTED NUMBER of i -> j transitions it makes during the step:
    #
    #     E[# i -> j | started in s] = q_ij * INT_0^dt [exp(Qt)]_{s,i} dt
    #
    # exact for a generator held constant over the step, which is the same
    # assumption the sampler itself makes. Multi-hop traversals and repeat
    # crossings are both counted correctly. See xb_expected_crossings().
    N = xb_expected_crossings(trace.xb_bins, trace.state.thick.xb_states)
    N10, N12, N21, N34, N04, N40 = [jnp.sum(N[k]) for k in range(6)]

    # Expected ATP consumed this step.
    #
    # ATP is booked on the 3 -> 4 crossing (xb_r34_coeff; Tight_2 -> Free_2 is
    # where the nucleotide is exchanged), plus one per strong closure tear.
    #
    # NO MASK, AND EVERY HEAD IS READ. Counting the 3 -> 4 EDGE is exact in
    # expectation (multi-hop and repeat crossings included), and needs no mask:
    #
    #   * Heads starting in 0, 4 or 5 are INCLUDED. A head that runs
    #     0 -> 1 -> 2 -> 3 -> 4 inside one step really does spend an ATP, and
    #     r04 (the reverse recovery stroke, which consumes nothing) is the
    #     0 -> 4 edge, not this one.
    #   * A head torn off a closing site is in state 4 in `trace.state`, so it is
    #     inside the sum. That is NOT double-counting the tear: r43 is always
    #     zero, so G[4, 3] is identically 0 and a head in Free_2 cannot reach
    #     Tight_2 within the step at all.
    #
    # THE TEAR TERM STAYS A REALISED COUNT (see the module docstring).
    #
    # WHAT IS STILL APPROXIMATE. Q is held constant across the step; only
    # halving dt tests it.
    atp_expected = n_tear_strong + N34

    # ========================================================================
    # ASSEMBLE RESULT DICT (fixed keys — same pytree every call)
    # ========================================================================
    return MetricsDict({
        # Driver / protocol values
        'axial_force': force,
        'solver_residual': solver_residual,
        # Raw max|F| above, in pN, kept unchanged. The two below are what make
        # it readable: `solver_tolerance` gives it a physical scale, and
        # `solver_residual_norm` is the pass/fail number — <= 1 is converged,
        # in fixed AND dynamic LS. With `newton_iters` they separate
        # "converged comfortably", "converged at the wire", "hit the Newton
        # cap" and "hit the cap against an unreachable target".
        'solver_residual_norm': solver_residual_norm,
        'solver_tolerance': solver_tolerance,
        # The PRESCRIBED z_line and pCa are fields on SimulationResult
        # (result.z_line / result.pCa). result.lattice_spacing is the prescribed
        # d0; this is the SOLVED d, and under dynamic LS they differ.
        'lattice_spacing': lattice_spacing,

        # Crossbridge state counts (bound = loose + tight_1 + tight_2).
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

        # TM state counts
        'n_tm_state_0': n_tm_0,
        'n_tm_state_1': n_tm_1,
        'n_tm_state_2': n_tm_2,
        'n_tm_state_3': n_tm_3,

        # TM occupancy over sites a crossbridge could actually reach. All-site
        # fractions: divide the counts above by n_thin * n_tm.
        'frac_tm_state_2_overlap': overlap_tm_fractions['frac_tm_state_2_overlap'],
        'frac_tm_state_3_overlap': overlap_tm_fractions['frac_tm_state_3_overlap'],
        'n_overlap_sites': overlap_tm_fractions['n_overlap_sites'],

        # Transition events
        'atp_consumed': atp_consumed,
        'newly_bound': newly_bound,

        # Heads tropomyosin tore off this step, split by what they had spent.
        # `closure_detach_free` is free; each `closure_detach_atp` is one ATP,
        # already included in atp_consumed and atp_expected. Both are
        # identically zero when xb_tm_K2 is jnp.inf (the hard lock makes closure
        # over a bound head unreachable).
        'closure_detach_free': n_tear_weak,
        'closure_detach_atp': n_tear_strong,

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
        'titin_energy_avg': titin_energy_avg,
        'titin_energy_delta_avg': titin_energy_delta_avg,

        # Work done ON the lattice BY the crossbridges (pN*nm) — see the WORK
        # METRICS block above. The EXTERNAL work is
        # -0.5*(F[t-1]+F[t])*dz from the axial_force and z_line traces.
        'xb_work_on_filaments': work_xb,

        # A per-step ratio cannot be time-averaged (mean(w/a) !=
        # mean(w)/mean(a)). Reduce the numerator and denominator over the
        # window and divide afterwards.
        'atp_expected': atp_expected,

        # THE CYCLE FLUXES, as exact expected crossing counts per step. Each is
        # summed over every head from its own mid state and its own generator;
        # see xb_expected_crossings(). They are informational — nothing else in
        # the model reads them — and they let the ATP books be checked on any
        # run.
        #
        # A Loose head falling off having never bound ATP: an ordinary failed
        # weak attachment, NOT a load-driven event, and it costs nothing.
        'xb_detach_free': N10,
        # NOT A DETACHMENT — the head stays weakly bound. This is the
        # strain-gated 2 -> 1 reversal (Pi rebinding; see xb_rate_21) by which a
        # badly-positioned head backs out of a cycle it cannot afford rather than
        # completing it. It REFUNDS a phosphate and costs no ATP: 3 -> 2 -> 1 -> 0
        # is the microscopic reverse of the forward path, so charging it would be
        # the opposite error.
        'xb_give_up': N21,
        # LEDGER CROSS-CHECK 1 — gross Pi release minus what was handed back.
        # ATP is spent on arrival in Tight_1 and refunded by going back down, so
        # at steady state this must equal atp_expected. A gap is a real leak in
        # the model's accounting, not a counter artefact.
        'atp_net_pi_release': N12 - N21,
        # LEDGER CROSS-CHECK 2 — net hydrolysis, a THIRD route that touches no
        # state 1, 2 or 3 at all. Equals atp_expected minus the change in the
        # Free_2 population. READ IT KNOWING IT IS A NET OF TWO LARGE NUMBERS:
        # the reverse recovery stroke N04 runs at 28-56 /ms at 8x8, which is
        # 29-82% of the booked rate.
        'atp_net_hydrolysis': N40 - N04,
        #
        # NOISE FLOOR ON BOTH CROSS-CHECKS. Realised-minus-expected occupancy is
        # a MARTINGALE, so the residual has sd ~ sqrt(n_4)/sqrt(n_steps) — about
        # 1.6 /ms at 8x8 over 300 ms. A within-one-sigma drift is not a bug.

        # Solver diagnostics
        'newton_iters': newton_iters,
    })
