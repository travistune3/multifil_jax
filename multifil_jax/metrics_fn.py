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

atp_expected BECAME EXACT ON 2026-09-11, AND IT WAS NOT BEFORE. Until then it
summed P(VISIT state 4 at least once) over heads in mid states 1-3, read from a
generator with rows 0 and 4 made absorbing. P(at least one) is a LOWER BOUND on
E(visits), so it was biased low by exactly the heads that cycled twice inside one
step — -0.07% cardiac, -1.2% skeletal at dt = 1 ms, measured by two independent
routes. It now counts the 3 -> 4 EDGE, over every head with no mask, which is
the exact quantity and needs no absorbing construction. The remaining
approximation is that Q is held constant across the step; that is a DIFFERENT
error, it did not shrink when this one was removed, and only halving dt tests it.

HOW WRONG IS atp_consumed? THE ANSWER IS PRESET-DEPENDENT AND THE SPREAD IS
LARGE. This docstring said "~5% low at dt = 1 ms" without qualification until
2026-09-11. That figure is CARDIAC ONLY. Measured against the exact book side
(local_projects/tension_cost/atp_balance_spy.py), 8x8, dt = 1 ms:

    cardiac    pCa 4.5  -5.0%    pCa 6.2  -5.3%
    skeletal   pCa 4.5 -27.6%    pCa 6.2 -25.2%

Skeletal turns over ~2.6x faster, so far more heads clear two or three stages
inside one step and land outside the (mid == 3) -> {4, 0} window this counter
reads. Do not carry the 5% figure into a skeletal tension-cost study; it is off
by a factor of five. Both are read against the MID state — the sarcomere as
thick_transitions found it — not against the state at the top of the step; see
compute_all_metrics.

THE CYCLE FLUXES ARE EXPORTED TOO, so the ATP books self-verify on any run
instead of needing a spy script. All are exact expected crossing counts per step
(see kernels/transitions.xb_expected_crossings):

    xb_detach_atp        N34, the Q-route ATP-consuming detachment alone
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
(= M.ADP.Pi) would hand back that phosphate for free. Before this was booked
(2026-09-10), 8.0% of all the ATP the model spent was never counted anywhere —
4.3 points from state-2 tears, 2.7 from state-3 tears, the remaining 0.4% being
the give-up route above, which is a genuine refund and not a leak.

THE CLOSURE CHARGE STAYS A REALISED COUNT and must not become an expectation.
The tear is fully observed and its charge is deterministic given it, so there is
nothing to take an expectation over; charging an expectation would bill heads
that are still bound. The 3 -> 4 term is an expectation for the opposite reason:
the sampler reports only endpoints, so a head that traverses 3 -> 4 -> 0 inside
one step is invisible to any realised count.

(A retired metric, `xb_tear_expected`, was removed on 2026-09-11 along with the
absorbing generator it was the sole reason for. It asked "reached DRX WITHOUT
passing through Free_2", read over mid states 2-3 — a first-passage question, and
its own docstring had to shout that despite the name it did not count closure
tears. `xb_give_up` is what it was reaching for, measured directly as a crossing
count rather than inferred from a first-passage probability.)

(A third metric, atp_expected_q, was removed in Session 108. It capped each
head's detachment rate at its zero-load value, which silently encoded a DIFFERENT
model — "load-accelerated detachment is mechanical, not ATP-driven" — that this
model does not hold: xb_rate_34 is load-dependent and it is the ATP-consuming
step, so capping it at its zero-load value discards real cycling either way.
That the shipped sign is now a CATCH bond (xb_delta_34 = -0.80 since S129, so
load SLOWS detachment) does not rescue the removed metric — it inverts which
direction the cap bites, nothing more. Its documented use as a timestep-adequacy
check was also wrong; the p/q gap tracked load, not dt.)

Usage:
    from multifil_jax.metrics_fn import compute_all_metrics
    metrics = compute_all_metrics(old_state, new_state, constants, drivers,
                                  topology, force, solver_residual,
                                  newton_iters, dt, trace)
"""

import jax
import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

from multifil_jax.kernels.forces import (axial_force_at_mline, xb_axial_force_by_state,
                                         xb_axial_work)
from multifil_jax.kernels.transitions import xb_expected_crossings
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
    force: jnp.ndarray,
    solver_residual: jnp.ndarray,
    newton_iters,
    dt: float,
    trace: 'KineticsTrace',
    delta_z: jnp.ndarray,
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
        force: Scalar M-line force (already computed)
        solver_residual: Scalar equilibrium solver residual (pN)
        newton_iters: Number of Newton iterations used by solver
        dt: Timestep size (ms)
        trace: KineticsTrace from kinetics_step — the MID state (post
            thin_transitions, pre thick_transitions), the driver-resolved
            constants at the PRE-solve lattice spacing, the resolved
            subpopulation tuple, and the closure-tear mask. Required, not
            optional: every realised transition count and every Q-matrix metric
            below reads it, because those are the state and the generator that
            actually drove this step.
        delta_z: scalar z-line displacement applied at the START of this step
            (nm). Negative is shortening. Needed for `sarcomere_work` and for
            nothing else.

    THE TWO CONSTANTS OBJECTS DIFFER ON PURPOSE. `constants`/`drivers` carry the
    SOLVED lattice spacing, and the mechanics metrics must use them —
    `xb_axial_force_by_state` and the reported `lattice_spacing` are post-solve
    quantities. `trace.constants` carries the PRE-solve spacing, and the Q-matrix
    metrics must use that, because it is the spacing the rates were evaluated at.
    Do not "unify" them; in fixed-LS mode they agree anyway, and in dynamic-LS
    mode each is right for its own question.

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
    # READ AGAINST THE MID STATE, NOT old_state. Every transition counted here
    # happens inside thick_transitions, so the "before" endpoint is the state
    # thick_transitions was handed — after update_nearest_neighbors and
    # thin_transitions have run. Reading old_state instead conflates the tear
    # (which happens in thin_transitions) with the cycle.
    #
    # No closure mask is needed and none is applied. A head torn off by
    # tropomyosin is already out of state 3 in `mid_xb`, because the tear
    # happened in thin_transitions, one phase earlier. The old expression
    # subtracted the mask precisely because it read the PRE-step states, where
    # the torn head was still in state 3.
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
    # state. It moved when the trace landed, for a reason that has nothing to do
    # with ATP: a head that ran 3 -> 0 in thin_transitions (torn) and then
    # 0 -> 1 in thick_transitions is a genuine new attachment, and the old
    # old_state-based form could not see it.
    newly_bound = jnp.sum((mid_xb == 0) & (new_xb == 1)).astype(jnp.float32)

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
    # WORK METRICS — TWO DIFFERENT QUANTITIES, REPORTED SEPARATELY
    # ========================================================================
    # These replace `work_thick`/`work_thick_mean`, which were M-line force
    # times the MEAN displacement of every thick crown — neither of the two
    # quantities below, and under an isometric hold dominated by internal
    # backbone strain redistribution rather than by anything a motor did.
    #
    # 1. xb_work_on_filaments: work done ON the lattice BY the crossbridges.
    #    Path-dependent and per-head, so it cannot be reconstructed afterwards
    #    from the returned traces — it has to be computed here. This is the
    #    numerator for an efficiency, because ATP is spent by crossbridges.
    #
    # 2. sarcomere_work: work done externally by the half-sarcomere at the
    #    driven z-line, -F*dz, positive when shortening against tension. This
    #    one IS exactly reconstructible afterwards from the axial_force and
    #    z_line traces (docs/README.md section 6 gives the expression);
    #    shipping it is a convenience and an anchor for the documentation, not
    #    independent information.
    #
    # old_state.thick.axial IS the pre-solve thick position: between the
    # previous step's solve and this one the only change is the z-line shift,
    # which touches thin.axial alone, and no kinetics call moves a filament.
    ls_old = trace.constants.lattice_spacing
    work_xb = xb_axial_work(
        old_state.thick.axial, old_state.thin.axial, ls_old,
        new_state.thick.axial, new_state.thin.axial, lattice_spacing,
        new_xb, new_state.thick.xb_bound_to, constants, topology)

    # An honest trapezoid without carrying a force through the scan.
    # `force_old` is one backbone spring strain and is EXACTLY the previous
    # step's reported axial_force: dz is applied to thin.axial only, while
    # axial_force_at_mline reads thick.axial[:, 0].
    force_old = axial_force_at_mline(old_state, constants, topology)
    sarcomere_work = -0.5 * (force_old + force) * delta_z

    # ========================================================================
    # ATP AND THE CYCLE FLUXES — exact expected crossing counts
    # ========================================================================
    # BUILT FROM THE TRACE, which is the whole point of the trace. Both the
    # state and the constants come from kinetics_step, so this generator is the
    # one thick_transitions actually sampled from — same mid state, same
    # driver-resolved constants at the same PRE-solve lattice spacing, same
    # already-resolved subpopulation tuple. Rebuilding any of the three here
    # (which is what this did until 2026-09-10) both duplicated work and got a
    # different answer: reading off old_state biased the ATP number by
    # 0.06%-0.46%, and in dynamic-LS mode the rebuilt constants carried the
    # SOLVED spacing, which the rates were never evaluated at.
    #
    # WHAT THIS RETURNS. For each head, and for each ordered pair (i, j), the
    # EXPECTED NUMBER of i -> j transitions it makes during the step:
    #
    #     E[# i -> j | started in s] = q_ij * INT_0^dt [exp(Qt)]_{s,i} dt
    #
    # exact for a generator held constant over the step, which is the same
    # assumption the sampler itself makes. Multi-hop traversals and repeat
    # crossings are both counted correctly. See xb_expected_crossings().
    N = xb_expected_crossings(
        trace.state, trace.constants, topology, dt, xb_subpop=trace.xb_subpop
    )
    N10, N12, N21, N34, N04, N40 = [jnp.sum(N[k]) for k in range(6)]

    # Expected ATP consumed this step.
    #
    # ATP is booked on the 3 -> 4 crossing (xb_r34_coeff; Tight_2 -> Free_2 is
    # where the nucleotide is exchanged), plus one per strong closure tear.
    #
    # NO MASK, AND EVERY HEAD IS READ. Until 2026-09-11 this summed
    # P_abs[s, 4] over heads in mid states 1-3 only, read from a generator with
    # rows 0 and 4 zeroed — i.e. P(VISIT state 4 at least once), not E(number of
    # visits). P(at least one) is a LOWER BOUND on E(visits), so the estimator
    # was biased low by exactly the heads that cycled twice inside one step:
    # -0.07% cardiac, -1.2% skeletal at dt = 1 ms, confirmed by two independent
    # routes (atp_estimator_spy head-by-head, atp_balance_spy whole-ledger).
    # Counting the EDGE removes that bias, and removes the need for the mask
    # along with it. Two arguments that used to be load-bearing FOR the mask no
    # longer apply, and both are worth stating because they look like they do:
    #
    #   * Heads starting in 0, 4 or 5 are now INCLUDED, and that is correct. A
    #     head that runs 0 -> 1 -> 2 -> 3 -> 4 inside one step really does spend
    #     an ATP. Row 0 was excluded because "reached state 4" from state 0 would
    #     have counted `r04` — the reverse recovery stroke, which re-primes a
    #     detached lever arm and consumes nothing. An edge count cannot confuse
    #     the two: r04 is the 0 -> 4 edge and this reads the 3 -> 4 edge. Row 4
    #     was excluded because a head parked in Free_2 sat in an absorbing state
    #     and would have been charged afresh every step it lingered; a crossing
    #     count charges it only when it crosses.
    #   * A head torn off a closing site is in state 4 in `trace.state`, so it is
    #     now inside the sum rather than outside it. That is NOT double-counting
    #     the tear: its own E[# 3 -> 4] is a SECOND, fresh cycle within the same
    #     step, not the one already charged, and from state 4 it would have to
    #     run 4 -> 3 -> 4 to earn it. MEASURED at 8x8, dt = 1 ms, both presets,
    #     pCa 4.5 and 6.2: it is EXACTLY zero, and structurally so rather than
    #     merely small — r43 is always zero (see _build_xb_Q_matrix_optimized:
    #     re-attaching directly into the post-stroke state would run the ATPase
    #     backwards), so G[4, 3] is identically 0 and a head in Free_2 cannot
    #     reach Tight_2 within the step at all.
    #
    # THE TEAR TERM STAYS A REALISED COUNT, and that is not an inconsistency.
    # The tear is fully observed and its charge is deterministic given it, so the
    # realised count IS this step's tear ATP — there is nothing to take an
    # expectation over, and charging an expectation would bill heads that are
    # still bound. The 3 -> 4 term must be an expectation for the opposite
    # reason: the sampler reports only endpoints, so a head that traverses
    # 3 -> 4 -> 0 inside one step is invisible to any realised count.
    #
    # WHAT IS STILL APPROXIMATE. Q is held constant across the step. That is a
    # different error from the estimator bias just removed, and it does not
    # shrink because this sum became exact — only halving dt tests it.
    atp_expected = n_tear_strong + N34

    # Work per ATP. The numerator is the CROSSBRIDGE work, because ATP is spent
    # by crossbridges — and because it stays meaningful under an isometric hold,
    # where external work is exactly zero while heads are still cycling and
    # spending. For whole-sarcomere efficiency divide the two exported keys
    # yourself: sarcomere_work / atp_expected. There is no third key for it.
    xb_work_per_atp = jnp.where(atp_expected > 0.01,
                                work_xb / atp_expected, 0.0)

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
        'thick_energy_first_delta_avg': thick_energy_first_delta_avg,
        'titin_energy_avg': titin_energy_avg,
        'titin_energy_delta_avg': titin_energy_delta_avg,

        # Work metrics (pN*nm). Different quantities — see the WORK METRICS
        # block above and docs/README.md section 6.
        'xb_work_on_filaments': work_xb,
        'sarcomere_work': sarcomere_work,

        # ATP expected metrics
        'atp_expected': atp_expected,
        'xb_work_per_atp': xb_work_per_atp,

        # THE CYCLE FLUXES, as exact expected crossing counts per step. Each is
        # summed over every head from its own mid state and its own generator;
        # see xb_expected_crossings(). They are informational — nothing else in
        # the model reads them — and they exist so the ATP books can be checked
        # on any run instead of needing a spy script.
        #
        # `xb_detach_atp` is the Q-route ATP-consuming detachment ALONE, i.e.
        # atp_expected minus the closure charge. Exported so the composition
        # of the ATP number is visible without doing arithmetic against
        # closure_detach_atp.
        'xb_detach_atp': N34,
        # A Loose head falling off having never bound ATP: an ordinary failed
        # weak attachment, NOT a load-driven event, and it costs nothing.
        'xb_detach_free': N10,
        # NOT A DETACHMENT — the head stays weakly bound. This is the
        # strain-gated 2 -> 1 reversal (Pi rebinding; see xb_rate_21) by which a
        # badly-positioned head backs out of a cycle it cannot afford rather than
        # completing it. It REFUNDS a phosphate and costs no ATP: 3 -> 2 -> 1 -> 0
        # is the microscopic reverse of the forward path, so charging it would be
        # the opposite error. This is the quantity the retired `xb_tear_expected`
        # was reaching for, and it is measured rather than inferred from a
        # first-passage probability.
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
        # 1.6 /ms at 8x8 over 300 ms. A within-one-sigma drift is not a bug; S138
        # nearly reported one as such.

        # Solver diagnostics
        'newton_iters': newton_iters,
    })
