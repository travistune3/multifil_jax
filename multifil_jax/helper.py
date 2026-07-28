"""
Diagnostic utilities, for checking the model rather than running it.

None of this is on the simulation path. These functions exist to answer
"is the model doing what I think it is doing", and are meant for interactive
use, debugging and testing.

    count_transitions          which state changes occurred between two states
    validate_forces_numerical  check the force kernel against an energy gradient
    validate_equilibrium       check that a solved state really is at force balance

validate_forces_numerical checks the analytic force kernel against the gradient
of an independently written energy function — force being minus the gradient of
potential energy. It is worth running after editing kernels/forces.py, because a
sign error in a spring term produces a plausible but wrong force-length curve
rather than an obvious failure.

It covers all three force sources: backbone springs, the crossbridge two-spring
element, and titin. The reference energy is a deliberate second implementation
of that physics — same equations, different code — because a check that reused
the kernel's own force routines would pass unconditionally and prove nothing.

  Crossbridge forces are only exercised if heads are actually attached in the
  state you pass. On a freshly realized state nothing is bound, so that part of
  the kernel goes untested; the function says so in its output.

validate_equilibrium is the complementary check: it asks whether a reported
force is an equilibrium force at all, or whether the solver stopped short.
Between the two, validate_forces_numerical tests whether the forces are right
and validate_equilibrium tests whether they have been balanced.

Neither is JIT-compiled, and the reference energy is written with explicit
Python loops rather than the vectorized forms used in the kernels. Both choices
are deliberate: a validator that reuses the machinery it validates cannot catch
errors in that machinery.
"""

import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams
    from multifil_jax.core.sarc_geometry import SarcTopology


# Elementary crossbridge transitions — exactly the non-zero off-diagonals of the
# rate matrix built in kernels/transitions.py. States are 0-based:
# 0 DRX, 1 Loose, 2 Tight_1, 3 Tight_2, 4 Free_2, 5 SRX.
_XB_TRANSITION_PAIRS = (
    (0, 1), (1, 0),   # attachment / weak detachment
    (1, 2), (2, 1),   # weak-to-strong isomerization (Pi release) / reverse
    (2, 3), (3, 2),   # working stroke / reverse stroke
    (3, 4), (4, 3),   # ADP release and detachment / reverse (structurally zero)
    (4, 0), (0, 4),   # recovery stroke / reverse
    (0, 5), (5, 0),   # SRX sequestration / Ca-dependent recruitment
)

# Elementary tropomyosin transitions. States: 0 Ca-free blocking,
# 1 Ca-bound blocking, 2 closed, 3 open. Note 3->0 exists but 0->3 does not:
# the cycle closes by calcium dissociating from the open state.
_TM_TRANSITION_PAIRS = (
    (0, 1), (1, 0),   # calcium binds / dissociates
    (1, 2), (2, 1),   # tropomyosin shifts to closed / back to blocking
    (2, 3), (3, 2),   # opens / closes
    (3, 0),           # cycle closes: calcium leaves the open state
)


def count_transitions(old_state: 'State', new_state: 'State') -> Dict[str, int]:
    """Count state changes between two consecutive states.

    A debugging counterpart to the aggregate metrics: rather than telling you
    how many units are in each state, it tells you which state CHANGES occurred.
    Useful for confirming that a kinetic pathway is carrying the traffic you
    expect, or for finding a transition that fires far more often than intended.

    WHAT IS COUNTED. Each unit is classified by comparing its state before and
    after, so every unit falls into exactly one bucket and the buckets sum to the
    total unit count:

        'xb_trans_XY'      changed from state X to state Y, X->Y being one of
                           the model's elementary transitions
        'xb_trans_other'   changed, but not by a single elementary transition
        'xb_trans_static'  did not change

    and likewise 'tm_trans_*'. State indices are 0-based throughout
    (0 DRX, 1 Loose, 2 Tight_1, 3 Tight_2, 4 Free_2, 5 SRX for crossbridges;
    0 Ca-free blocking, 1 Ca-bound blocking, 2 closed, 3 open for tropomyosin).

    THESE ARE NET CHANGES OVER THE STEP, NOT EVENT COUNTS. A unit that passes
    through an intermediate state within one timestep is classified by where it
    started and ended, not by the path it took. Two consequences worth
    internalizing:

      - A head completing 3 -> 4 -> 0 in one step lands in 'xb_trans_other',
        not in 'xb_trans_34'. That is precisely why ATP consumption is NOT
        measured this way; metrics_fn uses an absorbing-state construction to
        catch such traversals. Use 'atp_consumed' or 'atp_expected_p' for
        turnover, never these counts.
      - A head that leaves and returns within one step counts as 'static'.

    A large 'other' bucket means the timestep is long relative to the kinetics,
    and that endpoint comparison is losing information. It is a useful warning
    signal in its own right.

    'xb_trans_43' should always be zero: detachment is irreversible in this
    model, so a non-zero value indicates something has gone wrong upstream.

    Args:
        old_state: State before the timestep
        new_state: State after the timestep

    Returns:
        Dict of counts keyed as described above. Values are Python ints, so the
        result is directly JSON-serializable.
    """
    transitions = {}

    for prefix, old_arr, new_arr, pairs in (
        ('xb', old_state.thick.xb_states, new_state.thick.xb_states,
         _XB_TRANSITION_PAIRS),
        ('tm', old_state.thin.tm_states, new_state.thin.tm_states,
         _TM_TRANSITION_PAIRS),
    ):
        n_static = int(jnp.sum(old_arr == new_arr))
        n_elementary = 0
        for from_state, to_state in pairs:
            count = int(jnp.sum((old_arr == from_state) & (new_arr == to_state)))
            transitions[f'{prefix}_trans_{from_state}{to_state}'] = count
            n_elementary += count

        # Anything that changed but matched no elementary transition — a
        # multi-step traversal within the timestep. Deriving it by subtraction
        # rather than enumerating the remaining pairs guarantees the buckets
        # partition the units exactly.
        transitions[f'{prefix}_trans_static'] = n_static
        transitions[f'{prefix}_trans_other'] = (
            int(old_arr.size) - n_static - n_elementary
        )

    return transitions


def validate_forces_numerical(state: 'State', constants: 'DynamicParams',
                              topology: 'SarcTopology',
                              epsilon: float = 1e-4, tolerance: float = 1.0) -> bool:
    """Check the analytic force kernel against the gradient of an energy function.

    Force is minus the gradient of potential energy, so an independently written
    energy expression plus autodiff gives a reference the force kernel can be
    compared to. The reference here is built with explicit loops, deliberately
    not sharing code with the vectorized kernel it is checking.

    WHAT IS VALIDATED. All three force sources, at every node:

      - backbone springs on both filaments, including the M-line and Z-line
        boundary springs, rest lengths and the sign convention
      - the crossbridge two-spring element, both weak and strong configurations
      - titin, at each thick filament's tip crown

    In particular this covers the sign of the crossbridge angular term, which is
    the error most likely to survive casual inspection: getting it wrong still
    produces a smooth, plausible force-length curve.

    CROSSBRIDGE FORCES ARE ONLY EXERCISED IF HEADS ARE ATTACHED. A freshly
    realized state has every head detached, so a pass then says nothing about
    the crossbridge kernel. The printed report states how many heads were
    attached; if it says zero, that portion went untested. To exercise it, run a
    few timesteps first, or set xb_states and xb_bound_to by hand.

    SENSITIVITY. The tolerance is an ABSOLUTE force difference, so what counts
    as detectable depends on the magnitude of the forces present. With strongly
    bound heads carrying ~200 pN, the 1 pN default catches relative errors above
    roughly half a percent — comfortably enough for a sign flip or a wrong
    constant, but not for a subtle numerical drift. Correct code agrees to about
    1e-3 pN, the float32 floor, so tolerance can be tightened to 0.01 pN for a
    strict check; at that level a 0.01% error is caught.

    A note on coupling: the number of titin molecules per thick filament is
    written independently in this function and in the force kernel. If those
    ever diverge, this check fails at the tip crowns — which is intended, but
    means a tip-only failure points at that constant rather than at the titin
    force law.

    Prints a short report as well as returning the verdict, since it is intended
    for interactive use.

    Args:
        state: State to evaluate at, including crossbridge attachment
        constants: DynamicParams with spring constants, titin parameters,
            z_line and lattice_spacing
        topology: SarcTopology supplying rest spacings and crossbridge targets
        epsilon: Unused. Retained so existing call sites keep working; the
            comparison uses autodiff, which needs no step size.
        tolerance: Maximum tolerated discrepancy (pN). See SENSITIVITY above.

    Returns:
        True if every node agrees within tolerance
    """
    import jax
    from multifil_jax.kernels.forces import compute_forces_vectorized

    n_thick, n_crowns = state.thick.axial.shape
    n_thin, n_sites = state.thin.axial.shape
    n_thick_nodes = n_thick * n_crowns

    pos_thick = state.thick.axial
    pos_thin = state.thin.axial
    rests_thick = topology.crown_rests
    rests_thin = topology.binding_rests
    xb_states = state.thick.xb_states
    xb_bound_to = state.thick.xb_bound_to

    # Compute analytical forces
    forces_analytical = compute_forces_vectorized(
        pos_thick, pos_thin,
        constants.thick_k, constants.thin_k,
        constants.z_line, constants.lattice_spacing,
        constants.titin_a, constants.titin_b, constants.titin_rest,
        xb_states, xb_bound_to, constants, topology
    )

    # Compute numerical forces via jax.grad on total energy
    z_line = constants.z_line
    thick_k = constants.thick_k
    thin_k = constants.thin_k
    d = constants.lattice_spacing

    # ------------------------------------------------------------------ setup
    # Per-crossbridge constants. Which spring configuration applies depends on
    # the head's state: strong for Tight_1/Tight_2, weak for Loose. A head is
    # attached only if it is in a bound state AND has a recorded partner site.
    n_xb_per_crown = xb_states.shape[2]
    xb_states_flat = xb_states.reshape(-1)
    xb_bound_flat = xb_bound_to.reshape(-1)

    is_bound = (xb_states_flat >= 1) & (xb_states_flat <= 3) & (xb_bound_flat >= 0)
    is_strong = (xb_states_flat == 2) | (xb_states_flat == 3)
    bound_weight = is_bound.astype(jnp.float32)

    g_k = jnp.where(is_strong, constants.xb_g_k_strong, constants.xb_g_k_weak)
    g_rest = jnp.where(is_strong, constants.xb_g_rest_strong, constants.xb_g_rest_weak)
    c_k = jnp.where(is_strong, constants.xb_c_k_strong, constants.xb_c_k_weak)
    c_rest = jnp.where(is_strong, constants.xb_c_rest_strong, constants.xb_c_rest_weak)

    # Flat index of each head's partner site. Unbound heads carry -1, which would
    # wrap when used as an index, so clamp — their contribution is zeroed by
    # bound_weight regardless. No singularity results: r is never smaller than
    # the lattice spacing, so the energy stays smooth even at garbage offsets.
    flat_site = (jnp.clip(topology.xb_to_thin_id, 0, n_thin - 1) * n_sites
                 + jnp.clip(xb_bound_flat, 0, n_sites - 1))

    # Titin molecules per thick filament. Must track the n_titin_per_thick
    # default in compute_thick_passive_forces_vectorized; if the two ever
    # diverge, this check fails at the tip crowns, which is the intended alarm.
    n_titin_per_thick = 6
    titin_a_over_b = constants.titin_a / constants.titin_b

    def compute_total_energy(pos_flat):
        """Total potential energy: backbone springs, crossbridges, and titin.

        Written independently of kernels/forces.py — same physics, different
        code — so that agreement between this gradient and the force kernel is
        evidence rather than tautology.
        """
        pt = pos_flat[:n_thick_nodes].reshape(n_thick, n_crowns)
        pn = pos_flat[n_thick_nodes:].reshape(n_thin, n_sites)

        # Thick filament spring energy. The M-line is a fixed anchor at 0.
        thick_energy = 0.0
        for i in range(n_thick):
            dx = pt[i, 0] - rests_thick[i, 0]
            thick_energy += 0.5 * thick_k * dx ** 2
            for j in range(1, n_crowns):
                dx = pt[i, j] - pt[i, j-1] - rests_thick[i, j]
                thick_energy += 0.5 * thick_k * dx ** 2

        # Thin filament spring energy. The Z-line is the fixed anchor here.
        thin_energy = 0.0
        for i in range(n_thin):
            for j in range(n_sites - 1):
                dx = pn[i, j+1] - pn[i, j] - rests_thin[i, j]
                thin_energy += 0.5 * thin_k * dx ** 2
            dx = z_line - pn[i, -1] - rests_thin[i, -1]
            thin_energy += 0.5 * thin_k * dx ** 2

        # Crossbridge two-spring energy, for attached heads only:
        #     U = 0.5*g_k*(r - g_rest)^2 + 0.5*c_k*(theta - c_rest)^2
        # with r and theta the head's polar geometry relative to its bound site.
        # Vectorized rather than looped — there are far too many heads for a
        # Python loop — but the expression is written out here rather than
        # borrowed from the kernel under test.
        crown_pos = jnp.repeat(pt.reshape(-1), n_xb_per_crown)
        site_pos = pn.reshape(-1)[flat_site]
        x = site_pos - crown_pos
        r = jnp.sqrt(x ** 2 + d ** 2)
        theta = jnp.arctan2(d, x)
        xb_energy = jnp.sum(bound_weight * (
            0.5 * g_k * (r - g_rest) ** 2 + 0.5 * c_k * (theta - c_rest) ** 2
        ))

        # Titin energy at each thick filament tip. The force law is
        # F = a*exp(b*(L - rest)), so its potential is (a/b)*exp(b*(L - rest)),
        # with L the true 3D tether length. Always positive and always
        # attractive, so no one-sided clamp is needed in the energy.
        axial = z_line - pt[:, -1]
        L = jnp.sqrt(axial ** 2 + d ** 2)
        titin_energy = jnp.sum(
            n_titin_per_thick * (titin_a_over_b) * jnp.exp(
                constants.titin_b * (L - constants.titin_rest))
        )

        return thick_energy + thin_energy + xb_energy + titin_energy

    pos_flat = jnp.concatenate([pos_thick.flatten(), pos_thin.flatten()])
    forces_numerical = -jax.grad(compute_total_energy)(pos_flat)

    diff = jnp.abs(forces_analytical - forces_numerical)

    # Every node is comparable now: the reference energy covers all three force
    # sources, so nothing has to be masked out of the comparison.
    tip_indices = jnp.arange(n_thick) * n_crowns + (n_crowns - 1)
    max_diff_thick = float(jnp.max(diff[:n_thick_nodes]))
    max_diff_tips = float(jnp.max(diff[tip_indices]))
    max_diff_thin = float(jnp.max(diff[n_thick_nodes:]))
    max_diff = float(jnp.max(diff))
    passed = max_diff < tolerance

    n_attached = int(jnp.sum(is_bound))
    print("Force validation (analytic kernel vs independent energy gradient):")
    print(f"  Crossbridges attached in this state: {n_attached}")
    print(f"  Max diff, thick crowns:      {max_diff_thick:.2e} pN")
    print(f"    of which tip crowns:       {max_diff_tips:.2e} pN  (titin acts here)")
    print(f"  Max diff, thin nodes:        {max_diff_thin:.2e} pN")
    print(f"  Tolerance:                   {tolerance:.2e} pN")
    print(f"  Match: {passed}")
    if n_attached == 0:
        print("  NOTE: no crossbridges attached, so crossbridge forces were not"
              " exercised by this check.")

    return passed


def validate_equilibrium(state: 'State', constants: 'DynamicParams',
                         topology: 'SarcTopology', tolerance: float = 1.0) -> bool:
    """Check whether a state is actually at mechanical equilibrium.

    Evaluates the FULL force residual — backbone springs, crossbridges and titin
    — and reports the largest net force on any node. At equilibrium that should
    be zero to solver tolerance.

    Worth running on a state returned by the solver when a force reading looks
    suspicious. `axial_force_at_mline` infers force from the strain in one
    spring, an identity that holds only at equilibrium, so a state that has not
    converged will still report a plausible-looking force that means nothing.

    Unlike validate_forces_numerical, this shares the force kernel with the
    simulation, so it verifies that the solve CONVERGED, not that the forces are
    correct. A model with a sign error would pass this check happily.

    Note the achievable residual has a floor: node positions are ~1000 nm, where
    float32 resolves to ~1e-4 nm, so the residual cannot go below roughly
    thick_k * 1e-4 pN regardless of how hard the solver works. At default
    stiffness that is ~0.75 pN, so a tolerance below about 1 pN is not
    meaningful. See MIN_FLOAT32_TOLERANCE in kernels/solver.py.

    Args:
        state: State to check, normally one returned by solve_equilibrium
        constants: DynamicParams with physics values
        topology: SarcTopology with structural index maps
        tolerance: Maximum allowed residual (pN)

    Returns:
        True if the largest residual is below tolerance
    """
    from multifil_jax.kernels.forces import compute_forces_from_state_vectorized

    forces = compute_forces_from_state_vectorized(state, constants, topology)
    max_residual = float(jnp.max(jnp.abs(forces)))

    print(f"Equilibrium Validation:")
    print(f"  Max force residual: {max_residual:.2e} pN")
    print(f"  Tolerance: {tolerance:.2e} pN")
    print(f"  At equilibrium: {max_residual < tolerance}")

    return max_residual < tolerance
