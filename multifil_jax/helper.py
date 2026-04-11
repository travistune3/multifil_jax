"""
Helper functions for JAX Half-Sarcomere simulation.

This module contains utility functions that complement the main simulation
but are not part of the core timestep workflow.

Functions:
    count_transitions - Count state transitions between timesteps
    validate_forces_numerical - Verify forces via finite difference
    validate_equilibrium - Check if state is at force equilibrium
"""

import jax.numpy as jnp
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.state import State
    from multifil_jax.core.params import DynamicParams
    from multifil_jax.core.sarc_geometry import SarcTopology


def count_transitions(old_state: 'State', new_state: 'State') -> Dict[str, int]:
    """Count state transitions between two timesteps.

    Compare crossbridge and tropomyosin states between old and new state
    to count how many transitions of each type occurred.

    Args:
        old_state: State NamedTuple before timestep.
        new_state: State NamedTuple after timestep.

    Returns:
        Dict with transition counts.
        Keys are 'xb_trans_XY' for crossbridge X->Y transitions,
        'tm_trans_XY' for tropomyosin transitions.
    """
    old_xb = old_state.thick.xb_states
    new_xb = new_state.thick.xb_states

    transitions = {}

    # XB transitions (6 states: 1=DRX, 2=loose, 3=tight_1, 4=tight_2, 5=free_2, 6=SRX)
    # Track transitions between adjacent states in the kinetic scheme
    xb_transition_pairs = [
        (1, 2), (2, 1),  # DRX <-> loose (binding/unbinding)
        (2, 3), (3, 2),  # loose <-> tight_1 (isomerization)
        (3, 4), (4, 3),  # tight_1 <-> tight_2 (power stroke)
        (4, 5), (5, 4),  # tight_2 <-> free_2 (unbinding)
        (5, 1), (1, 5),  # free_2 <-> DRX (recovery)
        (1, 6), (6, 1),  # DRX <-> SRX (super-relaxed)
    ]

    for from_state, to_state in xb_transition_pairs:
        key = f'xb_trans_{from_state}{to_state}'
        # JAX: Boolean comparison creates element-wise mask.
        # (old_xb == from_state) & (new_xb == to_state) finds positions
        # where the state changed from from_state to to_state.
        # jnp.sum counts True values (each True counts as 1).
        count = jnp.sum((old_xb == from_state) & (new_xb == to_state))
        # JAX: Convert to Python int for JSON serialization compatibility.
        transitions[key] = int(count)

    # Static count (no transition)
    transitions['xb_trans_static'] = int(jnp.sum(old_xb == new_xb))

    # TM transitions (4 states: 0=unbound, 1=bound, 2=closed, 3=open)
    old_tm = old_state.thin.tm_states
    new_tm = new_state.thin.tm_states

    tm_transition_pairs = [
        (0, 1), (1, 0),  # unbound <-> bound (Ca binding)
        (1, 2), (2, 1),  # bound <-> closed
        (2, 3), (3, 2),  # closed <-> open (permissive)
        (3, 0), (0, 3),  # open <-> unbound (if XB unbinds)
    ]

    for from_state, to_state in tm_transition_pairs:
        key = f'tm_trans_{from_state}{to_state}'
        count = jnp.sum((old_tm == from_state) & (new_tm == to_state))
        transitions[key] = int(count)

    # TM static count
    transitions['tm_trans_static'] = int(jnp.sum(old_tm == new_tm))

    return transitions


def validate_forces_numerical(state: 'State', constants: 'DynamicParams',
                              topology: 'SarcTopology' = None,
                              epsilon: float = 1e-4, tolerance: float = 1.0) -> bool:
    """Validate force computation using numerical differentiation.

    Uses jax.grad on total energy as ground truth, compared against
    compute_forces_vectorized (our analytical implementation).

    Args:
        state: State NamedTuple with thick/thin fields.
        constants: DynamicParams with physics values.
        topology: SarcTopology (passed to compute_forces_vectorized).
        epsilon: Step size for finite difference (nm) - unused, kept for API.
        tolerance: Maximum allowed difference (pN).

    Returns:
        True if forces match within tolerance.
    """
    import jax
    from multifil_jax.kernels.forces import compute_forces_vectorized

    n_thick, n_crowns = state.thick.axial.shape
    n_thin, n_sites = state.thin.axial.shape
    n_thick_nodes = n_thick * n_crowns

    pos_thick = state.thick.axial
    pos_thin = state.thin.axial
    rests_thick = jnp.broadcast_to(topology.crown_rests[None, :], (n_thick, n_crowns))
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

    def compute_total_energy(pos_flat):
        """Compute total potential energy (passive springs only)."""
        pt = pos_flat[:n_thick_nodes].reshape(n_thick, n_crowns)
        pn = pos_flat[n_thick_nodes:].reshape(n_thin, n_sites)

        # Thick filament spring energy
        thick_energy = 0.0
        for i in range(n_thick):
            dx = pt[i, 0] - rests_thick[i, 0]
            thick_energy += 0.5 * thick_k * dx ** 2
            for j in range(1, n_crowns):
                dx = pt[i, j] - pt[i, j-1] - rests_thick[i, j]
                thick_energy += 0.5 * thick_k * dx ** 2

        # Thin filament spring energy
        thin_energy = 0.0
        for i in range(n_thin):
            for j in range(n_sites - 1):
                dx = pn[i, j+1] - pn[i, j] - rests_thin[i, j]
                thin_energy += 0.5 * thin_k * dx ** 2
            dx = z_line - pn[i, -1] - rests_thin[i, -1]
            thin_energy += 0.5 * thin_k * dx ** 2

        return thick_energy + thin_energy

    pos_flat = jnp.concatenate([pos_thick.flatten(), pos_thin.flatten()])
    forces_numerical = -jax.grad(compute_total_energy)(pos_flat)

    # Compare interior thick nodes where passive forces dominate
    interior_thick_mask = jnp.ones(n_thick_nodes, dtype=bool)
    for i in range(n_thick):
        interior_thick_mask = interior_thick_mask.at[i * n_crowns + n_crowns - 1].set(False)

    diff = jnp.abs(forces_analytical - forces_numerical)
    max_diff_interior = jnp.max(diff[:n_thick_nodes] * interior_thick_mask[:n_thick_nodes])

    print(f"Force Validation (Numerical Gradient):")
    print(f"  Max diff (interior thick): {float(max_diff_interior):.2e} pN")
    print(f"  Tolerance: {tolerance:.2e} pN")
    print(f"  Match: {float(max_diff_interior) < tolerance}")

    return float(max_diff_interior) < tolerance


def validate_equilibrium(state: 'State', constants: 'DynamicParams',
                         topology: 'SarcTopology', tolerance: float = 1.0) -> bool:
    """Validate that current state is at force equilibrium.

    Checks that the maximum force residual is below tolerance.

    Args:
        state: State NamedTuple
        constants: DynamicParams with physics values
        topology: SarcTopology with structural index maps
        tolerance: Maximum allowed force residual (pN)

    Returns:
        True if state is at equilibrium within tolerance
    """
    from multifil_jax.kernels.forces import compute_forces_from_state_vectorized

    forces = compute_forces_from_state_vectorized(state, constants, topology)
    max_residual = float(jnp.max(jnp.abs(forces)))

    print(f"Equilibrium Validation:")
    print(f"  Max force residual: {max_residual:.2e} pN")
    print(f"  Tolerance: {tolerance:.2e} pN")
    print(f"  At equilibrium: {max_residual < tolerance}")

    return max_residual < tolerance
