"""Mechanical strain enhancement of SRX release.

Baseline SRX release uses the existing xb_srx_k0 parameter.
Calcium does not directly control SRX release.

LDA adds a strain-dependent release opportunity.
Internal crown-to-crown strain only; excludes the M-line bare zone.
Each crown receives signals from its preceding/following crown.
Missing neighbors at filament ends contribute zero.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class LDASignals(NamedTuple):
    crown_strain: object
    source_signal: object
    neighbor_signal: object
    release_probability: object


def validate_lda_params(params):
    """Validate parameters outside jit/vmap."""
    names = (
        "lda_enabled",
        "lda_strain_half",
        "lda_p_max",
        "lda_dt_ref_ms",
        "xb_srx_k0",
    )

    values = {
        name: np.asarray(getattr(params, name))
        for name in names
    }

    if any(
        not np.all(np.isfinite(value))
        for value in values.values()
    ):
        raise ValueError("LDA and baseline-release parameters must be finite")

    if not np.all(
        np.isin(values["lda_enabled"], [0.0, 1.0])
    ):
        raise ValueError("lda_enabled must be 0 or 1")

    if np.any(values["lda_strain_half"] <= 0):
        raise ValueError("lda_strain_half must be positive")

    if np.any(values["lda_dt_ref_ms"] <= 0):
        raise ValueError("lda_dt_ref_ms must be positive")

    p_max = values["lda_p_max"]
    if np.any(p_max < 0) or np.any(p_max >= 1):
        raise ValueError("Require 0 <= lda_p_max < 1")

    if np.any(values["xb_srx_k0"] < 0):
        raise ValueError("xb_srx_k0 must be nonnegative")


def send_to_adjacent_crowns(source):
    """Send source signals to adjacent crowns on the same filament."""
    left = jnp.zeros_like(source)
    right = jnp.zeros_like(source)

    left = left.at[:, 1:].set(source[:, :-1])
    right = right.at[:, :-1].set(source[:, 1:])

    return 0.5 * (left + right)


def mechanical_srx_signals(state, topology, params, dt):
    """Calculate total SRX-release probability for the current timestep.

    Total release combines:
    1. Baseline release from xb_srx_k0.
    2. Additional mechanical release when LDA is enabled.

    The two opportunities are combined through their survival probabilities.
    """
    # 1. Internal segment strain.
    u = state.thick.displacement
    rest = topology.crown_offsets

    lengths = rest[:, 1:] - rest[:, :-1]
    extension = u[:, 1:] - u[:, :-1]

    strain = extension / lengths
    tensile = jnp.maximum(strain, 0.0)

    # 2. Assign nearby segment strain to each crown.
    total = jnp.zeros_like(u)
    total = total.at[:, :-1].add(tensile)
    total = total.at[:, 1:].add(tensile)

    count = jnp.zeros_like(u)
    count = count.at[:, :-1].add(1.0)
    count = count.at[:, 1:].add(1.0)

    crown = total / jnp.maximum(count, 1.0)

    # 3. Convert strain into a bounded source signal.
    source = crown / (
        params.lda_strain_half + crown
    )

    # 4. Receive signals from adjacent crowns.
    neighbor = send_to_adjacent_crowns(source)

    # 5. Baseline SRX release, present with LDA both off and on.
    # xb_srx_k0 has units ms^-1; dt has units ms.
    p_base = -jnp.expm1(
        -params.xb_srx_k0 * dt
    )

    # 6. Additional mechanical release.
    # lda_p_max is the maximum mechanical probability per reference timestep.
    p_lda_ref = jnp.where(
        params.lda_enabled == 1.0,
        params.lda_p_max * neighbor,
        0.0,
    )

    p_lda = -jnp.expm1(
        (dt / params.lda_dt_ref_ms)
        * jnp.log1p(-p_lda_ref)
    )

    # 7. Combine independent release opportunities.
    # Equivalent to: 1 - (1 - p_base) * (1 - p_lda).
    p_crown = p_base + (1.0 - p_base) * p_lda

    # 8. All heads on a crown receive the same probability.
    # Their random draws remain independent in thick_transitions().
    p_head = jnp.broadcast_to(
        p_crown[:, :, None],
        state.thick.xb_states.shape,
    )

    return LDASignals(
        crown_strain=crown,
        source_signal=source,
        neighbor_signal=neighbor,
        release_probability=p_head,
    )


def apply_srx_probability(
    prob_vectors,
    current_states,
    p_release,
):
    """Replace probabilities only for heads starting this step in SRX."""
    p = p_release.reshape(-1)

    row = jnp.zeros_like(prob_vectors)
    row = row.at[:, 0].set(p)        # SRX -> DRX
    row = row.at[:, 5].set(1.0 - p)  # Remain in SRX

    return jnp.where(
        (current_states == 5)[:, None],
        row,
        prob_vectors,
    )