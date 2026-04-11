"""
Geometry Calculations for JAX Half-Sarcomere

This module handles spatial calculations:
1. Nearest binding site for each crossbridge
2. Distances between crossbridges and binding sites
3. Connectivity-aware distance calculations

Optimizations:
- Fixed-width arrays for GPU parallelization
- Pre-computed flattened index maps from SarcTopology
- Unified Gather operations via vmap

From current code:
- hs.py: set_xb_nearest_binding_site()
- mf.py: Crown.nearest property
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from multifil_jax.core.sarc_geometry import SarcTopology


# ============================================================================
# FIXED-WIDTH NEAREST NEIGHBOR SEARCH (OPTIMIZED FOR GPU)
# ============================================================================

def find_nearest_binding_sites_fixed_width(
    xb_positions: jnp.ndarray,
    bs_positions: jnp.ndarray,
    xb_to_thin_id: jnp.ndarray,
    xb_to_site_indices: jnp.ndarray,
    n_sites_per_face_flat: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find nearest binding site using fixed-width Gather for GPU parallelization.

    KEY OPTIMIZATION: xb_to_site_indices has shape (n_xb, max_sites_per_face)
    which is CONSTANT across all XBs. This enables:
    - Single unified jnp.take Gather operation
    - No dynamic-size slicing per XB
    - Full GPU parallelization via vmap

    Args:
        xb_positions: (n_xb,) axial positions of crossbridges
        bs_positions: (n_thin, n_sites) axial positions of binding sites
        xb_to_thin_id: (n_xb,) target thin filament for each XB
        xb_to_site_indices: (n_xb, max_sites_per_face) FIXED-WIDTH site indices
        n_sites_per_face_flat: (n_xb,) pre-gathered valid counts per XB

    Returns:
        nearest_thin_idx: (n_xb,) which thin filament
        nearest_site_idx: (n_xb,) which site on that thin filament
    """
    n_xb = xb_positions.shape[0]
    max_sites_per_face = xb_to_site_indices.shape[1]

    # Unified Gather: bs_positions[thin_id, site_indices] for ALL XBs at once
    def gather_face_positions(thin_idx, site_indices):
        return bs_positions[thin_idx, site_indices]  # (max_sites_per_face,)

    face_bs_positions = jax.vmap(gather_face_positions)(
        xb_to_thin_id, xb_to_site_indices
    )  # (n_xb, max_sites_per_face)

    # Compute distances for ALL sites in fixed-width window
    # Shape: (n_xb, max_sites_per_face)
    distances = jnp.abs(face_bs_positions - xb_positions[:, None])

    # Mask: (1) beyond valid site count, (2) site is behind M-line (position <= 0)
    count_mask = jnp.arange(max_sites_per_face) < n_sites_per_face_flat[:, None]
    visible_mask = face_bs_positions > 0.0
    valid_mask = count_mask & visible_mask
    masked_distances = jnp.where(valid_mask, distances, jnp.inf)

    # Find argmin across fixed-width dimension (fully parallelized)
    min_local_idx = jnp.argmin(masked_distances, axis=1)  # (n_xb,)

    # Gather actual site indices
    nearest_site_idx = jnp.take_along_axis(
        xb_to_site_indices, min_local_idx[:, None], axis=1
    ).squeeze(1)

    return xb_to_thin_id, nearest_site_idx


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

def calculate_xb_to_bs_distances(xb_base_positions: jnp.ndarray,
                                 bs_positions: jnp.ndarray,
                                 nearest_thin: jnp.ndarray,
                                 nearest_site: jnp.ndarray,
                                 lattice_spacing: float) -> jnp.ndarray:
    """Calculate (axial, radial) distance from XBs to their nearest BS.

    Args:
        xb_base_positions: (n_xb,) XB base (crown) axial positions WITHOUT +13 offset
        bs_positions: (n_thin, n_sites) BS axial positions
        nearest_thin: (n_xb,) which thin filament
        nearest_site: (n_xb,) which site
        lattice_spacing: Radial distance (nm)

    Returns:
        distances: (n_xb, 2) array of (axial, radial) distances

    Note:
        The +13 nm offset is used when FINDING the nearest binding site
        (to account for the myosin head reaching out), but NOT when
        calculating the distance TO that binding site. The distance is
        measured from the crown base position (xb.axial_location in original).
    """

    n_xb = xb_base_positions.shape[0]

    bs_pos = bs_positions[nearest_thin, nearest_site]
    axial_dist = bs_pos - xb_base_positions
    radial_dist = jnp.full(n_xb, lattice_spacing)

    distances = jnp.stack([axial_dist, radial_dist], axis=1)

    return distances


# ============================================================================
# CONNECTIVITY-AWARE GEOMETRY
# ============================================================================

# ============================================================================
# UPDATE FUNCTIONS WITH SARC_GEOMETRY
# ============================================================================

def update_nearest_neighbors(
    state,
    constants,
    topology: 'SarcTopology'
):
    """Update nearest binding site for all crossbridges using topology.

    Uses fixed-width xb_to_site_indices from topology for GPU efficiency.

    Args:
        state: State NamedTuple with thick/thin sub-states.
        constants: Constants NamedTuple with lattice_spacing.
        topology: SarcTopology object with pre-computed index maps.

    Returns:
        new_state: State with updated nearest neighbor info
    """
    thick_axial = state.thick.axial
    thin_axial = state.thin.axial

    n_thick, n_crowns = thick_axial.shape

    # Flatten thick base positions (each crown has 3 XBs)
    xb_base_positions = jnp.repeat(thick_axial, 3, axis=1).reshape(-1)

    # For FINDING nearest binding site, use head position (+13 nm offset)
    xb_head_positions = xb_base_positions + 13.0

    # Pre-gather n_sites_per_face for each XB
    n_sites_per_face_flat = topology.n_sites_per_face[
        topology.xb_to_thin_id, topology.xb_to_thin_face
    ]

    # Find nearest binding sites using fixed-width gather
    nearest_thin, nearest_site = find_nearest_binding_sites_fixed_width(
        xb_head_positions,
        thin_axial,
        topology.xb_to_thin_id,
        topology.xb_to_site_indices,
        n_sites_per_face_flat,
    )

    # Calculate distances using BASE position (WITHOUT +13 offset)
    distances = calculate_xb_to_bs_distances(
        xb_base_positions,
        thin_axial,
        nearest_thin,
        nearest_site,
        constants.lattice_spacing
    )

    # Store site_idx only - thin filament is implicit from topology.xb_to_thin_id
    nearest_reshaped = nearest_site.reshape(n_thick, n_crowns, 3)
    distances_reshaped = distances.reshape(n_thick, n_crowns, 3, 2)

    # 1. Update the inner ThickState NamedTuple
    new_thick = state.thick._replace(
        xb_nearest_bs=nearest_reshaped,
        xb_distances=distances_reshaped
    )

    # 2. Update the outer State NamedTuple
    new_state = state._replace(thick=new_thick)

    return new_state


# Backward-compatible alias
update_nearest_neighbors_with_geometry = update_nearest_neighbors


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    from multifil_jax.core.sarc_geometry import SarcTopology
    from multifil_jax.core.params import get_skeletal_params

    print("Testing geometry module...")
    print("="*60)

    # Create test state using SarcTopology
    static_params, dynamic_params = get_skeletal_params()
    np.random.seed(42)
    geometry = SarcTopology.create(nrows=2, ncols=2, static_params=static_params, dynamic_params=dynamic_params)

    print(f"\nSarcTopology created: {geometry}")
    print(f"  total_xbs: {geometry.total_xbs}")
    print(f"  xb_to_thin_id shape: {geometry.xb_to_thin_id.shape}")
    print(f"  xb_to_site_indices shape: {geometry.xb_to_site_indices.shape}")

    # Test fixed-width nearest neighbor search
    print("\nTest: Fixed-width nearest neighbor search")
    print("-"*60)

    # Create mock positions
    n_thin = geometry.n_thin
    n_sites = geometry.n_sites
    bs_positions = jnp.linspace(50, 1200, n_thin * n_sites).reshape(n_thin, n_sites)
    xb_positions = jnp.linspace(100, 1100, geometry.total_xbs)

    n_sites_per_face_flat = geometry.n_sites_per_face[
        geometry.xb_to_thin_id, geometry.xb_to_thin_face
    ]

    nearest_thin, nearest_site = find_nearest_binding_sites_fixed_width(
        xb_positions,
        bs_positions,
        geometry.xb_to_thin_id,
        geometry.xb_to_site_indices,
        n_sites_per_face_flat,
    )

    print(f"Nearest thin shape: {nearest_thin.shape}")
    print(f"Nearest site shape: {nearest_site.shape}")
    print(f"First 5 nearest thin: {nearest_thin[:5]}")
    print(f"First 5 nearest site: {nearest_site[:5]}")

    # Test vmap broadcast with geometry
    print("\nTest: vmap broadcast with geometry")
    print("-"*60)

    batch_size = 10
    batched_xb_positions = jnp.tile(xb_positions, (batch_size, 1))
    batched_bs_positions = jnp.tile(bs_positions, (batch_size, 1, 1))

    def find_nearest_kernel(xb_pos, bs_pos):
        return find_nearest_binding_sites_fixed_width(
            xb_pos,
            bs_pos,
            geometry.xb_to_thin_id,
            geometry.xb_to_site_indices,
            n_sites_per_face_flat,
        )

    batched_results = jax.vmap(find_nearest_kernel)(batched_xb_positions, batched_bs_positions)
    print(f"Batched nearest thin shape: {batched_results[0].shape}")
    print(f"Batched nearest site shape: {batched_results[1].shape}")

    print("\nAll geometry tests passed!")
