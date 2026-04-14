"""
SarcTopology: Consolidated Sarcomere Topology for JAX Simulations
==================================================================

This module consolidates topology.py and template.py into a single SarcTopology
class with flattened index maps optimized for JAX kernel operations.

Core Principle: Fixed-Width Arrays for XLA Efficiency
------------------------------------------------------
XLA requires concrete shapes at compile-time. Dynamic-size slicing per XB
triggers serialized execution or recompilation. The solution is to pre-compute
fixed-width arrays that enable single unified Gather operations across all XBs.

PyTree Registration
-------------------
SarcTopology is a registered JAX PyTree where:
- Arrays are "children" (traced by JAX, flow through vmap/scan as data)
- Integers are "aux_data" (static, used for compilation decisions)

This allows the topology to flow through vmap/scan as a single unit while
ensuring shape-defining integers remain static for compilation.

Usage
-----
>>> from multifil_jax.core.sarc_geometry import SarcTopology
>>> from multifil_jax.core.params import get_skeletal_params
>>>
>>> static, dynamic = get_skeletal_params()
>>> topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
>>> topo = jax.device_put(topo)  # Move to GPU once
>>>
>>> # For parameter sweeps, use same topology for all sweep points:
>>> for params in sweep_grid:
>>>     state = realize_state(topo, params, z_line, pCa, lattice_spacing)
>>>     result = run(topo, ...)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

from .params import StaticParams, DynamicParams


# =============================================================================
# CONSTANTS
# =============================================================================

# Hexagonal geometry constants (float32 for JAX compatibility)
SQRT3 = np.float32(np.sqrt(3))
THICK_THIN_DISTANCE = np.float32(1.0 / np.sqrt(3))
THICK_THIN_DISTANCE_INVERTEBRATE = np.float32(0.5)

# Face orientation vectors (index 0-5)
ORIENTATION_VECTORS = np.array([
    [0.866, -0.5],    # 0: -30 deg (down-right)
    [0.0, -1.0],      # 1: -90 deg (down)
    [-0.866, -0.5],   # 2: -150 deg (down-left)
    [-0.866, 0.5],    # 3: 150 deg (up-left)
    [0.0, 1.0],       # 4: 90 deg (up)
    [0.866, 0.5]      # 5: 30 deg (up-right)
], dtype=np.float32)

# Thick face angles
THICK_FACE_ANGLES = np.array([150, 90, 30, 330, 270, 210], dtype=np.float32) * np.pi / 180
THICK_FACE_ANGLES_INVERTEBRATE = np.array([0, 60, 120, 180, 240, 300], dtype=np.float32) * np.pi / 180

# Thin face orientation patterns
THIN_ORIENTATION_UPWARD = (4, 0, 2)
THIN_ORIENTATION_DOWNWARD = (3, 5, 1)


# =============================================================================
# SARC GEOMETRY CLASS
# =============================================================================

@jax.tree_util.register_pytree_node_class
class SarcTopology:
    """Consolidated geometry for half-sarcomere simulation.

    REGISTERED JAX PYTREE: Arrays are "children" (traced), integers are "aux_data" (static).

    This class consolidates topology and template into a single unit with
    flattened index maps optimized for GPU parallelization.

    Attributes (Structural Integers - aux_data, static):
        n_thick: Number of thick filaments
        n_crowns: Number of crowns per thick filament
        n_thin: Number of thin filaments
        n_sites: Maximum binding sites per thin filament
        n_titin: Number of titin connections
        n_xb_per_crown: Number of crossbridges per crown (always 3)
        n_faces_per_thin: Number of faces per thin filament (3 vertebrate, 2 invertebrate)
        max_sites_per_face: Maximum binding sites per thin face
        total_xbs: Total number of crossbridges (n_thick * n_crowns * 3)

    Attributes (Flattened Index Maps - children, traced):
        xb_to_thin_id: (total_xbs,) - Global XB -> thin filament ID
        xb_to_thin_face: (total_xbs,) - Global XB -> thin face index
        xb_to_site_indices: (total_xbs, max_sites_per_face) - FIXED-WIDTH search array

    Attributes (Connectivity Arrays - children, traced):
        thick_to_thin: (n_thick, 6, 2) - [thick, face, (thin_idx, thin_face)]
        thin_to_thick: (n_thin, n_faces_per_thin, 2) - [thin, face, (thick_idx, thick_face)]
        face_to_sites: (n_thin, n_faces_per_thin, max_sites_per_face) - face to site mapping
        n_sites_per_face: (n_thin, n_faces_per_thin) - count of sites per face
        titin_connections: (n_titin, 4) - titin connectivity

    Attributes (Structural Arrays - children, traced):
        crown_offsets: (n_crowns,) - axial offset for each crown from M-line
        crown_rests: (n_crowns,) - rest spacing for each crown
        binding_offsets: (n_thin, n_sites) - z_line - site_position offsets
        binding_rests: (n_thin, n_sites) - rest spacing between binding sites
        tm_chains: (n_thin, n_sites) - TM chain assignment (0 or 1)
        thick_starts: (n_thick,) - crown level start offset (1-3)
        thin_starts: (n_thin,) - helical twist start offset (0-25)
        eye_4: (4, 4) - identity matrix for TM matrix exponential
        eye_6: (6, 6) - identity matrix for XB matrix exponential
    """

    __slots__ = (
        # Structural integers (aux_data)
        'n_thick', 'n_crowns', 'n_thin', 'n_sites', 'n_titin',
        'n_xb_per_crown', 'n_faces_per_thin', 'max_sites_per_face',
        'total_xbs',
        # Flattened index maps (children)
        'xb_to_thin_id', 'xb_to_thin_face', 'xb_to_site_indices',
        # Connectivity arrays (children)
        'thick_to_thin', 'thin_to_thick', 'face_to_sites', 'n_sites_per_face',
        'titin_connections',
        # Structural arrays (children)
        'crown_offsets', 'crown_rests', 'binding_offsets', 'binding_rests',
        'tm_chains', 'thick_starts', 'thin_starts',
        'eye_4', 'eye_6',
        # XB binning arrays (children)
        'n_xb_bins',         # int — number of bins (= n_xb_bins from StaticParams)
        'xb_bin_edges',      # (n_xb_bins+1,) float32 — bin boundaries
        'xb_bin_centers',    # (n_xb_bins,)   float32 — bin midpoints
    )

    def __init__(
        self,
        # === STRUCTURAL INTEGERS (aux_data - not traced) ===
        n_thick: int,
        n_crowns: int,
        n_thin: int,
        n_sites: int,
        n_titin: int,
        n_xb_per_crown: int,
        n_faces_per_thin: int,
        max_sites_per_face: int,
        total_xbs: int,

        # === FLATTENED INDEX MAPS (children - traced) ===
        xb_to_thin_id: jnp.ndarray,        # (total_xbs,)
        xb_to_thin_face: jnp.ndarray,      # (total_xbs,)
        xb_to_site_indices: jnp.ndarray,   # (total_xbs, max_sites_per_face)

        # === CONNECTIVITY ARRAYS (children - traced) ===
        thick_to_thin: jnp.ndarray,        # (n_thick, 6, 2)
        thin_to_thick: jnp.ndarray,        # (n_thin, n_faces_per_thin, 2)
        face_to_sites: jnp.ndarray,        # (n_thin, n_faces_per_thin, max_sites_per_face)
        n_sites_per_face: jnp.ndarray,     # (n_thin, n_faces_per_thin)
        titin_connections: jnp.ndarray,    # (n_titin, 4)

        # === STRUCTURAL ARRAYS (children - traced) ===
        crown_offsets: jnp.ndarray,        # (n_crowns,)
        crown_rests: jnp.ndarray,          # (n_crowns,)
        binding_offsets: jnp.ndarray,      # (n_thin, n_sites)
        binding_rests: jnp.ndarray,        # (n_thin, n_sites)
        tm_chains: jnp.ndarray,            # (n_thin, n_sites)
        thick_starts: jnp.ndarray,         # (n_thick,)
        thin_starts: jnp.ndarray,          # (n_thin,)

        # === STRUCTURAL CONSTANTS (children - traced) ===
        eye_4: jnp.ndarray,                # (4, 4)
        eye_6: jnp.ndarray,                # (6, 6)

        # === XB BINNING (children - traced) ===
        n_xb_bins: int,
        xb_bin_edges: jnp.ndarray,         # (n_xb_bins+1,) float32
        xb_bin_centers: jnp.ndarray,       # (n_xb_bins,)   float32
    ):
        """Initialize SarcTopology with structural dimensions and pre-allocated arrays."""
        # Store integers
        self.n_thick = n_thick
        self.n_crowns = n_crowns
        self.n_thin = n_thin
        self.n_sites = n_sites
        self.n_titin = n_titin
        self.n_xb_per_crown = n_xb_per_crown
        self.n_faces_per_thin = n_faces_per_thin
        self.max_sites_per_face = max_sites_per_face
        self.total_xbs = total_xbs

        # Store flattened index maps
        self.xb_to_thin_id = xb_to_thin_id
        self.xb_to_thin_face = xb_to_thin_face
        self.xb_to_site_indices = xb_to_site_indices

        # Store connectivity arrays
        self.thick_to_thin = thick_to_thin
        self.thin_to_thick = thin_to_thick
        self.face_to_sites = face_to_sites
        self.n_sites_per_face = n_sites_per_face
        self.titin_connections = titin_connections

        # Store structural arrays
        self.crown_offsets = crown_offsets
        self.crown_rests = crown_rests
        self.binding_offsets = binding_offsets
        self.binding_rests = binding_rests
        self.tm_chains = tm_chains
        self.thick_starts = thick_starts
        self.thin_starts = thin_starts
        self.eye_4 = eye_4
        self.eye_6 = eye_6
        self.n_xb_bins = n_xb_bins
        self.xb_bin_edges = xb_bin_edges
        self.xb_bin_centers = xb_bin_centers

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[int, ...]]:
        """Flatten for JAX: arrays are children, integers are aux_data."""
        children = (
            # Flattened index maps
            self.xb_to_thin_id, self.xb_to_thin_face, self.xb_to_site_indices,
            # Connectivity arrays
            self.thick_to_thin, self.thin_to_thick, self.face_to_sites,
            self.n_sites_per_face, self.titin_connections,
            # Structural arrays
            self.crown_offsets, self.crown_rests, self.binding_offsets,
            self.binding_rests, self.tm_chains, self.thick_starts, self.thin_starts,
            self.eye_4, self.eye_6,
            self.xb_bin_edges, self.xb_bin_centers,
        )
        aux_data = (
            self.n_thick, self.n_crowns, self.n_thin, self.n_sites,
            self.n_titin, self.n_xb_per_crown,
            self.n_faces_per_thin, self.max_sites_per_face, self.total_xbs,
            self.n_xb_bins,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, ...], children: Tuple[jnp.ndarray, ...]) -> 'SarcTopology':
        """Reconstruct SarcTopology from flattened representation."""
        return cls(
            # Integers from aux_data
            n_thick=aux_data[0],
            n_crowns=aux_data[1],
            n_thin=aux_data[2],
            n_sites=aux_data[3],
            n_titin=aux_data[4],
            n_xb_per_crown=aux_data[5],
            n_faces_per_thin=aux_data[6],
            max_sites_per_face=aux_data[7],
            total_xbs=aux_data[8],
            n_xb_bins=aux_data[9],
            # Arrays from children
            xb_to_thin_id=children[0],
            xb_to_thin_face=children[1],
            xb_to_site_indices=children[2],
            thick_to_thin=children[3],
            thin_to_thick=children[4],
            face_to_sites=children[5],
            n_sites_per_face=children[6],
            titin_connections=children[7],
            crown_offsets=children[8],
            crown_rests=children[9],
            binding_offsets=children[10],
            binding_rests=children[11],
            tm_chains=children[12],
            thick_starts=children[13],
            thin_starts=children[14],
            eye_4=children[15],
            eye_6=children[16],
            xb_bin_edges=children[17],
            xb_bin_centers=children[18],
        )

    @classmethod
    def create(
        cls,
        nrows: int,
        ncols: int,
        static_params: StaticParams,
        dynamic_params: DynamicParams,
        periodic: bool = True,
        lattice_spacing: float = 14.0,
        thin_starts: List[int] = None,
        thick_starts: List[int] = None,
    ) -> 'SarcTopology':
        """Create SarcTopology using pure NumPy (runs once on CPU).

        This factory method:
        1. Generates hexagonal thick positions
        2. Finds thin positions based on actin geometry
        3. Computes connectivity
        4. Calculates binding site offsets
        5. Computes FIXED-WIDTH flattened index maps
        6. Computes adaptive bin edges

        Args:
            nrows: Number of thick filament rows
            ncols: Number of thick filament columns
            static_params: StaticParams with structural configuration
            dynamic_params: DynamicParams with physical parameters
            periodic: Whether to use periodic boundary conditions
            lattice_spacing: Lattice spacing in nm (for adaptive binning)
            thin_starts: Optional list of helical twist start offsets (0-25)
            thick_starts: Optional list of crown level start offsets (1-3)

        Returns:
            SarcTopology ready for device_put
        """
        # Extract parameters
        n_crowns = static_params.n_crowns
        n_polymers_per_thin = static_params.n_polymers_per_thin
        actin_geometry = static_params.actin_geometry

        # Set geometry-dependent parameters
        n_faces_per_thin = 3 if actin_geometry == "vertebrate" else 2

        # 1. Generate hexagonal thick positions
        thick_positions, box_x, box_y = _generate_hexagonal_thick_positions(nrows, ncols)
        n_thick = len(thick_positions)

        # Generate thick_starts if not provided
        if thick_starts is None:
            thick_starts_arr = np.random.randint(1, 4, size=n_thick)
        else:
            thick_starts_arr = np.array(thick_starts, dtype=np.int32)

        # 2. Find thin positions based on actin geometry
        thin_thick_pairs = None
        if actin_geometry == "vertebrate":
            thin_positions, thin_orientations = _find_thin_positions_at_interstices(
                thick_positions, nrows, ncols, box_x, box_y, periodic
            )
        else:
            thin_positions, thin_orientations, thin_thick_pairs = _find_thin_positions_at_edges(
                thick_positions, nrows, ncols, box_x, box_y, periodic
            )
        n_thin = len(thin_positions)

        # Generate thin_starts if not provided
        if thin_starts is None:
            thin_starts_arr = np.random.randint(0, 26, size=n_thin)
        else:
            if len(thin_starts) != n_thin:
                thin_starts_arr = np.random.randint(0, 26, size=n_thin)
            else:
                thin_starts_arr = np.array(thin_starts, dtype=np.int32)

        # 3. Compute connectivity
        thick_to_thin_list, thin_to_thick_list = _compute_connectivity(
            thick_positions, thin_positions, thin_orientations,
            box_x, box_y, periodic, n_faces_per_thin,
            THICK_THIN_DISTANCE if actin_geometry == "vertebrate" else THICK_THIN_DISTANCE_INVERTEBRATE,
            thin_thick_pairs
        )

        # Convert to arrays
        thick_to_thin_arr = np.full((n_thick, 6, 2), -1, dtype=np.int32)
        for thick_idx, faces in enumerate(thick_to_thin_list):
            for face_idx, conn in enumerate(faces):
                if conn is not None:
                    thick_to_thin_arr[thick_idx, face_idx] = [conn[0], conn[1]]

        thin_to_thick_arr = np.full((n_thin, n_faces_per_thin, 2), -1, dtype=np.int32)
        for thin_idx, faces in enumerate(thin_to_thick_list):
            for face_idx, conn in enumerate(faces):
                if conn is not None:
                    thin_to_thick_arr[thin_idx, face_idx] = [conn[0], conn[1]]

        # Generate titin connections
        titin_connections_list = []
        for thick_idx in range(n_thick):
            for face_idx in range(6):
                conn = thick_to_thin_list[thick_idx][face_idx]
                if conn is not None:
                    titin_connections_list.append((thick_idx, face_idx, conn[0], conn[1]))
        n_titin = len(titin_connections_list)
        titin_arr = np.array(titin_connections_list, dtype=np.int32) if titin_connections_list else np.zeros((0, 4), dtype=np.int32)

        # 4. Calculate crown offsets
        crown_offsets, crown_rests = _calculate_crown_offsets(
            n_crowns, static_params.thick_bare_zone, static_params.thick_crown_spacing
        )

        # 5. Calculate binding site offsets
        (binding_offsets_list, binding_rests_list, face_to_sites_list,
         n_sites_per_face_list, tm_chains_list, max_sites) = _calculate_binding_site_offsets(
            thin_orientations, thin_starts_arr, n_thin, n_polymers_per_thin, n_faces_per_thin
        )

        # Pad to uniform arrays
        binding_offsets_arr = np.zeros((n_thin, max_sites), dtype=np.float32)
        binding_rests_arr = np.full((n_thin, max_sites), 2.77, dtype=np.float32)
        tm_chains_arr = np.zeros((n_thin, max_sites), dtype=np.int32)
        for i in range(n_thin):
            n = len(binding_offsets_list[i])
            binding_offsets_arr[i, :n] = binding_offsets_list[i]
            binding_rests_arr[i, :n] = binding_rests_list[i]
            tm_chains_arr[i, :n] = tm_chains_list[i]

        # Find max sites per face
        max_sites_per_face = max(
            len(sites) for thin_faces in face_to_sites_list for sites in thin_faces
        ) if face_to_sites_list else 0

        # Pad face_to_sites
        face_to_sites_arr = np.full((n_thin, n_faces_per_thin, max_sites_per_face), -1, dtype=np.int32)
        n_sites_per_face_arr = np.zeros((n_thin, n_faces_per_thin), dtype=np.int32)
        for i in range(n_thin):
            for j in range(len(face_to_sites_list[i])):
                sites = face_to_sites_list[i][j]
                face_to_sites_arr[i, j, :len(sites)] = sites
                n_sites_per_face_arr[i, j] = len(sites)

        # 6. Compute FIXED-WIDTH flattened index maps
        total_xbs = n_thick * n_crowns * 3
        xb_to_thin_id, xb_to_thin_face, xb_to_site_indices = _compute_flat_index_maps_fixed_width(
            thick_to_thin_arr, face_to_sites_arr, n_sites_per_face_arr,
            thick_starts_arr, n_thick, n_crowns, max_sites_per_face
        )

        # Pre-allocate structural constants
        eye_4 = np.eye(4, dtype=np.float32)
        eye_6 = np.eye(6, dtype=np.float32)

        # XB bin edges and centers (baked in at topology creation time)
        bin_edges = jnp.linspace(
            static_params.xb_bin_lo,
            static_params.xb_bin_hi,
            static_params.n_xb_bins + 1,
            dtype=jnp.float32,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return cls(
            # Structural integers
            n_thick=n_thick,
            n_crowns=n_crowns,
            n_thin=n_thin,
            n_sites=max_sites,
            n_titin=n_titin,
            n_xb_per_crown=3,
            n_faces_per_thin=n_faces_per_thin,
            max_sites_per_face=max_sites_per_face,
            total_xbs=total_xbs,
            n_xb_bins=static_params.n_xb_bins,
            # Flattened index maps
            xb_to_thin_id=jnp.asarray(xb_to_thin_id),
            xb_to_thin_face=jnp.asarray(xb_to_thin_face),
            xb_to_site_indices=jnp.asarray(xb_to_site_indices),
            # Connectivity arrays
            thick_to_thin=jnp.asarray(thick_to_thin_arr),
            thin_to_thick=jnp.asarray(thin_to_thick_arr),
            face_to_sites=jnp.asarray(face_to_sites_arr),
            n_sites_per_face=jnp.asarray(n_sites_per_face_arr),
            titin_connections=jnp.asarray(titin_arr),
            # Structural arrays
            crown_offsets=jnp.asarray(crown_offsets),
            crown_rests=jnp.asarray(crown_rests),
            binding_offsets=jnp.asarray(binding_offsets_arr),
            binding_rests=jnp.asarray(binding_rests_arr),
            tm_chains=jnp.asarray(tm_chains_arr),
            thick_starts=jnp.asarray(thick_starts_arr),
            thin_starts=jnp.asarray(thin_starts_arr),
            # Structural constants
            eye_4=jnp.asarray(eye_4),
            eye_6=jnp.asarray(eye_6),
            # XB binning
            xb_bin_edges=bin_edges,
            xb_bin_centers=bin_centers,
        )

    def visualize(self, filename: str = None):
        """Create a visual representation of the sarcomere geometry.

        Args:
            filename: Optional file to save detailed connectivity info
        """
        print("=" * 60)
        print(f"SARC GEOMETRY: {self.n_thick} thick x {self.n_thin} thin")
        print("=" * 60)

        if self.n_thick == 4 and self.n_thin == 8:
            print("""
        Hexagonal lattice arrangement:

             T1       T3
               \\     /
                M0-M1
               /  X  \\
             T0   |   T2
               \\  |  /
                M3-M2
               /     \\
             T4       T6
               \\     /
                \\   /
             T5       T7

        M = Thick (Myosin), T = Thin (Actin)
            """)
        else:
            print(f"(Lattice: {self.n_thick} thick, {self.n_thin} thin)")

        print(f"\nStructure:")
        print(f"  Crowns per thick: {self.n_crowns}")
        print(f"  Sites per thin: {self.n_sites}")
        print(f"  Faces per thin: {self.n_faces_per_thin}")
        print(f"  Total crossbridges: {self.total_xbs}")
        print(f"  Total titin: {self.n_titin}")
        print(f"  Max sites per face: {self.max_sites_per_face}")

        if filename:
            with open(filename, 'w') as f:
                f.write("SARC GEOMETRY DETAILS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"n_thick: {self.n_thick}\n")
                f.write(f"n_thin: {self.n_thin}\n")
                f.write(f"n_crowns: {self.n_crowns}\n")
                f.write(f"total_xbs: {self.total_xbs}\n\n")

                f.write("THICK TO THIN CONNECTIONS\n")
                f.write("-" * 40 + "\n")
                thick_to_thin = np.asarray(self.thick_to_thin)
                for thick_idx in range(self.n_thick):
                    f.write(f"\nThick[{thick_idx}]:\n")
                    for face_idx in range(6):
                        thin_idx, thin_face = thick_to_thin[thick_idx, face_idx]
                        if thin_idx >= 0:
                            f.write(f"  Face {face_idx} -> Thin[{thin_idx}].face[{thin_face}]\n")
                        else:
                            f.write(f"  Face {face_idx} -> (no connection)\n")

            print(f"\nDetailed connectivity saved to: {filename}")

    def __repr__(self) -> str:
        """String representation showing key dimensions."""
        return (
            f"SarcTopology("
            f"n_thick={self.n_thick}, n_crowns={self.n_crowns}, "
            f"n_thin={self.n_thin}, n_sites={self.n_sites}, "
            f"total_xbs={self.total_xbs}, n_titin={self.n_titin})"
        )



# =============================================================================
# GEOMETRY GENERATION FUNCTIONS
# =============================================================================

def _generate_hexagonal_thick_positions(
    n_rows: int,
    n_cols: int,
    lattice_spacing: float = 1.0
) -> Tuple[np.ndarray, float, float]:
    """Generate thick filament positions on hexagonal grid."""
    lattice_spacing = np.float32(lattice_spacing)
    positions = []

    for row in range(n_rows):
        for col in range(n_cols):
            x_offset = np.float32(0.5) * lattice_spacing if row % 2 == 1 else np.float32(0.0)
            x = col * lattice_spacing + x_offset
            y = -row * SQRT3 / np.float32(2.0) * lattice_spacing
            positions.append([x, y])

    positions = np.array(positions, dtype=np.float32)
    box_x = np.float32(n_cols) * lattice_spacing
    box_y = np.float32(n_rows) * SQRT3 / np.float32(2.0) * lattice_spacing

    return positions, box_x, box_y


def _wrap_y(y, box_y):
    """Wrap y coordinate to [-box_y/2, box_y/2] range."""
    while y < -box_y / 2:
        y += box_y
    while y > box_y / 2:
        y -= box_y
    return np.float32(y)


def _wrap_x(x: float, box_x: float, tol: float = 1e-6) -> float:
    """Wrap x coordinate to [0, box_x) with tolerance."""
    x = x % box_x
    if abs(x - box_x) < tol or abs(x) < tol:
        x = 0.0
    return x


def _periodic_distance(pos1, pos2, box_x, box_y):
    """Compute minimum distance with periodic boundaries."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dx = dx - box_x * np.round(dx / box_x)
    dy = dy - box_y * np.round(dy / box_y)
    return np.float32(np.sqrt(dx * dx + dy * dy))


def _find_thin_at_position(thin_positions, x, y, box_x, box_y, periodic, tol=0.01):
    """Find thin filament index at the given position."""
    for idx, pos in enumerate(thin_positions):
        if periodic:
            dist = _periodic_distance([x, y], pos, box_x, box_y)
        else:
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if dist < tol:
            return idx
    return None


def _find_matching_thin_face(thick_pos, thin_pos, thin_orientations, box_x, box_y, periodic):
    """Find which thin face points toward the thick filament."""
    dx = thick_pos[0] - thin_pos[0]
    dy = thick_pos[1] - thin_pos[1]
    if periodic:
        dx = dx - box_x * np.round(dx / box_x)
        dy = dy - box_y * np.round(dy / box_y)
    angle_to_thick = np.arctan2(dy, dx)

    best_face = 0
    best_diff = np.inf

    for face_idx, orientation_idx in enumerate(thin_orientations):
        vec = ORIENTATION_VECTORS[orientation_idx]
        face_angle = np.arctan2(vec[1], vec[0])
        diff = abs(angle_to_thick - face_angle)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        if diff < best_diff:
            best_diff = diff
            best_face = face_idx

    return best_face


def _find_thin_positions_at_interstices(
    thick_positions: np.ndarray,
    n_rows: int,
    n_cols: int,
    box_x: float,
    box_y: float,
    periodic: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Find thin filament positions at trigonal interstices (vertebrate geometry)."""
    tol = 0.05
    pos_to_data = {}

    for thick_idx, thick_pos in enumerate(thick_positions):
        for face_idx in range(6):
            angle = THICK_FACE_ANGLES[face_idx]
            tx = thick_pos[0] + THICK_THIN_DISTANCE * np.cos(angle)
            ty = thick_pos[1] + THICK_THIN_DISTANCE * np.sin(angle)

            if periodic:
                tx = _wrap_x(tx, box_x)
                ty = _wrap_y(ty, box_y)

            pos_key = (round(tx / tol), round(ty / tol))
            if pos_key not in pos_to_data:
                pos_to_data[pos_key] = {'pos': (np.float32(tx), np.float32(ty)), 'thick_faces': []}
            pos_to_data[pos_key]['thick_faces'].append((thick_idx, face_idx))

    thin_data = []
    seen_positions = set()
    upper_faces = [0, 1, 2]
    lower_faces = [5, 4, 3]

    thick_by_row = {}
    for thick_idx, thick_pos in enumerate(thick_positions):
        row_key = round(thick_pos[1] / 0.1)
        if row_key not in thick_by_row:
            thick_by_row[row_key] = []
        thick_by_row[row_key].append((thick_idx, thick_pos))

    sorted_rows = sorted(thick_by_row.keys(), reverse=True)

    for row_key in sorted_rows:
        thick_in_row = sorted(thick_by_row[row_key], key=lambda t: t[1][0])

        for thick_idx, thick_pos in thick_in_row:
            for face_idx in upper_faces:
                angle = THICK_FACE_ANGLES[face_idx]
                tx = thick_pos[0] + THICK_THIN_DISTANCE * np.cos(angle)
                ty = thick_pos[1] + THICK_THIN_DISTANCE * np.sin(angle)

                if periodic:
                    tx = _wrap_x(tx, box_x)
                    ty = _wrap_y(ty, box_y)

                pos_key = (round(tx / tol), round(ty / tol))
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    data = pos_to_data[pos_key]
                    thin_data.append((data['pos'][0], data['pos'][1], data['thick_faces']))

        for thick_idx, thick_pos in thick_in_row:
            for face_idx in lower_faces:
                angle = THICK_FACE_ANGLES[face_idx]
                tx = thick_pos[0] + THICK_THIN_DISTANCE * np.cos(angle)
                ty = thick_pos[1] + THICK_THIN_DISTANCE * np.sin(angle)

                if periodic:
                    tx = _wrap_x(tx, box_x)
                    ty = _wrap_y(ty, box_y)

                pos_key = (round(tx / tol), round(ty / tol))
                if pos_key not in seen_positions:
                    seen_positions.add(pos_key)
                    data = pos_to_data[pos_key]
                    thin_data.append((data['pos'][0], data['pos'][1], data['thick_faces']))

    thin_positions_list = []
    thin_orientations = []

    for tx, ty, thick_faces in thin_data:
        thick_above = 0
        thick_below = 0
        for thick_idx, _ in thick_faces:
            thick_y = thick_positions[thick_idx][1]
            if periodic:
                dy = thick_y - ty
                dy = dy - box_y * np.round(dy / box_y)
                if dy > tol:
                    thick_above += 1
                elif dy < -tol:
                    thick_below += 1
            else:
                if thick_y > ty + tol:
                    thick_above += 1
                elif thick_y < ty - tol:
                    thick_below += 1

        is_upward = thick_above <= thick_below
        thin_positions_list.append([tx, ty])
        thin_orientations.append(THIN_ORIENTATION_UPWARD if is_upward else THIN_ORIENTATION_DOWNWARD)

    return np.array(thin_positions_list, dtype=np.float32), thin_orientations


def _find_thin_positions_at_edges(
    thick_positions: np.ndarray,
    n_rows: int,
    n_cols: int,
    box_x: float,
    box_y: float,
    periodic: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Find thin filament positions at edge midpoints (invertebrate geometry)."""
    tol = 0.1
    thin_data = []
    seen_edge_positions = set()

    neighbor_angles = np.array([0, 60, 120, 180, 240, 300], dtype=np.float32) * np.pi / 180
    neighbor_distance = 1.0

    orientation_angles = np.array([np.arctan2(v[1], v[0]) for v in ORIENTATION_VECTORS])

    def find_best_orientation(target_angle):
        diffs = np.abs(orientation_angles - target_angle)
        diffs = np.minimum(diffs, 2 * np.pi - diffs)
        return int(np.argmin(diffs))

    def find_thick_at_position(target_x, target_y):
        for idx, pos in enumerate(thick_positions):
            if periodic:
                dx = target_x - pos[0]
                dy = target_y - pos[1]
                dx = dx - box_x * np.round(dx / box_x)
                dy = dy - box_y * np.round(dy / box_y)
                dist = np.sqrt(dx**2 + dy**2)
            else:
                dist = np.sqrt((target_x - pos[0])**2 + (target_y - pos[1])**2)
            if dist < tol:
                return idx
        return None

    n_thick = len(thick_positions)
    for i in range(n_thick):
        pos_i = thick_positions[i]

        for angle in neighbor_angles:
            neighbor_x = pos_i[0] + neighbor_distance * np.cos(angle)
            neighbor_y = pos_i[1] + neighbor_distance * np.sin(angle)

            if periodic:
                neighbor_x = _wrap_x(neighbor_x, box_x)
                neighbor_y = _wrap_y(neighbor_y, box_y)

            j = find_thick_at_position(neighbor_x, neighbor_y)
            if j is None:
                continue

            thin_x = pos_i[0] + 0.5 * neighbor_distance * np.cos(angle)
            thin_y = pos_i[1] + 0.5 * neighbor_distance * np.sin(angle)
            if periodic:
                thin_x = _wrap_x(thin_x, box_x)
                thin_y = _wrap_y(thin_y, box_y)

            key_x = round(thin_x, 4) % box_x
            key_y = round(thin_y, 4) % box_y
            edge_key = (round(key_x, 3), round(key_y, 3))
            if edge_key in seen_edge_positions:
                continue
            seen_edge_positions.add(edge_key)

            angle_to_i = angle + np.pi
            if angle_to_i > np.pi:
                angle_to_i -= 2 * np.pi
            angle_to_j = angle

            orient_i = find_best_orientation(angle_to_i)
            orient_j = find_best_orientation(angle_to_j)

            if orient_i == orient_j:
                orient_j = (orient_i + 3) % 6

            orientation = (orient_i, orient_j)
            thick_pair = (i, j)
            thin_data.append((np.float32(thin_x), np.float32(thin_y), orientation, thick_pair))

    thin_data.sort(key=lambda t: (-t[1], t[0]))

    if len(thin_data) == 0:
        return np.zeros((0, 2), dtype=np.float32), [], []

    thin_positions = np.array([[t[0], t[1]] for t in thin_data], dtype=np.float32)
    thin_orientations = [t[2] for t in thin_data]
    thin_thick_pairs = [t[3] for t in thin_data]

    return thin_positions, thin_orientations, thin_thick_pairs


def _compute_connectivity(
    thick_positions: np.ndarray,
    thin_positions: np.ndarray,
    thin_face_orientations: List[Tuple[int, ...]],
    box_x: float,
    box_y: float,
    periodic: bool = True,
    n_faces_per_thin: int = 3,
    thick_thin_distance: float = None,
    thin_thick_pairs: List[Tuple[int, int]] = None
) -> Tuple[List[List[Optional[Tuple[int, int]]]], List[List[Optional[Tuple[int, int]]]]]:
    """Compute thick-thin connectivity."""
    if thick_thin_distance is None:
        thick_thin_distance = THICK_THIN_DISTANCE

    n_thick = len(thick_positions)
    n_thin = len(thin_positions)

    thick_to_thin = [[None] * 6 for _ in range(n_thick)]
    thin_to_thick = [[None] * n_faces_per_thin for _ in range(n_thin)]

    if n_faces_per_thin == 2 and thin_thick_pairs is not None:
        for thin_idx in range(n_thin):
            thin_pos = thin_positions[thin_idx]
            thick_i, thick_j = thin_thick_pairs[thin_idx]

            for face_idx, thick_idx in enumerate([thick_i, thick_j]):
                thick_pos = thick_positions[thick_idx]

                if periodic:
                    dx = thin_pos[0] - thick_pos[0]
                    dy = thin_pos[1] - thick_pos[1]
                    dx = dx - box_x * np.round(dx / box_x)
                    dy = dy - box_y * np.round(dy / box_y)
                else:
                    dx = thin_pos[0] - thick_pos[0]
                    dy = thin_pos[1] - thick_pos[1]
                angle_from_thick = np.arctan2(dy, dx)

                best_thick_face = 0
                best_diff = np.inf
                for thick_face_idx in range(6):
                    thick_angle = THICK_FACE_ANGLES_INVERTEBRATE[thick_face_idx]
                    diff = abs((angle_from_thick - thick_angle + np.pi) % (2 * np.pi) - np.pi)
                    if diff < best_diff:
                        best_diff = diff
                        best_thick_face = thick_face_idx

                thin_to_thick[thin_idx][face_idx] = (thick_idx, best_thick_face)
                thick_to_thin[thick_idx][best_thick_face] = (thin_idx, face_idx)
    else:
        for thick_idx in range(n_thick):
            thick_pos = thick_positions[thick_idx]

            for thick_face in range(6):
                angle = THICK_FACE_ANGLES[thick_face]
                expected_x = thick_pos[0] + thick_thin_distance * np.cos(angle)
                expected_y = thick_pos[1] + thick_thin_distance * np.sin(angle)

                if periodic:
                    expected_x = expected_x % box_x
                    expected_y = _wrap_y(expected_y, box_y)

                thin_idx = _find_thin_at_position(
                    thin_positions, expected_x, expected_y,
                    box_x, box_y, periodic, tol=0.1
                )

                if thin_idx is None:
                    continue

                thin_face = _find_matching_thin_face(
                    thick_pos, thin_positions[thin_idx],
                    thin_face_orientations[thin_idx],
                    box_x, box_y, periodic
                )

                thick_to_thin[thick_idx][thick_face] = (thin_idx, thin_face)
                thin_to_thick[thin_idx][thin_face] = (thick_idx, thick_face)

    return thick_to_thin, thin_to_thick


# =============================================================================
# OFFSET CALCULATION FUNCTIONS
# =============================================================================

def _calculate_crown_offsets(n_crowns: int, bare_zone: float, crown_spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate crown axial offsets relative to M-line."""
    bare_zone = np.float32(bare_zone)
    crown_spacing = np.float32(crown_spacing)

    offsets = bare_zone + np.arange(n_crowns, dtype=np.float32) * crown_spacing
    rests = np.full(n_crowns, crown_spacing, dtype=np.float32)
    rests[0] = bare_zone

    return offsets, rests


def _calculate_binding_site_offsets(
    thin_face_orientations: List[Tuple[int, ...]],
    thin_starts: np.ndarray,
    n_thin: int,
    n_polymers_per_thin: int,
    n_faces_per_thin: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]], List[List[int]], List[np.ndarray], int]:
    """Calculate binding site offsets for each thin filament."""
    mono_per_poly = 26
    polymer_base_length = np.float32(72.0)
    polymer_base_turns = np.float32(12.0)

    rev = np.float32(2 * np.pi)
    pitch = polymer_base_turns * rev / mono_per_poly
    rise = polymer_base_length / mono_per_poly

    total_monomers = mono_per_poly * n_polymers_per_thin
    monomer_offsets = (total_monomers - np.arange(total_monomers, dtype=np.float32)) * rise

    all_orientation_vectors = np.array([
        (0.866, -0.5), (0, -1.0), (-0.866, -0.5),
        (-0.866, 0.5), (0, 1.0), (0.866, 0.5)
    ], dtype=np.float32)

    offsets_list = []
    rests_list = []
    all_face_to_sites = []
    all_n_sites_per_face = []
    all_tm_chains = []

    for thin_idx in range(n_thin):
        active_faces = list(thin_face_orientations[thin_idx])
        start = thin_starts[thin_idx]

        monomer_angles = np.array([
            ((m + start + 1) % mono_per_poly) * pitch % rev
            for m in range(total_monomers)
        ], dtype=np.float32)

        orientation_vectors = all_orientation_vectors[active_faces]
        face_angles = np.arctan2(orientation_vectors[:, 1], orientation_vectors[:, 0])
        face_angles = np.where(face_angles < 0, face_angles + rev, face_angles)

        wiggle = rev / 24
        mono_in_faces = []

        for face_angle in face_angles:
            angle_diff = np.abs(monomer_angles - face_angle)
            within_wiggle = angle_diff < wiggle
            face_matches = np.where(within_wiggle)[0]
            mono_in_faces.append(face_matches)

        offsets_by_face = [monomer_offsets[mono_ind] for mono_ind in mono_in_faces]

        if len(offsets_by_face) > 0 and any(len(f) > 0 for f in offsets_by_face):
            offsets_flat = np.sort(np.hstack(offsets_by_face))[::-1]
        else:
            offsets_flat = monomer_offsets[::-1]

        tm_chain_this_thin = []
        for offset in offsets_flat:
            mono_idx_matches = np.where(np.abs(monomer_offsets - offset) < 1e-6)[0]
            if len(mono_idx_matches) > 0:
                mono_index = mono_idx_matches[0]
            else:
                mono_index = 0
            tm_chain_this_thin.append(mono_index % 2)

        node_index_by_face = []
        for face_offsets in offsets_by_face:
            site_indices = []
            for offset in face_offsets:
                idx = np.where(np.abs(offsets_flat - offset) < 1e-6)[0]
                if len(idx) > 0:
                    site_indices.append(idx[0])
            node_index_by_face.append(np.array(site_indices, dtype=np.int32))

        if len(offsets_flat) > 1:
            rests = offsets_flat[:-1] - offsets_flat[1:]
            rests = np.append(rests, offsets_flat[-1])
        else:
            rests = np.array([offsets_flat[0] if len(offsets_flat) > 0 else rise], dtype=np.float32)

        n_sites_this_face = [len(sites) for sites in node_index_by_face]

        offsets_list.append(offsets_flat.astype(np.float32))
        rests_list.append(rests.astype(np.float32))
        all_face_to_sites.append(node_index_by_face)
        all_n_sites_per_face.append(n_sites_this_face)
        all_tm_chains.append(np.array(tm_chain_this_thin, dtype=np.int32))

    max_sites = max(len(offsets) for offsets in offsets_list)

    return offsets_list, rests_list, all_face_to_sites, all_n_sites_per_face, all_tm_chains, max_sites


# =============================================================================
# FLATTENED INDEX MAP COMPUTATION
# =============================================================================

def _compute_flat_index_maps_fixed_width(
    thick_to_thin: np.ndarray,
    face_to_sites: np.ndarray,
    n_sites_per_face: np.ndarray,
    thick_starts: np.ndarray,
    n_thick: int,
    n_crowns: int,
    max_sites_per_face: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert nested Thick->Face->Thin to flat XB->Thin maps with fixed-width arrays.

    KEY OPTIMIZATION: xb_to_site_indices has shape (total_xbs, max_sites_per_face)
    which is CONSTANT across all XBs. This enables:
    - Single unified jnp.take Gather operation
    - No dynamic-size slicing per XB
    - Full GPU parallelization via vmap

    Returns:
        xb_to_thin_id: (total_xbs,) - Target thin filament for each XB
        xb_to_thin_face: (total_xbs,) - Target thin face for each XB
        xb_to_site_indices: (total_xbs, max_sites_per_face) - FIXED-WIDTH site indices
            Padded with neutral value (first valid site) for GPU parallelization
    """
    total_xbs = n_thick * n_crowns * 3
    face_pattern = np.array([[0, 2, 4], [1, 3, 5], [0, 2, 4]])  # Level 1, 2, 3

    xb_to_thin_id = np.zeros(total_xbs, dtype=np.int32)
    xb_to_thin_face = np.zeros(total_xbs, dtype=np.int32)
    xb_to_site_indices = np.zeros((total_xbs, max_sites_per_face), dtype=np.int32)

    for xb_idx in range(total_xbs):
        thick_idx = xb_idx // (n_crowns * 3)
        local_idx = xb_idx % (n_crowns * 3)
        crown_idx = local_idx // 3
        xb_in_crown = local_idx % 3

        # Crown level based on thick_starts (1-indexed: 1, 2, 3)
        crown_level = (crown_idx + thick_starts[thick_idx] - 1) % 3 + 1
        face_idx = face_pattern[crown_level - 1, xb_in_crown]

        # Get thin filament and face from thick_to_thin connectivity
        thin_idx = thick_to_thin[thick_idx, face_idx, 0]
        thin_face = thick_to_thin[thick_idx, face_idx, 1]

        # Handle unconnected faces (thin_idx == -1)
        if thin_idx < 0:
            thin_idx = 0
            thin_face = 0

        xb_to_thin_id[xb_idx] = thin_idx
        xb_to_thin_face[xb_idx] = thin_face

        # FIXED-WIDTH: Copy all site indices, pad with first site for unused slots
        n_valid = n_sites_per_face[thin_idx, thin_face]
        site_indices = face_to_sites[thin_idx, thin_face, :]

        # Get first valid site for padding (or 0 if no valid sites)
        first_valid = site_indices[0] if n_valid > 0 else 0

        # Pad invalid slots with first valid site (neutral for distance calculation)
        padded_indices = np.where(
            np.arange(max_sites_per_face) < n_valid,
            site_indices,
            first_valid
        )
        xb_to_site_indices[xb_idx] = padded_indices

    return xb_to_thin_id, xb_to_thin_face, xb_to_site_indices


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing SarcTopology...")
    print("=" * 60)

    from .params import get_skeletal_params

    # Test 1: Create geometry with skeletal parameters
    print("\nTest 1: Generate geometry with skeletal parameters")
    print("-" * 60)
    static, dynamic = get_skeletal_params()
    np.random.seed(42)
    geometry = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
    print(f"Geometry: {geometry}")
    print(f"  thick_to_thin shape: {geometry.thick_to_thin.shape}")
    print(f"  thin_to_thick shape: {geometry.thin_to_thick.shape}")
    print(f"  xb_to_thin_id shape: {geometry.xb_to_thin_id.shape}")
    print(f"  xb_to_site_indices shape: {geometry.xb_to_site_indices.shape}")

    # Test 2: PyTree roundtrip
    print("\nTest 2: PyTree flatten/unflatten")
    print("-" * 60)
    leaves, treedef = jax.tree_util.tree_flatten(geometry)
    print(f"Number of leaves (arrays): {len(leaves)}")
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    print(f"Reconstructed: {reconstructed}")
    print(f"n_thick match: {reconstructed.n_thick == geometry.n_thick}")
    print(f"xb_to_thin_id match: {jnp.allclose(reconstructed.xb_to_thin_id, geometry.xb_to_thin_id)}")

    # Test 3: JIT with geometry
    print("\nTest 3: JIT function using geometry")
    print("-" * 60)

    @jax.jit
    def test_fn(geometry: SarcTopology) -> jnp.ndarray:
        return geometry.crown_offsets.sum() * geometry.n_thick * geometry.n_crowns

    result = test_fn(geometry)
    print(f"JIT result: {result}")

    # Test 4: vmap with geometry
    print("\nTest 4: vmap with geometry (None in_axes = broadcast)")
    print("-" * 60)

    @jax.jit
    def per_xb_fn(geometry: SarcTopology, xb_idx: jnp.ndarray) -> jnp.ndarray:
        return geometry.xb_to_thin_id[xb_idx]

    vmapped = jax.vmap(per_xb_fn, in_axes=(None, 0))
    xb_indices = jnp.arange(min(10, geometry.total_xbs))
    results = vmapped(geometry, xb_indices)
    print(f"vmap results: {results}")

    print("\nAll tests passed!")
