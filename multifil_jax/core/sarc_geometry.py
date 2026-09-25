"""
Sarcomere structure: which filament is where, and what can reach what.

Everything geometric and unchanging lives here. A SarcTopology is built once on
the CPU, moved to the GPU, and then reused for every simulation that shares its
structure. It never changes during a run.

WHAT IS BEING BUILT
-------------------
Real muscle packs filaments into a hexagonal lattice. Thick filaments sit on the
lattice points; thin filaments sit between them, and how many thin filaments
surround each thick one is a genuine anatomical difference between species:

    vertebrate      1 thick : 2 thin, thin filaments at lattice interstices,
                    each thin facing 3 thick filaments
    invertebrate    1 thick : 3 thin, thin filaments at edge midpoints,
                    each thin facing 2 thick filaments

From that arrangement everything else follows: which thin filament each myosin
head faces, which actin monomers are close enough to the right azimuth to be
binding sites, where crowns and sites sit axially, which sites are neighbours on
the same tropomyosin strand, and where titin attaches.

The lattice is periodic by default, so a modest number of filaments approximates
the interior of a much larger myofibril rather than a fibre bundle dominated by
its own free surfaces.

WHY IT IS ALL PRECOMPUTED INTO FIXED-WIDTH ARRAYS
-------------------------------------------------
XLA needs shapes known at compile time. Anything that would vary in length per
crossbridge — "the list of sites this head can reach" — has to become a
rectangular array, padded to a common width, or the whole simulation degrades
into per-element control flow. So the geometry is resolved once here, into
arrays the kernels can gather from uniformly.

The cost of that is padding, and padding is not always benign. A head with no
real target still occupies a slot in xb_to_thin_id and xb_to_thin_face, filled
with a placeholder (0, 0) that is indistinguishable from a genuine reference to
thin filament 0. The companion boolean array xb_valid is what separates the two,
and analysis code that reads the index arrays without consulting it will silently
overcount filament 0. Use valid_xb_targets() instead. This is not an edge case in
asymmetric lattices: with fourfold crown symmetry and a rotation that does not
align with the hexagonal directions, a majority of slots can be placeholders.

PYTREE REGISTRATION
-------------------
SarcTopology is a registered JAX PyTree. Arrays are children, so they are traced
and can move through vmap and scan as data; the shape-defining integers are
aux_data, so they stay static and available to the compiler. This lets the whole
topology be passed around as one object without JAX trying to trace its
dimensions.

Usage
-----
>>> from multifil_jax.core.sarc_geometry import SarcTopology
>>> from multifil_jax.core.params import get_skeletal_params
>>>
>>> static, dynamic, z0, d0 = get_skeletal_params()
>>> topo = SarcTopology.create(nrows=2, ncols=2, static_params=static,
...                            dynamic_params=dynamic)
>>> topo = jax.device_put(topo)   # move to GPU once, reuse for every run


REFERENCE
---------
Squire JM, Luther PK, Knupp C (2006), "The myosin filament superlattice in the
    flight muscles of flies: A-band lattice optimisation for stretch-activation?",
    J Mol Biol 361:823-838, doi:10.1016/j.jmb.2006.06.072.
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

# Squire 2006 3-fold actin (thin-filament) registration: the 6 actins around
# each myosin sit at 3 systematic axial phases (actin_half_pitch/3 apart, with
# 60°/120° rotations). A hex lattice has exactly 3 edge directions, and each
# invertebrate thin filament's edge direction IS its registration class — so the
# count is structural (# hex edge directions), never a tunable knob.
N_ACTIN_REGISTRATION_CLASSES = 3  # hex edge directions = Squire 3-fold

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

    THE THIN FILAMENT IS THREE LAYERS ON ONE MONOMER LIST
    -----------------------------------------------------
    Every actin monomer is kept. Three independent layers index them:

      mechanical nodes    the unknowns of the thin backbone; thin.displacement is
                          per node. A monomer's displacement is interpolated
                          linearly between the node at or M-line-side of it
                          (mono_node) and the next one toward the Z-disc, with
                          weight mono_xi on the latter; the Z-disc is a fixed
                          node past the last one. core.state.monomer_axial is
                          the one place that does it.
      binding candidates  the monomers a head may bind (face_to_monomers,
                          xb_to_mono_indices). xb_bound_to and xb_nearest_bs
                          hold MONOMER indices; thin.bound_to is stored per
                          CANDIDATE (cand_mono / mono_cand translate).
      tropomyosin units   the regulatory switches; thin.tm_states is per unit.
                          mono_tm names the unit covering each monomer, so the
                          gate a head reads is tm_states[thin, mono_tm[thin, m]].

    Offsets are rest distances from the Z-disc, so node index 0 is the pointed
    (M-line) end and offsets DECREASE with index.

    Attributes (aux_data, static):
        n_thick, n_crowns, n_thin, n_titin, n_xb_per_crown, n_faces_per_thin,
        total_xbs, n_xb_bins
        n_mono: monomers per thin filament
        n_cand: binding candidates per thin filament (union of its faces)
        n_nodes: mechanical nodes per thin filament
        n_tm: tropomyosin units per thin filament
        max_mono_per_face: width of the fixed-width candidate lists
        thin_node_spacing: rest spacing of the thin mechanical nodes (nm)

    Attributes (children, traced):
        xb_to_thin_id, xb_to_thin_face: (total_xbs,) target thin and face
        xb_to_mono_indices: (total_xbs, max_mono_per_face) candidate monomers
        xb_valid: (total_xbs,) bool - False where the XB has no real geometric
            thin-filament partner this crown (continuous-formula miss, or a
            genuinely unconnected thick face); always True for the legacy path
            and for vertebrate defaults. Kinetics must gate binding on this.
        thick_to_thin: (n_thick, 6, 2) - [thick, face, (thin_idx, thin_face)]
        thin_to_thick: (n_thin, n_faces_per_thin, 2) - [thin, face, (thick_idx, thick_face)]
        face_to_monomers: (n_thin, n_faces_per_thin, max_mono_per_face), -1 padded
        n_mono_per_face: (n_thin, n_faces_per_thin)
        cand_mono: (n_thin, n_cand) int32 monomer index of each candidate, ascending
        mono_cand: (n_thin, n_mono) int32 candidate index of each monomer, -1 if none
        titin_connections: (n_titin, 4)
        crown_offsets: (n_thick, n_crowns) - axial offset for each crown from
            M-line, per thick filament (superlattice-aware)
        crown_rests: (n_thick, n_crowns) - rest spacing for each crown
        mono_offsets: (n_thin, n_mono) rest distance of each monomer from the Z-disc
        mono_angle: (n_thin, n_mono) monomer azimuth (rad)
        mono_strand: (n_thin, n_mono) long-pitch strand, monomer index % 2
        mono_node: (n_thin, n_mono) int32 bracketing node on the M-line side
        mono_xi: (n_thin, n_mono) float32 in [0, 1], weight of node mono_node + 1
        mono_tm: (n_thin, n_mono) int32 tropomyosin unit covering the monomer
        node_offsets: (n_thin, n_nodes) rest distance of each node from the Z-disc
        node_rests: (n_thin, n_nodes) rest length of each node's Z-side segment
        tm_chains: (n_thin, n_tm) strand of each unit
        tm_prev_neighbor, tm_next_neighbor: (n_thin, n_tm) int32 adjacent
            same-strand unit (self-referencing at strand ends)
        tm_rep_mono: (n_thin, n_tm) int32 the monomer that stands for the
            unit's axial position in position-based metrics and masks
        thick_starts: (n_thick,) - crown level start offset (1..n_xb_per_crown)
        thin_starts: (n_thin,) - helical twist start offset (0-25)
        eye_4, eye_6: identities for the TM / XB matrix exponentials
        xb_bin_edges, xb_bin_centers: crossbridge strain bins
    """

    _AUX = (
        'n_thick', 'n_crowns', 'n_thin', 'n_mono', 'n_cand', 'n_nodes', 'n_tm', 'n_titin',
        'n_xb_per_crown', 'n_faces_per_thin', 'max_mono_per_face', 'total_xbs',
        'n_xb_bins', 'thin_node_spacing', 'tm_max_heads',
    )
    _CHILDREN = (
        'xb_to_thin_id', 'xb_to_thin_face', 'xb_to_mono_indices', 'xb_valid',
        'thick_to_thin', 'thin_to_thick', 'face_to_monomers', 'n_mono_per_face',
        'cand_mono', 'mono_cand',
        'titin_connections',
        'crown_offsets', 'crown_rests', 'thick_starts', 'thin_starts',
        'mono_offsets', 'mono_angle', 'mono_strand', 'mono_node', 'mono_xi', 'mono_tm',
        'node_offsets', 'node_rests',
        'tm_chains', 'tm_prev_neighbor', 'tm_next_neighbor', 'tm_rep_mono',
        'eye_4', 'eye_6', 'xb_bin_edges', 'xb_bin_centers',
    )
    __slots__ = _AUX + _CHILDREN

    def __init__(self, **fields):
        """Every name in __slots__, by keyword. Integers in _AUX, arrays in _CHILDREN."""
        assert set(fields) == set(self.__slots__), \
            f"SarcTopology fields: -{set(self.__slots__) - set(fields)} +{set(fields) - set(self.__slots__)}"
        for name, value in fields.items():
            setattr(self, name, value)

    def valid_xb_targets(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Crossbridge connectivity with placeholder entries removed.

        USE THIS RATHER THAN READING xb_to_thin_id / xb_to_thin_face DIRECTLY.

        Those two arrays must have an entry for every crossbridge slot, because
        JAX gathers need static shapes. But not every slot corresponds to a head
        that can actually reach a thin filament: a crown's arm may point into a
        gap in the lattice, which happens whenever the azimuthal rotation between
        crowns is not a multiple of the 60 degrees separating hexagonal
        neighbours. Those slots are filled with a PLACEHOLDER of
        (thin_idx=0, thin_face=0).

        Nothing about the placeholder value distinguishes it from a real
        reference to thin filament 0, face 0. The separate boolean array
        `xb_valid` is the only thing that does. So any code that tallies
        connectivity from the raw arrays — counting how many heads target a
        given filament, or which crowns supply a particular binding site — will
        overcount filament 0 by exactly the number of placeholders unless it
        masks first.

        The magnitude is not small in asymmetric lattices. With fourfold crown
        symmetry and a 33.75 degree rotation, over half of all slots can be
        placeholders. In large symmetric vertebrate lattices every slot is valid
        and the distinction is moot, which is precisely why the bug is easy to
        introduce while working on one and only notice on the other.

        The simulation kernels already gate on xb_valid. This helper exists so
        analysis and debugging code does not have to re-derive the mask.

        Returns:
            xb_index: (n_valid,) flat indices into the total_xbs-length arrays
            thin_id: (n_valid,) the thin filament each of those heads targets
            thin_face: (n_valid,) the face on that filament
        """
        idx = jnp.where(self.xb_valid)[0]
        return idx, self.xb_to_thin_id[idx], self.xb_to_thin_face[idx]

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[int, ...]]:
        """Flatten for JAX: arrays are children, integers are aux_data."""
        return (tuple(getattr(self, n) for n in self._CHILDREN),
                tuple(getattr(self, n) for n in self._AUX))

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[int, ...], children: Tuple[jnp.ndarray, ...]) -> 'SarcTopology':
        """Reconstruct SarcTopology from flattened representation."""
        return cls(**dict(zip(cls._AUX, aux_data)), **dict(zip(cls._CHILDREN, children)))

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
            thin_starts: Optional list of helical twist start offsets (0-25).
                Omit (None) for the default deterministic unbiased spread; pass a
                list of length n_thin to override (must match n_thin or ValueError).
            thick_starts: Optional list of crown level start offsets
                (1..static_params.n_xb_per_crown).
                Omit (None) for the default deterministic unbiased spread; pass a
                list of length n_thick to override (must match n_thick or ValueError).

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

        # thick_starts: default deterministic unbiased spread, else validated override
        if thick_starts is None:
            thick_starts_arr = _spread_starts(n_thick, 1, static_params.n_xb_per_crown + 1)
        else:
            if len(thick_starts) != n_thick:
                raise ValueError(f"thick_starts must have length n_thick={n_thick}, got {len(thick_starts)}")
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

        # Squire 3-fold actin registration class per thin filament (edge
        # direction). None for vertebrate (interstice geometry) → all class 0.
        thin_class_arr = _compute_thin_registration_classes(
            thick_positions, thin_thick_pairs, box_x, box_y, periodic
        )

        # thin_starts default is geometry-aware:
        #   invertebrate → base_k = 0 (crystalline lattice is locked; the 3-fold
        #     class supplies the only systematic phase variation),
        #   vertebrate  → S83 deterministic decorrelated spread.
        # An explicit thin_starts= still overrides either default.
        if thin_starts is None:
            if actin_geometry == "vertebrate":
                thin_starts_arr = _spread_starts(n_thin, 0, 26)
            else:
                thin_starts_arr = np.zeros(n_thin, dtype=np.int32)
        else:
            if len(thin_starts) != n_thin:
                raise ValueError(f"thin_starts must have length n_thin={n_thin}, got {len(thin_starts)}")
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

        # thick_to_thin completeness check: under periodic boundaries every thick
        # face must reach a real thin filament (a genuinely missing spatial
        # neighbor here is a malformed lattice, e.g. invertebrate geometry at
        # odd nrows — independent of n_xb_per_crown/legacy_crown_geometry).
        # Non-periodic lattices legitimately have unconnected boundary faces,
        # so this check only applies when periodic=True.
        if periodic:
            missing = np.argwhere(thick_to_thin_arr[:, :, 0] < 0)
            if len(missing) > 0:
                thick_idx0, face_idx0 = missing[0]
                raise ValueError(
                    f"thick_to_thin has {len(missing)} unconnected (thick, face) pair(s) "
                    f"under periodic=True (e.g. thick={thick_idx0}, face={face_idx0}) — "
                    f"malformed lattice for nrows={nrows}, ncols={ncols}, "
                    f"actin_geometry='{actin_geometry}'. Invertebrate geometry requires "
                    "an even nrows."
                )

        if periodic and static_params.n_superlattice_classes == 3:
            if nrows % 2 != 0 or ncols % 3 != 0:
                raise ValueError(
                    f"n_superlattice_classes=3 requires nrows even AND ncols a multiple "
                    f"of 3 under periodic=True (got nrows={nrows}, ncols={ncols}) — "
                    "otherwise the 3-coloring has same-class neighbor pairs at the "
                    "periodic seam. Known-good examples: 2x3, 4x3, 6x6, 4x6, 6x9, 8x3."
                )

        # Generate titin connections
        titin_connections_list = []
        for thick_idx in range(n_thick):
            for face_idx in range(6):
                conn = thick_to_thin_list[thick_idx][face_idx]
                if conn is not None:
                    titin_connections_list.append((thick_idx, face_idx, conn[0], conn[1]))
        n_titin = len(titin_connections_list)
        titin_arr = np.array(titin_connections_list, dtype=np.int32) if titin_connections_list else np.zeros((0, 4), dtype=np.int32)

        # 4. Calculate crown offsets (per-filament, superlattice-aware)
        superlattice_class = _compute_superlattice_classes(
            n_thick, ncols, static_params.n_superlattice_classes
        )
        bare_zone_arr = (
            static_params.thick_bare_zone
            + superlattice_class.astype(np.float32)
            * (static_params.thick_crown_spacing / static_params.n_superlattice_classes)
        )
        crown_offsets, crown_rests = _calculate_crown_offsets(
            n_crowns, bare_zone_arr, static_params.thick_crown_spacing
        )

        # 5. The thin filament: every monomer, and the three layers built on it
        thin = _build_thin_layers(
            thin_orientations, thin_starts_arr, n_thin, n_polymers_per_thin,
            static_params.actin_half_pitch, static_params.mono_per_poly,
            static_params.polymer_base_turns, static_params.target_zone_wiggle,
            static_params.thin_node_spacing, static_params.tm_monomers_per_unit,
            thin_class=thin_class_arr,
        )

        # 6. Compute FIXED-WIDTH flattened index maps
        total_xbs = n_thick * n_crowns * static_params.n_xb_per_crown
        xb_to_thin_id, xb_to_thin_face, xb_to_mono_indices, xb_valid = _compute_flat_index_maps_fixed_width(
            thick_to_thin_arr, thin['face_to_monomers'], thin['n_mono_per_face'],
            thick_starts_arr, n_thick, n_crowns,
            static_params.n_xb_per_crown, static_params.crown_rotation_deg,
            static_params.crown_face_wiggle_deg, static_params.legacy_crown_geometry,
        )

        # XB bin edges and centers (baked in at topology creation time)
        bin_edges = jnp.linspace(
            static_params.xb_bin_lo,
            static_params.xb_bin_hi,
            static_params.n_xb_bins + 1,
            dtype=jnp.float32,
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        return cls(
            n_thick=n_thick,
            n_crowns=n_crowns,
            n_thin=n_thin,
            n_mono=thin['mono_offsets'].shape[1],
            n_cand=thin['cand_mono'].shape[1],
            n_nodes=thin['node_offsets'].shape[1],
            n_tm=thin['tm_chains'].shape[1],
            tm_max_heads=_max_candidates_per_unit(thin['cand_mono'], thin['mono_tm'], thin['tm_chains'].shape[1]),
            n_titin=n_titin,
            n_xb_per_crown=static_params.n_xb_per_crown,
            n_faces_per_thin=n_faces_per_thin,
            max_mono_per_face=thin['face_to_monomers'].shape[2],
            total_xbs=total_xbs,
            n_xb_bins=static_params.n_xb_bins,
            thin_node_spacing=static_params.thin_node_spacing,
            xb_to_thin_id=jnp.asarray(xb_to_thin_id),
            xb_to_thin_face=jnp.asarray(xb_to_thin_face),
            xb_to_mono_indices=jnp.asarray(xb_to_mono_indices),
            xb_valid=jnp.asarray(xb_valid),
            thick_to_thin=jnp.asarray(thick_to_thin_arr),
            thin_to_thick=jnp.asarray(thin_to_thick_arr),
            titin_connections=jnp.asarray(titin_arr),
            crown_offsets=jnp.asarray(crown_offsets),
            crown_rests=jnp.asarray(crown_rests),
            thick_starts=jnp.asarray(thick_starts_arr),
            thin_starts=jnp.asarray(thin_starts_arr),
            eye_4=jnp.eye(4, dtype=jnp.float32),
            eye_6=jnp.eye(6, dtype=jnp.float32),
            xb_bin_edges=bin_edges,
            xb_bin_centers=bin_centers,
            **{name: jnp.asarray(arr) for name, arr in thin.items()},
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
        print(f"  Monomers / nodes / Tm units per thin: {self.n_mono} / {self.n_nodes} / {self.n_tm}")
        print(f"  Faces per thin: {self.n_faces_per_thin}")
        print(f"  Total crossbridges: {self.total_xbs}")
        print(f"  Total titin: {self.n_titin}")
        print(f"  Candidate monomers per face: {self.max_mono_per_face}")

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
            f"n_thin={self.n_thin}, n_mono={self.n_mono}, n_nodes={self.n_nodes}, n_tm={self.n_tm}, "
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

def _spread_step(nv):
    """Low-discrepancy step coprime to nv, nearest to nv/golden-ratio.

    Stepping through 0..nv-1 by this amount spreads consecutive filaments evenly
    over the value space (a 1-D low-discrepancy sequence), decorrelating the
    registrations of spatially-adjacent filaments.
    """
    target = max(1, round(nv / 1.6180339887))
    for d in range(nv):
        for s in (target - d, target + d):
            if 1 <= s < nv and np.gcd(s, nv) == 1:
                return s
    return 1


def _spread_starts(n, low, high):
    """Deterministic low-discrepancy spread of starts on [low, high).
    Single-lattice-unbiased: the lattice's connections sample the full
    registration space, so its mean ~= the phase-ensemble mean."""
    nv = high - low
    return (low + (np.arange(n) * _spread_step(nv)) % nv).astype(np.int32)


def _compute_superlattice_classes(n_thick: int, ncols: int, n_superlattice_classes: int) -> np.ndarray:
    """Axial-coordinate 3-coloring for the Drosophila myosin superlattice.

    thick_idx -> (row, col) via row-major reconstruction, matching
    _generate_hexagonal_thick_positions's iteration order exactly.
    At n_superlattice_classes=1, class[i]=0 for every i — the formula
    degenerates for free, no special-casing needed at any call site.
    """
    thick_idx = np.arange(n_thick)
    row = thick_idx // ncols
    col = thick_idx % ncols
    q = col - (row - (row & 1)) // 2
    r = row
    return ((q - r) % n_superlattice_classes).astype(np.int32)


def _compute_thin_registration_classes(
    thick_positions: np.ndarray,
    thin_thick_pairs: Optional[List[Tuple[int, int]]],
    box_x: float,
    box_y: float,
    periodic: bool,
) -> Optional[np.ndarray]:
    """Squire 2006 3-fold actin registration class per thin filament.

    The thin-filament analog of _compute_superlattice_classes (thick). Each
    invertebrate thin filament sits at an edge midpoint between two thick
    filaments (thin_thick_pairs = (i, j)); a hex lattice has exactly 3 edge
    directions (0°/60°/120° mod 180°), and that direction IS the Squire
    registration class. Returns (n_thin,) int32 in {0, 1, 2}.

    Arrangement validated against Squire 2006 (JMB 361:823, p.826): this rule
    makes the six actins around each myosin run 0,1,2,0,1,2 by angular position
    — adjacent actins one phase apart, same phase on opposite (collinear) edges —
    reproducing Squire's stated "systematic relative rotations of 60°/120° and
    axial shifts of 38.7/3 = 12.9 nm" for the six surrounding actins.

    Vertebrate geometry has no thin_thick_pairs (thins sit at interstices, not
    edges) — returns None, treated downstream as all class 0, so registration is
    a no-op and byte-identical to the pre-registration behavior.
    """
    if thin_thick_pairs is None:
        return None
    classes = np.empty(len(thin_thick_pairs), dtype=np.int32)
    for idx, (i, j) in enumerate(thin_thick_pairs):
        dx = thick_positions[j][0] - thick_positions[i][0]
        dy = thick_positions[j][1] - thick_positions[i][1]
        if periodic:
            dx = dx - box_x * np.round(dx / box_x)
            dy = dy - box_y * np.round(dy / box_y)
        angle = np.degrees(np.arctan2(dy, dx)) % 180.0   # edge direction, undirected
        classes[idx] = int(np.round(angle / 60.0)) % N_ACTIN_REGISTRATION_CLASSES
    return classes


def _calculate_crown_offsets(
    n_crowns: int, bare_zone: np.ndarray, crown_spacing: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate crown axial offsets relative to M-line, per thick filament.

    bare_zone: (n_thick,) array, one value per filament.
    """
    bare_zone = np.asarray(bare_zone, dtype=np.float32)   # (n_thick,)
    crown_spacing = np.float32(crown_spacing)
    n_thick = bare_zone.shape[0]

    offsets = (
        bare_zone[:, None]
        + np.arange(n_crowns, dtype=np.float32)[None, :] * crown_spacing
    )  # (n_thick, n_crowns)
    rests = np.full((n_thick, n_crowns), crown_spacing, dtype=np.float32)
    rests[:, 0] = bare_zone

    return offsets, rests


def _build_thin_layers(
    thin_face_orientations: List[Tuple[int, ...]],
    thin_starts: np.ndarray,
    n_thin: int,
    n_polymers_per_thin: int,
    actin_half_pitch: float,
    mono_per_poly: int,
    polymer_base_turns: float,
    target_zone_wiggle: float,
    thin_node_spacing: float,
    tm_monomers_per_unit: Optional[int],
    thin_class: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Every actin monomer of every thin filament, and the three layers on them.

    MONOMERS. mono_per_poly * n_polymers_per_thin per filament, on a 1-start
    genetic helix: monomer m sits (n_mono - m) * rise from the Z-disc at azimuth
    (m + start + 1) * pitch, and alternates between the two long-pitch strands
    (m % 2). Helix geometry (actin_half_pitch/mono_per_poly/polymer_base_turns)
    and the acceptance window (target_zone_wiggle) come from StaticParams.

    CANDIDATES. A monomer is a binding candidate for a face when its azimuth is
    within target_zone_wiggle of the face direction.

    NODES. An even grid thin_node_spacing apart from the Z-disc, the same for
    every thin filament; node 0 is the pointed end. Every monomer is placed on
    it by linear interpolation (_interpolate_monomers).

    TROPOMYOSIN UNITS. With tm_monomers_per_unit None, one unit per binding
    candidate (no monomer may then belong to two faces). With an integer,
    consecutive runs of that many monomers on each strand (_tropomyosin_units).
    Either way the adjacent units on the same strand are Ising neighbours, a head
    bound to any monomer of a unit locks it by one factor (1 + xb_tm_K2), and a
    unit that leaves the open state releases every head on its monomers.

    thin_class: (n_thin,) int32 in {0,1,2}, the Squire 3-fold registration class
    per thin filament (None → all class 0). Each class shifts that filament's
    monomer grid by Squire's measured inter-filament screw:
      angular: start += class · round(mono_per_poly/3)  (≈ 60°/120° rotation)
      axial:   grid  += class · (actin_half_pitch/3)    (12.9/25.8 nm, +Z; see
               the sign/handedness rationale in the body)
    At class 0 both shifts vanish, so vertebrate/cardiac is byte-identical.

    Returns a dict keyed by SarcTopology field name (NumPy arrays).
    """
    if thin_class is None:
        thin_class = np.zeros(n_thin, dtype=np.int32)

    # Per-class 3-fold offsets. Magnitudes are Squire 2006 (JMB 361:823, p.826):
    # the six actins around each myosin sit at relative rotations of 60°/120° AND
    # axial shifts of 38.7/3 = 12.9 nm — one rigid inter-filament screw per step.
    #   angular: class · round(mono_per_poly/3) monomers = best integer approx to
    #            Squire's 60° (9 mono → 64.3°, ~4.3° off; intrinsic to the 28/13
    #            helix, since 60° is not an exact multiple of the 12.857° grid).
    #   axial:   class · (actin_half_pitch/3) = 12.9 / 25.8 nm.
    # SIGN (+, toward the Z-line): a phase shift on a *finite* monomer array must
    #   truncate at one end; + puts that boundary effect in the crown-free I-band
    #   (past crown_offsets.max(), already unreachable/masked) rather than across
    #   the M-line into the crown zone. Measured: '−' pushes 4 sites per class-2
    #   thin behind the M-line (a class-asymmetric artifact), '+' leaves all three
    #   classes symmetric. Squire fixes the magnitude and relative arrangement,
    #   not our coordinate sign, so this is a free, artifact-avoiding choice.
    # RESIDUAL (does NOT affect the 3-fold structure or match/mismatch magnitude):
    #   the absolute chirality of the screw vs real Lethocerus is not fixed by
    #   Squire's relative low-angle X-ray data, nor by the Hu 2016 cryo-EM (myosin
    #   only); pinning it would need a thin-filament 3D reconstruction. It mirrors
    #   the whole lattice only. The absolute AZIMUTHAL actin-vs-crown registration
    #   (base_k) is likewise unmeasured and gauge-degenerate with the crown-face
    #   convention → base_k defaults to 0 (see create()). Its EFFECT is real, not
    #   noise: Squire Fig.6b/7d show head-attachment probability oscillates with
    #   registration — that IS stretch activation, so it must not be averaged out.
    angular_step = int(round(mono_per_poly / N_ACTIN_REGISTRATION_CLASSES))
    axial_step = np.float32(actin_half_pitch / N_ACTIN_REGISTRATION_CLASSES)

    polymer_base_length = np.float32(2 * actin_half_pitch)
    polymer_base_turns = np.float32(polymer_base_turns)

    rev = np.float32(2 * np.pi)
    pitch = polymer_base_turns * rev / mono_per_poly
    rise = polymer_base_length / mono_per_poly

    n_mono = mono_per_poly * n_polymers_per_thin
    monomer_offsets = (n_mono - np.arange(n_mono, dtype=np.float32)) * rise
    wiggle = np.float32(target_zone_wiggle)

    # ------------------------------------------------------------------ monomers
    mono_offsets = np.empty((n_thin, n_mono), dtype=np.float32)
    mono_angle = np.empty((n_thin, n_mono), dtype=np.float32)
    faces = []   # per thin: one ascending monomer-index array per face
    for thin_idx in range(n_thin):
        cls = int(thin_class[thin_idx])
        start = thin_starts[thin_idx] + cls * angular_step
        # This thin's axially-shifted monomer grid (Squire screw translation per
        # class, +Z toward the Z-line — see the sign rationale above).
        mono_offsets[thin_idx] = monomer_offsets + cls * axial_step
        mono_angle[thin_idx] = np.array([
            ((m + start + 1) % mono_per_poly) * pitch % rev
            for m in range(n_mono)
        ], dtype=np.float32)

        # ------------------------------------------------------------ candidates
        orientation_vectors = ORIENTATION_VECTORS[list(thin_face_orientations[thin_idx])]
        face_angles = np.arctan2(orientation_vectors[:, 1], orientation_vectors[:, 0])
        face_angles = np.where(face_angles < 0, face_angles + rev, face_angles)
        faces.append([np.where(np.abs(mono_angle[thin_idx] - fa) < wiggle)[0]
                      for fa in face_angles])
    mono_strand = np.tile(np.arange(n_mono, dtype=np.int32) % 2, (n_thin, 1))

    n_faces = len(faces[0])
    max_mono_per_face = max(len(f) for thin_faces in faces for f in thin_faces)
    face_to_monomers = np.full((n_thin, n_faces, max_mono_per_face), -1, dtype=np.int32)
    n_mono_per_face = np.zeros((n_thin, n_faces), dtype=np.int32)
    for thin_idx, thin_faces in enumerate(faces):
        for face_idx, mono in enumerate(thin_faces):
            face_to_monomers[thin_idx, face_idx, :len(mono)] = mono
            n_mono_per_face[thin_idx, face_idx] = len(mono)

    # ------------------------------------------------------------------ candidates, per thin
    # The union of a thin filament's face lists, ascending. thin.bound_to is
    # stored per candidate (only a candidate can ever be bound), so the
    # per-step binding bookkeeping scales with the candidates, not with every
    # monomer. mono_cand inverts it, -1 for a monomer no face admits.
    cand_mono = [np.unique(np.concatenate(thin_faces)) for thin_faces in faces]
    assert len({len(c) for c in cand_mono}) == 1, "thin filaments differ in candidate count"
    cand_mono = np.stack(cand_mono).astype(np.int32)                  # (n_thin, n_cand)
    mono_cand = np.full((n_thin, n_mono), -1, dtype=np.int32)
    np.put_along_axis(mono_cand, cand_mono,
                      np.tile(np.arange(cand_mono.shape[1], dtype=np.int32), (n_thin, 1)), axis=1)

    # ------------------------------------------------------------------ nodes
    # An even grid from the Z-disc, the same for every thin filament, so one
    # thin factorization serves the whole lattice. Node j sits
    # thin_node_spacing * (n_nodes - j) from the Z-disc, and the grid reaches
    # the longest filament's last monomer; a node beyond a shorter filament's
    # last monomer is an unloaded free end.
    spacing = np.float32(thin_node_spacing)
    n_nodes = int(np.ceil(float(mono_offsets.max()) / float(spacing)))
    node_offsets = np.tile(spacing * (n_nodes - np.arange(n_nodes, dtype=np.float32)), (n_thin, 1))
    node_rests = np.full((n_thin, n_nodes), spacing, dtype=np.float32)
    mono_node, mono_xi = _interpolate_monomers(mono_offsets, node_offsets)

    # ------------------------------------------------------------------ tropomyosin units
    if tm_monomers_per_unit is None:
        # One unit per binding candidate, on the candidate's strand; every other
        # monomer maps to the nearest unit on its own strand (never read by the
        # kinetics, since only candidates can be bound).
        assert all(sum(len(f) for f in thin_faces) == cand_mono.shape[1] for thin_faces in faces), \
            "a monomer is a candidate for two faces"
        tm_rep_mono = cand_mono
        tm_chains = tm_rep_mono % 2
        mono_tm = _nearest_same_strand_unit(mono_offsets, mono_strand, tm_rep_mono, tm_chains)
    else:
        mono_tm, tm_chains, tm_rep_mono = _tropomyosin_units(
            mono_offsets, mono_strand, tm_monomers_per_unit)
    tm_prev_neighbor, tm_next_neighbor = _same_strand_neighbours(tm_chains)

    return dict(
        mono_offsets=mono_offsets, mono_angle=mono_angle, mono_strand=mono_strand,
        mono_node=mono_node, mono_xi=mono_xi, mono_tm=mono_tm,
        face_to_monomers=face_to_monomers, n_mono_per_face=n_mono_per_face,
        cand_mono=cand_mono, mono_cand=mono_cand,
        node_offsets=node_offsets, node_rests=node_rests,
        tm_chains=tm_chains, tm_prev_neighbor=tm_prev_neighbor,
        tm_next_neighbor=tm_next_neighbor, tm_rep_mono=tm_rep_mono,
    )


def _max_candidates_per_unit(cand_mono: np.ndarray, mono_tm: np.ndarray, n_tm: int) -> int:
    """The most binding candidates any tropomyosin unit covers = the most heads that can be bound
    on one unit at once (a candidate holds at most one head). Sets how many lock levels
    (1 + xb_tm_K2)^n the tropomyosin generator needs; static, so it is an aux field."""
    counts = np.zeros((cand_mono.shape[0], n_tm), dtype=np.int64)
    for t in range(cand_mono.shape[0]):
        m = cand_mono[t][cand_mono[t] >= 0]
        np.add.at(counts[t], mono_tm[t, m], 1)
    return int(counts.max())


def _interpolate_monomers(mono_offsets: np.ndarray, node_offsets: np.ndarray):
    """Where each monomer sits on the node grid: (mono_node, mono_xi).

    A monomer between node e and node e+1 (toward the Z-disc; index n_nodes is
    the Z-disc anchor at offset 0) takes the displacement
    (1 - xi) * u[e] + xi * u[e+1]. Computed in float64 from the rest offsets, so
    a monomer that coincides with a node gets xi = 0 exactly. A monomer on the
    M-line side of node 0 clamps to (0, 0) and follows node 0 rigidly.
    """
    n_thin, n_mono = mono_offsets.shape
    n_nodes = node_offsets.shape[1]
    grid = np.concatenate([node_offsets.astype(np.float64), np.zeros((n_thin, 1))], axis=1)
    off = mono_offsets.astype(np.float64)
    mono_node = np.empty((n_thin, n_mono), dtype=np.int32)
    mono_xi = np.empty((n_thin, n_mono), dtype=np.float32)
    for t in range(n_thin):
        e = np.clip(np.searchsorted(-grid[t], -off[t], side='right') - 1, 0, n_nodes - 1)
        xi = (grid[t, e] - off[t]) / (grid[t, e] - grid[t, e + 1])
        mono_node[t] = e
        mono_xi[t] = np.clip(xi, 0.0, 1.0)
    return mono_node, mono_xi


def _same_strand_neighbours(tm_chains: np.ndarray):
    """Adjacent unit on the same strand, in index order, for every unit.

    Units must be indexed in axial order within each strand. Strand ends
    self-reference rather than using -1, since a -1 index silently wraps to the
    last element on gather.
    """
    n_thin, n_tm = tm_chains.shape
    prev_neighbor = np.tile(np.arange(n_tm, dtype=np.int32), (n_thin, 1))
    next_neighbor = prev_neighbor.copy()
    for t in range(n_thin):
        for c in (0, 1):
            idx = np.where(tm_chains[t] == c)[0]
            prev_neighbor[t, idx[1:]] = idx[:-1]
            next_neighbor[t, idx[:-1]] = idx[1:]
    return prev_neighbor, next_neighbor


def _nearest_same_strand_unit(mono_offsets, mono_strand, tm_rep_mono, tm_chains):
    """(n_thin, n_mono) index of the unit on each monomer's strand whose
    representative monomer is axially nearest to it."""
    n_thin, n_mono = mono_offsets.shape
    rep_off = np.take_along_axis(mono_offsets, tm_rep_mono, axis=1).astype(np.float64)
    mono_tm = np.empty((n_thin, n_mono), dtype=np.int32)
    for t in range(n_thin):
        for c in (0, 1):
            units = np.where(tm_chains[t] == c)[0]
            mono = np.where(mono_strand[t] == c)[0]
            d = np.abs(mono_offsets[t][mono][:, None].astype(np.float64) - rep_off[t][units][None, :])
            mono_tm[t, mono] = units[np.argmin(d, axis=1)]
    return mono_tm


def _tropomyosin_units(mono_offsets: np.ndarray, mono_strand: np.ndarray, per_unit: int):
    """Partition every strand's monomers into tropomyosin units.

    A unit is per_unit consecutive monomers on one strand, counted from the
    monomer nearest the Z-disc; a strand whose length is not a multiple of
    per_unit ends in one short unit at the pointed end. Units are indexed strand
    by strand, and within a strand from the pointed end toward the Z-disc — the
    same direction as the node index, and the axial order
    _same_strand_neighbours relies on.

    Returns:
        mono_tm: (n_thin, n_mono) int32 unit covering each monomer
        tm_chains: (n_thin, n_tm) int32 strand of each unit
        tm_rep_mono: (n_thin, n_tm) int32 the unit's fourth monomer from its
            Z-disc end (its last, for a unit shorter than that)
    """
    n_thin, n_mono = mono_offsets.shape
    assert n_mono % 2 == 0, "the two strands must hold equal monomer counts"
    per_strand = -(-(n_mono // 2) // per_unit)
    mono_tm = np.empty((n_thin, n_mono), dtype=np.int32)
    tm_rep_mono = np.empty((n_thin, 2 * per_strand), dtype=np.int32)
    tm_chains = np.tile(np.repeat(np.arange(2, dtype=np.int32), per_strand), (n_thin, 1))
    for t in range(n_thin):
        for c in (0, 1):
            mono = np.where(mono_strand[t] == c)[0]
            mono = mono[np.argsort(mono_offsets[t, mono], kind='stable')]   # Z-disc end first
            group = np.arange(len(mono)) // per_unit
            unit = c * per_strand + (per_strand - 1 - group)
            mono_tm[t, mono] = unit
            for g in range(per_strand):
                members = mono[group == g]
                tm_rep_mono[t, c * per_strand + per_strand - 1 - g] = members[min(3, len(members) - 1)]
    return mono_tm, tm_chains, tm_rep_mono


# =============================================================================
# FLATTENED INDEX MAP COMPUTATION
# =============================================================================

_LEGACY_FACE_PATTERN = np.array([[0, 2, 4], [1, 3, 5], [0, 2, 4]])  # Level 1, 2, 3


def _compute_flat_index_maps_fixed_width(
    thick_to_thin: np.ndarray,
    face_to_monomers: np.ndarray,
    n_mono_per_face: np.ndarray,
    thick_starts: np.ndarray,
    n_thick: int,
    n_crowns: int,
    n_xb_per_crown: int,
    crown_rotation_deg: float,
    crown_face_wiggle_deg: float,
    legacy_crown_geometry: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert nested Thick->Face->Thin to flat XB->Thin maps with fixed-width arrays.

    KEY OPTIMIZATION: xb_to_mono_indices has shape (total_xbs, max_mono_per_face)
    which is CONSTANT across all XBs. This enables:
    - Single unified jnp.take Gather operation
    - No dynamic-size slicing per XB
    - Full GPU parallelization via vmap

    HOW A HEAD IS ASSIGNED A TARGET. Each crown is rotated azimuthally relative
    to the one below it, and each head on a crown points in its own direction.
    A head can only reach a thin filament that its arm actually points at, so
    the assignment is: compute the head's azimuth, find the nearest of the six
    hexagonal neighbour directions, and accept the match if it falls within
    crown_face_wiggle_deg.

    Whether every head finds a partner depends entirely on the arithmetic:

      3 heads per crown, 60 degrees rotation — every head lands exactly on a
        hexagonal direction. All heads match, and the six faces are used evenly.

      4 heads per crown, 33.75 degrees rotation — 33.75 is not a multiple of
        60, so heads drift out of alignment as the rotation accumulates and
        only about half find a partner. This is a real structural property of
        the geometry, not a modelling failure: the unmatched heads are marked
        invalid in xb_valid and are permanently excluded from binding.

    Setting legacy_crown_geometry=True substitutes a fixed lookup table for the
    azimuth calculation, which is only meaningful for 3 heads per crown. It is
    retained for compatibility with configurations that assume it.

    face_idx here is an abstract 0-5 label, not a physical angle. The mapping
    from label to direction — which differs by 30 degrees between the vertebrate
    and invertebrate lattices — was already resolved when thick_to_thin was
    built, which is what lets this function stay agnostic about which lattice it
    is working on.

    Returns:
        xb_to_thin_id: (total_xbs,) - Target thin filament for each XB
        xb_to_thin_face: (total_xbs,) - Target thin face for each XB
        xb_to_mono_indices: (total_xbs, max_mono_per_face) - FIXED-WIDTH candidate monomers
            Padded with neutral value (first candidate) for GPU parallelization
        xb_valid: (total_xbs,) bool - False where the XB has no real geometric
            thin-filament partner this crown (face didn't match, or the thick
            filament genuinely has no neighbor there). Always True for the
            legacy path. Downstream kinetics (transitions.py) must gate the
            binding rate on this — it is NOT enforced here.
    """
    total_xbs = n_thick * n_crowns * n_xb_per_crown
    max_mono_per_face = face_to_monomers.shape[2]

    xb_to_thin_id = np.zeros(total_xbs, dtype=np.int32)
    xb_to_thin_face = np.zeros(total_xbs, dtype=np.int32)
    xb_to_mono_indices = np.zeros((total_xbs, max_mono_per_face), dtype=np.int32)
    xb_valid = np.ones(total_xbs, dtype=bool)

    for xb_idx in range(total_xbs):
        thick_idx = xb_idx // (n_crowns * n_xb_per_crown)
        local_idx = xb_idx % (n_crowns * n_xb_per_crown)
        crown_idx = local_idx // n_xb_per_crown
        xb_in_crown = local_idx % n_xb_per_crown

        if legacy_crown_geometry:
            # Verbatim original table — bit-identical, only valid at n_xb_per_crown==3
            crown_level = (crown_idx + thick_starts[thick_idx] - 1) % 3 + 1
            face_idx = _LEGACY_FACE_PATTERN[crown_level - 1, xb_in_crown]
            matched = True
        else:
            eff_crown = crown_idx + (thick_starts[thick_idx] - 1)
            azimuth_deg = (xb_in_crown * (360.0 / n_xb_per_crown)
                           + eff_crown * crown_rotation_deg) % 360.0
            nearest_face = int(round(azimuth_deg / 60.0)) % 6
            residual_deg = abs(azimuth_deg - nearest_face * 60.0)
            residual_deg = min(residual_deg, 360.0 - residual_deg)
            matched = residual_deg < crown_face_wiggle_deg
            face_idx = nearest_face

        if matched:
            # Get thin filament and face from thick_to_thin connectivity
            thin_idx = thick_to_thin[thick_idx, face_idx, 0]
            thin_face = thick_to_thin[thick_idx, face_idx, 1]
        else:
            thin_idx = -1  # no geometric partner this crown

        # Record validity BEFORE the unconnected-face remap below — covers both
        # causes uniformly (formula miss, or thick_to_thin itself being -1).
        xb_valid[xb_idx] = thin_idx >= 0

        # Handle unconnected faces (thin_idx == -1)
        if thin_idx < 0:
            thin_idx = 0
            thin_face = 0

        xb_to_thin_id[xb_idx] = thin_idx
        xb_to_thin_face[xb_idx] = thin_face

        # FIXED-WIDTH: Copy all site indices, pad with first site for unused slots
        n_valid = n_mono_per_face[thin_idx, thin_face]
        mono_indices = face_to_monomers[thin_idx, thin_face, :]

        # Get first candidate for padding (or 0 if no valid sites)
        first_valid = mono_indices[0] if n_valid > 0 else 0

        # Pad invalid slots with first candidate (neutral for distance calculation)
        padded_indices = np.where(
            np.arange(max_mono_per_face) < n_valid,
            mono_indices,
            first_valid
        )
        xb_to_mono_indices[xb_idx] = padded_indices

    return xb_to_thin_id, xb_to_thin_face, xb_to_mono_indices, xb_valid


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
    static, dynamic, *_ = get_skeletal_params()
    geometry = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
    print(f"Geometry: {geometry}")
    print(f"  thick_to_thin shape: {geometry.thick_to_thin.shape}")
    print(f"  thin_to_thick shape: {geometry.thin_to_thick.shape}")
    print(f"  xb_to_thin_id shape: {geometry.xb_to_thin_id.shape}")
    print(f"  xb_to_mono_indices shape: {geometry.xb_to_mono_indices.shape}")

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
