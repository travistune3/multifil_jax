"""
Mixed populations of motors with different kinetics.

Real muscle is rarely uniform. A heterozygous mutation puts mutant and wild-type
myosin side by side on the same filament. Accessory proteins such as cMyBP-C
decorate only the central stripes of the thick filament, so heads there behave
differently from heads at the ends. This module lets a simulation contain
several populations of motors that share the same mechanics but obey different
rates.

WHAT A SUBPOPULATION IS
-----------------------
A list of K populations, index 0 being wild-type. Each carries a dict of
MULTIPLICATIVE scale factors applied to whatever DynamicParams was passed to
run(). Scaling rather than absolute values means one Subpopulation describes a
perturbation that can be applied on top of any baseline — a mutation defined
once works against a skeletal, cardiac or fitted parameter set alike.

Scales must name xb_* or tm_* fields. Mechanical parameters always take the
wild-type value: populations differ in how fast they cycle, never in how stiff
they are. That is a modelling limitation, and an intentional one — differing
stiffness would mean differing forces at the same strain, which the force
kernels are not structured to represent per motor.

TWO WAYS TO MIX
---------------
mean_field
    No spatial assignment at all. The K rate matrices are averaged with the
    population fractions as weights, Q_eff = sum_k f_k Q_k, and every motor uses
    the result. Note this averages GENERATORS, before exponentiating — not
    parameters, and not probabilities. Averaging parameters would be wrong
    because rates depend nonlinearly on them; averaging probabilities would be
    wrong because it double-counts within-step transitions. Deterministic, and
    the cheapest option: still one matrix exponential.

random / c_zone
    Genuine spatial assignment. Every motor and site carries an integer label,
    each population is exponentiated separately, and each unit selects by label.
    Costs K times the exponentials but captures something mean-field cannot:
    local clustering. A patch of mutant heads on one filament interacts through
    filament compliance and cooperativity in a way that a uniformly averaged
    motor does not.

    random draws labels per simulation from seed + simulation index, so
    replicates sample different arrangements and provide statistical power.
    c_zone assigns by axial position and is fully deterministic.

Comparing mean_field against random with the same fraction is a direct measure
of how much spatial heterogeneity matters for a given question.

EXACTNESS AT THE EDGES
----------------------
fraction = 0 reproduces wild-type bit-for-bit, and fraction = 1 reproduces a
global parameter scale bit-for-bit, in all three modes. These are not
approximations that happen to be close; they are exact, and worth using as
sanity checks when adding a new perturbation.

BOUNDARY
--------
Subpopulation is imported only here and in simulation.py. The kernels never see
this class — they receive plain DynamicParams lists and integer mask arrays,
types they already understand. Keeping the abstraction out of the kernels is
what allows the whole feature to add zero overhead when unused.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from multifil_jax.core.params import DynamicParams, DYNAMIC_FIELDS

# Default cardiac C-zone axial band (nm from the M-line). cMyBP-C decorates the
# central stripes of the thick filament; crowns whose axial offset falls in this
# range are assigned to the mutant/modified population. Adjustable per-call.
C_ZONE_MIN_NM = 350.0
C_ZONE_MAX_NM = 650.0


def _validate_scales(scales: Dict[str, float]) -> Dict[str, float]:
    """Reject unknown field names early (before any device work)."""
    invalid = set(scales) - set(DYNAMIC_FIELDS)
    if invalid:
        raise ValueError(
            f"Unknown subpopulation scale field(s): {sorted(invalid)}. "
            f"Valid DynamicParams fields: {list(DYNAMIC_FIELDS)}"
        )
    return dict(scales)


@dataclass
class Subpopulation:
    """K-general subpopulation config (index 0 = WT).

    Attributes:
        scales: K scale dicts (field name -> multiplicative factor). scales[0]
                is WT ({}); scales[1..] are mutant populations.
        fractions: (K,) population weights, sum = 1. Mean-field blend weights;
                   also the label probabilities used to draw random masks.
        mode: 'mean_field' | 'random' | 'c_zone'.
        xb_mask: (total_xbs,) INT labels 0..K-1, or None (generated at run() for
                 'random', unused for 'mean_field').
        tm_mask: (n_sites_total,) INT labels, or None.
        seed: base seed for run()-time random mask generation.
    """

    scales: List[Dict[str, float]]
    fractions: jnp.ndarray
    mode: str
    xb_mask: Optional[jnp.ndarray] = None
    tm_mask: Optional[jnp.ndarray] = None
    seed: Optional[int] = None

    # ------------------------------------------------------------------ K & fields
    @property
    def K(self) -> int:
        return len(self.scales)

    def scaled_fields(self) -> tuple:
        """Sorted union of every field any population scales (static)."""
        fields = set()
        for s in self.scales:
            fields |= set(s)
        return tuple(sorted(fields))

    def scale_array(self, field_names: tuple) -> jnp.ndarray:
        """(K, F) multiplicative factors in the given field order (WT rows = 1)."""
        return jnp.asarray(
            [[float(s.get(f, 1.0)) for f in field_names] for s in self.scales],
            dtype=jnp.float32,
        )

    # ------------------------------------------------------------------ scaling
    def apply_scales(self, constants: DynamicParams) -> List[DynamicParams]:
        """Multiply each population's scale dict onto the WT baseline.

        Returns a length-K list of DynamicParams (entry 0 == constants unchanged).
        Works on scalar or batched constants and inside JIT (uses .copy()).
        """
        return [_scale_one(constants, s) for s in self.scales]

    # ------------------------------------------------------------------ constructors
    @classmethod
    def mean_field(cls, fraction: float, **scales) -> "Subpopulation":
        """Mean-field blend: all sites see the fraction-weighted average of WT
        and mutant rate matrices. Deterministic, no mask."""
        _validate_scales(scales)
        return cls(
            scales=[{}, dict(scales)],
            fractions=jnp.asarray([1.0 - fraction, fraction], dtype=jnp.float32),
            mode="mean_field",
        )

    @classmethod
    def random(cls, fraction: float, seed: int = 0, **scales) -> "Subpopulation":
        """Stochastic binary: each XB / TM site is independently mutant with
        probability ``fraction``. Masks are drawn fresh per replicate in run()
        (seed + replicate index), so replicates give statistical power. Can be
        passed as a list to sweep severity/fraction, provided every entry in
        the list shares the same ``seed``."""
        _validate_scales(scales)
        return cls(
            scales=[{}, dict(scales)],
            fractions=jnp.asarray([1.0 - fraction, fraction], dtype=jnp.float32),
            mode="random",
            seed=seed,
        )

    @classmethod
    def c_zone(cls, topology, c_zone_min_nm: float = C_ZONE_MIN_NM,
               c_zone_max_nm: float = C_ZONE_MAX_NM, **scales) -> "Subpopulation":
        """Deterministic C-zone band: crossbridges on crowns whose axial offset
        falls in [c_zone_min_nm, c_zone_max_nm] (nm from M-line) are assigned to
        the mutant population. If any tm_* scales are given, thin binding sites in
        the same axial band are also labelled. Can be passed as a list to sweep
        band position/severity across sims."""
        _validate_scales(scales)
        xb_mask, tm_mask = _c_zone_masks(topology, c_zone_min_nm, c_zone_max_nm, scales)
        frac = float(np.mean(np.asarray(xb_mask)))  # empirical mutant fraction
        return cls(
            scales=[{}, dict(scales)],
            fractions=jnp.asarray([1.0 - frac, frac], dtype=jnp.float32),
            mode="c_zone",
            xb_mask=xb_mask,
            tm_mask=tm_mask,
        )


def _scale_one(constants: DynamicParams, scale: Dict[str, float]) -> DynamicParams:
    """Apply one population's multiplicative scale dict to constants."""
    if not scale:
        return constants  # WT — share reference, zero cost
    return constants.copy(**{
        f: jnp.asarray(getattr(constants, f)) * jnp.float32(v) for f, v in scale.items()
    })


def _c_zone_masks(topology, lo_nm: float, hi_nm: float, scales: Dict[str, float]):
    """Build per-XB and per-site INT-label masks for a C-zone band.

    XB label = 1 where the crown axial offset is in [lo, hi]. TM sites are
    labelled by the same axial band only when tm_* scales are present (otherwise
    all WT, so the TM path is untouched).
    """
    crown_off = np.asarray(topology.crown_offsets)                 # (n_thick, n_crowns)
    crown_in = ((crown_off >= lo_nm) & (crown_off <= hi_nm)).astype(np.int32)  # (n_thick, n_crowns)
    # Broadcast crown pattern to every XB: (n_thick, n_crowns, n_xb_per_crown).
    # Per-filament now (not per-crown-index alone) — under a myosin superlattice
    # (n_superlattice_classes > 1), filaments in different classes sit at
    # different axial positions for the same crown index, so C-zone membership
    # is genuinely filament-dependent for the first time.
    xb_mask = np.broadcast_to(
        crown_in[:, :, None],
        (topology.n_thick, topology.n_crowns, topology.n_xb_per_crown),
    ).reshape(-1)

    has_tm = any(f.startswith("tm_") for f in scales)
    if has_tm:
        # Thin binding-site axial rest positions (same frame metrics_fn uses).
        site_pos = np.cumsum(np.asarray(topology.binding_rests), axis=1)  # (n_thin, n_sites)
        tm_mask = ((site_pos >= lo_nm) & (site_pos <= hi_nm)).astype(np.int32).reshape(-1)
    else:
        tm_mask = np.zeros(topology.n_thin * topology.n_sites, dtype=np.int32)

    return jnp.asarray(xb_mask, jnp.int32), jnp.asarray(tm_mask, jnp.int32)


def generate_random_masks_batch(seed: int, p_mut_b: jnp.ndarray, topology, total_batch: int):
    """Vectorized per-sim mask draw for a whole batch.

    Each of the ``total_batch`` simulations (sweep cell × replicate) gets an
    independent Bernoulli(p_mut) label draw keyed by ``seed + sim_index``, so
    replicates provide statistical power and mean_field vs random are comparable.
    ``p_mut_b`` is (total_batch,) — per-sim mutant probability, letting fraction
    vary across a swept sweep axis while sharing one ``seed``.

    Returns:
        xb_mask_b: (total_batch, total_xbs) INT labels
        tm_mask_b: (total_batch, n_sites_total) INT labels
    """
    sim_keys = jax.random.split(jax.random.PRNGKey(int(seed)), total_batch)
    total_xbs = topology.n_thick * topology.n_crowns * topology.n_xb_per_crown
    n_sites_total = topology.n_thin * topology.n_sites

    def _mk(key, p_mut):
        kx, kt = jax.random.split(key)
        xb = (jax.random.uniform(kx, (total_xbs,)) < p_mut).astype(jnp.int32)
        tm = (jax.random.uniform(kt, (n_sites_total,)) < p_mut).astype(jnp.int32)
        return xb, tm

    return jax.vmap(_mk)(sim_keys, p_mut_b)
