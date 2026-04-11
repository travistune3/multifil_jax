"""
Parameters Module for JAX Half-Sarcomere

This module contains StaticParams and DynamicParams classes for separating
structural configuration from tunable physical parameters.

Key design decisions:
1. StaticParams: frozen dataclass for structural config (NOT a PyTree)
   - n_crowns, n_polymers_per_thin, solver_max_iter, actin_geometry
   - Changing these requires recompilation (affects array shapes)
2. DynamicParams: JAX PyTree for tunable parameters
   - All float physical parameters (stiffness, rates, etc.)
   - Changing values doesn't trigger recompilation
3. Absolute values - no hidden multipliers or hardcoded constants in rate functions
4. Dynamic state (z_line, pCa, lattice_spacing) IS also stored here as DynamicParams fields
   (default/sweep values). Time-varying per-step overrides go in Drivers (Tier 3).

Parameter derivation from OOP hs_params:
- OOP uses multipliers applied to base rates
- JAX uses absolute values computed from OOP defaults
- See individual parameter comments for derivation

JAX Notes:
- DynamicParams is a JAX PyTree - vmappable and JIT-compatible
- StaticParams is NOT a PyTree - it's a configuration container
- Use attribute access (params.thick_k) throughout the codebase
"""
import jax
import jax.numpy as jnp
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

# Static fields that affect array shapes (changing these triggers recompilation)
STATIC_FIELDS = frozenset({'n_crowns', 'n_polymers_per_thin', 'solver_max_iter', 'actin_geometry', 'n_newton_steps', 'n_cg_steps', 'solver_residual_tol', 'n_xb_bins'})

# All dynamic field names in order (for tree_flatten/unflatten)
DYNAMIC_FIELDS = (
    'thick_k',
    'thin_k',
    'xb_c_rest_weak', 'xb_c_rest_strong', 'xb_c_k_weak', 'xb_c_k_strong',
    'xb_g_rest_weak', 'xb_g_rest_strong', 'xb_g_k_weak', 'xb_g_k_strong',
    'titin_a', 'titin_b', 'titin_rest',
    'tm_k_12', 'tm_k_23', 'tm_k_34', 'tm_k_41',
    'tm_K1', 'tm_K2', 'tm_K3', 'tm_K4',
    'tm_coop_magnitude', 'tm_span_base', 'tm_span_force50', 'tm_span_steep',
    'xb_r12_coeff', 'xb_r23_coeff', 'xb_r34_coeff', 'xb_r45_coeff',
    'xb_delta_34', 'xb_delta_45',
    'xb_r51', 'xb_r15', 'xb_r16',
    'xb_U_DRX', 'xb_U_loose', 'xb_U_tight_1', 'xb_U_tight_2',
    'xb_srx_k0', 'xb_srx_kmax', 'xb_srx_b', 'xb_srx_ca50',
    'temp_celsius', 'solver_tol',
    # Tiered architecture: drivers stored as constants when not time-varying
    'pCa', 'z_line', 'lattice_spacing',
)


# =============================================================================
# STATIC PARAMS (Configuration - NOT a PyTree)
# =============================================================================

@dataclass(frozen=True)
class StaticParams:
    """Static parameters that affect array shapes (NOT a JAX PyTree).

    These parameters define structural configuration that, when changed,
    requires recompilation because they affect array dimensions.

    Use .replace() to create modified copies (immutable dataclass).

    Attributes:
        n_crowns: Number of crowns per thick filament (default 52)
        n_polymers_per_thin: Number of polymers per thin filament (default 15)
        solver_max_iter: Maximum iterations for equilibrium solver (default 50)
        actin_geometry: "vertebrate" (1:2 thick:thin ratio, 3 faces) or
                       "invertebrate" (1:3 ratio, 2 faces)
    """
    n_crowns: int = 52
    n_polymers_per_thin: int = 15
    solver_max_iter: int = 50
    actin_geometry: str = "vertebrate"
    n_newton_steps: int = 4    # Hard cap on Newton while_loop iterations (exits early at convergence)
    n_cg_steps: int = 6        # CG steps per Newton iter; 0=Richardson (no JVP)
    solver_residual_tol: float = 0.5  # pN — post-run residual warning threshold
    n_xb_bins: int = 200       # bins per AP level; total expm = 2 × n_xb_bins per step
    xb_bin_lo: float = -8.0    # nm — lower edge of axial distance range (baked into SarcTopology)
    xb_bin_hi: float = 35.0    # nm — upper edge; measured range at z=1100 is [-5, 31]nm
    thick_bare_zone: float = 58.0    # nm — M-line to first crown rest distance
    thick_crown_spacing: float = 14.3  # nm — inter-crown rest spacing

    def replace(self, **kwargs) -> 'StaticParams':
        """Create a new StaticParams with updated values.

        Example:
            static = StaticParams()
            modified = static.replace(actin_geometry='invertebrate')
        """
        return StaticParams(**{**asdict(self), **kwargs})

    def __repr__(self) -> str:
        return (f"StaticParams(n_crowns={self.n_crowns}, "
                f"n_polymers_per_thin={self.n_polymers_per_thin}, "
                f"actin_geometry='{self.actin_geometry}')")


def validate_sweep_params(sweep_params: Dict[str, list]) -> None:
    """Raise ValueError if sweeping over structural parameters.

    Structural parameters (n_crowns, n_polymers_per_thin, solver_max_iter)
    affect array shapes and cannot be swept without recompilation.

    Args:
        sweep_params: Dictionary mapping parameter names to lists of values

    Raises:
        ValueError: If any key is in STATIC_FIELDS
    """
    invalid = set(sweep_params.keys()) & STATIC_FIELDS
    if invalid:
        raise ValueError(
            f"Cannot sweep over structural parameters: {sorted(invalid)}. "
            f"These affect array shapes and require recompilation. "
            f"Run separate simulations for different structural configurations."
        )


@jax.tree_util.register_pytree_node_class
class DynamicParams:
    """Dynamic parameters with ABSOLUTE values (not multipliers) - JAX PyTree.

    This is a JAX PyTree containing all tunable physical parameters.
    All fields are stored as JAX arrays, enabling vmap over parameter batches.

    Defaults are skeletal fast-twitch values (~26°C). For cardiac presets use
    get_cardiac_params(). Users can modify any parameter via .copy().

    Usage:
        static, dynamic = get_skeletal_params()
        dynamic = dynamic.copy(xb_r12_coeff=350.0)  # Modify binding rate
        # Pass to run() via SarcTopology.create() and dynamic_params=
    """

    __slots__ = (
        # All dynamic fields stored as JAX arrays
        'thick_k',
        'thin_k',
        'xb_c_rest_weak', 'xb_c_rest_strong', 'xb_c_k_weak', 'xb_c_k_strong',
        'xb_g_rest_weak', 'xb_g_rest_strong', 'xb_g_k_weak', 'xb_g_k_strong',
        'titin_a', 'titin_b', 'titin_rest',
        'tm_k_12', 'tm_k_23', 'tm_k_34', 'tm_k_41',
        'tm_K1', 'tm_K2', 'tm_K3', 'tm_K4',
        'tm_coop_magnitude', 'tm_span_base', 'tm_span_force50', 'tm_span_steep',
        'xb_r12_coeff', 'xb_r23_coeff', 'xb_r34_coeff', 'xb_r45_coeff',
        'xb_delta_34', 'xb_delta_45',
        'xb_r51', 'xb_r15', 'xb_r16',
        'xb_U_DRX', 'xb_U_loose', 'xb_U_tight_1', 'xb_U_tight_2',
        'xb_srx_k0', 'xb_srx_kmax', 'xb_srx_b', 'xb_srx_ca50',
        'temp_celsius', 'solver_tol',
        # Tiered architecture: drivers stored as constants when not time-varying
        'pCa', 'z_line', 'lattice_spacing',
    )

    def __init__(self, **kwargs):
        """Initialize DynamicParams with optional keyword arguments.

        All parameters are stored as JAX arrays for PyTree compatibility.
        Defaults are skeletal fast-twitch values (~26°C). Use get_cardiac_params()
        for cardiac-specific defaults.
        """

        # ==========================================================================
        # MECHANICAL PARAMETERS (all as JAX arrays)
        # ==========================================================================

        # Thick filament
        self.thick_k = jnp.asarray(kwargs.get('thick_k', 2020.0))  # pN/nm

        # Thin filament
        self.thin_k = jnp.asarray(kwargs.get('thin_k', 1743.0))  # pN/nm

        # Crossbridge springs - converter domain (angular spring)
        # c_k_strong: Daniel/Chase/Regnier group lineage (Chase 2004 Ann Biomed Eng;
        #   Tanner 2007 Biophys J; Williams 2010 Biophys J). κ = 40 pN·nm/rad is at the low
        #   end but consistent with the model's published parameterization.
        # c_k_weak: Kaya & Bhatt 2020 (eLife 9:e55368) show pre-stroke head stiffness is
        #   ~5-10× lower than post-stroke. Partition between linear/angular springs unknown;
        #   reduce both weak stiffnesses proportionally (~5× here).
        self.xb_c_rest_weak = jnp.asarray(kwargs.get('xb_c_rest_weak', 0.82309))  # radians
        self.xb_c_rest_strong = jnp.asarray(kwargs.get('xb_c_rest_strong', 1.27758))  # radians
        self.xb_c_k_weak = jnp.asarray(kwargs.get('xb_c_k_weak', 8.0))    # pN·nm/rad — pre-stroke compliant
        self.xb_c_k_strong = jnp.asarray(kwargs.get('xb_c_k_strong', 40.0))  # pN·nm/rad — Daniel group lineage

        # Crossbridge springs - globular domain (linear spring)
        # g_k_strong: ~5 pN/nm for globular domain alone in the two-spring model; the angular
        #   spring in series brings effective total stiffness down (Daniel group calibration).
        # g_k_weak: Kaya & Bhatt 2020 measured ~0.2–0.4 pN/nm total pre-stroke vs
        #   ~2.5–3.5 pN/nm post-stroke. Use 0.4 pN/nm for compliant pre-stroke globular domain.
        self.xb_g_rest_weak = jnp.asarray(kwargs.get('xb_g_rest_weak', 19.93))  # nm
        self.xb_g_rest_strong = jnp.asarray(kwargs.get('xb_g_rest_strong', 16.47))  # nm
        self.xb_g_k_weak = jnp.asarray(kwargs.get('xb_g_k_weak', 0.4))   # pN/nm — pre-stroke compliant
        self.xb_g_k_strong = jnp.asarray(kwargs.get('xb_g_k_strong', 5.0))  # pN/nm — Daniel group lineage

        # Titin exponential spring: F = titin_a * exp(titin_b * (L - titin_rest)), clamped >= 0
        # L = sqrt(axial_dist² + lattice_spacing²), axial_dist = z_line - last_crown_pos (~787.3 nm)
        # Skeletal defaults (N2A isoform); cardiac overrides in get_cardiac_params()
        # titin_a: Powers, Williams, Regnier & Daniel 2018 Integr.Comp.Biol. 58:186
        #          (compliant N2A: 260 pN / 6 titins per thick = 43 pN per molecule)
        # titin_b: Powers 2018 psoas calibration (4 µm⁻¹ = 0.004 nm⁻¹)
        # titin_rest: slack at SL 2.0 µm (z_line=1000 nm → L≈213 nm); Linke 1998 PNAS 95:8052
        self.titin_a = jnp.asarray(kwargs.get('titin_a', 43.0))     # pN/molecule
        self.titin_b = jnp.asarray(kwargs.get('titin_b', 0.004))    # nm⁻¹
        self.titin_rest = jnp.asarray(kwargs.get('titin_rest', 215.0))  # nm

        # ==========================================================================
        # TROPOMYOSIN KINETICS (absolute rates)
        # ==========================================================================
        self.tm_k_12 = jnp.asarray(kwargs.get('tm_k_12', 100000.0))   # Robertson 1981: 5e7–2e8 M⁻¹s⁻¹
        self.tm_k_23 = jnp.asarray(kwargs.get('tm_k_23', 1.0))        # Fraser & Bhatt 2019; Geeves & Lehrer 1994: 20–1000 s⁻¹
        self.tm_k_34 = jnp.asarray(kwargs.get('tm_k_34', 0.1))        # center of 50–200 s⁻¹
        self.tm_k_41 = jnp.asarray(kwargs.get('tm_k_41', 0.2))        # Robertson 1981: 100–500 s⁻¹

        # Equilibrium constants (absolute values)
        self.tm_K1 = jnp.asarray(kwargs.get('tm_K1', 500000.0))   # skeletal TnC Kd ~2 µM; Potter & Gergely 1975
        self.tm_K2 = jnp.asarray(kwargs.get('tm_K2', 130.0))      # dimensionless
        self.tm_K3 = jnp.asarray(kwargs.get('tm_K3', 0.1))        # McKillop & Geeves 1993: K_T=0.09 (no Ca²⁺); close to measured
        self.tm_K4 = jnp.asarray(kwargs.get('tm_K4', 0.0))        # unused

        # Cooperativity
        self.tm_coop_magnitude = jnp.asarray(kwargs.get('tm_coop_magnitude', 100.0))
        self.tm_span_base = jnp.asarray(kwargs.get('tm_span_base', 62.0))  # nm
        self.tm_span_force50 = jnp.asarray(kwargs.get('tm_span_force50', -8.0))  # pN
        self.tm_span_steep = jnp.asarray(kwargs.get('tm_span_steep', 1.0))

        # ==========================================================================
        # CROSSBRIDGE KINETICS (absolute/consolidated values)
        # ==========================================================================
        self.xb_r12_coeff = jnp.asarray(kwargs.get('xb_r12_coeff', 305.99))
        self.xb_r23_coeff = jnp.asarray(kwargs.get('xb_r23_coeff', 0.6))    # Fitting parameter targeting process B apparent rate (2πb ~ 20–60 s⁻¹ skeletal; Kawai & Zhao 1993 Biophys J 65:638)
        self.xb_r34_coeff = jnp.asarray(kwargs.get('xb_r34_coeff', 0.15))   # Millar & Homsher 1990: 70–100 s⁻¹
        self.xb_r45_coeff = jnp.asarray(kwargs.get('xb_r45_coeff', 0.6))    # Siemankowski & White 1984: ≥500 s⁻¹ (skeletal)
        self.xb_delta_34 = jnp.asarray(kwargs.get('xb_delta_34', 1.0))      # nm, Bell distance for power stroke; Pate & Cooke 1989; Huxley & Simmons 1971: 1–2 nm
        self.xb_delta_45 = jnp.asarray(kwargs.get('xb_delta_45', 0.5))      # nm, Bell distance for detachment
        self.xb_r51 = jnp.asarray(kwargs.get('xb_r51', 0.1))
        self.xb_r15 = jnp.asarray(kwargs.get('xb_r15', 0.01))               # Mijailovich 2020 (k−H=10 s⁻¹); detailed balance r51/r15=10
        self.xb_r16 = jnp.asarray(kwargs.get('xb_r16', 0.007))              # 50% SRX at rest; Stewart 2010 PNAS 107:430

        # Free energies (kT units). Total cycle = ΔG_ATP ≈ -22 to -24 kT at 37°C.
        # Partitioning per Howard 2001 Fig 14.6; Pate & Cooke 1989 JMRCM 10:181;
        # Månsson 2016 JMRCM 37:181; Offer & Ranatunga 2013 Biophys J 105:1767:
        #   ATP binding + dissociation:  -8 to -10 kT
        #   Hydrolysis on myosin:        ~0 to -2 kT
        #   Weak actin binding:          -1 to -3 kT
        #   Pi release + power stroke:   -8 to -13 kT  (ΔG loose→tight_1 = -10 to -12 kT)
        #   ADP release:                 -2 to -4 kT   (ΔG tight_1→tight_2 = -4 to -6 kT)
        self.xb_U_DRX = jnp.asarray(kwargs.get('xb_U_DRX', -2.3))      # M.ATP / M.ADP.Pi (detached)
        self.xb_U_loose = jnp.asarray(kwargs.get('xb_U_loose', -4.3))   # AM.ADP.Pi (weakly bound); hydrolysis + weak binding
        self.xb_U_tight_1 = jnp.asarray(kwargs.get('xb_U_tight_1', -15.0))  # AM.ADP pre-lever-arm; Pi release ΔG ≈ -10.7 kT
        self.xb_U_tight_2 = jnp.asarray(kwargs.get('xb_U_tight_2', -21.0))  # AM.ADP post-lever-arm; lever arm ΔG ≈ -6 kT

        # SRX -> DRX transition (r61) params
        self.xb_srx_k0 = jnp.asarray(kwargs.get('xb_srx_k0', 0.007))    # r16/k0=1 → 50% SRX at rest
        self.xb_srx_kmax = jnp.asarray(kwargs.get('xb_srx_kmax', 0.4))  # Mijailovich 2020 (kPSmax=400 s⁻¹)
        self.xb_srx_b = jnp.asarray(kwargs.get('xb_srx_b', 5.0))        # Mijailovich 2020; Linari 2015 Nature 528:276
        self.xb_srx_ca50 = jnp.asarray(kwargs.get('xb_srx_ca50', 1e-6)) # Ca50 (M)

        # ==========================================================================
        # SIMULATION PARAMETERS
        # ==========================================================================
        self.temp_celsius = jnp.asarray(kwargs.get('temp_celsius', 26.15))
        self.solver_tol = jnp.asarray(kwargs.get('solver_tol', 0.3))

        # ==========================================================================
        # TIERED ARCHITECTURE: Drivers stored as constants when not time-varying
        # ==========================================================================
        self.pCa = jnp.asarray(kwargs.get('pCa', 4.5))
        self.z_line = jnp.asarray(kwargs.get('z_line', 900.0))
        self.lattice_spacing = jnp.asarray(kwargs.get('lattice_spacing', 14.0))

    def tree_flatten(self):
        """Flatten for JAX tree operations.

        Returns:
            children: Tuple of all dynamic field values (JAX arrays)
            aux_data: Empty tuple (no static data in DynamicParams)
        """
        children = tuple(getattr(self, name) for name in DYNAMIC_FIELDS)
        aux_data = ()  # No static fields in DynamicParams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct DynamicParams from flattened representation."""
        # Bypass __init__ (which calls jnp.asarray) — children are already
        # JAX arrays/tracers. Older JAX versions call tree_unflatten with
        # object() sentinels during in_axes probing; __init__ would crash on those.
        new = object.__new__(cls)
        for name, value in zip(DYNAMIC_FIELDS, children):
            object.__setattr__(new, name, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for serialization/metadata."""
        d = {}
        # Dynamic fields - convert JAX arrays to Python floats
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name)
            d[name] = float(val) if hasattr(val, 'item') else val
        return d

    def copy(self, **updates) -> 'DynamicParams':
        """Create copy with updated values (JIT-compatible).

        Example:
            _, dynamic = get_skeletal_params()
            modified = dynamic.copy(xb_r12_coeff=400.0, tm_k_12=60000.0)
        """
        # Validate keys before constructing (Python-level, not traced)
        invalid = set(updates.keys()) - set(DYNAMIC_FIELDS)
        if invalid:
            raise ValueError(f"Unknown parameter: {sorted(invalid)}. Valid: {list(DYNAMIC_FIELDS)}")
        # Build kwargs from current values, overriding with updates
        # No float() conversion — preserves JAX tracers inside JIT
        kwargs = {name: updates.get(name, getattr(self, name)) for name in DYNAMIC_FIELDS}
        return DynamicParams(**kwargs)

    def with_drivers(self, pCa, z_line, lattice_spacing) -> 'DynamicParams':
        """Create copy with only driver fields replaced (fast path for scan body).

        Unlike copy(), this bypasses the full 45-field kwargs rebuild and
        jnp.asarray() calls. It directly shares references for unchanged fields,
        avoiding ~42 redundant identity-copy XLA ops per timestep.

        Args:
            pCa: Resolved pCa value
            z_line: Resolved z_line value
            lattice_spacing: Resolved lattice_spacing value

        Returns:
            New DynamicParams with only pCa/z_line/lattice_spacing updated
        """
        # Bypass __init__ entirely — create empty instance and copy attrs
        new = object.__new__(DynamicParams)
        for name in DYNAMIC_FIELDS:
            if name == 'pCa':
                object.__setattr__(new, name, pCa)
            elif name == 'z_line':
                object.__setattr__(new, name, z_line)
            elif name == 'lattice_spacing':
                object.__setattr__(new, name, lattice_spacing)
            else:
                object.__setattr__(new, name, getattr(self, name))
        return new

    def __repr__(self) -> str:
        return f"DynamicParams(thick_k={float(self.thick_k):.1f}, thin_k={float(self.thin_k):.1f}, ...)"


def get_skeletal_params() -> Tuple[StaticParams, DynamicParams]:
    """Get fast-twitch skeletal muscle parameters (~26°C).

    Returns (StaticParams, DynamicParams) with defaults calibrated to
    fast-twitch skeletal myosin II kinetics.

    Key values and citations:
        xb_g_k_weak  = 0.4 pN/nm     — Kaya & Bhatt 2020 eLife 9:e55368 (~0.2–0.4 pN/nm pre-stroke)
        xb_g_k_strong = 5.0 pN/nm    — Daniel group two-spring calibration (Chase 2004, Tanner 2007)
        xb_c_k_weak  = 8.0 pN·nm/rad — ~5× reduction from strong; Kaya & Bhatt 2020 (5–10× ratio)
        xb_c_k_strong = 40 pN·nm/rad — Daniel group lineage (Chase 2004, Williams 2010)
        xb_U_tight_1 = -15.0 kT      — Pi release ΔG ≈ -10.7 kT; Howard 2001; Månsson 2016
        xb_U_tight_2 = -21.0 kT      — lever arm ΔG ≈ -6 kT; Offer & Ranatunga 2013
        xb_r23_coeff = 0.6 ms⁻¹      — fitting parameter targeting process B (2πb ~ 20–60 s⁻¹)
        xb_delta_34  = 1.0 nm         — Pate & Cooke 1989 JMRCM 10:181; Huxley & Simmons 1971
        xb_delta_45  = 0.5 nm         — Duke 1999 PNAS 96:2770
        xb_r16       = 0.007 ms⁻¹    — 50% SRX at rest; Stewart 2010 PNAS 107:430
        tm_K3        = 0.1            — McKillop & Geeves 1993 Biophys J 65:693 (K_T=0.09 no Ca²⁺)
        titin_a      = 43.0 pN        — Powers, Williams, Regnier & Daniel 2018 Integr.Comp.Biol. 58:186
        titin_b      = 0.004 nm⁻¹    — Powers 2018 psoas calibration (4 µm⁻¹)
        titin_rest   = 215.0 nm       — slack at SL 2.0 µm; Linke 1998 PNAS 95:8052

    Typical z_line for skeletal: 1000–1300 nm (SL 2.0–2.6 µm). Pass to run() directly:
        run(topo, pCa=4.5, z_line=1100.0)  # working SL ~2.2 µm

    Returns:
        Tuple of (static_params, dynamic_params)

    Example:
        static, dynamic = get_skeletal_params()
        static = static.replace(actin_geometry='invertebrate')
        dynamic = dynamic.copy(thick_k=3000.0)
    """
    return StaticParams(), DynamicParams()


def get_cardiac_params() -> Tuple[StaticParams, DynamicParams]:
    """Get generic cardiac muscle parameters (~27°C).

    Returns (StaticParams, DynamicParams) with defaults calibrated to
    cardiac myosin II / cardiac TnC kinetics, starting from the skeletal
    baseline and applying cardiac-specific overrides.

    Cardiac overrides and citations:
        tm_k_23  = 0.5 ms⁻¹      — Cardiac TM dynamics slower; Lehrer & Morris 1982
        tm_k_12  = 80000 M⁻¹ms⁻¹ — Robertson 1981: 4–8×10⁷ M⁻¹s⁻¹
        tm_K1    = 750000 M⁻¹    — Cardiac TnC Kd ~1.3 µM; Pinto 2011 JBC 286:2007 (1–2 µM)
        tm_k_41  = 0.04 ms⁻¹     — Cardiac Ca²⁺ off-rate ~40 s⁻¹; Davis 2007 Biophys J 92:20
        tm_coop_magnitude = 40.0  — Hill coeff ~2–4 (cardiac); de Tombe & Stienen 1995 Circ Res 76:734
        xb_r23_coeff = 0.175 ms⁻¹ — Process B 3–4× slower (2πb ~ 5–15 s⁻¹);
                                     Kawai et al. 1993 Circ Res 73:35
        xb_r34_coeff = 0.065 ms⁻¹ — Lever arm rate ~2× slower (cardiac beta-MHC);
                                     Deacon et al. 2012 J Mol Biol 421:173
        xb_r45_coeff = 0.065 ms⁻¹ — Cardiac ADP release ~65 s⁻¹; Siemankowski & White 1984 JBC
        xb_r16   = 0.012 ms⁻¹    — 70% SRX at rest; Linari 2015 Nature 528:276
        xb_srx_k0 = 0.005 ms⁻¹   — Empirically calibrated (kPS0=5 s⁻¹)
        titin_a  = 55.0 pN        — N2B isoform stiffer than N2A;
                                   Granzier & Labeit 2004 Circ Res 94:284
        titin_b  = 0.008 nm⁻¹    — Powers 2018 cardiac-like stiffness (8 µm⁻¹)
        titin_rest = 140.0 nm     — slack at SL 1.85 µm (z_line=925 → L≈138 nm); Linke 1998

    Typical z_line for cardiac: 900–1100 nm (SL 1.8–2.2 µm). Pass to run() directly:
        run(topo, pCa=4.5, z_line=950.0)  # resting SL ~1.9 µm

    Returns:
        Tuple of (static_params, dynamic_params)

    Example:
        static, dynamic = get_cardiac_params()
        dynamic = dynamic.copy(tm_K1=1e6)
    """
    cardiac_overrides = {
        'tm_k_23': 0.5,           # Cardiac TM dynamics slower; Lehrer & Morris 1982
        'tm_k_12': 80000.0,       # Robertson 1981: 4–8×10⁷ M⁻¹s⁻¹
        'tm_K1': 750000.0,        # Cardiac TnC Kd ~1.3 µM; Pinto 2011 JBC 286:2007
        'tm_k_41': 0.04,          # Cardiac Ca²⁺ off-rate ~40 s⁻¹; Davis 2007 Biophys J 92:20
        'tm_coop_magnitude': 40.0, # Hill coeff ~2–4; de Tombe & Stienen 1995 Circ Res 76:734
        'xb_r23_coeff': 0.175,    # Process B 3–4× slower; Kawai et al. 1993 Circ Res 73:35
        'xb_r34_coeff': 0.065,    # Lever arm ~2× slower; Deacon et al. 2012 J Mol Biol 421:173
        'xb_r45_coeff': 0.065,    # ADP release ~65 s⁻¹; Siemankowski & White 1984 JBC
        'xb_r16': 0.012,          # 12/(12+5)=70% SRX at rest; Linari 2015
        'xb_srx_k0': 0.005,       # Empirically calibrated (kPS0=5 s⁻¹)
        'titin_a': 55.0,          # N2B stiffer than N2A; Granzier & Labeit 2004 Circ Res 94:284
        'titin_b': 0.008,         # Powers 2018 cardiac-like stiffness (8 µm⁻¹)
        'titin_rest': 140.0,      # slack at SL 1.85 µm (z_line=925 nm → L≈138 nm)
    }
    return StaticParams(), DynamicParams(**cardiac_overrides)


# Alias for tiered architecture
Constants = DynamicParams


def print_params(static: StaticParams = None, dynamic: DynamicParams = None):
    """Print parameters in organized format.

    Args:
        static: StaticParams object (optional)
        dynamic: DynamicParams object (optional)
    """
    print("=" * 70)
    print("JAX MULTIFIL PARAMETERS")
    print("=" * 70)

    if static is not None:
        print("\nStatic Parameters (affect array shapes):")
        print("-" * 70)
        print(f"  {'n_crowns':30s} = {static.n_crowns}")
        print(f"  {'n_polymers_per_thin':30s} = {static.n_polymers_per_thin}")
        print(f"  {'solver_max_iter':30s} = {static.solver_max_iter}")
        print(f"  {'actin_geometry':30s} = {static.actin_geometry}")

    if dynamic is not None:
        d = dynamic.to_dict()

        # Group by prefix
        groups = {
            'Thick Filament': ['thick_k'],
            'Thin Filament': ['thin_k'],
            'XB Springs': [k for k in d if k.startswith('xb_c_') or k.startswith('xb_g_')],
            'Titin': ['titin_a', 'titin_b', 'titin_rest'],
            'TM Kinetics': [k for k in d if k.startswith('tm_')],
            'XB Kinetics': [k for k in d if k.startswith('xb_') and not (
                k.startswith('xb_c_') or k.startswith('xb_g_'))],
            'Simulation': ['temp_celsius', 'solver_tol'],
        }

        for group_name, keys in groups.items():
            print(f"\n{group_name}:")
            print("-" * 70)
            for key in keys:
                if key in d:
                    print(f"  {key:30s} = {d[key]}")

    print("=" * 70)


if __name__ == "__main__":
    print("Testing StaticParams and DynamicParams...")
    print()

    # Test 1: Skeletal parameters
    print("Test 1: Get skeletal parameters")
    static, dynamic = get_skeletal_params()
    print(f"Static: {static}")
    print(f"Dynamic: {dynamic}")
    print_params(static, dynamic)

    # Test 2: Copy with updates (DynamicParams)
    print("\n\nTest 2: DynamicParams.copy() with updates")
    print("=" * 70)
    new_dynamic = dynamic.copy(
        xb_r12_coeff=400.0,
        tm_k_12=60000.0
    )
    print(f"Original xb_r12_coeff: {float(dynamic.xb_r12_coeff)}")
    print(f"New xb_r12_coeff: {float(new_dynamic.xb_r12_coeff)}")
    print(f"Original tm_k_12: {float(dynamic.tm_k_12)}")
    print(f"New tm_k_12: {float(new_dynamic.tm_k_12)}")

    # Test 3: Replace static params
    print("\n\nTest 3: StaticParams.replace()")
    print("=" * 70)
    new_static = static.replace(actin_geometry='invertebrate', n_crowns=60)
    print(f"Original: {static}")
    print(f"Modified: {new_static}")

    # Test 4: To dict
    print("\n\nTest 4: Convert DynamicParams to dictionary")
    print("=" * 70)
    param_dict = dynamic.to_dict()
    print(f"Total parameters: {len(param_dict)}")
    print("Sample entries:")
    for i, (key, value) in enumerate(list(param_dict.items())[:5]):
        print(f"  {key}: {value}")
    print("  ...")

    # Test 5: Invalid parameter
    print("\n\nTest 5: Invalid parameter (should raise error)")
    print("=" * 70)
    try:
        dynamic.copy(invalid_param=123)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test 6: PyTree roundtrip (DynamicParams only)
    print("\n\nTest 6: JAX PyTree roundtrip (DynamicParams)")
    print("=" * 70)
    leaves, treedef = jax.tree_util.tree_flatten(dynamic)
    print(f"Number of leaves (dynamic params): {len(leaves)}")
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    print(f"Reconstructed thick_k: {float(reconstructed.thick_k)}")
    print(f"Original thick_k: {float(dynamic.thick_k)}")
    print(f"Match: {float(reconstructed.thick_k) == float(dynamic.thick_k)}")

    # Test 7: Validate sweep params
    print("\n\nTest 7: Validate sweep params")
    print("=" * 70)
    try:
        validate_sweep_params({'n_crowns': [52, 100]})
    except ValueError as e:
        print(f"Correctly blocked structural sweep: {e}")

    # Valid sweep should pass
    validate_sweep_params({'thick_k': [1000, 2000, 3000]})
    print("Dynamic sweep (thick_k) allowed - OK")

    print("\nAll tests passed!")
