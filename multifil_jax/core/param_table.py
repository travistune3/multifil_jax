"""The parameter table: the single source of truth for every physical,
structural and numerical value, per muscle class.

One `row()` per parameter, naming every class explicitly:
    P(nominal, tag, basis, lo=, hi=, scale=, source=)   the class's own value
    inh(parent, lo=, hi=, scale=)                       the parent's value, tag G;
                                                        a range given here must
                                                        contain the parent's
    ratio(parent, factor, lo_factor, hi_factor, tag, basis)
                                                        parent's value x factor;
                                                        range = parent's range
                                                        x [lo_factor, hi_factor]
Tags: M measured, I inferred, G guess / carried over, F fitted (see params.py).
lo = hi = None means no defensible range: `draw` and `sample_params` refuse it.
`scale` is how a range is sampled: 'log' (default) or 'linear'.

Kinds: 'static' -> StaticParams (shapes, compile-time geometry, solver caps),
'dynamic' -> DynamicParams (sweepable), 'driver' -> the preset's z0 / d0.
"""
from typing import NamedTuple, Optional
import numpy as np

CLASSES = ('skeletal', 'cardiac_beta', 'cardiac_alpha', 'insect')

KINDS = ('static', 'dynamic', 'driver')


class P(NamedTuple):
    nominal: object
    tag: str
    basis: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    scale: str = 'log'
    source: str = ''


class Inherit(NamedTuple):
    parent: str
    lo: Optional[float] = None
    hi: Optional[float] = None
    scale: Optional[str] = None


def inh(parent, lo=None, hi=None, scale=None):
    return Inherit(parent, lo, hi, scale)


class Ratio(NamedTuple):
    parent: str
    factor: float
    lo_factor: float
    hi_factor: float
    tag: str
    basis: str


def ratio(parent, factor, lo_factor, hi_factor, tag, basis):
    return Ratio(parent, factor, lo_factor, hi_factor, tag, basis)


class Row(NamedTuple):
    name: str
    kind: str
    unit: str
    entries: dict      # class -> P | Inherit | Ratio


ROWS = {}


def row(name, kind, unit, **entries):
    assert kind in KINDS, (name, kind)
    assert set(entries) == set(CLASSES), (name, sorted(set(CLASSES) ^ set(entries)))
    assert name not in ROWS, name
    ROWS[name] = Row(name, kind, unit, entries)


BG = 'range: generic striated-muscle fitting bounds'
BC = 'range: beta-cardiac fitting bounds (27 C)'

# =============================================================================
# DRIVERS — the preset's natural operating point (run() takes them explicitly)
# =============================================================================
row('z0', 'driver', 'nm',
    skeletal=P(900.0, 'G', 'short for skeletal (SL 1.8 um); working range 1000-1300'),
    cardiac_beta=P(950.0, 'I', 'SL 1.9 um, the length d0 was set for'),
    cardiac_alpha=inh('cardiac_beta'),
    insect=inh('skeletal'))
row('d0', 'driver', 'nm',
    skeletal=P(14.0, 'I', 'd = (2/3) d10 - radii; d10 40.5 nm at SL 1.9 um'),
    cardiac_beta=P(10.3, 'G', 'intact-lattice gap for the crossbridge geometry set'),
    cardiac_alpha=inh('cardiac_beta'),
    insect=inh('skeletal'))

# =============================================================================
# STRUCTURE (static) — array shapes and compile-time geometry
# =============================================================================
row('actin_geometry', 'static', '-',
    skeletal=P('vertebrate', 'M', '1 thick : 2 thin, 3 faces per thin'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P('invertebrate', 'M', '1 thick : 3 thin, 2 faces per thin'))
row('n_crowns', 'static', '-',
    skeletal=P(52, 'I', '~0.79 um half thick filament at 14.3 nm'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(100, 'I', '3.04 um Drosophila thick filament /2 /14.5 nm'))
row('n_polymers_per_thin', 'static', '-',
    skeletal=P(15, 'I', '15 x 72 nm = 1.08 um thin filament'),
    cardiac_beta=P(14, 'I', '14 x 72 nm = 1.01 um cardiac thin filament'),
    cardiac_alpha=inh('cardiac_beta'),
    insect=P(20, 'I', '1.5 um Drosophila thin filament / 77.4 nm'))
row('thick_bare_zone', 'static', 'nm',
    skeletal=P(58.0, 'G', 'M-line to first crown'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('thick_crown_spacing', 'static', 'nm',
    skeletal=P(14.3, 'M', 'inter-crown axial repeat'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(14.5, 'M', 'IFM crown repeat (145 A)'))
row('n_superlattice_classes', 'static', '-',
    skeletal=P(1, 'M', 'simple lattice'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(1, 'M', 'Lethocerus; set 3 for the Drosophila superlattice'))
row('actin_half_pitch', 'static', 'nm',
    skeletal=P(36.0, 'M', 'long-pitch half repeat'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(38.7, 'M', 'IFM half repeat'))
row('mono_per_poly', 'static', '-',
    skeletal=P(26, 'M', '13/6 helix as 26 monomers per 12 turns'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(28, 'M', '28/13 IFM helix'))
row('polymer_base_turns', 'static', '-',
    skeletal=P(12.0, 'M', 'turns per pseudo-repeat'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(13, 'M', 'turns per IFM pseudo-repeat'))
row('target_zone_wiggle', 'static', 'rad',
    skeletal=P(float(np.float32(2 * np.pi) / np.float32(24)), 'G', '15 deg binding window'),
    cardiac_beta=P(float(np.radians(26.0)), 'I', '26 deg window, 2-monomer target zone'),
    cardiac_alpha=inh('cardiac_beta'),
    insect=P(float(np.radians(26.0)), 'I', '26 deg window, 2-monomer target zone'))
row('n_xb_per_crown', 'static', '-',
    skeletal=P(3, 'M', 'one per myosin molecule'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(4, 'M', '4-fold IFM crown'))
row('crown_rotation_deg', 'static', 'deg',
    skeletal=P(60.0, 'M', 'successive crowns on hexagonal neighbours'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'),
    insect=P(33.75, 'I', '3 x 360/32 from the 32/3 helix'))
row('crown_face_wiggle_deg', 'static', 'deg',
    skeletal=P(15.0, 'G', 'arm-to-neighbour acceptance; inert at 3 heads'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('n_titin_per_thick', 'static', '-',
    skeletal=P(6, 'M', 'six titin molecules per vertebrate thick filament'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_head_reach', 'static', 'nm',
    skeletal=P(13.0, 'G', 'axial reach of S1 from its crown (binding search, overlap zone)'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('filament_radii_sum', 'static', 'nm',
    skeletal=P(13.0, 'M', 'thick + thin radius (8 + 5): surface gap <-> centre spacing'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('c_zone_lo', 'static', 'nm',
    skeletal=P(350.0, 'G', 'cMyBP-C band inner edge from the M-line'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('c_zone_hi', 'static', 'nm',
    skeletal=P(650.0, 'G', 'cMyBP-C band outer edge from the M-line'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))

# Resolution choices (not physics; the same for every class)
row('thin_node_spacing', 'static', 'nm',
    skeletal=P(12.0, 'G', 'thin backbone node grid (resolution)'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_monomers_per_unit', 'static', '-',
    skeletal=P(None, 'G', 'None = one Tm unit per binding candidate'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('n_xb_bins', 'static', '-',
    skeletal=P(200, 'G', 'axial-distance bins for the rate matrices'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_bin_lo', 'static', 'nm',
    skeletal=P(-8.0, 'G', 'lower edge of the binned axial range'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_bin_hi', 'static', 'nm',
    skeletal=P(35.0, 'G', 'upper edge of the binned axial range'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))

# =============================================================================
# NUMERICS — solver caps (static) and tolerances (dynamic)
# =============================================================================
row('n_newton_steps', 'static', '-',
    skeletal=P(4, 'G', 'Newton cap; the loop exits at convergence'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('n_cg_steps', 'static', '-',
    skeletal=P(6, 'G', 'CG steps per Newton step; 0 diverges with heads bound'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('solver_atol', 'dynamic', 'pN',
    skeletal=P(0.01, 'G', 'absolute term of tol = atol + rtol * scale'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('solver_rtol', 'dynamic', '-',
    skeletal=P(1e-3, 'G', 'relative term, against the RMS backbone force'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))

# =============================================================================
# FILAMENT AND TITIN MECHANICS (dynamic)
# =============================================================================
row('thick_k', 'dynamic', 'pN/nm',
    skeletal=P(7500.0, 'I', 'per crown segment; whole filament ~144 pN/nm'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('thin_EA', 'dynamic', 'pN',
    skeletal=P(66000.0, 'I', 'axial rigidity; whole filament ~61 pN/nm'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('titin_a', 'dynamic', 'pN',
    skeletal=P(43.0, 'I', 'N2A exponential prefactor per molecule'),
    cardiac_beta=P(55.0, 'I', 'N2B stiffer than N2A', lo=20.0, hi=150.0, source=BC),
    cardiac_alpha=inh('cardiac_beta'),
    insect=P(40.0, 'G', 'kettin/projectin; kept near the vertebrate values'))
row('titin_b', 'dynamic', '1/nm',
    skeletal=P(0.004, 'G', 'exponential steepness (a swept value, not a measurement)'),
    cardiac_beta=P(0.008, 'G', '8 um^-1, interpolated', lo=0.003, hi=0.02, source=BC),
    cardiac_alpha=inh('cardiac_beta'),
    insect=P(0.025, 'I', 'derived from two converging chains'))
row('titin_rest', 'dynamic', 'nm',
    skeletal=P(215.0, 'I', 'exponent offset (slack length)'),
    cardiac_beta=P(140.0, 'I', 'slack at SL 1.85 um', lo=100.0, hi=200.0, scale='linear',
                   source=BC),
    cardiac_alpha=inh('cardiac_beta'),
    insect=P(50.0, 'G', 'IFM I-band length, unverified'))

# =============================================================================
# CROSSBRIDGE SPRINGS (dynamic)
# =============================================================================
row('xb_c_rest_weak', 'dynamic', 'rad',
    skeletal=P(0.82309, 'G', 'pre-stroke converter angle'),
    cardiac_beta=P(0.6906753, 'I', 'd_rest 11.20 nm, x_rest 13.53 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_c_rest_strong', 'dynamic', 'rad',
    skeletal=P(1.27758, 'G', 'post-stroke converter angle'),
    cardiac_beta=P(1.0058527, 'M', 'd_rest 11.27 nm, x_rest 7.14 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_c_k_weak', 'dynamic', 'pN nm/rad',
    skeletal=P(8.0, 'I', 'pre-stroke compliant angular spring', lo=0.4, hi=20.0, source=BG),
    cardiac_beta=P(32.0, 'G', '4x skeletal: no binding behind the weak zero-force point'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_c_k_strong', 'dynamic', 'pN nm/rad',
    skeletal=P(40.0, 'I', 'strong angular spring', lo=4.0, hi=250.0, source=BG),
    cardiac_beta=P(18.0634, 'I', 'tangent axial stiffness 1.50 pN/nm at 5.02 pN'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_rest_weak', 'dynamic', 'nm',
    skeletal=P(19.93, 'G', 'pre-stroke head length'),
    cardiac_beta=P(17.5808, 'I', 'd_rest 11.20 nm, x_rest 13.53 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_rest_strong', 'dynamic', 'nm',
    skeletal=P(16.47, 'G', 'post-stroke head length'),
    cardiac_beta=P(13.3433, 'M', 'd_rest 11.27 nm, x_rest 7.14 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_k_weak', 'dynamic', 'pN/nm',
    skeletal=P(0.4, 'I', 'pre-stroke compliant axial spring'),
    cardiac_beta=P(1.6, 'G', '4x skeletal, with xb_c_k_weak'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_k_strong', 'dynamic', 'pN/nm',
    skeletal=P(5.0, 'I', 'strong axial spring', lo=1.5, hi=10.0, source=BG),
    cardiac_beta=P(2.2578, 'I', 'tangent axial stiffness 1.50 pN/nm at 5.02 pN'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_c_rest_tight_1', 'dynamic', 'rad',
    skeletal=P(1.1775922, 'I', 'Tight_1 converter angle (split stroke)'),
    cardiac_beta=P(0.9253923, 'M', 'd_rest 11.27 nm, x_rest 8.49 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_c_k_tight_1', 'dynamic', 'pN nm/rad',
    skeletal=P(40.0, 'I', 'equals Tight_2'),
    cardiac_beta=P(18.0634, 'I', 'equals Tight_2'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_rest_tight_1', 'dynamic', 'nm',
    skeletal=P(17.2312, 'I', 'Tight_1 head length (split stroke)'),
    cardiac_beta=P(14.1077, 'M', 'd_rest 11.27 nm, x_rest 8.49 nm'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_g_k_tight_1', 'dynamic', 'pN/nm',
    skeletal=P(5.0, 'I', 'equals Tight_2'),
    cardiac_beta=P(2.2578, 'I', 'equals Tight_2'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))

# =============================================================================
# THIN FILAMENT REGULATION (dynamic)
# =============================================================================
row('tm_k_01', 'dynamic', '1/(M ms)',
    skeletal=P(100000.0, 'G', 'Ca on-rate to the regulatory site, 1e8 /M/s',
               lo=10000.0, hi=1000000.0, source=BG),
    cardiac_beta=P(80000.0, 'M', 'cardiac Ca on-rate, 8e7 /M/s', lo=50000.0, hi=600000.0,
                   source=BC),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_k_12', 'dynamic', '1/ms',
    skeletal=P(1.0, 'G', 'blocked -> closed; no source'),
    cardiac_beta=P(0.5, 'G', 'half the (unsourced) skeletal value'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_k_23', 'dynamic', '1/ms',
    skeletal=P(0.1, 'G', 'closed -> open', lo=0.01, hi=1.0, source=BG),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_k_30', 'dynamic', '1/ms',
    skeletal=P(0.2, 'G', 'cycle close (Ca off)'),
    cardiac_beta=P(0.04, 'M', 'cardiac Ca off-rate, 40 /s', lo=0.02, hi=0.3, source=BC),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_Keq_01', 'dynamic', '1/M',
    skeletal=P(500000.0, 'G', 'Kd 2 uM regulatory site'),
    cardiac_beta=P(750000.0, 'G', 'Kd 1.33 uM', lo=200000.0, hi=1.2e6, source=BC),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_Keq_12', 'dynamic', '-',
    skeletal=P(130.0, 'G', 'blocked -> closed equilibrium', lo=13.0, hi=1300.0, source=BG),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_Keq_23', 'dynamic', '-',
    skeletal=P(0.1, 'M', 'closed -> open equilibrium (-Ca)', lo=0.01, hi=1.0, source=BG),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_J_C', 'dynamic', 'kT',
    skeletal=P(0.0, 'I', 'Ising coupling to closed neighbours; inert'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('tm_J_M', 'dynamic', 'kT',
    skeletal=P(2.70, 'F', 'Ising coupling to open neighbours'),
    cardiac_beta=inh('skeletal'),
    cardiac_alpha=inh('cardiac_beta', lo=0.0, hi=5.0, scale='linear'),
    insect=inh('skeletal'))
row('xb_tm_K2', 'dynamic', '-',
    skeletal=P(79.0, 'M', 'finite crossbridge lock, per head (1+K2)^n'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('thin_thin_overlap_screening', 'dynamic', '-',
    skeletal=P(1.0, 'G', 'double-overlap blocking of attachment, 0-1'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))

# =============================================================================
# CROSSBRIDGE KINETICS (dynamic)
# =============================================================================
row('xb_r01_coeff', 'dynamic', '1/ms',
    skeletal=P(305.99, 'G', 'attachment pre-exponential; inherited, no source',
               lo=30.0, hi=3000.0, source=BG),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_r12_coeff', 'dynamic', '1/ms',
    skeletal=P(0.6, 'F', 'Pi release + working stroke, lumped', lo=0.1, hi=1.0, source=BG),
    cardiac_beta=P(0.175, 'G', 'process B 3-4x slower than skeletal', lo=0.02, hi=0.2,
                   source=BC),
    cardiac_alpha=ratio('cardiac_beta', 1.4, 1.3, 1.5, 'I',
                        'alpha/beta force-generation rate k4 1.3-1.5 (Kawai 2016, rat, 25 C)'),
    insect=inh('skeletal'))
row('xb_r23_coeff', 'dynamic', '1/ms',
    skeletal=P(0.15, 'G', 'Tight_1 -> Tight_2 isomerization; no range (bounds were stale-index)'),
    cardiac_beta=P(0.065, 'G', 'Tight_1 -> Tight_2; value unjustified'),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_r34_coeff', 'dynamic', '1/ms',
    skeletal=P(0.6, 'I', 'ADP release / detachment at zero load', lo=0.3, hi=1.5, source=BG),
    cardiac_beta=P(0.065, 'M', 'cardiac ADP release ~65 /s (bovine ventricle)', lo=0.04, hi=0.2,
                   source=BC),
    cardiac_alpha=ratio('cardiac_beta', 2.8, 1.67, 5.6, 'G',
                        'alpha/beta detachment: k-ADP 4.2-5.6 (Wang 2013, 17 C) vs k2 1.7-2 '
                        '(Kawai 2016, 25 C); nominal splits them'),
    insect=inh('skeletal'))
row('xb_delta_34', 'dynamic', 'nm',
    skeletal=P(-0.80, 'M', 'detachment distance; negative = catch bond'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_r40', 'dynamic', '1/ms',
    skeletal=P(0.1, 'G', 'recovery stroke', lo=0.03, hi=0.3, source=BG),
    cardiac_beta=inh('skeletal', lo=0.015, hi=0.3), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_r04', 'dynamic', '1/ms',
    skeletal=P(0.01, 'M', 'reverse recovery stroke', lo=0.003, hi=0.1, source=BG),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_r05', 'dynamic', '1/ms',
    skeletal=P(0.007, 'I', 'DRX -> SRX sequestration'),
    cardiac_beta=P(0.2, 'G', 'DRX -> SRX; Mijailovich k-PS (Assumed)', lo=0.05, hi=0.4,
                   source=BC),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_U_DRX', 'dynamic', 'kT',
    skeletal=P(-2.3, 'I', 'detached M.ADP.Pi'),
    cardiac_beta=inh('skeletal', lo=-4.0, hi=-1.0, scale='linear'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_U_loose', 'dynamic', 'kT',
    skeletal=P(-4.3, 'I', 'weakly bound AM.ADP.Pi'),
    cardiac_beta=inh('skeletal', lo=-6.0, hi=-2.0, scale='linear'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_U_tight_1', 'dynamic', 'kT',
    skeletal=P(-15.0, 'I', 'AM.ADP after Pi release'),
    cardiac_beta=inh('skeletal', lo=-18.0, hi=-12.0, scale='linear'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_U_tight_2', 'dynamic', 'kT',
    skeletal=P(-21.0, 'G', 'second strong substate; 6 kT gap is a choice'),
    cardiac_beta=inh('skeletal', lo=-24.0, hi=-16.0, scale='linear'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_srx_k0', 'dynamic', '1/ms',
    skeletal=P(0.007, 'G', 'SRX -> DRX basal rate (= r05: 50% SRX at rest)'),
    cardiac_beta=P(0.005, 'G', 'Mijailovich kPS0 (Assumed)', lo=0.003, hi=0.03, source=BC),
    cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_srx_kmax', 'dynamic', '1/ms',
    skeletal=P(0.4, 'G', 'SRX -> DRX saturating rate (Assumed)'),
    cardiac_beta=inh('skeletal', lo=0.05, hi=0.5), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_srx_b', 'dynamic', '-',
    skeletal=P(5.0, 'G', 'Hill exponent of SRX recruitment (Assumed)'),
    cardiac_beta=inh('skeletal', lo=1.0, hi=8.0), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('xb_srx_ca50', 'dynamic', 'M',
    skeletal=P(1e-6, 'G', 'half recruitment at pCa 6 (Assumed)'),
    cardiac_beta=inh('skeletal', lo=3e-7, hi=3e-6), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
row('temp_celsius', 'dynamic', 'C',
    skeletal=P(26.15, 'G', 'sets kT only; rates are not temperature-scaled'),
    cardiac_beta=inh('skeletal'), cardiac_alpha=inh('cardiac_beta'), insect=inh('skeletal'))
