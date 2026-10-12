"""
Physical, structural and numerical parameters for the half-sarcomere model.

Every value comes from ONE table (core/param_table.py), one row per parameter,
with a nominal value, a confidence tag, a one-line basis and (where one can be
defended) a range, for each muscle class:

    skeletal       fast-twitch skeletal, ~26 °C
    cardiac_beta   beta-myosin cardiac (large mammals, human), ~26 °C
    cardiac_alpha  alpha-myosin cardiac (rodents): cardiac_beta with the
                   crossbridge rates that have a matched alpha/beta measurement
    insect         insect flight muscle (Lethocerus); the Drosophila thick-filament
                   superlattice is static.replace(n_superlattice_classes=3)

get_params(cls) is the only constructor. Nothing has a default: a missing field
is a TypeError, an unknown one too. Change a value with .copy(**updates)
(DynamicParams) or .replace(**updates) (StaticParams).

StaticParams — structural configuration
    Array shapes and compile-time geometry: crowns per thick filament, heads per
    crown, the actin helix, solver iteration caps. Frozen and hashable, NOT a JAX
    PyTree; the topology keeps it, and changing it means rebuilding the topology
    and recompiling.

DynamicParams — physics values
    Spring stiffnesses, rates, free energies, couplings, solver tolerances. JAX
    arrays in a registered PyTree, so they sweep across a batch without
    recompiling. Values are ABSOLUTE, in model units: a rate here is the rate the
    kernel uses.

pCa, z_line and lattice_spacing are drivers, not parameters: run() takes them
explicitly. get_params also returns the class's natural operating point (z0, d0).

UNITS
-----
    time            ms          (a rate of 0.6 means 0.6 ms^-1 = 600 s^-1)
    length          nm
    force           pN
    stiffness       pN/nm (linear) or pN*nm/rad (angular)
    energy          kT, except spring energies which are pN*nm
    calcium         pCa = -log10([Ca2+] in M)
    temperature     degrees Celsius

CONFIDENCE TIERS
----------------
Muscle models are underdetermined: many parameters have no direct measurement,
and several that do have measurements spanning an order of magnitude. Every row
is tagged so a reader can tell what is anchored and what is a placeholder:

    [M] MEASURED   a literature value used essentially as reported
    [I] INFERRED   derived by arithmetic or geometry from measured value(s),
                   in this model's particular parameterization
    [G] GUESS      no source for this system: a value carried over from another
                   species/muscle type, a plausibility estimate, or a number
                   inherited from an earlier implementation
    [F] FITTED     tuned so the model reproduces some emergent measurement
                   (a Hill coefficient, an apparent rate), not measured directly

A [F] or [G] value is not wrong — it is simply not evidence. resolve(cls) returns
every row with its tag, basis and range.

SAMPLING
--------
draw(params, name, n, seed) returns n values of one field, uniform (log or linear,
per the row) across its range for params' class — pass the list to run() as a
field keyword for a random grid. sample_params(params, n, seed, vary=...) returns
n DynamicParams with several fields drawn jointly — pass the list as run()'s
params for one 'candidates' axis.
"""
from dataclasses import asdict, make_dataclass
from typing import Dict, Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from .param_table import ROWS, CLASSES, P, Inherit, Ratio

STATIC_FIELDS = tuple(n for n, r in ROWS.items() if r.kind == 'static')
DYNAMIC_FIELDS = tuple(n for n, r in ROWS.items() if r.kind == 'dynamic')


def poisson_spacing(d0, z0, z, nu, radii_sum):
    """Surface gap at Z-line z, scaled from gap d0 at z0 with Poisson ratio nu.

    The isovolumic argument constrains filament PACKING, i.e. the
    centre-to-centre spacing, so the scaling applies to d + radii_sum
    (StaticParams.filament_radii_sum) and the radii come back off afterwards.
    nu = 0.5 is isovolumic. The one place this formula is written: the Poisson
    LS mode and the dynamic-LS reference spacing both call it.
    """
    return (d0 + radii_sum) * (z0 / z) ** nu - radii_sum


# =============================================================================
# THE TABLE, RESOLVED PER CLASS
# =============================================================================

class Row(NamedTuple):
    """One parameter for one muscle class. lo = hi = None: no range."""
    nominal: Any
    tag: str
    basis: str
    lo: Optional[float]
    hi: Optional[float]
    scale: str
    source: str
    unit: str
    kind: str


def _resolve_entry(name, cls, seen=()):
    if cls in seen:
        raise ValueError(f"{name}: inheritance cycle {seen + (cls,)}")
    e = ROWS[name].entries[cls]
    if isinstance(e, P):
        return e
    par = _resolve_entry(name, e.parent, seen + (cls,))
    if isinstance(e, Ratio):
        if par.lo is None:
            raise ValueError(f"{name}[{cls}]: a ratio needs a ranged parent")
        return P(par.nominal * e.factor, e.tag, e.basis, par.lo * e.lo_factor,
                 par.hi * e.hi_factor, par.scale, f"{e.parent} x {e.factor}")
    lo, hi = e.lo, e.hi
    if lo is None and hi is None:
        lo, hi = par.lo, par.hi
    elif par.lo is not None and not (lo <= par.lo and hi >= par.hi):
        raise ValueError(f"{name}[{cls}]: inherited range [{lo}, {hi}] narrower than "
                         f"{e.parent}'s [{par.lo}, {par.hi}]")
    return P(par.nominal, 'G', f"as {e.parent}: {par.basis}", lo, hi, e.scale or par.scale,
             f"inherits {e.parent}")


def resolve(cls: str) -> Dict[str, Row]:
    """Every row for one muscle class, inheritance resolved. Validates the table."""
    if cls not in CLASSES:
        raise ValueError(f"unknown muscle class {cls!r}; valid: {CLASSES}")
    out = {}
    for name, r in ROWS.items():
        p = _resolve_entry(name, cls)
        if p.tag not in ('M', 'I', 'G', 'F'):
            raise ValueError(f"{name}[{cls}]: tag {p.tag!r} not in M/I/G/F")
        if (p.lo is None) != (p.hi is None):
            raise ValueError(f"{name}[{cls}]: half a range")
        if p.lo is not None:
            if p.scale not in ('log', 'linear'):
                raise ValueError(f"{name}[{cls}]: scale {p.scale!r}")
            if p.scale == 'log' and p.lo <= 0:
                raise ValueError(f"{name}[{cls}]: a log range must be positive")
            if not (p.lo <= p.nominal <= p.hi):
                raise ValueError(f"{name}[{cls}]: nominal {p.nominal} outside [{p.lo}, {p.hi}]")
        out[name] = Row(*p, unit=r.unit, kind=r.kind)
    return out


# =============================================================================
# STATIC PARAMS (structural configuration — NOT a PyTree)
# =============================================================================

def _static_replace(self, **updates) -> 'StaticParams':
    """A copy with the named fields changed. Unknown name -> TypeError."""
    unknown = set(updates) - set(STATIC_FIELDS)
    if unknown:
        raise TypeError(f"unknown StaticParams field(s) {sorted(unknown)}")
    return StaticParams(**{**asdict(self), **updates})


def _static_post_init(self):
    if self.n_superlattice_classes not in (1, 3):
        raise ValueError(
            f"n_superlattice_classes must be 1 (no superlattice) or 3 (Drosophila "
            f"3-class axial superlattice), got {self.n_superlattice_classes}: the class "
            "formula (q-r) mod n gives a geometrically wrong sublattice for any other n.")


StaticParams = make_dataclass(
    'StaticParams', [(n, Any) for n in STATIC_FIELDS], frozen=True,
    namespace={'replace': _static_replace, '__post_init__': _static_post_init})
StaticParams.__module__ = __name__
StaticParams.__doc__ = """Structural configuration. Frozen, hashable, NOT a JAX PyTree.

Every field is required; the fields are the 'static' rows of the table
(resolve(cls) gives each one's unit, tag and basis). Everything here sets an
array shape or a compile-time constant, so the topology keeps it and a change
means rebuilding the topology and recompiling. Derive a variant with
.replace(**updates).
"""


# =============================================================================
# DYNAMIC PARAMS (physics values — a JAX PyTree)
# =============================================================================

@jax.tree_util.register_pytree_node_class
class DynamicParams:
    """All sweepable physics values, as a JAX PyTree.

    Every field is required (the 'dynamic' rows of the table); a missing or
    unknown field is a TypeError. `muscle_class` names the table class the
    values came from, so draw / sample_params can find the ranges. It is not
    part of the PyTree: a DynamicParams rebuilt inside a JAX transform has
    muscle_class None.

    Usage:
        static, dynamic, z0, d0 = get_params('skeletal')
        dynamic = dynamic.copy(xb_r01_coeff=350.0)
    """

    __slots__ = DYNAMIC_FIELDS + ('muscle_class',)

    def __init__(self, *, muscle_class: Optional[str], **fields):
        missing = set(DYNAMIC_FIELDS) - set(fields)
        unknown = set(fields) - set(DYNAMIC_FIELDS)
        if missing or unknown:
            raise TypeError(f"DynamicParams: missing {sorted(missing)}, unknown {sorted(unknown)}")
        if muscle_class is not None and muscle_class not in CLASSES:
            raise ValueError(f"unknown muscle class {muscle_class!r}; valid: {CLASSES}")
        object.__setattr__(self, 'muscle_class', muscle_class)
        for name in DYNAMIC_FIELDS:
            object.__setattr__(self, name, jnp.asarray(fields[name]))

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in DYNAMIC_FIELDS), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Bypass __init__: children may be tracers or JAX's object() sentinels.
        new = object.__new__(cls)
        object.__setattr__(new, 'muscle_class', None)
        for name, value in zip(DYNAMIC_FIELDS, children):
            object.__setattr__(new, name, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Field values as Python floats (scalar params only)."""
        return {name: float(getattr(self, name)) for name in DYNAMIC_FIELDS}

    def copy(self, **updates) -> 'DynamicParams':
        """A copy with the named fields changed (JIT-safe). Unknown name -> TypeError."""
        return DynamicParams(muscle_class=self.muscle_class,
                             **{**{n: getattr(self, n) for n in DYNAMIC_FIELDS}, **updates})

    def __repr__(self) -> str:
        return (f"DynamicParams({self.muscle_class}, thick_k={float(self.thick_k):.1f}, "
                f"thin_EA={float(self.thin_EA):.1f}, ...)")


# Alias kept for kernel signatures.
Constants = DynamicParams


# =============================================================================
# CONSTRUCTOR AND SAMPLING
# =============================================================================

def get_params(cls: str) -> Tuple[StaticParams, DynamicParams, float, float]:
    """(StaticParams, DynamicParams, z0, d0) for one muscle class.

    z0 is the class's natural half-sarcomere length (z_line, nm from the M-line),
    d0 its lattice spacing (thick-to-thin surface gap, nm).

        static, dynamic, z0, d0 = get_params('cardiac_beta')
        topo = SarcTopology.create(8, 8, static)
        result = run(topo, dynamic, pCa=4.5, z_line=z0, lattice_spacing=d0,
                     dt=1.0, duration_ms=1000)
    """
    rows = resolve(cls)
    static = StaticParams(**{n: rows[n].nominal for n in STATIC_FIELDS})
    dynamic = DynamicParams(muscle_class=cls, **{n: rows[n].nominal for n in DYNAMIC_FIELDS})
    return static, dynamic, rows['z0'].nominal, rows['d0'].nominal


def _check_seed(seed):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")


def _ranged_row(params: DynamicParams, name: str) -> Row:
    if params.muscle_class is None:
        raise ValueError("params has no muscle_class (built inside a JAX transform?)")
    rows = resolve(params.muscle_class)
    if name not in rows:
        raise ValueError(f"cannot draw {name!r}: unknown parameter")
    if name not in DYNAMIC_FIELDS:
        raise ValueError(f"cannot draw {name!r}: not a DynamicParams field")
    if rows[name].lo is None:
        raise ValueError(f"cannot draw {name!r}: no range in the {params.muscle_class} table")
    return rows[name]


def _sample(row: Row, u):
    if row.scale == 'log':
        return np.exp(np.log(row.lo) + u * (np.log(row.hi) - np.log(row.lo)))
    return row.lo + u * (row.hi - row.lo)


def draw(params: DynamicParams, name: str, n: int, seed: int) -> List[float]:
    """n random values of one field, uniform (log or linear, per its row) across
    its range for params' class. Pass the list to run() as a field keyword."""
    _check_seed(seed)
    row = _ranged_row(params, name)
    return [float(x) for x in _sample(row, np.random.default_rng(seed).random(n))]


def sample_params(params: DynamicParams, n: int, seed: int,
                  vary: Union[str, Sequence[str]], hold: Sequence[str] = ()) -> List[DynamicParams]:
    """n DynamicParams with the `vary` fields drawn jointly, every other field as params.

    vary: field names, or a string of tags (e.g. 'GF') selecting every ranged
    field with one of those tags in params' class. hold: names taken back out of
    the varied set. Pass the list as run()'s params (one 'candidates' axis; the
    drawn values are in result.metadata['candidates']).
    """
    _check_seed(seed)
    if isinstance(vary, str):
        if not set(vary) <= set('MIGF'):
            raise ValueError(f"vary={vary!r}: a string selects tags from 'MIGF'")
        rows = resolve(params.muscle_class) if params.muscle_class else {}
        names = [k for k in DYNAMIC_FIELDS if k in rows and rows[k].tag in vary
                 and rows[k].lo is not None]
    else:
        names = list(vary)
    unknown = set(hold) - set(DYNAMIC_FIELDS)
    if unknown:
        raise ValueError(f"hold names unknown fields {sorted(unknown)}")
    names = [k for k in names if k not in set(hold)]
    if not names:
        raise ValueError("nothing to vary")
    rows = [_ranged_row(params, k) for k in names]
    u = np.random.default_rng(seed).random((n, len(names)))
    return [params.copy(**{k: float(_sample(r, u[i, j])) for j, (k, r) in enumerate(zip(names, rows))})
            for i in range(n)]
