"""
multifil_jax — a GPU-accelerated spatially-explicit half-sarcomere model.

Simulates the molecular machinery of muscle contraction: individual myosin heads
cycling through a six-state chemomechanical cycle, tropomyosin regulating access
to actin under calcium control and cooperative coupling, and the whole filament
lattice held in mechanical equilibrium at every timestep. Force emerges from the
interaction of thousands of stochastic motors with a compliant elastic
structure; it is never prescribed.

The model is spatially explicit, which is the point of it. Each head has a real
position in a hexagonal filament lattice, reaches for real binding sites, and
competes with its neighbours through filament compliance. Questions about
geometry — lattice spacing, filament registration, sarcomere length, species
differences in packing — can be asked directly rather than parameterized away.

Everything is JAX. The full simulation compiles to a single GPU kernel, and
parameter sweeps run as one batched computation rather than a loop, which makes
sweeping hundreds of conditions little more expensive than one.

QUICK START
-----------
    from multifil_jax import SarcTopology, run
    from multifil_jax.core.params import get_skeletal_params

    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(nrows=2, ncols=2,
                               static_params=static, dynamic_params=dynamic)

    result = run(topo, pCa=4.5, z_line=1100.0, duration_ms=1000,
                 dynamic_params=dynamic, static_params=static)
    print(result.summary())

Pass a list instead of a scalar to sweep it — the axes are Cartesian-producted
and run in parallel:

    result = run(topo, pCa=[9.0, 6.0, 5.5, 4.5], replicates=5)

WHERE THINGS LIVE
-----------------
    core/params.py        every physical parameter, with citations and an
                          explicit confidence tier; four species presets
    core/sarc_geometry.py the filament lattice and all structural index maps
    core/state.py         what changes each timestep, and nothing else
    kernels/              the physics: rate laws, forces, transitions, solver
    metrics_fn.py         everything a run reports
    simulation.py         run(), the batching machinery, and SimulationResult

A NOTE ON PARAMETERS
--------------------
Muscle models are underdetermined. Several parameters here have no direct
measurement, and some that do have measurements spanning an order of magnitude.
Every value in core/params.py is therefore tagged [M] measured, [I] inferred,
[G] guess, or [F] fitted. Check the tag before treating a number as evidence.
"""

# Version info
__version__ = '3.0.0'  # Major version bump for API refactor

# Hardware detection and persistent XLA compilation cache (must run early)
from multifil_jax.utils.hardware import detect_hardware

# Core API
from multifil_jax.timestep import timestep
from multifil_jax.metrics_fn import compute_all_metrics
from multifil_jax.core.state import realize_state, get_ca_concentration
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import get_skeletal_params, get_cardiac_params
from multifil_jax.kernels.forces import axial_force_at_mline

# Top-level simulation API
from multifil_jax.simulation import run, SimulationResult

# Subpopulation feature (modified-kinetics subsets of XB / TM units)
from multifil_jax.core.subpopulation import Subpopulation

# Helper functions
from multifil_jax.helper import count_transitions

__all__ = [
    # Top-level simulation
    'run',
    'SimulationResult',
    'Subpopulation',

    # Core simulation
    'timestep',
    'compute_all_metrics',
    'axial_force_at_mline',

    # State creation
    'SarcTopology',
    'realize_state',
    'get_ca_concentration',
    'get_skeletal_params',
    'get_cardiac_params',

    # Helper functions
    'count_transitions',
]
