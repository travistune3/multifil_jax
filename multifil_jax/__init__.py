"""
MultiFilament JAX Implementation

High-performance JAX implementation of half-sarcomere muscle contraction model.

Basic usage:
    from multifil_jax import SarcTopology, get_skeletal_params
    from multifil_jax.core.params import StaticParams

    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(nrows=2, ncols=2,
                               static_params=static, dynamic_params=dynamic)

    # Run simulation via top-level run() API
    from multifil_jax import run
    result = run(topo, pCa=4.5, z_line=900.0, duration_ms=100)
    print(result.summary())

For parameter sweeps:
    result = run(topo, pCa=[4.5, 5.0, 6.0], replicates=5)

All metrics are always computed (no metric selection needed).
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

# Helper functions
from multifil_jax.helper import count_transitions

__all__ = [
    # Top-level simulation
    'run',
    'SimulationResult',

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
