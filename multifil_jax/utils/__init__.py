"""
Utility functions for JAX Half-Sarcomere simulation.

Modules:
    hardware: Hardware detection and batch mapping utilities
"""
from multifil_jax.utils.hardware import (
    detect_hardware,
    get_batch_mapper,
    get_platform_info,
)
