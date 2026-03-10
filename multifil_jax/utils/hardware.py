"""
Hardware detection and batch mapping utilities.

Provides hardware-agnostic parallelization with OS-aware fallbacks:
- Linux/Mac + GPU: Uses jax.vmap on GPU
- Windows (any): CPU only (JAX CUDA not supported on Windows)
- Linux/Mac + CPU: Uses jax.pmap with XLA multi-device

Key constraint: Windows cannot use JAX GPU even if GPU is present.

Usage:
    from multifil_jax.utils.hardware import detect_hardware, get_batch_mapper

    # Get hardware info
    has_gpu, is_windows, device_count = detect_hardware()

    # Get appropriate batch mapper for parallel execution
    batch_map, parallel_mode = get_batch_mapper()
    results = batch_map(my_function)(batched_inputs)
"""
import os
import sys
import subprocess
import multiprocessing

# JAX: Must configure XLA_FLAGS BEFORE importing JAX.
# XLA_FLAGS controls the XLA compiler behavior, including multi-device CPU.
# Setting it after JAX import has no effect.
_is_windows = sys.platform.startswith('win') or sys.platform == 'cygwin'


def _configure_xla_for_cpu():
    """Configure XLA for multi-device CPU parallelism.

    JAX: On CPU-only systems, XLA can simulate multiple devices for pmap.
    This enables data parallelism even without a GPU by treating each
    CPU core as a separate "device".
    """
    if 'XLA_FLAGS' not in os.environ:
        cpu_count = multiprocessing.cpu_count()
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count}"


def _configure_xla_for_gpu():
    """Configure XLA flags for GPU.

    Note: autotune_level=2 was tried (Session 3→4) to reduce compile time
    but may have contributed to a ~20x runtime regression. Reverted to
    default (level 4) which lets XLA find optimal kernel variants.
    """
    pass


def _check_gpu_available():
    """Check if an NVIDIA GPU is available via nvidia-smi.

    Returns:
        bool: True if nvidia-smi runs successfully, False otherwise.

    Note:
        This only checks for NVIDIA GPUs. AMD/Intel GPUs are not supported
        by JAX CUDA and will return False.
    """
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def detect_hardware():
    """Detect available hardware and configure JAX accordingly.

    This function must be called BEFORE importing JAX to properly configure
    XLA_FLAGS for multi-device CPU parallelism.

    Returns:
        Tuple of (has_gpu, is_windows, device_count):
            - has_gpu (bool): True if GPU is available AND usable
            - is_windows (bool): True if running on Windows
            - device_count (int): Number of devices (CPU cores or GPUs)

    Note:
        Windows always returns has_gpu=False regardless of hardware,
        because JAX CUDA is not supported on Windows.

    Example:
        # Call before importing JAX
        from multifil_jax.utils.hardware import detect_hardware
        has_gpu, is_windows, device_count = detect_hardware()

        import jax  # Now JAX is configured properly
    """
    # Windows cannot use JAX GPU - force CPU mode
    # JAX: JAX's CUDA backend requires Linux or macOS. Windows uses CPU only.
    if _is_windows:
        _configure_xla_for_cpu()
        return (False, True, multiprocessing.cpu_count())

    # Linux/Mac: check for actual GPU
    has_gpu = _check_gpu_available()

    if has_gpu:
        _configure_xla_for_gpu()
    else:
        _configure_xla_for_cpu()

    # Return device count (will be accurate after JAX import)
    device_count = multiprocessing.cpu_count() if not has_gpu else 1
    return (has_gpu, False, device_count)


def get_batch_mapper():
    """Return appropriate batch mapper (vmap or pmap) for parallel execution.

    This function must be called AFTER importing JAX.

    Returns:
        Tuple of (batch_map_fn, parallel_mode_string):
            - batch_map_fn: jax.vmap or jax.pmap
            - parallel_mode_string: 'vmap' or 'pmap'

    JAX: vmap vs pmap
        - vmap: Vectorizes a function over a batch dimension. Works on single
                device (GPU or CPU). Adds a batch axis to all inputs/outputs.
        - pmap: Parallelizes a function across multiple devices. Each device
                processes one element of the batch. Requires multi-device setup.

    Usage:
        batch_map, mode = get_batch_mapper()

        # vmap usage (GPU or single CPU)
        @jax.jit
        def process(x):
            return x ** 2
        results = batch_map(process)(batch_of_x)

        # pmap usage (multi-device CPU)
        # Same API, but data is distributed across devices
        results = batch_map(process)(batch_of_x)

    Example:
        import jax
        from multifil_jax.utils.hardware import get_batch_mapper

        batch_map, parallel_mode = get_batch_mapper()
        print(f"Using {parallel_mode} for parallel execution")
    """
    import jax

    devices = jax.local_devices()
    is_cpu = devices[0].platform == 'cpu'
    multi_device = len(devices) > 1

    # JAX: Use pmap for multi-device CPU, vmap otherwise.
    # pmap distributes computation across devices, vmap vectorizes on one device.
    if is_cpu and multi_device:
        return (jax.pmap, 'pmap')
    else:
        return (jax.vmap, 'vmap')


def get_platform_info():
    """Get human-readable platform information string.

    Returns:
        str: Description of the current platform and JAX configuration.

    Example:
        print(get_platform_info())
        # Output: "Platform: Linux | GPU: Available | Devices: 1 (cuda)"
    """
    import jax

    has_gpu, is_windows, _ = detect_hardware()
    devices = jax.local_devices()
    platform = devices[0].platform

    os_name = "Windows" if is_windows else ("Linux" if sys.platform.startswith('linux') else "macOS")
    gpu_status = "Available" if has_gpu else "Not available (CPU mode)"

    return f"Platform: {os_name} | GPU: {gpu_status} | Devices: {len(devices)} ({platform})"


# Module-level initialization
# JAX: Configure XLA before any potential JAX imports elsewhere.
# This ensures consistent behavior regardless of import order.
_has_gpu, _is_windows, _device_count = detect_hardware()

# Persistent compilation cache — eliminates recompilation on restarts.
# Different lattice sizes / iteration counts get separate cache entries.
# Override location via MULTIFIL_JAX_CACHE_DIR environment variable.
import jax
_cache_dir = os.environ.get(
    "MULTIFIL_JAX_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "multifil_jax", "xla"),
)
os.makedirs(_cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
