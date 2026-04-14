# Installation

## Requirements

- Python 3.10 or later
- NVIDIA GPU (any architecture from Turing / RTX 20xx onwards)
- A Linux environment — see below if you are on Windows or Mac

## Getting a Linux environment

JAX CUDA requires Linux. If you are not already on Linux, the two practical options are:

**WSL2 (Windows)** — Windows Subsystem for Linux 2 runs a full Linux kernel inside Windows with near-native performance. NVIDIA supports CUDA inside WSL2 with the standard Windows GPU driver — no separate Linux driver needed.

```bash
# In PowerShell (run as administrator)
wsl --install          # installs Ubuntu by default
# Restart, then open the Ubuntu app and continue below
```

See [Microsoft's WSL2 CUDA guide](https://docs.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl) for driver setup.

**Dual-boot Linux (Mac or Windows)** — install Ubuntu alongside your existing OS. On Mac (Intel), use a USB installer and partition the disk. On Windows, the Ubuntu installer can shrink the Windows partition automatically. Once booted into Linux the setup is identical to native Linux.

Ubuntu 22.04 LTS is a reliable choice for both options.

## Installing

Once you are in a Linux environment (native, WSL2, or dual-boot), clone the repository and install:

```bash
git clone https://github.com/travistune3/multifil_jax.git
cd multifil_jax
pip install -e ".[cuda13]"
```

This installs `multifil_jax` as a symlinked package and pulls JAX with all required CUDA libraries via pip. No separate CUDA toolkit installation is required.

### Choosing between CUDA 12 and CUDA 13

Check your driver version first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

| Command | When to use |
|---|---|
| `pip install -e ".[cuda13]"` | Driver ≥ 580, GPU is Turing (RTX 20xx) or newer — recommended |
| `pip install -e ".[cuda12]"` | Driver ≥ 525, GPU is older (Volta V100, Pascal GTX 10xx) |

CUDA 13 is the direction JAX is heading — CUDA 12 support will eventually be dropped. Use CUDA 13 if your driver supports it.

## Verifying the install

```python
import multifil_jax  # must come before `import jax`
import jax
print(jax.devices())   # should show e.g. [CudaDevice(id=0)]
from multifil_jax import run
print("multifil_jax ready")
```

If `jax.devices()` shows a CPU device instead of CUDA, JAX is not seeing the GPU. Common causes in WSL2: the Windows NVIDIA driver is not installed (the Linux driver inside WSL2 is provided by Windows — do not install a separate Linux driver inside WSL2).

**Import order:** always `import multifil_jax` before `import jax` in your scripts. `multifil_jax` sets XLA environment flags at import time that must be in place before JAX initializes. If JAX is imported first, GPU detection and multi-CPU configuration may not take effect.

## Upgrading JAX

```bash
pip install --upgrade "jax[cuda13]"
```

The `multifil_jax` package does not need to be reinstalled after a JAX upgrade.

### JAX version notes

- **Minimum**: JAX 0.4.25
- **Recommended**: JAX 0.7.0 or later — includes compiler improvements that benefit the vmap-outside-scan simulation loop
- **Latest tested**: JAX 0.9.x with CUDA 13

If you see compilation errors after upgrading JAX, clear the XLA cache:

```bash
rm -rf ~/.cache/multifil_jax/xla/
```

## GPU memory (VRAM) usage

Peak VRAM scales with batch size and simulation duration:

```
peak VRAM (GB) ≈ minibatch_size × n_steps × 45 metrics × 4 bytes × 2 / 1e9
```

| minibatch_size | Peak VRAM (approx, at 1000 steps) |
|---|---|
| 256 | ~0.1 GB |
| 1024 | ~0.4 GB |
| 4096 | ~1.5 GB |
| 8192 | ~3 GB |
| 16384 | ~6 GB |

For an **8 GB GPU (e.g. RTX 4060):** the default `minibatch_size="auto"` is safe up to batch=16384 at 1000 steps. For longer simulations at large batch, set it explicitly:

```python
# ~3 GB peak for 2000-step sim with large batch:
result = run(topo, pCa=4.5, replicates=16384, duration_ms=2000, minibatch_size=2048)
```

`minibatch_size` splits the batch into sequential chunks — it does not affect results, only peak memory and marginally performance. Non-power-of-2 values are snapped down to the nearest power of 2 automatically.

## XLA compilation cache

The first time you run a simulation with a given configuration, JAX compiles the GPU kernel. This takes 1–5 minutes. Compiled kernels are cached in `~/.cache/multifil_jax/xla/` and reused in future sessions.

Different configurations (lattice size, batch size, `StaticParams`) each get their own cache entry automatically — you do not need to clear the cache when switching between them. Clear it only after upgrading JAX:

```bash
rm -rf ~/.cache/multifil_jax/xla/
```

