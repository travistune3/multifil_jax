# multifil_jax

JAX-accelerated half-sarcomere muscle contraction simulator.

Simulates thick and thin filament mechanics, crossbridge cycling, and tropomyosin cooperativity in a half-sarcomere lattice. Runs parameter sweeps of hundreds of conditions in parallel on GPU via `jax.vmap`.

See `docs/README.md` for a detailed walkthrough and `docs/INSTALL.md` for installation instructions.

## Quick start

```python
from multifil_jax import run, get_skeletal_params, SarcTopology
import jax

static, dynamic = get_skeletal_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000)
print(result.axial_force.mean())
```

## Examples

- `examples/quickstart.py` — isometric contraction, sweeps, time traces
- `examples/dynamic_lattice_spacing.py` — emergent lattice spacing from radial force balance
- `examples/sinusoidal_analysis.py` — Nyquist frequency sweep (Kawai & Brandt 1980)
- `examples/hysteresis.py` — stiffness parameter sweep, vertebrate vs invertebrate
- `examples/benchmarks/benchmark_dynamic_ls.py` — dynamic LS performance and lattice scaling
- `examples/benchmarks/benchmark_minibatch.py` — GPU memory / throughput tuning
- `examples/benchmarks/profile_jax.py` — Perfetto GPU trace generation
