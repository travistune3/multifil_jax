# multifil_jax

JAX-accelerated half-sarcomere muscle contraction simulator.

Simulates thick and thin filament mechanics, crossbridge cycling, and tropomyosin cooperativity in a half-sarcomere lattice. Runs parameter sweeps of hundreds of conditions in parallel on GPU via `jax.vmap`.

See `docs/README.md` for a detailed walkthrough and `docs/INSTALL.md` for installation instructions.

## Quick start

```python
from multifil_jax import run, get_default_params, SarcTopology
import jax

static, dynamic = get_default_params()
topo = SarcTopology.create(nrows=2, ncols=2, static_params=static, dynamic_params=dynamic)
topo = jax.device_put(topo)

result = run(topo, pCa=4.5, z_line=900.0, duration_ms=1000)
print(result.axial_force.mean())
```

## Examples

- `examples/quickstart.py` — isometric contraction, sweeps, time traces
- `examples/sinusoidal_analysis.py` — Nyquist frequency sweep (Kawai & Brandt 1980)
- `examples/hysteresis.py` — pCa-force hysteresis loops
- `benchmarking/benchmark_minibatch.py` — GPU memory / throughput tuning
- `benchmarking/profile_jax.py` — Perfetto GPU trace generation
