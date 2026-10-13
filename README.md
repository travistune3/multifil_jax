# multifil_jax

JAX-accelerated half-sarcomere muscle contraction simulator.

Simulates thick and thin filament mechanics, crossbridge cycling and tropomyosin
cooperativity in a half-sarcomere lattice, and runs parameter sweeps of thousands
of conditions in parallel on a GPU.

- [docs/INSTALL.md](docs/INSTALL.md) — installation (Linux / WSL2, CUDA)
- [docs/README.md](docs/README.md) — user guide: running, sweeping, reading results, the physics
- [docs/CODE_Architecture.md](docs/CODE_Architecture.md) — how the code is organised

## Quick start

```python
from multifil_jax import get_params, SarcTopology, run

static, dynamic, z0, d0 = get_params('cardiac_beta')   # structure, physics, operating point
topo = SarcTopology.create(2, 2, static)

result = run(topo, dynamic, pCa=4.5, z_line=z0, lattice_spacing=d0,
             dt=1.0, duration_ms=1000)
print(result.axial_force[..., -100:].mean())
```

Muscle classes: `skeletal`, `cardiac_beta`, `cardiac_alpha`, `insect`.

## Examples

- `examples/quickstart.py` — the `run()` patterns: sweeps, overrides, time courses, sampling
- `examples/dynamic_lattice_spacing.py` — lattice spacing solved from radial force balance
- `examples/subpopulation.py` — mixed motor populations (mean-field, random, C-zone)
- `examples/sinusoidal_analysis.py` — complex stiffness from a sinusoidal frequency sweep
- `examples/vertebrate_vs_invertebrate_geometry.py` — skeletal vs insect lattice geometry
- `examples/stiffness_sweep_cardiac.py` — cardiac thick/thin stiffness sweep
- `examples/stiffness_corner_sweep.py` — three-way stiffness sweep as a corner plot
- `examples/benchmarks/` — dynamic-LS scaling, minibatch tuning, profiling
