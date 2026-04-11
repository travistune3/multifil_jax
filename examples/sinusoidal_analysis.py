"""
Kawai-Brandt Sinusoidal Analysis

User generates sine wave traces and passes them to Protocol.
Core library is signal-agnostic - it just runs whatever traces you give it.

This script runs a Nyquist frequency sweep to characterize the complex
stiffness of the simulated half-sarcomere model.

Reference: Kawai & Brandt, 1980, J. Muscle Research and Cell Motility

Usage:
    python examples/sinusoidal_analysis.py
"""


import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Persistent XLA compilation cache — survives process restarts.
# Shared with tm_rate_sweep.py when both use the same batch/n_steps/dt.
_JAX_CACHE = os.path.expanduser("~/.cache/jax_compile_cache")
jax.config.update("jax_compilation_cache_dir", _JAX_CACHE)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from multifil_jax.simulation import run
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import get_skeletal_params, StaticParams

# ============================================================================
# USER CONFIGURATION - Edit these values
# ============================================================================

# Frequency sweep parameters (Hz).
# Must be integer multiples of bin_width = min_freq / N_CYCLES to avoid spectral leakage.
# With min_freq=1 Hz and N_CYCLES=5: bin_width=0.2 Hz, so all freqs must be multiples of 0.2.
# (3.13→3.2, 7.14→7.0, 16.7→16.6 relative to Kawai & Brandt values)
FREQUENCIES = [1.0, 2.0, 3.2, 5.0, 7.0, 10.0, 16.6, 25.0, 33.0, 50.0]

# Muscle activation — fully activated, matching Kawai & Brandt (1980)
PCA = 4.1

# Oscillation parameters
AMPLITUDE_NM = 1.0           # Oscillation amplitude (nm), linear regime
N_CYCLES = 5                 # Cycles of lowest frequency to simulate
SETTLING_CYCLES = 1          # Transient cycles to discard before FFT

# Replicate runs for statistics (vmapped for parallel execution)
# Multiple replicates capture variance from stochastic crossbridge dynamics
REPLICATES = 3               # Number of independent runs
                             # 1 = fast, no error bars
                             # 3-5 = good error bars

# Lattice size (affects number of crossbridges)
# IMPORTANT: nrows/ncols must be >= 2 (1x1 lattice behavior is unverified)
NROWS = 8                    # Lattice rows
NCOLS = 8                    # Lattice columns

# Baseline sarcomere geometry
MEAN_Z = 1100.0              # Mean z-line position (nm) — skeletal working SL ~2.2 µm
LATTICE_SPACING = 14.0       # Lattice spacing (nm)

# Output options
SAVE_PLOT = True
PLOT_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sinusoidal_analysis.png')

# ============================================================================
# Kawai & Brandt (1980) reference data — Fig. 3c
# Activated rabbit psoas, 20°C, pCa 4.1. Units: 10^5 N m^-2.
# ============================================================================

KAWAI_1980_FIG3C = {
    'freq_hz': np.array([0.25, 0.5, 1.0, 2.0, 3.13, 5.0, 7.14,
                         10.0, 16.7, 25.0, 33.0, 50.0, 80.0, 100.0, 133.0, 167.0]),
    'elastic': np.array([23.91, 35.06, 60.20, 78.06, 86.34, 90.36, 86.81,
                         82.83, 68.75, 52.23, 45.93, 53.31, 73.90, 81.74, 91.72, 96.91]),
    'viscous': np.array([26.56, 35.64, 35.79, 26.34, 15.19, 4.17, -6.72,
                         -13.73, -18.41, -10.64, 3.36, 26.45, 38.27, 41.38, 41.00, 40.49]),
}

# ============================================================================
# WAVEFORM GENERATION (User-driven "Science")
# ============================================================================

def generate_sine_traces(frequencies, mean_z, amplitude_nm, n_cycles):
    """Generate sinusoidal z-line traces for each frequency.

    dt is set to exactly 20 samples/cycle at the highest frequency so that
    each stimulus frequency falls on an exact FFT bin (no spectral leakage).
    Frequencies must be integer multiples of min_freq/n_cycles.

    Args:
        frequencies: List of frequencies in Hz (must be bin-aligned)
        mean_z: Mean z-line position (nm)
        amplitude_nm: Oscillation amplitude (nm)
        n_cycles: Number of cycles of the lowest frequency

    Returns:
        Tuple of (z_traces, duration_ms, dt, n_steps, time_ms)
    """
    min_freq = min(frequencies)
    max_freq = max(frequencies)

    # Duration captures n_cycles of the lowest frequency
    duration_ms = (1000.0 / min_freq) * n_cycles

    # dt: exactly 20 samples/cycle at max_freq — defines the FFT bin grid
    dt = 1000.0 / (max_freq * 20)
    n_steps = int(duration_ms / dt)

    print(f"Duration: {duration_ms:.1f} ms, dt: {dt:.3f} ms, n_steps: {n_steps}")

    # Generate time array
    time_ms = np.arange(n_steps) * dt

    # Generate sine wave traces for each frequency
    z_traces = [
        mean_z + amplitude_nm * np.sin(2 * np.pi * freq / 1000.0 * time_ms)
        for freq in frequencies
    ]

    return z_traces, duration_ms, dt, n_steps, time_ms

# ============================================================================
# ANALYSIS FUNCTIONS (User-defined post-processing)
# ============================================================================

def compute_complex_stiffness(result, frequencies, settling_cycles=1):
    """Compute complex stiffness via FFT.

    Analyzes the force response to sinusoidal length perturbations.

    Args:
        result: SimulationResult from Protocol
        frequencies: List of frequencies used
        settling_cycles: Number of initial cycles to discard

    Returns:
        Dict with freq_hz, magnitude, magnitude_std, phase, phase_std,
        elastic, viscous components
    """
    dt = result.dt
    min_freq = min(frequencies)

    # Discard settling period
    settling_samples = int(settling_cycles * 1000.0 / min_freq / dt)
    force = result.axial_force[..., settling_samples:]  # (..., replicates, time)
    n_samples = force.shape[-1]

    # FFT at each frequency
    magnitudes = []
    phases = []

    for i, freq in enumerate(frequencies):
        # Get force for this frequency, all replicates
        f_data = np.array(force[i])  # (replicates, time)

        # Truncate to an exact integer number of cycles for this frequency so
        # the FFT window is perfectly periodic — avoids spectral leakage.
        n_complete = int(n_samples * dt / 1000.0 * freq)
        n_use = int(round(n_complete * 1000.0 / freq / dt))
        n_use = min(n_use, n_samples)
        f_data = f_data[..., :n_use]

        # Reconstruct z_line for this frequency at the correct simulation time.
        # Force data starts at settling_samples*dt — the reconstructed z must
        # match that time origin or non-integer Hz frequencies accumulate a
        # phase offset of -360° * freq * settling_s (pure artifact).
        settling_time_ms = settling_samples * dt
        time_ms = np.arange(n_use) * dt + settling_time_ms
        z_data = MEAN_Z + AMPLITUDE_NM * np.sin(2 * np.pi * freq / 1000.0 * time_ms)

        # Remove DC component
        f_detrend = f_data - f_data.mean(axis=-1, keepdims=True)
        z_detrend = z_data - z_data.mean()

        # FFT
        f_fft = np.fft.fft(f_detrend, axis=-1)
        z_fft = np.fft.fft(z_detrend)
        freqs = np.fft.fftfreq(n_use, dt / 1000.0)

        # Find stimulus frequency bin
        idx = np.argmin(np.abs(freqs - freq))

        # Complex stiffness Y = F / L
        Y = f_fft[:, idx] / z_fft[idx]

        magnitudes.append(np.abs(Y))
        phases.append(np.angle(Y))

    # Compute statistics
    mag_mean = np.array([m.mean() for m in magnitudes])
    mag_std = np.array([m.std() for m in magnitudes])
    phase_mean = np.array([p.mean() for p in phases])
    phase_std = np.array([p.std() for p in phases])

    # Elastic and viscous components
    elastic = mag_mean * np.cos(phase_mean)
    viscous = mag_mean * np.sin(phase_mean)

    return {
        'freq_hz': np.array(frequencies),
        'magnitude': mag_mean,
        'magnitude_std': mag_std,
        'phase': phase_mean,
        'phase_std': phase_std,
        'elastic': elastic,
        'viscous': viscous,
    }


def plot_stiffness_all(stiffness, reference=None, save_path=None, suptitle=None):
    """Generate comprehensive stiffness plots.

    Args:
        stiffness: Dict from compute_complex_stiffness, units 10^5 N/m²
        reference: Optional dict with 'freq_hz', 'elastic', 'viscous' to overlay
                   (e.g. KAWAI_1980_FIG3C). Same units as stiffness.
        save_path: Optional path to save figure
        suptitle: Optional figure title

    Returns:
        Figure and axes
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    freq = stiffness['freq_hz']
    mag = stiffness['magnitude']
    phase_deg = np.degrees(stiffness['phase'])
    elastic = stiffness['elastic']
    viscous = stiffness['viscous']

    has_std = 'magnitude_std' in stiffness and stiffness['magnitude_std'] is not None

    # 1. Bode magnitude
    ax = axes[0, 0]
    if has_std:
        ax.errorbar(freq, mag, yerr=stiffness['magnitude_std'],
                   fmt='o-', capsize=3, markersize=6, label='Model')
    else:
        ax.plot(freq, mag, 'o-', markersize=6, label='Model')
    if reference is not None:
        ref_mag = np.sqrt(reference['elastic']**2 + reference['viscous']**2)
        ax.plot(reference['freq_hz'], ref_mag, 's--', color='tab:orange',
                markersize=6, label='Kawai & Brandt (1980)')
        ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Y| (10⁵ N/m²)')
    ax.set_title('Stiffness Magnitude')
    ax.grid(True, alpha=0.3)

    # 2. Bode phase
    ax = axes[0, 1]
    if has_std:
        ax.errorbar(freq, phase_deg, yerr=np.degrees(stiffness['phase_std']),
                   fmt='o-', capsize=3, markersize=6, label='Model')
    else:
        ax.plot(freq, phase_deg, 'o-', markersize=6, label='Model')
    if reference is not None:
        ref_phase = np.degrees(np.arctan2(reference['viscous'], reference['elastic']))
        ax.plot(reference['freq_hz'], ref_phase, 's--', color='tab:orange',
                markersize=6, label='Kawai & Brandt (1980)')
        ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Stiffness Phase')
    ax.grid(True, alpha=0.3)

    # 3. Elastic vs Viscous
    ax = axes[1, 0]
    ax.plot(freq, elastic, 'o-', label='Model elastic', markersize=6)
    ax.plot(freq, viscous, 's-', label='Model viscous', markersize=6)
    if reference is not None:
        ax.plot(reference['freq_hz'], reference['elastic'], 'o--', color='tab:orange',
                alpha=0.7, label='K&B elastic', markersize=5)
        ax.plot(reference['freq_hz'], reference['viscous'], 's--', color='tab:red',
                alpha=0.7, label='K&B viscous', markersize=5)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Stiffness Component (10⁵ N/m²)')
    ax.set_title('Elastic vs Viscous')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Nyquist plot
    ax = axes[1, 1]
    sc = ax.scatter(elastic, viscous, c=np.log10(freq), cmap='viridis', s=80,
                    zorder=3, label='Model')
    for i, f in enumerate(freq):
        ax.annotate(f'{f:.2g}', (elastic[i], viscous[i]),
                   textcoords='offset points', xytext=(5, 5), fontsize=7)
    if reference is not None:
        ref_mask = (reference['freq_hz'] >= freq.min()) & (reference['freq_hz'] <= freq.max())
        ax.scatter(reference['elastic'][ref_mask], reference['viscous'][ref_mask],
                   c=np.log10(reference['freq_hz'][ref_mask]), cmap='viridis',
                   s=60, marker='s', alpha=0.6, zorder=2)
        ax.plot(reference['elastic'], reference['viscous'], '--',
                color='tab:orange', alpha=0.5, linewidth=1.5, label='Kawai & Brandt (1980)')
        for i in np.where(ref_mask)[0]:
            ax.annotate(f'{reference["freq_hz"][i]:.2g}',
                       (reference['elastic'][i], reference['viscous'][i]),
                       textcoords='offset points', xytext=(5, -10),
                       fontsize=7, color='tab:orange')
        ax.legend(fontsize=8)
    ax.set_xlabel('Elastic (Real) [10⁵ N/m²]')
    ax.set_ylabel('Viscous (Imaginary) [10⁵ N/m²]')
    ax.set_title('Nyquist Plot')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, axes

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Kawai-Brandt Sinusoidal Stiffness Analysis")
    print("=" * 60)
    print("\nThis example demonstrates the SIGNAL-AGNOSTIC design:")
    print("  - User generates sine wave traces (the 'Science')")
    print("  - Core library just runs whatever traces you give it")
    print()

    # ========================================================================
    # Step 1: USER GENERATES WAVEFORMS (the "Science")
    # ========================================================================
    print("-" * 60)
    print("Step 1: Generating sinusoidal waveforms")
    print("-" * 60)

    z_traces, duration_ms, dt, n_steps, time_ms = generate_sine_traces(
        frequencies=FREQUENCIES,
        mean_z=MEAN_Z,
        amplitude_nm=AMPLITUDE_NM,
        n_cycles=N_CYCLES,
    )
    bin_width = min(FREQUENCIES) / N_CYCLES
    print(f"FFT bin width: {bin_width:.3f} Hz — all stimulus freqs are exact multiples")

    print(f"Generated {len(z_traces)} sinusoidal traces")
    print(f"Frequencies: {FREQUENCIES} Hz")

    # ========================================================================
    # Step 2: EXECUTE VIA run() (the "Engine")
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Executing grid via run()")
    print("-" * 60)

    static, dynamic = get_skeletal_params()
    topo = SarcTopology.create(
        nrows=NROWS,
        ncols=NCOLS,
        static_params=static,
        dynamic_params=dynamic,
    )

    result = run(
        topo,
        z_line=z_traces,          # List of arrays -> sweep axis
        pCa=PCA,                  # Scalar -> broadcast
        lattice_spacing=LATTICE_SPACING,
        duration_ms=duration_ms,
        dt=dt,
        replicates=REPLICATES,
        verbose=True,
    )

    # result.axial_force.shape -> (n_freq, replicates, n_steps)
    # result.coords['z_line'] -> list of n_freq sine arrays

    print(f"Result shape: {result.axial_force.shape}")
    print(f"Axes: {result._axis_names}")

    # ========================================================================
    # Step 3: ANALYSIS (user-defined post-processing)
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 3: Computing complex stiffness")
    print("-" * 60)

    stiffness = compute_complex_stiffness(result, FREQUENCIES, settling_cycles=SETTLING_CYCLES)

    # Convert units: pN/nm → 10^5 N/m²
    # Y [pN/nm] * z_line [nm] / CSA [nm²] * 10 = modulus [10^5 N/m²]
    # (pN/nm * nm / nm² = pN/nm² = MPa = 10 × 10^5 N/m²)
    from math import sqrt as _sqrt
    R_THICK, R_THIN = 8.0, 4.5
    d_cc = LATTICE_SPACING + R_THICK + R_THIN
    csa = topo.n_thick * (3 * _sqrt(3) / 2) * d_cc ** 2   # nm², vertebrate hexagonal
    unit_scale = MEAN_Z / csa * 10  # pN/nm → 10^5 N/m²
    for key in ('elastic', 'viscous', 'magnitude'):
        stiffness[key] = stiffness[key] * unit_scale
    if stiffness.get('magnitude_std') is not None:
        stiffness['magnitude_std'] = stiffness['magnitude_std'] * unit_scale
    print(f"\nUnit conversion: CSA={csa:.1f} nm², scale={unit_scale:.4f} (pN/nm → 10⁵ N/m²)")

    # Print summary table
    print("\n" + "-" * 70)
    header = f"{'Freq (Hz)':>10} {'|Y| (10⁵N/m²)':>14} {'Phase (deg)':>12} {'Elastic':>10} {'Viscous':>10}"
    if REPLICATES > 1:
        header += " (mean +/- std)"
    print(header)
    print("-" * 70)

    for i, f in enumerate(FREQUENCIES):
        mag = float(stiffness['magnitude'][i])
        phase = np.degrees(float(stiffness['phase'][i]))
        elastic = float(stiffness['elastic'][i])
        viscous = float(stiffness['viscous'][i])

        if REPLICATES > 1 and 'magnitude_std' in stiffness:
            mag_std = float(stiffness['magnitude_std'][i])
            print(f"{f:>10.2f} {mag:>8.2f}+/-{mag_std:<4.2f} {phase:>12.1f} "
                  f"{elastic:>10.2f} {viscous:>10.2f}")
        else:
            print(f"{f:>10.2f} {mag:>12.2f} {phase:>12.1f} {elastic:>10.2f} {viscous:>10.2f}")

    # ========================================================================
    # Step 4: PLOTTING
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 4: Generating plots")
    print("-" * 60)

    save_path = PLOT_FILENAME if SAVE_PLOT else None
    fig, axes = plot_stiffness_all(
        stiffness,
        reference=KAWAI_1980_FIG3C,
        save_path=save_path,
        suptitle=f'Sinusoidal Stiffness (pCa {PCA}, n={REPLICATES}) vs Kawai & Brandt (1980)',
    )

    # ========================================================================
    # DEMONSTRATE SIMULATIONRESULT FEATURES
    # ========================================================================
    print("\n" + "-" * 60)
    print("Bonus: SimulationResult Grid Features Demo")
    print("-" * 60)

    # mean() collapses replicate axis
    mean_result = result.mean()
    print(f"result.mean().shape: {mean_result.axial_force.shape}")

    # std() gives standard deviation
    std_result = result.std()
    print(f"result.std().shape: {std_result.axial_force.shape}")

    # Slicing
    first_freq = result[0]
    print(f"result[0].shape (first frequency): {first_freq.axial_force.shape}")

    # Access coords
    print(f"Replicate indices: {result.coords['replicates']}")

    print("\nDone!")
