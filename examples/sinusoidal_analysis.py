"""
Kawai-Brandt Sinusoidal Analysis

User generates sine wave traces and passes them to Protocol.
Core library is signal-agnostic - it just runs whatever traces you give it.

This script runs a Nyquist frequency sweep to characterize the complex
stiffness of the simulated half-sarcomere model.

Reference: Kawai & Brandt, 1980, J. Muscle Research and Cell Motility

Usage:
    python sinusoidal_analysis.py
"""


import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from multifil_jax.simulation import run
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import get_default_params, StaticParams

# ============================================================================
# USER CONFIGURATION - Edit these values
# ============================================================================

# Frequency sweep parameters (Hz)
# Typical range covers 0.1-100 Hz; reduced here for faster demo
FREQUENCIES = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

# Muscle activation
PCA = 5.0                    # Calcium concentration (-log10[Ca])
                             # pCa 5.0 = half-maximal activation

# Oscillation parameters
AMPLITUDE_NM = 5.0           # Oscillation amplitude (nm)
                             # Keep small (1-10 nm) for linear regime
N_CYCLES = 5                 # Cycles of lowest frequency to simulate
                             # More cycles = better FFT resolution
SETTLING_CYCLES = 1          # Transient cycles to discard before FFT

# Replicate runs for statistics (vmapped for parallel execution)
# Multiple replicates capture variance from stochastic crossbridge dynamics
REPLICATES = 3               # Number of independent runs
                             # 1 = fast, no error bars
                             # 3-5 = good error bars

# Lattice size (affects number of crossbridges)
# IMPORTANT: nrows/ncols must be >= 2 (1x1 lattice behavior is unverified)
NROWS = 2                    # Lattice rows
NCOLS = 2                    # Lattice columns

# Baseline sarcomere geometry
MEAN_Z = 900.0               # Mean z-line position (nm)
LATTICE_SPACING = 14.0       # Lattice spacing (nm)

# Output options
SAVE_PLOT = True
PLOT_FILENAME = 'sinusoidal_analysis.png'

# ============================================================================
# WAVEFORM GENERATION (User-driven "Science")
# ============================================================================

def generate_sine_traces(frequencies, mean_z, amplitude_nm, n_cycles, min_dt=0.1):
    """Generate sinusoidal z-line traces for each frequency.

    This is the "user-driven science" part. The core library doesn't know
    about sinusoids - it just runs whatever traces you give it.

    Args:
        frequencies: List of frequencies in Hz
        mean_z: Mean z-line position (nm)
        amplitude_nm: Oscillation amplitude (nm)
        n_cycles: Number of cycles of the lowest frequency
        min_dt: Minimum time step (ms)

    Returns:
        Tuple of (z_traces, duration_ms, dt, n_steps, time_ms)
    """
    min_freq = min(frequencies)
    max_freq = max(frequencies)

    # Compute duration to capture n_cycles of the lowest frequency
    duration_ms = (1000.0 / min_freq) * n_cycles

    # Compute dt for proper Nyquist sampling (20 samples per cycle minimum)
    dt = min(min_dt, 1000.0 / (max_freq * 20))
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

        # Reconstruct z_line for this frequency (we know the formula)
        time_ms = np.arange(n_samples) * dt
        z_data = MEAN_Z + AMPLITUDE_NM * np.sin(2 * np.pi * freq / 1000.0 * time_ms)

        # Remove DC component
        f_detrend = f_data - f_data.mean(axis=-1, keepdims=True)
        z_detrend = z_data - z_data.mean()

        # FFT
        f_fft = np.fft.fft(f_detrend, axis=-1)
        z_fft = np.fft.fft(z_detrend)
        freqs = np.fft.fftfreq(n_samples, dt / 1000.0)

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


def plot_stiffness_all(stiffness, save_path=None, suptitle=None):
    """Generate comprehensive stiffness plots.

    Args:
        stiffness: Dict from compute_complex_stiffness
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

    # Check for error bars
    has_std = 'magnitude_std' in stiffness and stiffness['magnitude_std'] is not None

    # 1. Bode magnitude
    ax = axes[0, 0]
    if has_std:
        ax.errorbar(freq, mag, yerr=stiffness['magnitude_std'],
                   fmt='o-', capsize=3, markersize=6)
    else:
        ax.plot(freq, mag, 'o-', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|Y| (pN/nm)')
    ax.set_title('Stiffness Magnitude')
    ax.grid(True, alpha=0.3)

    # 2. Bode phase
    ax = axes[0, 1]
    if has_std:
        ax.errorbar(freq, phase_deg, yerr=np.degrees(stiffness['phase_std']),
                   fmt='o-', capsize=3, markersize=6)
    else:
        ax.plot(freq, phase_deg, 'o-', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Stiffness Phase')
    ax.grid(True, alpha=0.3)

    # 3. Elastic vs Viscous
    ax = axes[1, 0]
    ax.plot(freq, elastic, 'o-', label='Elastic', markersize=6)
    ax.plot(freq, viscous, 's-', label='Viscous', markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Stiffness Component (pN/nm)')
    ax.set_title('Elastic vs Viscous')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Nyquist plot (Real vs Imaginary)
    ax = axes[1, 1]
    ax.scatter(elastic, viscous, c=np.log10(freq), cmap='viridis', s=80)
    for i, f in enumerate(freq):
        ax.annotate(f'{f:.1f} Hz', (elastic[i], viscous[i]),
                   textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Elastic (Real) [pN/nm]')
    ax.set_ylabel('Viscous (Imaginary) [pN/nm]')
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

    print(f"Generated {len(z_traces)} sinusoidal traces")
    print(f"Frequencies: {FREQUENCIES} Hz")

    # ========================================================================
    # Step 2: EXECUTE VIA run() (the "Engine")
    # ========================================================================
    print("\n" + "-" * 60)
    print("Step 2: Executing grid via run()")
    print("-" * 60)

    static, dynamic = get_default_params()
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

    # Print summary table
    print("\n" + "-" * 70)
    header = f"{'Freq (Hz)':>10} {'|Y| (pN/nm)':>12} {'Phase (deg)':>12} {'Elastic':>10} {'Viscous':>10}"
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
        save_path=save_path,
        suptitle=f'Sinusoidal Stiffness (pCa {PCA}, n={REPLICATES})'
    )

    plt.show()

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
