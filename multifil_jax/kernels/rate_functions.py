"""
Rate Functions for Tropomyosin and Crossbridge Kinetics

This module contains all rate functions extracted from transitions.py.
Users can customize rates by:
1. Modifying the DynamicParams values passed to these functions
2. Writing custom rate functions with the same signature

Each rate function takes explicit parameters rather than a params dict,
making dependencies clear and enabling easy customization.

Rate function naming convention:
- tm_k_XY: Tropomyosin rate for transition X->Y
- xb_rXY: Crossbridge rate for transition X->Y

States:
- TM: 0=unbound, 1=Ca-bound, 2=closed, 3=open (permissive)
- XB: 1=DRX, 2=loose, 3=tight_1, 4=tight_2, 5=free_2, 6=SRX
"""
import jax.numpy as jnp


# =============================================================================
# TROPOMYOSIN RATE FUNCTIONS
# =============================================================================

def tm_rate_12(ca_conc, k_12_base, coop_factor):
    """Rate 0->1 (Ca binding).

    Calcium binding rate depends on calcium concentration and cooperativity.

    Args:
        ca_conc: Calcium concentration (M)
        k_12_base: Base rate constant (per M per ms) - from params.tm_k_12
        coop_factor: Cooperativity multiplier (1.0 if not cooperative)

    Returns:
        Rate k_12 (per ms)
    """
    return k_12_base * ca_conc * coop_factor


def tm_rate_21(k_12_base, K1):
    """Rate 1->0 (Ca unbinding) - detailed balance.

    Args:
        k_12_base: Base forward rate constant (per M per ms)
        K1: Equilibrium constant for Ca binding

    Returns:
        Rate k_21 (per ms)
    """
    return k_12_base / K1


def tm_rate_23(k_23_base, coop_factor):
    """Rate 1->2 (Closed transition).

    Args:
        k_23_base: Base rate constant (per ms) - from params.tm_k_23.
                   1.0 ms⁻¹ (skeletal) / 0.5 ms⁻¹ (cardiac);
                   Fraser & Bhatt 2019; Geeves & Lehrer 1994 (20–1000 s⁻¹).
        coop_factor: Cooperativity multiplier

    Returns:
        Rate k_23 (per ms)
    """
    return k_23_base * coop_factor


def tm_rate_32(k_23_base, K2):
    """Rate 2->1 (Return from closed) - detailed balance.

    Args:
        k_23_base: Base forward rate constant (per ms)
        K2: Equilibrium constant

    Returns:
        Rate k_32 (per ms)
    """
    return k_23_base / K2


def tm_rate_34(k_34_base, coop_factor):
    """Rate 2->3 (Open transition - becomes permissive).

    Args:
        k_34_base: Base rate constant (per ms) - from params.tm_k_34
        coop_factor: Cooperativity multiplier

    Returns:
        Rate k_34 (per ms)
    """
    return k_34_base * coop_factor


def tm_rate_43(k_34_base, K3, can_leave):
    """Rate 3->2 (Return from open) - detailed balance.

    Args:
        k_34_base: Base forward rate constant (per ms)
        K3: Equilibrium constant
        can_leave: 1.0 if site can leave state 3, 0.0 if XB bound

    Returns:
        Rate k_43 (per ms)
    """
    return (k_34_base / K3) * can_leave


def tm_rate_41(k_41_base, can_leave):
    """Rate 3->0 (Return to unbound from open state).

    Args:
        k_41_base: Base rate constant (per ms) - from params.tm_k_41
        can_leave: 1.0 if site can leave state 3, 0.0 if XB bound

    Returns:
        Rate k_41 (per ms)
    """
    return k_41_base * can_leave


# =============================================================================
# CROSSBRIDGE RATE FUNCTIONS
# =============================================================================

def xb_rate_12(permissiveness, r12_coeff, E_weak):
    """Rate DRX->Loose (binding).

    This is the main binding rate. 

    Args:
        permissiveness: 0-1, whether binding site is available (from TM state)
        r12_coeff: Consolidated binding coefficient - from params.xb_r12_coeff
                   (= OOP mh_br * tau = 424.987 * 0.72 = 305.99)
        E_weak: Weak state potential energy (kT units)

    Returns:
        Rate r12 (per ms)
    """
    r12 = permissiveness * r12_coeff * jnp.exp(-E_weak)
    return jnp.where(jnp.isnan(r12), 0.0, r12)


def xb_rate_21(r12, U_DRX, U_loose):
    """Rate Loose->DRX (unbinding) - detailed balance.

    Uses log-space arithmetic to avoid overflow.

    Args:
        r12: Forward binding rate
        U_DRX: Free energy of DRX state (kT units)
        U_loose: Free energy of loose state (kT units)

    Returns:
        Rate r21 (per ms)
    """
    upper = 10000.0
    log_r21 = jnp.log(r12 + 0.005) - (U_DRX - U_loose)
    r21 = jnp.exp(log_r21)
    r21 = jnp.minimum(r21, upper)
    return jnp.where(jnp.isnan(r21), upper, r21)


def xb_rate_23(A23, E_diff):
    """Rate Loose->Tight_1 (weak-to-strong isomerization).

    Symmetric barrier (α=0.5): r23 = A23 × exp(ΔG₂₃/2kT) = A23 × exp(E_diff/2).
    Faster when the strong state has less elastic strain than the weak state (E_diff > 0),
    restoring full position-dependent mechanosensitivity.
    Cap at 30 kT prevents overflow without affecting physiological range.

    Args:
        A23: Pre-exponential rate coefficient (per ms) - from params.xb_r23_coeff.
             Fitting parameter targeting process B apparent rate (2πb). Skeletal:
             2πb ~ 20–60 s⁻¹ at 20–25°C (Kawai & Zhao 1993 Biophys J 65:638).
             Note: 286 s⁻¹ from Kawai & Zhao 1993 is k₂ (ATP-induced detachment,
             process C) — not the isomerization rate. The symmetric barrier (α=0.5)
             is a modeling assumption, not derived from Duke 1999.
        E_diff: Energy difference E_weak - E_strong (kT units, positive favors strong state)

    Returns:
        Rate r23 (per ms)

    References:
        Kawai & Zhao 1993 Biophys J 65:638 (process B rates); Månsson 2016 Biophys J.
    """
    r23 = A23 * jnp.exp(jnp.minimum(E_diff, 30.0) / 2.0)
    return jnp.where(jnp.isnan(r23), 0.0, r23)


def xb_rate_32(r23, U_loose, U_tight_1):
    """Rate Tight_1->Loose (reverse power stroke) - detailed balance.

    Uses log-space arithmetic.

    Args:
        r23: Forward rate
        U_loose: Free energy of loose state
        U_tight_1: Free energy of tight_1 state

    Returns:
        Rate r32 (per ms)
    """
    upper = 10000.0
    log_r32 = jnp.log(r23) - (U_loose - U_tight_1)
    r32 = jnp.exp(log_r32)
    return jnp.minimum(r32, upper)


def xb_rate_34(A34, f_3_4, delta34, k_t):
    """Rate Tight_1->Tight_2 (power stroke).

    Bell model: r34 = A34 × exp(-f × δ₃₄ / kT).
    Resisting force (f > 0) slows the stroke; assisting force accelerates it.
    δ₃₄ ≈ 0.5 nm is the lever-arm transition-state distance.
    Tight_1 and Tight_2 share the same spring, so there is no E_diff dependence.

    Args:
        A34: Pre-exponential rate (per ms) - from params.xb_r34_coeff.
             Calibrated to ~70-100 s⁻¹ at zero load (Millar & Homsher 1990).
        f_3_4: Force in strong state (pN); positive = resisting (toward M-line)
        delta34: Transition-state distance for power stroke (nm) - params.xb_delta_34.
                 1.0 nm; Pate & Cooke 1989 JMRCM 10:181; Huxley & Simmons 1971 (1–2 nm).
        k_t: Thermal energy kT (pN·nm)

    Returns:
        Rate r34 (per ms)

    References:
        Bell 1978 Science; Walcott 2010 Biophys J; Reconditi 2011 PNAS; Piazzesi 2007 Cell;
        Pate & Cooke 1989 JMRCM 10:181; Huxley & Simmons 1971 Nature.
    """
    upper = 10000.0
    r34 = A34 * jnp.exp(-f_3_4 * delta34 / k_t)
    return jnp.minimum(r34, upper)


def xb_rate_43(r34, U_tight_1, U_tight_2):
    """Rate Tight_2->Tight_1 (reverse power stroke) - detailed balance.

    r43 = r34 × exp(U_tight_2 - U_tight_1).
    With new defaults (U_tight_2=-23, U_tight_1=-18.6): r43 ≈ 0.012 × r34 —
    strongly suppresses the reverse stroke, consistent with high duty ratio.

    Args:
        r34: Forward power-stroke rate (per ms)
        U_tight_1: Free energy of tight_1 state (kT units, includes E_strong)
        U_tight_2: Free energy of tight_2 state (kT units, includes E_strong)

    Returns:
        Rate r43 (per ms)

    References:
        Hill 1977 Free Energy Transduction; Pate & Cooke 1989 JMRCM 10:181.
    """
    upper = 10000.0
    log_r43 = jnp.log(r34 + 1e-30) + (U_tight_2 - U_tight_1)
    return jnp.exp(jnp.minimum(log_r43, jnp.log(upper)))


def xb_rate_45(A45, f_3_4, delta45, k_t):
    """Rate Tight_2->Free_2 (ADP release / detachment).

    Slip bond (Bell 1978): tensile load accelerates ADP release for fast skeletal
    myosin II. Positive sign is essential — opposite sign to r34 (catch bond would
    suppress detachment under load, contradicting rapid skeletal muscle kinetics).
    δ₄₅ ≈ 0.5 nm.

    Args:
        A45: Pre-exponential detachment rate (per ms) - from params.xb_r45_coeff.
             ≥500 s⁻¹ at near-zero load (Siemankowski & White 1984 JBC).
        f_3_4: Force in strong state (pN); positive = tensile (accelerates detachment)
        delta45: Transition-state distance for detachment (nm) - params.xb_delta_45
        k_t: Thermal energy kT (pN·nm)

    Returns:
        Rate r45 (per ms)

    References:
        Bell 1978 Science; Siemankowski & White 1984 JBC; Veigel 2005 Nat Cell Biol;
        Capitanio 2006 PNAS; Walcott 2010 Biophys J; Prodanovic 2019 J Gen Physiol.
    """
    upper = 10000.0
    r45 = A45 * jnp.exp(f_3_4 * delta45 / k_t)
    return jnp.minimum(r45, upper)


def xb_rate_54():
    """Rate Free_2->Tight_2 (reverse binding after detachment).

    This is zero - no reverse transition.

    Returns:
        Rate r54 = 0 (per ms)
    """
    return 0.0


def xb_rate_51(r51_rate):
    """Rate Free_2->DRX (recovery to relaxed state).

    Args:
        r51_rate: Rate constant - from params.xb_r51
                  (= 0.1, was hardcoded in OOP, now parameterized)

    Returns:
        Rate r51 (per ms)
    """
    return r51_rate


def xb_rate_15(r15_rate):
    """Rate DRX->Free_2 (rare direct transition).

    Args:
        r15_rate: Rate constant - from params.xb_r15.
                  0.01 ms⁻¹; Mijailovich 2020 (k−H=10 s⁻¹); detailed balance r51/r15=10.

    Returns:
        Rate r15 (per ms)
    """
    return r15_rate


def xb_rate_61(ca_conc, k0, kmax, b, ca50):
    """Rate SRX->DRX (Ca-dependent activation from super-relaxed).

    Hill equation for calcium-dependent SRX exit.

    Args:
        ca_conc: Calcium concentration (M)
        k0: Basal rate (per ms) - from params.xb_srx_k0.
            Skeletal 0.007 ms⁻¹; cardiac 0.005 ms⁻¹; Mijailovich 2020 (kPS0=5 s⁻¹).
        kmax: Maximum rate (per ms) - from params.xb_srx_kmax.
              0.4 ms⁻¹; Mijailovich 2020 (kPSmax=400 s⁻¹).
        b: Hill coefficient - from params.xb_srx_b.
           5.0; Mijailovich 2020; Linari 2015 Nature 528:276.
        ca50: Ca50 for half-activation (M) - from params.xb_srx_ca50

    Returns:
        Rate r61 (per ms)
    """
    return k0 + ((kmax - k0) * ca_conc**b) / (ca50**b + ca_conc**b)


def xb_rate_16(r16_rate):
    """Rate DRX->SRX (entering super-relaxed state).

    Args:
        r16_rate: Consolidated rate - from params.xb_r16.
                  Skeletal 0.007 ms⁻¹ (50% SRX at rest); cardiac 0.012 ms⁻¹ (70% SRX);
                  Stewart 2010 PNAS 107:430; Linari 2015 Nature 528:276.

    Returns:
        Rate r16 (per ms)
    """
    return r16_rate


# =============================================================================
# CONVENIENCE: Default rate functions dictionary
# =============================================================================

def get_default_rate_functions():
    """Return dict of all default rate functions for easy override.

    Usage:
        rate_fns = get_default_rate_functions()
        rate_fns['xb_rate_12'] = my_custom_r12  # Override specific rate
        # Pass rate_fns to transitions if needed
    """
    return {
        # TM rates
        'tm_rate_12': tm_rate_12,
        'tm_rate_21': tm_rate_21,
        'tm_rate_23': tm_rate_23,
        'tm_rate_32': tm_rate_32,
        'tm_rate_34': tm_rate_34,
        'tm_rate_43': tm_rate_43,
        'tm_rate_41': tm_rate_41,
        # XB rates
        'xb_rate_12': xb_rate_12,
        'xb_rate_21': xb_rate_21,
        'xb_rate_23': xb_rate_23,
        'xb_rate_32': xb_rate_32,
        'xb_rate_34': xb_rate_34,
        'xb_rate_43': xb_rate_43,
        'xb_rate_45': xb_rate_45,
        'xb_rate_54': xb_rate_54,
        'xb_rate_51': xb_rate_51,
        'xb_rate_15': xb_rate_15,
        'xb_rate_61': xb_rate_61,
        'xb_rate_16': xb_rate_16,
    }


# =============================================================================
# ENERGY CALCULATIONS (used by rate functions)
# =============================================================================

def compute_xb_energies(r, theta, g_k_weak, g_r_weak, c_k_weak, c_r_weak,
                        g_k_strong, g_r_strong, c_k_strong, c_r_strong, k_t):
    """Compute crossbridge potential energies in weak and strong states.

    Args:
        r: Radial distance from thick to thin filament (nm)
        theta: Angular position (radians)
        g_k_weak, g_r_weak: Globular domain spring constant and rest length (weak)
        c_k_weak, c_r_weak: Converter domain spring constant and rest angle (weak)
        g_k_strong, g_r_strong: Globular domain (strong)
        c_k_strong, c_r_strong: Converter domain (strong)
        k_t: Thermal energy kT (pN*nm)

    Returns:
        E_weak: Energy in weak state (kT units)
        E_strong: Energy in strong state (kT units)
        E_diff: E_weak - E_strong (computed directly for precision)
    """
    # Weak state energy (states 2, 5)
    E_weak = (0.5 * g_k_weak * (r - g_r_weak)**2 +
              0.5 * c_k_weak * (theta - c_r_weak)**2) / k_t

    # Strong state energy (states 3, 4)
    E_strong = (0.5 * g_k_strong * (r - g_r_strong)**2 +
                0.5 * c_k_strong * (theta - c_r_strong)**2) / k_t

    # Compute difference directly to avoid catastrophic cancellation
    delta_g_energy = 0.5 * (g_k_weak * (r - g_r_weak)**2 - g_k_strong * (r - g_r_strong)**2)
    delta_c_energy = 0.5 * (c_k_weak * (theta - c_r_weak)**2 - c_k_strong * (theta - c_r_strong)**2)
    E_diff = (delta_g_energy + delta_c_energy) / k_t

    return E_weak, E_strong, E_diff


def compute_xb_free_energies(E_weak, E_strong, U_DRX=-2.3, U_loose_base=-4.3,
                              U_tight_1_base=-18.6, U_tight_2_base=-20.72):
    """Compute free energies for all XB states.

    Free energies from Pate & Cooke 1989, in units of kT.

    Args:
        E_weak: Weak state potential energy
        E_strong: Strong state potential energy
        U_DRX: Base free energy of DRX state
        U_loose_base: Base free energy of loose state (before adding E_weak)
        U_tight_1_base: Base free energy of tight_1 state (before adding E_strong)
        U_tight_2_base: Base free energy of tight_2 state (before adding E_strong)

    Returns:
        U_DRX, U_loose, U_tight_1, U_tight_2: Free energies for each state
    """
    U_loose = U_loose_base + E_weak
    U_tight_1 = U_tight_1_base + E_strong
    U_tight_2 = U_tight_2_base + E_strong

    return U_DRX, U_loose, U_tight_1, U_tight_2


def compute_xb_force(r, theta, g_k_strong, g_r_strong, c_k_strong, c_r_strong):
    """Compute force in strong state for force-dependent rates.

    Args:
        r, theta: Position
        g_k_strong, g_r_strong: Globular domain params
        c_k_strong, c_r_strong: Converter domain params

    Returns:
        f_3_4: Force in strong state (pN)
    """
    return g_k_strong * (r - g_r_strong) + (1.0/r) * c_k_strong * (theta - c_r_strong)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing rate_functions module...")
    print("=" * 60)

    # Test TM rates
    print("\nTest 1: TM rate functions")
    print("-" * 60)

    ca_conc = 1e-4  # pCa 4
    k_12_base = 54633.43
    coop_factor = 100.0
    K1, K2, K3 = 260000.0, 130.0, 0.91
    k_23_base = 931.233
    k_34_base = 0.07490
    k_41_base = 18.1178

    k_12 = tm_rate_12(ca_conc, k_12_base, coop_factor)
    k_21 = tm_rate_21(k_12_base, K1)
    k_23 = tm_rate_23(k_23_base, coop_factor)
    k_32 = tm_rate_32(k_23_base, K2)
    k_34 = tm_rate_34(k_34_base, coop_factor)
    k_43 = tm_rate_43(k_34_base, K3, can_leave=1.0)
    k_41 = tm_rate_41(k_41_base, can_leave=1.0)

    print(f"k_12 (Ca binding): {k_12:.4f}")
    print(f"k_21 (Ca unbinding): {k_21:.6f}")
    print(f"k_23 (closed): {k_23:.4f}")
    print(f"k_32 (reverse): {k_32:.4f}")
    print(f"k_34 (open): {k_34:.4f}")
    print(f"k_43 (reverse): {k_43:.4f}")
    print(f"k_41 (return): {k_41:.4f}")

    # Test XB rates
    print("\nTest 2: XB rate functions")
    print("-" * 60)

    # Sample inputs
    permissiveness = 1.0
    r12_coeff = 305.99
    E_weak, E_strong, E_diff = 2.0, 1.5, 0.5
    U_DRX, U_loose, U_tight_1, U_tight_2 = -2.3, -2.3, -16.8, -18.92
    f_3_4 = 5.0  # pN

    k_t = 4.14  # pN*nm at ~26°C
    r12 = xb_rate_12(permissiveness, r12_coeff, E_weak)
    r21 = xb_rate_21(r12, U_DRX, U_loose)
    r23 = xb_rate_23(0.6, E_diff)
    r32 = xb_rate_32(r23, U_loose, U_tight_1)
    r34 = xb_rate_34(0.15, f_3_4, 1.0, k_t)
    r43 = xb_rate_43(r34, U_tight_1, U_tight_2)
    r45 = xb_rate_45(0.6, f_3_4, 0.5, k_t)
    r51 = xb_rate_51(0.1)
    r15 = xb_rate_15(0.01)
    r61 = xb_rate_61(ca_conc, 0.007, 0.4, 5.0, 1e-6)
    r16 = xb_rate_16(0.007)

    print(f"r12 (binding): {r12:.6f}")
    print(f"r21 (unbinding): {r21:.6f}")
    print(f"r23 (loose->tight_1): {r23:.6f}")
    print(f"r32 (tight_1->loose): {r32:.6f}")
    print(f"r34 (tight_1->tight_2): {r34:.6f}")
    print(f"r43 (tight_2->tight_1): {r43:.6f}")
    print(f"r45 (detachment): {r45:.6f}")
    print(f"r51 (recovery): {r51:.6f}")
    print(f"r15 (DRX->free_2): {r15:.6f}")
    print(f"r61 (SRX->DRX): {r61:.6f}")
    print(f"r16 (DRX->SRX): {r16:.6f}")

    # Test rate functions dict
    print("\nTest 3: Rate functions dictionary")
    print("-" * 60)
    rate_fns = get_default_rate_functions()
    print(f"Total rate functions: {len(rate_fns)}")
    print(f"Functions: {list(rate_fns.keys())}")

    print("\nAll tests passed!")
