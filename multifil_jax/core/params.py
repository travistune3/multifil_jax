"""
Physical and structural parameters for the half-sarcomere model.

The model needs two kinds of parameter, and they behave very differently under
JIT compilation, so they live in two different containers:

StaticParams — structural configuration
    How many crowns on a thick filament, how many crossbridges on a crown, what
    the actin helix pitch is, how many Newton iterations the solver may take.
    These determine array *shapes*, so they are baked into the compiled kernel:
    changing one forces a recompile and (usually) a rebuilt SarcTopology. It is a
    frozen dataclass, deliberately NOT a JAX PyTree.

DynamicParams — physics values
    Spring stiffnesses, reaction rates, free energies, cooperativity couplings.
    These are JAX arrays inside a registered PyTree, so they can be swept across
    a batch dimension without recompiling — the whole point of the split. Run a
    thousand stiffness values in one call and JAX sees a single batched kernel.

Rates and stiffnesses here are ABSOLUTE values in model units (ms, nm, pN, kT),
not multipliers on some hidden base. A rate constant in this file is the rate
the kernel uses. Nothing is scaled again downstream.

pCa, z_line, and lattice_spacing appear here too, holding their default (or
swept) values. When they need to vary within a single simulation they are
overridden per timestep through Drivers; see core/state.py.

UNITS
-----
    time            ms          (a rate of 0.6 means 0.6 ms^-1 = 600 s^-1)
    length          nm
    force           pN
    stiffness       pN/nm (linear) or pN*nm/rad (angular)
    energy          kT, except spring energies which are pN*nm
    calcium         pCa = -log10([Ca2+] in M)
    temperature     degrees Celsius

CONFIDENCE TIERS
----------------
Muscle models are underdetermined: many parameters have no direct measurement,
and several that do have measurements spanning an order of magnitude. Every
value below is therefore tagged, so a reader can tell what is anchored and what
is a placeholder:

    [M] MEASURED   a literature value used essentially as reported
    [I] INFERRED   derived by arithmetic or geometry from measured value(s),
                   in this model's particular parameterization
    [G] GUESS      no source for this system: a value carried over from another
                   species/muscle type, a plausibility estimate, or a number
                   inherited from an earlier implementation
    [F] FITTED     tuned so the model reproduces some emergent measurement
                   (a Hill coefficient, an apparent rate), not measured directly

A [F] or [G] value is not wrong — it is simply not evidence. Treat the tag as
part of the parameter.
"""
import jax
import jax.numpy as jnp
import numpy as np
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

# Static fields that affect array shapes (changing these triggers recompilation)
STATIC_FIELDS = frozenset({'n_crowns', 'n_polymers_per_thin', 'solver_max_iter', 'actin_geometry', 'n_newton_steps', 'n_cg_steps', 'solver_residual_tol', 'n_xb_bins', 'xb_bin_lo', 'xb_bin_hi', 'thick_bare_zone', 'thick_crown_spacing', 'actin_half_pitch', 'mono_per_poly', 'polymer_base_turns', 'target_zone_wiggle', 'n_xb_per_crown'})

# Single source of truth: every DynamicParams field with its skeletal default.
# Citations and confidence tiers ([M]/[I]/[G]/[F], see module docstring) live
# inline against each entry. Insertion order is preserved (Python 3.7+), so
# DYNAMIC_FIELDS below matches tree_flatten/unflatten order.
# To add a new field: add ONE entry here. __slots__, DYNAMIC_FIELDS, and
# __init__ all derive from this dict.
_DYNAMIC_DEFAULTS = {
    # ==========================================================================
    # MECHANICAL PARAMETERS
    #
    # Both filaments are modelled as chains of linear springs between nodes:
    # crowns on the thick filament, binding sites on the thin. Filament
    # compliance is not a detail — a myosin head straining against a compliant
    # backbone transmits less force than one anchored to a rigid rod, and
    # filament compliance is comparable to crossbridge compliance in real
    # muscle, so it redistributes strain among attached heads and couples them
    # mechanically even when they are far apart.
    # ==========================================================================

    # Thick filament — per-segment spring constant (between adjacent crowns)
    # Whole-filament axial stiffness = thick_k / n_segments. Model has 52 crowns
    # → 52 segments (M-line + 51 inter-crown) over ~730 nm.
    #
    # Brunello et al. 2014 J Physiol 592:3881 (PMC4192709, skeletal X-ray interferometry):
    #     specific compliance c_M = 17.5 nm·MPa⁻¹·µm⁻¹, half-sarc thick length l_M = 0.8 µm.
    #     Per-thick cross-section area A_M = (√3/2)·d_{thick-thick}² ≈ 1754 nm² (d_{TT}≈45.9 nm).
    #     k_whole = A_M / (c_M · l_M) = 1754 / (17.5 × 0.8) = 125 pN/nm.
    # Mijailovich et al. 2021 PMC7852458 Table 1 (cardiac MUSICO model):
    #     AE_m = 132 nN, l_m = 0.8 µm → k_whole = AE_m / l_m = 165 pN/nm.
    # Lit consensus midpoint ≈ 145 pN/nm whole-filament.
    # Conversion to per-segment: thick_k = k_whole × n_segments = 145 × 52 ≈ 7540 → round to 7500.
    # NOTE: this per-segment value is tied to n_crowns=52. A preset with a different
    # crown count (e.g. insect flight muscle at 100) inherits the same per-segment
    # stiffness, which is the right invariant here — crown spacing is a physical
    # ~15 nm in every species, so equal per-segment stiffness means equal stiffness
    # per unit length, and the longer filament is correctly more compliant overall.
    'thick_k': 7500.0,  # [I] pN/nm per segment (whole-filament ≈ 144 pN/nm)

    # Thin filament — per-segment spring constant (between adjacent binding-site nodes)
    # Whole-filament axial stiffness = thin_k / n_segments. At the vertebrate
    # defaults there are 90 binding-site nodes per thin filament + a z-line anchor
    # → 90 segments over ~1077 nm. The node count is NOT a constant of the model:
    # it follows from actin_half_pitch, mono_per_poly and target_zone_wiggle.
    #
    # Brunello et al. 2014 J Physiol 592:3881:
    #     specific compliance c_A = 14.3 nm·MPa⁻¹·µm⁻¹, half-sarc thin length l_A = 0.975 µm.
    #     BOTH VERIFIED verbatim 2026-08-19: "sarcomere length 2.15 μm, lA = 0.975 μm,
    #     lM = 0.800 μm, ζ = 0.700 μm, cA = 14.3 ± 1.9 nm MPa⁻¹ μm⁻¹". Frog, intact fibres.
    #     Per-thin area A_A = A_M / 2 = 877 nm² (1:2 thick:thin stoichiometry, vertebrate).
    #     k_whole = A_A / (c_A · l_A) = 877 / (14.3 × 0.975) = 63 pN/nm.
    #   (Note: do NOT use an overlap-corrected C_A of 6.9 — the string "6.9" does
    #    not appear anywhere in Brunello 2014, so it must not be attributed to that
    #    paper; the caution itself still stands — it removes the
    #    non-overlap segment from the load path; our uniform model spring of length l_A
    #    represents the whole filament, so use c × L without the correction.)
    # Mijailovich et al. 2021 Table 1: AE_a = 65 nN, l_a = 1.1 µm → k_whole = 59 pN/nm.
    # Lit consensus ≈ 60 pN/nm whole-filament (Brunello and Mij agree).
    # Conversion: thin_k = k_whole × n_segments = 60 × 90 = 5400 → round to 5500.
    # Thin is ~2× more compliant per length than thick (consistent with smaller cross-
    # section: actin double helix ~7 nm Ø vs myosin backbone ~15 nm Ø).
    #
    # KNOWN LIMITATION — per-segment stiffness is a leaky invariant on the thin
    # filament. Unlike crown spacing (a physical ~15 nm in every species), thin
    # segment length is an artifact of which monomers the angular acceptance window
    # (target_zone_wiggle) admits as binding sites. Widen that window and segments
    # get shorter, so the same thin_k silently yields a more compliant filament per
    # unit length. Comparing the vertebrate defaults against the invertebrate
    # presets: segment length 11.97 nm vs 9.73 nm, giving axial rigidity
    # (thin_k × L) of 65,800 pN vs 53,500 pN — the insect thin filament ends up
    # ~19% more compliant per unit length purely from a binding-geometry choice.
    # The equivalent thick-filament comparison agrees to ~1%, as it should.
    # If you change target_zone_wiggle, or compare mechanics across presets that
    # differ in it, rescale thin_k to hold thin_k × segment_length fixed.
    'thin_k': 5500.0,   # [I] pN/nm per segment (whole-filament ≈ 61 pN/nm)

    # --------------------------------------------------------------------------
    # CROSSBRIDGE SPRINGS — the two-spring myosin head
    #
    # A myosin head is not a simple axial spring. It is a lever arm hinged at the
    # converter domain, projecting away from the thick filament backbone to reach
    # actin. The model captures this with two springs acting on the head's polar
    # geometry relative to its bound site:
    #
    #     r     = sqrt(x^2 + d^2)     head length      (x = axial offset to the
    #     theta = atan2(d, x)         head angle        bound site, d = lattice
    #                                                   spacing)
    #
    #     U = 0.5*g_k*(r - g_rest)^2  +  0.5*c_k*(theta - c_rest)^2
    #          globular (linear)           converter (angular)
    #
    # The power stroke is a change in the springs' REST configuration, not an
    # applied force: weak and strong states have different (g_rest, c_rest), so a
    # head that transitions to the strong state suddenly finds itself displaced
    # from equilibrium and pulls. The two rest configurations sit at
    #
    #     weak:   r=19.93 nm, theta=47.2 deg  ->  axial 13.55 nm, radial 14.61 nm
    #     strong: r=16.47 nm, theta=73.2 deg  ->  axial  4.76 nm, radial 15.77 nm
    #
    # i.e. an ~8.8 nm separation between the rest configurations. There are only
    # TWO of them, so the whole stroke is on the weak->strong isomerization
    # (state 1 -> 2); states 2 and 3 share the strong configuration and the
    # 2 -> 3 transition moves nothing. See kernels/rate_functions.py.
    #
    # 8.8 nm is NOT the stroke the mechanics delivers. A rest configuration is a
    # point in (r, theta) and is only reachable at its own natural radial
    # distance; at a fixed lattice spacing only x can vary, so what matters is
    # where f_axial = 0 along that line. At the default d = 14 nm:
    #
    #     weak   force zero at x = 14.126 nm  (rest config says 13.55)
    #     strong force zero at x =  8.288 nm  (rest config says  4.76)
    #
    # so the EFFECTIVE axial working stroke is 5.84 nm, not 8.8 — both zeros
    # move, and the weak one moves the other way, so the stroke shrinks. 5.84 nm
    # sits comfortably in the measured 5-11 nm range (Huxley & Simmons 1971
    # Nature 233:533; Piazzesi 2002 Biophys J 82:3118) and matches the ~6 nm
    # usually quoted for beta-cardiac (Sung 2015 Nat Commun 6:7931; Woody 2019
    # eLife 8:e49266). Quote 5.84, not 8.79, as this model's working stroke — and
    # note it is lattice-spacing dependent, which is the mechanism by which
    # lattice spacing feeds back into crossbridge kinetics.
    #
    # Also at d = 14 nm the local axial stiffness of the strong state is
    # ~1.3-1.4 pN/nm rather than g_k = 5. THIS is why single-head
    # stiffness measurements (~2-3 pN/nm; Woody 2019 eLife 8:e49266 VERIFIED:
    # "we estimate the stiffness of the motor to be 2.3 pN/nm (68% CI, 2.0-2.7
    # pN/nm) or greater", recombinant human beta-cardiac MYH7, and Woody notes it
    # is "at the upper end of previous estimates") must
    # be compared against the projected two-spring stiffness, never against g_k
    # directly — and it is also the mechanism by which lattice spacing feeds back
    # into crossbridge kinetics, since strain-dependent rates read this geometry.
    #
    # PROVENANCE — this is the 2sXB head of Williams CD, Regnier M & Daniel TL
    # 2010, "Axial and Radial Forces of Cross-Bridges Depend on Lattice Spacing",
    # PLoS Comput Biol 6(12):e1001018. Its Table 1 gives, for the 2sXB model:
    #
    #       spring                    rest      k            this module
    #       h   converter, pre        47 deg    40 pN/rad    xb_c_rest_weak
    #       h'  converter, post       73 deg    40 pN/rad    xb_c_rest_strong
    #       r   linear, pre           20 nm      2 pN/nm     xb_g_rest_weak
    #       r'  linear, post          16 nm      2 pN/nm     xb_g_rest_strong
    #
    # The five-figure rest values here are not spurious precision: the paper's
    # Table 1 caption states the 2sXB rests are computed from the 4sXB geometry so
    # that both models' heads rest in the same place, and the table rounds them.
    # Upstream measurements behind that geometry (all three VERIFIED verbatim in
    # the Table 1 caption, 2026-08-19): the 125 -> 70 deg lever rotation is Taylor
    # et al. 1999; the light-chain-domain rest length is from Houdusse et al. 2000
    # (structure 1DFK); the S2 angle range is Liu et al. 2006 -- though Williams did
    # NOT adopt Liu's 51-60 deg, arguing it was "distorted by the compressive radial
    # force ... in the swollen lattice spacings that Liu et al. used" and choosing
    # 40-50 deg instead. The geometry inherited here is Williams' CORRECTION of Liu. The
    # lineage that reuses this head: Daniel, Trimble & Chase 1998 Biophys J
    # 74:1611; Chase, Macpherson & Daniel 2004 Ann Biomed Eng 32:1559; Tanner,
    # Daniel & Regnier 2007 PLoS Comput Biol 3(7):e115; Williams et al. 2013
    # Proc R Soc B 280:20130697.
    #
    # STIFFNESSES
    # c_k_strong = 40 pN*nm/rad [I]: as Williams 2010 (via Taylor 1999).
    # c_k_weak = 8.0, g_k_weak = 0.4 [G]: a 5x softening of the weak state relative
    #   to Williams 2010, which uses one stiffness per spring across both states
    #   (40 pN/rad, 2 pN/nm). State-dependent softening is physically motivated —
    #   a weakly bound, non-stereospecific head should bear less load than a
    #   stereospecifically bound one — but the factor of 5 is a modelling choice
    #   with no measurement behind it, and how any softening partitions between the
    #   linear and angular springs is not measured either.
    #   KNOWN MISMATCH: at these values the weak state carries ~28% of total
    #   crossbridge tension. The X-ray decomposition of Eakins et al. 2016
    #   (Biology 5:41) puts it at ~4%, while this model matches that study's state
    #   POPULATIONS well (24%/32.5% here vs 20%/32% there). So the weak state is
    #   roughly 7x too load-bearing, and moving toward Williams 2010's stiffer
    #   values would worsen it. Treat both as open, and see section 5 of
    #   .claude/lit_reviews/state_scheme_rate_function_audit.md before fitting them.
    # g_k_strong = 5.0 pN/nm [G]: taken from the 4sXB "c" spring of the same paper
    #   (5 pN/nm, Houdusse 2000) rather than the 2sXB r' value of 2 pN/nm. What
    #   matters mechanically is the projected two-spring stiffness, not g_k alone
    #   (see above): that projection is ~1.33 pN/nm at d = 14 nm, which agrees
    #   closely with the single crossbridge stiffness of 1.3 pN/nm used by MUSICO
    #   (Mijailovich et al. 2021 JGP 153:e202012604, after Duke 1999 -- Duke's
    #   K = 1.3 pN/nm verified 2026-08-19).
    #   BUT 1.3 pN/nm is another MODEL's chosen value, while the two direct
    #   MEASUREMENTS sit at 2.3-2.7 pN/nm: Woody 2019 >=2.3 (human beta-cardiac,
    #   single molecule) and Brunello et al. 2014 J Physiol 592:3881 Table 1,
    #   in-situ crossbridge stiffness eps = 2.7 +/- 0.9 pN/nm (frog). So this
    #   projection agrees with the models and is ~2x below the experiments. Open.
    # --------------------------------------------------------------------------
    'xb_c_rest_weak':   0.82309,   # [G] rad — pre-stroke converter angle
    'xb_c_rest_strong': 1.27758,   # [G] rad — post-stroke converter angle
    'xb_c_k_weak':      8.0,       # [I] pN·nm/rad — pre-stroke compliant
    'xb_c_k_strong':    40.0,      # [I] pN·nm/rad
    'xb_g_rest_weak':   19.93,     # [G] nm — pre-stroke head length
    'xb_g_rest_strong': 16.47,     # [G] nm — post-stroke head length
    'xb_g_k_weak':      0.4,       # [I] pN/nm — pre-stroke compliant
    'xb_g_k_strong':    5.0,       # [I] pN/nm

    # --------------------------------------------------------------------------
    # TITIN — the passive elastic element
    #
    # Titin is a single giant protein spanning from the Z-disc to the thick
    # filament. It is what makes resting muscle springy: stretch a relaxed fibre
    # and titin, not the crossbridges, resists. Its force-extension curve is
    # steeply non-linear (entropic unfolding of Ig domains, then backbone
    # stretching), modelled here as a one-sided exponential:
    #
    #     F = titin_a * exp(titin_b * (L - titin_rest)),  clamped to >= 0
    #
    # L is the true 3D length of the connection, sqrt(axial^2 + lattice_spacing^2),
    # so titin also pulls the filaments together radially — a term that matters
    # for dynamic lattice spacing. axial = z_line - last_crown_position.
    #
    # PRACTICAL WARNING: at long sarcomere lengths titin can dominate total force
    # (it exceeded 3/4 of total force in one insect-geometry configuration).
    # Any force-pCa or force-length analysis must subtract a relaxed (pCa 9-10)
    # baseline before reading active force, or the active signal is swamped.
    #
    # Skeletal defaults are the compliant N2A isoform; get_cardiac_params()
    # overrides them with the stiffer N2B values.
    # titin_a [I]: Powers, Williams, Regnier & Daniel 2018 Integr.Comp.Biol. 58:186
    #          (260 pN divided by the 6 titins per thick filament = 43 pN each).
    #          NB the 260 pN is ARBITRARY in the source, which states plainly:
    #          "a was arbitrarily fixed at 260 pN". The division is ours and the
    #          "6 titins per thick filament" divisor is not from that paper either.
    # titin_b [G]: RETAGGED FROM [M] 2026-08-19, and it is NOT a psoas calibration.
    #          Powers 2018 SWEPT this parameter: "b was set to 4, 7.5, or 10 µm⁻¹"
    #          to study the effect of titin stiffness. 4 µm⁻¹ = 0.004 nm⁻¹ is simply
    #          their most compliant case, paired with the arbitrary a above.
    # titin_rest [I]: slack length at SL 2.0 µm (z_line=1000 nm → L≈213 nm);
    #          Linke 1998 PNAS 95:8052
    # --------------------------------------------------------------------------
    'titin_a':    43.0,    # [I] pN per molecule (from an arbitrary 260 pN — see above)
    'titin_b':    0.004,   # [G] nm⁻¹ — a swept value in Powers 2018, not a measurement
    'titin_rest': 215.0,   # [I] nm

    # ==========================================================================
    # TROPOMYOSIN KINETICS — thin filament regulation
    #
    # Tropomyosin physically covers myosin's binding sites on actin. Calcium
    # binding to troponin lets it move aside. The four-state cycle is
    #   0 Ca-free/blocking -> 1 Ca-bound/blocking -> 2 closed -> 3 open,
    # closing 3 -> 0 when calcium dissociates. Only state 3 permits binding.
    # See kernels/rate_functions.py for each transition's rate law.
    #
    # Forward rates are given here; every reverse rate is derived from the
    # forward rate and the matching equilibrium constant, so detailed balance
    # cannot be broken by a careless parameter choice. The one exception is the
    # cycle-closing 3 -> 0 step, which is deliberately one-way.
    #
    # These rates are NOT well constrained. The measurements that exist span
    # more than an order of magnitude, so a value picked from within a cited
    # range is a modelling decision as much as a measurement.
    # ==========================================================================
    'tm_k_01': 100000.0,   # [L-compilation] M⁻¹ms⁻¹ = 1e8 M⁻¹s⁻¹. Robertson, Johnson
                           #     & Potter 1981 Biophys J 34:559, Table I: skeletal Tn
                           #     Ca-specific sites k_on = 1.15e8 M⁻¹s⁻¹. Model is within
                           #     13% of that. NOTE the previously quoted "5e7–2e8" range
                           #     is NOT in the paper — deleted 2026-08-19. Robertson is a
                           #     COMPILATION (a computer-modelling paper citing others),
                           #     and states no temperature and no ionic strength.
    'tm_k_12': 1.0,        # [G] ms⁻¹ blocking→closed. NO SOURCE EXISTS. Audit 2026-08-19:
                           #     Geeves & Lehrer 1994 contains no "20–1000 s⁻¹" range (its
                           #     only rate statement is a ONE-SIDED bound, k+T + k-T >> 500
                           #     s⁻¹, and it applies to the closed↔open step = our 2↔3, not
                           #     this one). Fraser & Marston 1995 contains ZERO rate
                           #     constants — it is a steady-state motility paper (its result
                           #     is pCa50 5.8–6.0 and all-or-none switching). Both citations
                           #     removed. The EQUILIBRIUM is constrained: McKillop & Geeves
                           #     1993 K_B = [closed]/[blocked] = 0.3 (−Ca), ≥16 (+Ca) —
                           #     see tm_Keq_12.
    'tm_k_23': 0.1,        # [G] ms⁻¹ closed→open. The "50–200 s⁻¹" previously noted here
                           #     had no citation at all. The one real constraint is
                           #     Geeves & Lehrer 1994 Biophys J 67:273 p.277: the switch is
                           #     in RAPID EQUILIBRIUM, k+T + k-T >> 500 s⁻¹ — a one-sided
                           #     bound on the SUM only. Model: 100 + k_32 1000 = 1100 s⁻¹,
                           #     which satisfies it. Only the sum and the ratio (tm_Keq_23)
                           #     are constrained, so both rates may be scaled together.
                           #     (rabbit skeletal, reconstituted Tm·Tn·actin + S1, 20 °C)
    'tm_k_30': 0.2,        # [G] ms⁻¹ cycle close (Ca off). The cited "Robertson 1981
                           #     (100–500 s⁻¹)" is NOT in that paper: no k_off in its
                           #     Table I lies in 100–500. Skeletal Tn Ca-specific k_off is
                           #     23.0 s⁻¹; the only 100–500 entry is 238 s⁻¹, which belongs
                           #     to CALMODULIN, a different protein. Model is 8.7× the cited
                           #     skeletal value. What IS measured for this physical quantity
                           #     (Ca off-rate from an assembled filament): Davis 2007 ssTn
                           #     7.8 s⁻¹ isolated, 85 s⁻¹ in thin filament (15 °C).
                           #     NB the cardiac preset uses 40 s⁻¹, so the two presets are
                           #     INVERTED relative to Davis, who measures cardiac Ca release
                           #     as FASTER than skeletal in every matched condition.
                           #     Dominates relaxation kinetics — see rate_functions.tm_rate_30

    # Equilibrium constants. These set every reverse rate (k_reverse = k_forward / Keq).
    'tm_Keq_01': 500000.0,     # [G] M⁻¹, i.e. Kd 2 µM for the regulatory site.
                               #     Potter & Gergely 1975 JBC 250:4628 SUMMARY: whole
                               #     troponin binds "two sites ... 5 × 10⁸ M⁻¹, and two ...
                               #     5 × 10⁶ M⁻¹", the latter being the Ca-specific
                               #     (regulatory) sites (Kd 0.2 µM); under 2 mM MgCl₂ all
                               #     four sites go to a single 5 × 10⁶ M⁻¹. (CORRECTION
                               #     2026-08-19: the 2×10⁶ figure previously quoted here is
                               #     isolated TnC, not Tn — the two were conflated.)
                               #     Conditions: rabbit skeletal, isolated protein, 100 mM
                               #     KCl, imidazole pH 7.0; TEMPERATURE NOT STATED.
                               #     This value is ~10× weaker than the isolated-protein
                               #     figure, on the reasoning that regulatory-site affinity
                               #     is lower in an intact thin filament — that DIRECTION is
                               #     now confirmed measured (Pinto 2011 Table 1: cTn pCa50
                               #     6.69 → thin filament 6.13, a 3.6× affinity loss), but
                               #     the 10× magnitude remains a choice, so still [G].
                               #     Note the model's Ca sensitivity is dominated by the
                               #     SRX gate, not by this constant: a 5× change moves
                               #     force pCa50 by ~0.02 units
    'tm_Keq_12': 130.0,        # [G] dimensionless. A SOURCE DOES EXIST (found 2026-08-19;
                               #     the old "no source located" note was wrong): McKillop &
                               #     Geeves 1993 Biophys J 65:693 abstract gives
                               #     K_B = [closed]/[blocked] = 0.3 without Ca²⁺ and ≥16
                               #     with Ca²⁺ — this model's 1↔2 step. Their K_B is
                               #     Ca-DEPENDENT; ours is not, because this scheme puts all
                               #     Ca-dependence in 0→1. So 130 is not directly comparable
                               #     to 0.3 or 16, and the magnitude remains a choice: [G].
    'tm_Keq_23': 0.1,          # [L] dimensionless; McKillop & Geeves 1993 Biophys J
                               #     65:693, abstract: "K_T = [open]/[closed] = 0.09 and
                               #     0.25 in the absence and presence of calcium,
                               #     respectively." VERIFIED verbatim 2026-08-19 — the one
                               #     thin-filament claim that checks out cleanly. <1 means
                               #     closed is favoured at rest, as it should be.
                               #     Conditions: rabbit skeletal S1 + reconstituted
                               #     actin·Tm·Tn, 140 mM KCl, 5 mM MgCl₂, MOPS pH 7.0, 20 °C.
                               #     TWO CAVEATS: (a) their K_T is Ca-dependent, ours is
                               #     fixed; (b) their Table 1 spans K_T = 0.013–0.60 across
                               #     nucleotide conditions (no-nucleotide 0.013, Pi 0.18,
                               #     ADP 0.12, sulfate 0.07 at −Ca), so a single fixed 0.1 is
                               #     a mid-range pick from a 46-fold spread, and a
                               #     contracting muscle is not the no-nucleotide condition.

    # ------------------------------------------------------------------------
    # SYMMETRIC ISING COOPERATIVITY — the tropomyosin cooperativity model.
    #
    # Tropomyosin is a continuous strand, not a set of independent switches:
    # opening one stretch mechanically strains its neighbours toward opening
    # too. That is what makes force rise far more steeply with calcium than
    # single-site binding could explain. The model treats each site as a spin
    # coupled to its two nearest neighbours ALONG ITS OWN TROPOMYOSIN STRAND
    # (topology.tm_prev_neighbor / tm_next_neighbor — a structural adjacency,
    # with no distance cutoff involved).
    #
    # Local field on site i:
    #     h = J_C * n_2 + J_M * n_3 - 0.5*(J_C + J_M) * n_closed
    # where n_2, n_3, n_closed count that site's two neighbours in each
    # category. Forward rates are scaled by exp(+h/2) and reverse rates by
    # exp(-h/2). Splitting the factor symmetrically means the equilibrium
    # constant shifts by exactly exp(h) — the Boltzmann factor for an energy
    # well deepened by h — so the coupling is a real free-energy term and
    # detailed balance survives. Boosting only forward rates would not.
    # The one-way cycle-closing rate k_30 is deliberately left unscaled.
    #
    # WHAT THE TWO COUPLINGS COUNT (see kernels/transitions.py,
    # count_neighbor_states_split): tm_J_C counts a site's neighbours in STATE 2,
    # which this scheme calls closed; tm_J_M counts neighbours in STATE 3 (open),
    # whether or not a crossbridge is attached to them; n_closed counts states 0
    # and 1 together, i.e. the two blocking states.
    #
    # So tm_J_M is a tropomyosin-to-tropomyosin strand coupling, not a
    # myosin-binding coupling: it operates with no crossbridges present at all
    # (with binding disabled it still lifts the Hill coefficient from 1.01 to
    # 1.54). Myosin-induced cooperativity does enter, but indirectly — an attached
    # head locks its site in state 3, and that site then contributes to its
    # neighbours' field. At pCa 4.0 roughly 47% of open sites carry a crossbridge
    # and 53% do not, so the two contributions are of comparable size.
    #
    # tm_J_C is empirically inert in this model and defaults to 0. That is an
    # observation here, not a theoretical result — but Saadat et al. 2026
    # (arXiv:2603.03866, Ising Models of Cooperativity in Muscle Contraction)
    # give a reason to expect it. Their Hamiltonian is the standard
    # H = -sum(J s_i s_{i+1} + h s_i): calcium enters as the FIELD h and motor
    # force sets the nearest-neighbour COUPLING J (their Eq. 4, h = 1/2 log(c)
    # and J = 1/2 log(n_H); their Fig. 3a plots J against motor force F_0). So
    # their J is the direct analogue of tm_J_M, not of a field. Their §III result
    # is that force-pCa data alone cannot determine more than two parameters —
    # meaning a SECOND coupling is unidentifiable from that observable, which is
    # exactly what tm_J_C's observed inertness looks like from inside the model.
    #
    # tm_J_M is the live knob. It has
    # no direct measurement — it is [F], tuned so the model reproduces an
    # observed Hill coefficient, and it is NOT transferable: its meaning depends
    # on the chain's discretization, so it must be re-tuned after any change to
    # how tropomyosin neighbours are defined or how many sites a filament has.
    #
    # tm_J_M IS NOT CURRENTLY CALIBRATED. The natural structural handle is the
    # correlation length of the open/closed state along the chain, measurable
    # straight from a tm_states snapshot with no fitting loop. Measured here
    # (chain driven alone, crossbridge binding DISABLED; 1 chain site ~ 4.35
    # actin monomers at the vertebrate site spacing):
    #
    #       tm_J_M   1.50   2.00   2.25   2.50   2.70   3.00
    #       xi (mon) 2.53   4.11   5.14   6.52   7.84   9.98
    #
    # These numbers are real, but they do NOT currently anchor to a literature
    # target, and an earlier reading of this table that put tm_J_M at 1.5-2.5 has
    # been WITHDRAWN. Saadat et al.'s correlation length of 2-7 is in REGULATORY
    # UNITS, not actin monomers (their §IV B: "correlation length between
    # neighboring regulatory units"; their Eq. 9 indexes spins, and one spin is
    # one RU = 7 monomers). That is 14-49 monomers — above, not below, the range
    # tabulated here, so the correction reverses direction. Two further
    # mismatches block a naive comparison even after the unit fix: their spin
    # requires an ATTACHED MOTOR whereas the rows above were measured with
    # binding disabled, and their discretization is 7 monomers per spin against
    # this model's ~4.35 per site.
    #
    # The shipped 2.70 came from a cardiac force-pCa calibration whose kinetics
    # could not be confirmed, so it is [F] and should be re-derived rather than
    # quoted. Both presets currently share it. If re-deriving, prefer a
    # correlation-length route over tuning to a Hill coefficient — nH is
    # dominated by the SRX gate, not by this coupling — but fix the observable
    # and the discretization first.
    # ------------------------------------------------------------------------
    'tm_J_C': 0.0,   # [I] kT, coupling to closed (state-2) neighbours; inert here
    'tm_J_M': 2.70,  # [F] kT, coupling to open (state-3) neighbours; see above

    # ==========================================================================
    # CROSSBRIDGE KINETICS
    #
    # The six-state cycle 0 DRX -> 1 Loose -> 2 Tight_1 -> 3 Tight_2 -> 4 Free_2
    # -> 0, plus state 5 SRX as an off-pathway reserve. One ATP per lap. Full
    # rate laws and their strain dependence: kernels/rate_functions.py.
    #
    # Only FORWARD rates appear here. Reverse rates are computed from these and
    # the free energies below, so the cycle cannot be parameterized into
    # violating thermodynamics. The coefficients named *_coeff are
    # pre-exponential factors: the actual rate is this value times a strain- or
    # load-dependent exponential, so none of them is directly comparable to a
    # measured rate except at zero strain and zero load.
    # ==========================================================================
    'xb_r01_coeff': 305.99,  # [G] ms⁻¹ attachment. NO literature source: inherited
                             #     from an earlier parameterization of this model,
                             #     where it arose as a binding rate times a duty
                             #     scaling. Sets the overall attachment timescale —
                             #     effectively a free parameter, and a natural first
                             #     candidate when fitting. The 5-figure precision is
                             #     an artifact of that derivation, not significance.
    'xb_r12_coeff': 0.6,     # [F] ms⁻¹ weak→strong (Pi release). Tuned to the
                             #     apparent rate of sinusoidal-analysis process B
                             #     (2πb ~ 20–60 s⁻¹ skeletal; Kawai & Zhao 1993
                             #     Biophys J 65:638), not measured directly
    'xb_r23_coeff': 0.15,    # [G] ms⁻¹ working stroke at zero load. The cited
                             #     "Millar & Homsher 1990 (70–100 s⁻¹)" is NOT in that
                             #     paper (audit 2026-08-19): JBC 265:20234 measures
                             #     k_Pi = 23.5 ± 1.7 s⁻¹ at 10 °C, 0.7 mM Pi, and
                             #     k_TR = 13.6 ± 0.2 s⁻¹ — the only two rates it
                             #     reports. HYPOTHESIS (not established): 70–100 s⁻¹
                             #     may be 23.5 s⁻¹ Q10-corrected from 10 °C to ~26 °C
                             #     (Q10 2–3 gives 68–120). If so, write that down;
                             #     otherwise drop the attribution.
    'xb_r34_coeff': 0.6,     # [I] ms⁻¹ ADP release/detachment at zero load.
                             #     Siemankowski & White 1984 JBC 259:5045 say
                             #     ">500 s⁻¹" for rabbit skeletal at 15 °C — but that
                             #     is (a) a ONE-SIDED BOUND, not a value, and (b) not
                             #     their own measurement: the sentence carries their
                             #     ref (14). Their own data are bovine ventricle.
                             #     Retagged [M] → [I] 2026-08-19.
    'xb_delta_23': 1.0,      # [I] nm, distance to the working-stroke transition
                             #     state; Pate & Cooke 1989 JMRCM 10:181.
                             #     Huxley & Simmons 1971 Nature 233:533 give
                             #     1/a = 2 nm and h = 8 nm (the stroke); there is no
                             #     "1–2 nm" range in the paper, and 1/a is one of "the
                             #     parameters used in obtaining this degree of
                             #     agreement", i.e. a FIT to tension transients (frog,
                             #     4 °C), not a measurement. Retagged [M] → [I].
    'xb_delta_34': 0.5,      # [I] nm, distance to the detachment transition state.
                             #     VERIFIED: Duke 1999 PNAS 96:2770 — with d = 11 nm
                             #     and K = 1.3 pN/nm chosen, "the value of ε₂ then
                             #     implies d' = 0.5 nm". [I] is the right tag: it is
                             #     implied inside a model, not measured. CAVEAT: Duke's
                             #     d' is a CONFORMATIONAL displacement of the lever on
                             #     ADP release, whereas this parameter is used as a
                             #     Bell transition-state distance. Measured Bell
                             #     distances are larger — Sung 2015 0.8 ± 0.1 nm,
                             #     Wang 2024 1.53–1.87 nm. Note this parameter
                             #     trades off against xb_g_k_strong when fitting —
                             #     stiffness and Bell distance can compensate for
                             #     each other to give the same load sensitivity
    'xb_r40': 0.1,           # [G] ms⁻¹ recovery stroke. Unsourced, inherited as a
                             #     hardcoded constant from an earlier version.
                             #     Caps the maximum cycling rate
    'xb_r04': 0.01,          # [M] ms⁻¹ reverse recovery; Mijailovich 2021
                             #     PMC7852458 (k−H = 10 s⁻¹). With r40 this gives
                             #     a hydrolysis equilibrium constant of 10
    'xb_r05': 0.007,         # [I] ms⁻¹ DRX→SRX sequestration. The "~50% SRX at rest"
                             #     is an ARITHMETIC CONSEQUENCE of setting xb_srx_k0
                             #     equal to this rate (r05/(r05+k0) = 0.5), not a target
                             #     taken from Stewart 2010. Stewart PNAS 107:430 Table 1
                             #     measures P2 = 0.31 ± 0.01 (n=24) in relaxed rabbit
                             #     psoas at ~24 °C — i.e. ~31% SRX, so the model sits
                             #     ~1.6× high. Stewart is strongly temperature-dependent:
                             #     P2 = 0.16 at 12 °C, 0.33 at 30 °C. What Stewart does
                             #     establish is the EXISTENCE and slow turnover of the
                             #     resting SRX pool (lifetime T2 = 230 ± 24 s).

    # Free energies of each state (kT), relative to a common reference. These fix
    # every reverse rate by detailed balance, so they encode the direction and
    # irreversibility of the cycle rather than merely labelling it. The total drop
    # around the cycle is ΔG_ATP ≈ -22 to -24 kT at 37 °C — the energy available
    # from one ATP, and therefore the ceiling on the work one crossbridge can do.
    #
    # The reference set is Howard 2001, Mechanics of Motor Proteins and the
    # Cytoskeleton (Sinauer), Table 14.2 "Actin-myosin hydrolysis cycle (rabbit
    # skeletal muscle)", p. 235. VERIFIED CELL BY CELL 2026-08-19 by reading the
    # page as an image (the PDF's OCR text layer corrupts digits and units and
    # must NOT be quoted; see .claude/papers/TEXT_LAYER_REPLACEMENTS.md).
    # All ten energies below are transcribed correctly.
    # CORRECTION: the table's note says [ATP] = 4 mM, not 2 mM as previously
    # stated here. Verbatim: "In parentheses are the approximate free energies of
    # the states assuming [ATP] = 4 mM, [P_i] = 2 mM, [ADP] = 20 uM."
    # CONDITIONS (same note): rabbit skeletal, 20 C, "isolated S1 and/or
    # myofibrils under roughly physiological ionic strength", overall ATPase
    # k_cat = 25 s^-1. Howard's own caveat: "The values vary by up to a factor of
    # 10 between laboratories."
    # State free energies in kT at [ATP] = 4 mM, [Pi] = 2 mM, [ADP] = 20 µM:
    #
    #   attached   A.M.T (+8)   A.M.D.P (0)   A.M.D (-12)   A.M (-15)   A.M.T (-17)
    #   detached   M.T (0)      M.D.P (-2)    M.D (-8)      M (-6)      M.T (-25)
    #
    # How this model maps onto it, and where it differs:
    #   xb_U_DRX     -2.3  <->  M.D.P    -2    close
    #   xb_U_loose   -4.3  <->  A.M.D.P   0    4.3 kT lower here
    #   xb_U_tight_1 -15.0 <->  A.M.D   -12    3 kT lower here
    #   xb_U_tight_2 -21.0 <->  (no counterpart — see below)
    #
    # Howard has ONE strongly-bound A.M.D state. This model splits it into
    # tight_1 and tight_2, so xb_U_tight_2 has no direct equivalent; the 6 kT gap
    # between them is what suppresses the reverse transition to ~0.25% of the
    # forward rate and makes the cycle effectively one-way. It is a modelling
    # choice, not a measured free energy.
    # Where the sources align:
    #   DRX -> loose     -2.0 kT here. Pate & Cooke 1989 JMRCM 10:181 put the first
    #                    bound state "only 2 RT below that of the detached M.D.P
    #                    state" — an exact match.
    #   Pi release      -10.7 kT here; -12 kT in Howard; 14 RT in Pate & Cooke
    #                    (from K34 = 1.89e-4 M^-1 at 3 mM Pi). Both VERIFIED, and
    #                    they bracket -10.7.
    #                    WITHDRAWN 2026-08-19: the "-8 to -13 kT range of Månsson
    #                    2016 JMRCM 37:181 and Offer & Ranatunga 2013" has NO
    #                    source. BOTH papers are now on disk and BOTH were checked:
    #                    Månsson 2016 (JMRCM 37:181-194, locator correct) contains
    #                    only three kT values -- 18 kBT for the MT-vs-AM free-energy
    #                    minima difference, and 18/7 kBT as an NEM-HMM binding
    #                    energy in a figure legend -- and gives NO Pi-release free
    #                    energy at all. Offer & Ranatunga 2013 contains no kT/RT
    #                    values whatsoever (and its locator was wrong: 105:928-940,
    #                    not 105:1767).
    #   cycle total     -23 RT in Pate & Cooke, -25 kT in Howard.
    #
    # KNOWN IMBALANCE: the return path (tight_2 -> free_2 -> DRX) is only ~-2.3 kT
    # net in these base energies, against -8 to -10 kT expected for ATP binding
    # alone. It does not corrupt the forward cycle because r34 is a Bell rate and
    # r40 a constant — neither is set by detailed balance — but the reverse-path
    # thermodynamics are not self-consistent, and the total around the cycle falls
    # short of Howard's -25 kT.
    'xb_U_DRX':     -2.3,    # [I] kT — M.ATP / M.ADP.Pi, detached
    'xb_U_loose':   -4.3,    # [I] kT — AM.ADP.Pi, weakly bound
    'xb_U_tight_1': -15.0,   # [I] kT — AM.ADP; Pi release ΔG ≈ -10.7 kT
    'xb_U_tight_2': -21.0,   # [G] kT — second strongly-bound substate; the 6 kT
                             #     gap is a modelling choice, not a measurement

    # SRX -> DRX recruitment (see rate_functions.xb_rate_50). Thick-filament
    # activation: the Hill exponent here contributes much of the model's calcium
    # sensitivity, independently of tropomyosin.
    # NOTE ON THIS WHOLE BLOCK (audit 2026-08-19): all five SRX parameters trace to
    # ONE table — Mijailovich 2021 JGP 153:e202012604 Table 1, "PS" block — and the
    # Reference column there reads "Assumed" for EVERY ONE of them (k−PS 200 s⁻¹,
    # k⁰PS 5 s⁻¹, kmax_PS 400 s⁻¹, b 5, [Ca²⁺]50 1 µM). Every other block in that
    # table carries real literature references; the PS block is the exception. So
    # nothing here is measured, and the subsystem that carries much of the model's
    # calcium sensitivity rests on another model's assumptions. Conditions there:
    # rat cardiac trabeculae, 27.2 °C (rat ventricle is predominantly α-MHC).
    'xb_srx_k0':   0.007,    # [G] ms⁻¹ basal rate at zero Ca. Set equal to xb_r05 by
                             #     construction — that is what forces 50% SRX at rest.
                             #     Mijailovich 2021 kPS0 = 5 s⁻¹ ("Assumed"); this is
                             #     7 s⁻¹, i.e. 1.4× the cited value, because the
                             #     construction takes precedence over the citation.
    'xb_srx_kmax': 0.4,      # [G] ms⁻¹ saturating rate; Mijailovich 2021 kmax_PS =
                             #     400 s⁻¹. RETAGGED [M] → [G] 2026-08-19: the number
                             #     is transcribed correctly but Table 1 marks it
                             #     "Assumed", so there is no measurement to be [M] about
                             #     — the same reasoning already applied to xb_srx_b below.
    'xb_srx_b':    5.0,      # [G] Hill exponent. Mijailovich 2021 Table 1 uses b = 5
                             #     and marks it "Assumed" — a shape parameter chosen
                             #     inside a model, not a measurement, hence [G].
                             #     Note this gate is a calcium PROXY for what is
                             #     believed to be a mechanosensitive process: Linari
                             #     2015 Nature 528:276 and Fusi 2016 Nat Commun 7:13281
                             #     show thick-filament activation tracks filament
                             #     STRESS and is independent of [Ca²⁺]. The proxy is
                             #     more defensible for cardiac, where stress-dependent
                             #     activation does require thin-filament activation
                             #     (Park-Holohan et al. 2021 PNAS 118:e2023706118,
                             #     PMC8072254 — citation resolved 2026-08-19), than for
                             #     skeletal. VERIFIED QUOTES: Fusi 2016 abstract — "both
                             #     the extent and kinetics of thick filament activation
                             #     depend on thick filament stress but are independent of
                             #     intracellular calcium concentration in the
                             #     physiological range" (rabbit psoas, skinned);
                             #     Park-Holohan 2021 abstract — folded motors are "not
                             #     directly switched on ... in the absence of thin
                             #     filament activation", but with thin-filament
                             #     activation stretch does activate them (rat trabeculae,
                             #     26 °C). So the Ca proxy is CONTRADICTED for skeletal
                             #     and SUPPORTED for cardiac.
                             #     It carries much of the model's calcium
                             #     sensitivity, so treat it as a fitting knob
    'xb_srx_ca50': 1e-6,     # [G] M — half-recruitment at pCa 6, as Mijailovich 2021
                             #     Table 1 ([Ca²⁺]50 = 1 µM, also marked "Assumed")

    # ==========================================================================
    # SIMULATION PARAMETERS
    # ==========================================================================
    'temp_celsius': 26.15,   # [G] °C. Sets kT in the Boltzmann and Bell terms, so
                             #     it scales every strain-dependent rate. Chosen to
                             #     match typical skinned-fibre experiments (~25 °C),
                             #     not derived; the two decimals are not meaningful.
                             #     NOTE: only kT depends on this — the rate constants
                             #     above are NOT temperature-corrected, so changing
                             #     it does not give a physically complete temperature
                             #     change (no Q10 on the pre-exponentials)
    'solver_tol':   0.3,     # [G] pN — mechanical equilibrium convergence target.
                             #     Numerical, not physical. Floored internally at
                             #     thick_k × 1e-4 (the float32 precision limit at
                             #     sarcomere-scale positions), so at stiff parameters
                             #     the floor, not this value, is what applies

    # ==========================================================================
    # DEFAULT DRIVER VALUES
    #
    # pCa, z_line and lattice_spacing live here so they can be swept like any
    # other parameter. When they need to vary WITHIN a simulation they are
    # overridden per timestep through Drivers (core/state.py); these values are
    # the fallback for any step that supplies no override.
    # ==========================================================================
    'pCa':             4.5,    # near-saturating activation (10^-4.5 M Ca²⁺)
    'z_line':          900.0,  # nm from M-line, i.e. sarcomere length 1.8 µm.
                               # Short for skeletal (typical working 1000–1300 nm);
                               # appropriate for cardiac. Set it explicitly
    'lattice_spacing': 14.0,   # [M] nm, thick-to-thin surface separation at
                               # typical vertebrate sarcomere lengths
}

# Field order for tree_flatten/unflatten (Python 3.7+ preserves dict order)
DYNAMIC_FIELDS = tuple(_DYNAMIC_DEFAULTS)


# =============================================================================
# STATIC PARAMS (Configuration - NOT a PyTree)
# =============================================================================

@dataclass(frozen=True)
class StaticParams:
    """Structural configuration. Frozen, and NOT a JAX PyTree.

    Everything here determines an array shape or a compile-time constant, so
    changing any field requires rebuilding the SarcTopology and recompiling the
    kernel. That is the whole reason these are separated from DynamicParams:
    physics values can be swept across a batch for free, structure cannot.

    Immutable — use .replace(**kwargs) to derive a modified copy.

    FILAMENT DIMENSIONS
        n_crowns: Crowns (rings of myosin heads) per thick filament. With the
            default 14.3 nm spacing, 52 crowns gives a ~0.79 µm half-filament,
            correct for vertebrate. Insect flight muscle needs ~100.
        n_polymers_per_thin: Actin pseudo-repeats per thin filament, setting its
            length. 15 repeats at 72 nm each gives ~1.08 µm.
        thick_bare_zone: Distance from the M-line to the first crown (nm). The
            bare zone has no heads — it is where the thick filaments of the two
            half-sarcomeres join, so nothing can bind there.
        thick_crown_spacing: Axial rest spacing between crowns (nm).

    ACTIN HELIX GEOMETRY
        These three define the actin double helix, which determines where
        binding sites sit both axially and azimuthally:
        actin_half_pitch: Long-pitch half-repeat (nm). The helix crossover
            distance; the full pseudo-repeat is twice this.
        mono_per_poly: Actin monomers per pseudo-repeat.
        polymer_base_turns: Helical turns per pseudo-repeat. Together with
            mono_per_poly this is the classic helix symmetry (vertebrate 13/6
            expressed here as 26 monomers per 12 turns; insect 28/13).
        target_zone_wiggle: Angular half-width (rad) of the acceptance window
            deciding which monomers count as binding sites for a given face.
            Because the helix is periodic, monomers fall into a few discrete
            azimuthal classes, so this window controls the site COUNT in
            discrete jumps rather than continuously. Widening it adds sites,
            which shortens the thin filament's spring segments and therefore
            changes its effective stiffness — see the thin_k note in
            _DYNAMIC_DEFAULTS before changing it.

    CROWN-FACE GEOMETRY
        n_xb_per_crown: Myosin heads per crown. Sets total_xbs, so it is a
            first-order cost driver. Vertebrate 3, insect flight muscle 4.
        crown_rotation_deg: Azimuthal rotation between successive crowns. With
            3 heads at 60°, successive crowns land exactly on hexagonal
            neighbours; non-multiples of 60° (e.g. the insect 33.75°) do not,
            so some heads end up pointing at no thin filament at all. Those are
            flagged False in SarcTopology.xb_valid.
        crown_face_wiggle_deg: Acceptance half-angle for matching a head's arm
            to one of the six hexagonal neighbour directions. A DIFFERENT
            question from target_zone_wiggle (which monomer, not which
            filament). Inert at n_xb_per_crown=3, where the match is exact.
        legacy_crown_geometry: Use the fixed pre-generalization face-assignment
            table instead of the continuous azimuth formula. Only valid with
            n_xb_per_crown=3.

    LATTICE ARRANGEMENT
        actin_geometry: "vertebrate" (1 thick : 2 thin, 3 faces per thin) or
            "invertebrate" (1:3, 2 faces). This is a real structural difference
            between vertebrate and insect flight muscle, not a modelling knob.
        n_superlattice_classes: 1 for a simple lattice, 3 for the Drosophila
            myosin superlattice, where thick filaments fall into three axial
            classes offset by thick_crown_spacing/3. Only {1, 3} are valid —
            other integers produce a geometrically wrong sublattice rather than
            merely an uncalibrated one.

    SOLVER
        n_newton_steps: Hard cap on Newton iterations. The loop exits early at
            convergence, so this is a safety bound, not a fixed cost.
        n_cg_steps: Conjugate-gradient iterations per Newton step. 0 degenerates
            to Richardson iteration, which converges only when no crossbridges
            are attached — do not use it as a default.
        solver_residual_tol: Post-run warning threshold (pN). Diagnostic only;
            it does not affect the solve. Calibrated just above the float32
            precision floor, which scales as ~thick_k × 2e-4, so raising
            filament stiffnesses may require raising this too.
        solver_max_iter: Legacy iteration bound retained for compatibility with
            configurations that set it; the Newton/CG caps above are what the
            current solver actually reads.

    XB TRANSITION-MATRIX BINNING
        n_xb_bins, xb_bin_lo, xb_bin_hi: Resolution and range of the
            axial-distance grid on which crossbridge rate matrices are
            evaluated. Heads then look up the bin they fall in rather than each
            computing its own matrix exponential. Finer bins mean more accurate
            strain dependence and more compute; the range should bracket the
            axial distances actually reachable at your sarcomere length.
    """
    n_crowns: int = 52
    n_polymers_per_thin: int = 15
    solver_max_iter: int = 50
    actin_geometry: str = "vertebrate"
    n_newton_steps: int = 4    # Hard cap on Newton while_loop iterations (exits early at convergence)
    n_cg_steps: int = 6        # CG steps per Newton iter; 0=Richardson (no JVP)
    solver_residual_tol: float = 1.5  # pN — post-run residual warning threshold
    # Calibrated to the float32 precision floor at the lit-consistent thick_k=7500.
    # Empirical floor scales as ~thick_k × 2e-4 pN; raising thick_k or thin_k from defaults
    # may push the floor above this tol and trigger warnings — adjust accordingly.
    n_xb_bins: int = 200       # bins per AP level; total expm = 2 × n_xb_bins per step
    xb_bin_lo: float = -8.0    # nm — lower edge of axial distance range (baked into SarcTopology)
    xb_bin_hi: float = 35.0    # nm — upper edge; measured range at z=1100 is [-5, 31]nm
    thick_bare_zone: float = 58.0    # nm — M-line to first crown rest distance
    thick_crown_spacing: float = 14.3  # nm — inter-crown rest spacing
    # Drosophila myosin superlattice: assigns each thick filament to one of
    # n_superlattice_classes via an axial-coordinate 3-coloring, each class
    # getting an axial offset of class * (thick_crown_spacing / n_superlattice_classes).
    # Only {1, 3} are valid (see __post_init__) — other integers give a
    # geometrically wrong sublattice under this formula, not merely uncalibrated.
    # Lethocerus: 1 (default, no superlattice). Drosophila: 3 (Squire 2006).
    n_superlattice_classes: int = 1
    # Thin-filament actin helix geometry, consumed by _calculate_binding_site_offsets.
    # Vertebrate defaults reproduce the prior hardcoded constants bit-for-bit.
    # (IFM values: actin_half_pitch=38.7, mono_per_poly=28, polymer_base_turns=13.)
    actin_half_pitch: float = 36.0     # nm — long-pitch half-repeat (polymer_base_length = 2×)
    mono_per_poly: int = 26            # monomers per polymer pseudo-repeat
    polymer_base_turns: float = 12.0   # turns per polymer pseudo-repeat
    # Angular half-width (rad) of the target-zone acceptance window. Default is the
    # exact float32 round-trip of the prior `rev/24`, so vertebrate stays byte-identical.
    target_zone_wiggle: float = float(np.float32(2 * np.pi) / np.float32(24))
    # Crown-face geometry (thick-filament crown -> thin-filament face assignment).
    # Consumed by SarcTopology's _compute_flat_index_maps_fixed_width. Vertebrate
    # defaults (n=3, rotation=60) give a byte-identical total_xbs to the prior
    # hardcoded-3 model; connectivity itself changes (see legacy_crown_geometry).
    # IFM values: n_xb_per_crown=4, crown_rotation_deg=33.75.
    n_xb_per_crown: int = 3           # cross-bridges per crown; affects total_xbs shape
    crown_rotation_deg: float = 60.0  # azimuthal step per crown (IFM: 33.75)
    # +/- acceptance half-angle for face matching (which of 6 hex neighbors a
    # crown's arm points at) — a different physical question from
    # target_zone_wiggle (which actin monomer is in a target zone). Inert for
    # n_xb_per_crown==3 (60-degree-multiples always match exactly regardless of
    # window), so its value doesn't affect any existing default behavior.
    crown_face_wiggle_deg: float = 15.0
    # True = exact pre-existing 2:1 face_pattern table (bit-identical to the
    # code prior to this field's introduction); False (default) = continuous
    # azimuth formula, uniform for vertebrate, partial-coverage for IFM. Only
    # valid combined with n_xb_per_crown==3 (the legacy table has no n!=3
    # analogue) — see __post_init__.
    legacy_crown_geometry: bool = False

    def __post_init__(self):
        if self.legacy_crown_geometry and self.n_xb_per_crown != 3:
            raise ValueError(
                "legacy_crown_geometry=True requires n_xb_per_crown == 3 "
                f"(the legacy face_pattern table has no n={self.n_xb_per_crown} analogue)"
            )
        if self.legacy_crown_geometry and (
            self.crown_rotation_deg != 60.0 or self.crown_face_wiggle_deg != 15.0
        ):
            warnings.warn(
                "legacy_crown_geometry=True ignores crown_rotation_deg/crown_face_wiggle_deg "
                "(the legacy face_pattern table is fixed); non-default values here are a silent no-op.",
                stacklevel=2,
            )
        if self.n_superlattice_classes not in (1, 3):
            raise ValueError(
                f"n_superlattice_classes must be 1 (no superlattice) or 3 (Drosophila "
                f"3-class axial superlattice), got {self.n_superlattice_classes}. Other "
                "integers are not merely uncalibrated — the class formula (q-r) mod n "
                "gives a geometrically wrong sublattice for any n outside {1, 3}."
            )

    def replace(self, **kwargs) -> 'StaticParams':
        """Create a new StaticParams with updated values.

        Example:
            static = StaticParams()
            modified = static.replace(actin_geometry='invertebrate')
        """
        return StaticParams(**{**asdict(self), **kwargs})

    def __repr__(self) -> str:
        return (f"StaticParams(n_crowns={self.n_crowns}, "
                f"n_polymers_per_thin={self.n_polymers_per_thin}, "
                f"actin_geometry='{self.actin_geometry}')")


@jax.tree_util.register_pytree_node_class
class DynamicParams:
    """All tunable physics values, as a JAX PyTree.

    Every field is a JAX array, which is what lets a whole parameter sweep run
    as one batched kernel: vmap maps over the leading axis of each field, so a
    thousand stiffness values cost one compilation, not a thousand.

    Values are ABSOLUTE, in the units listed in the module docstring. There are
    no hidden multipliers — a rate here is the rate the kernel uses.

    Defaults are fast-twitch skeletal at ~26 °C. See _DYNAMIC_DEFAULTS at module
    level for every value with its citation and confidence tier, and
    get_cardiac_params() / get_lethocerus_params() / get_drosophila_params() for
    the other presets.

    Field order is fixed by _DYNAMIC_DEFAULTS and is what tree_flatten and
    tree_unflatten rely on, so entries must never be reordered.

    Usage:
        static, dynamic = get_skeletal_params()
        dynamic = dynamic.copy(xb_r01_coeff=350.0)   # validates field names
        result = run(topo, dynamic_params=dynamic)
    """

    __slots__ = DYNAMIC_FIELDS

    def __init__(self, **kwargs):
        """Initialize DynamicParams with optional keyword arguments.

        All parameters are stored as JAX arrays for PyTree compatibility.
        Defaults are skeletal fast-twitch values (~26°C); see
        ``_DYNAMIC_DEFAULTS`` at module level for values + citations.
        Use get_cardiac_params() for cardiac-specific defaults.
        """
        for name, default in _DYNAMIC_DEFAULTS.items():
            object.__setattr__(self, name,
                jnp.asarray(kwargs.get(name, default)))

    def tree_flatten(self):
        """Flatten for JAX tree operations.

        Returns:
            children: Tuple of all dynamic field values (JAX arrays)
            aux_data: Empty tuple (no static data in DynamicParams)
        """
        children = tuple(getattr(self, name) for name in DYNAMIC_FIELDS)
        aux_data = ()  # No static fields in DynamicParams
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct DynamicParams from flattened representation."""
        # Bypass __init__ (which calls jnp.asarray) — children are already
        # JAX arrays/tracers. Older JAX versions call tree_unflatten with
        # object() sentinels during in_axes probing; __init__ would crash on those.
        new = object.__new__(cls)
        for name, value in zip(DYNAMIC_FIELDS, children):
            object.__setattr__(new, name, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for serialization/metadata."""
        d = {}
        # Dynamic fields - convert JAX arrays to Python floats
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name)
            d[name] = float(val) if hasattr(val, 'item') else val
        return d

    def copy(self, **updates) -> 'DynamicParams':
        """Create copy with updated values (JIT-compatible).

        Example:
            _, dynamic = get_skeletal_params()
            modified = dynamic.copy(xb_r01_coeff=400.0, tm_k_01=60000.0)
        """
        # Validate keys before constructing (Python-level, not traced)
        invalid = set(updates.keys()) - set(DYNAMIC_FIELDS)
        if invalid:
            raise ValueError(f"Unknown parameter: {sorted(invalid)}. Valid: {list(DYNAMIC_FIELDS)}")
        # Build kwargs from current values, overriding with updates
        # No float() conversion — preserves JAX tracers inside JIT
        kwargs = {name: updates.get(name, getattr(self, name)) for name in DYNAMIC_FIELDS}
        return DynamicParams(**kwargs)

    def with_drivers(self, pCa, z_line, lattice_spacing) -> 'DynamicParams':
        """Copy with only the three driver fields replaced — the scan-body fast path.

        Called once per timestep, so it is on the hottest path in the model.
        Unlike copy(), it skips the full kwargs rebuild and the jnp.asarray()
        conversions, and shares references for the other 46 fields rather than
        emitting an identity copy for each. Those copies are individually
        trivial but there is one per field per timestep, which XLA otherwise
        has to schedule.

        Args:
            pCa: Resolved calcium (from Drivers, or the constant default)
            z_line: Resolved z-line position (nm)
            lattice_spacing: Resolved lattice spacing (nm)

        Returns:
            New DynamicParams identical except for the three driver fields
        """
        # Bypass __init__ entirely — create empty instance and copy attrs
        new = object.__new__(DynamicParams)
        for name in DYNAMIC_FIELDS:
            if name == 'pCa':
                object.__setattr__(new, name, pCa)
            elif name == 'z_line':
                object.__setattr__(new, name, z_line)
            elif name == 'lattice_spacing':
                object.__setattr__(new, name, lattice_spacing)
            else:
                object.__setattr__(new, name, getattr(self, name))
        return new

    def __repr__(self) -> str:
        return f"DynamicParams(thick_k={float(self.thick_k):.1f}, thin_k={float(self.thin_k):.1f}, ...)"


def get_skeletal_params() -> Tuple[StaticParams, DynamicParams]:
    """Fast-twitch skeletal muscle, ~26 °C. The model's baseline.

    Vertebrate lattice geometry (1 thick : 2 thin, 3 heads per crown) with
    fast-twitch skeletal myosin II kinetics and the compliant N2A titin isoform.
    This preset is simply the unmodified defaults; every other preset is
    expressed as overrides on it.

    Rabbit psoas is the implicit reference animal, since that is what most of the
    single-molecule and fibre mechanics literature behind these numbers used.

    Every value, with its citation and confidence tier, is in _DYNAMIC_DEFAULTS
    at module level — that dict is the authority, not this docstring. In brief,
    the crossbridge spring constants come from the Daniel-group two-spring
    lineage (Chase 2004; Tanner 2007; Williams 2010); the weak-state softening is
    unsourced and measurably too stiff — see the xb_c_k_weak/xb_g_k_weak entry.
    Filament stiffnesses from Brunello 2014
    and Mijailovich 2021; free energies referenced to Howard 2001 Table 14.2. Two of the
    kinetic coefficients (xb_r01_coeff, xb_r40) are inherited placeholders with
    no literature source at all — see their entries before trusting them.

    OPERATING POINT: pass z_line explicitly. The default of 900 nm is short for
    skeletal muscle; the physiological working range is 1000-1300 nm
    (sarcomere length 2.0-2.6 µm), and force-length behaviour is strongly
    dependent on it.

        static, dynamic = get_skeletal_params()
        result = run(topo, pCa=4.5, z_line=1100.0)   # SL ~2.2 µm

    Returns:
        (StaticParams, DynamicParams)

    Example:
        static, dynamic = get_skeletal_params()
        static = static.replace(n_crowns=60)
        dynamic = dynamic.copy(thick_k=9000.0)
    """
    return StaticParams(), DynamicParams()


def get_cardiac_params() -> Tuple[StaticParams, DynamicParams]:
    """Generic cardiac muscle, ~27 °C.

    Same vertebrate lattice geometry as skeletal — the structural difference
    between the two is small — but different kinetics throughout. Cardiac
    myosin (beta-MHC) cycles several times more slowly than fast skeletal
    myosin, cardiac troponin C releases calcium faster, and cardiac titin is
    the shorter, stiffer N2B isoform.

    THE DEFINING FEATURE of this preset is not any single rate but the SRX
    balance. xb_r05 is ~30x the skeletal value, so at rest the great majority of
    cardiac heads are parked in the super-relaxed state rather than the ~50%
    parked in skeletal. Calcium then recruits them. That reserve is where much of
    cardiac contractile reserve and its steep calcium sensitivity come from, and
    it is why cardiac force-pCa curves can be steep without needing a large
    tropomyosin cooperativity term. Steepness here is meant to come from the SRX
    gate, not from the tropomyosin coupling.

    OPERATING POINT: cardiac sarcomeres work short, 900-1100 nm z-line
    (SL 1.8-2.2 µm):

        static, dynamic = get_cardiac_params()
        result = run(topo, pCa=4.5, z_line=950.0)   # SL ~1.9 µm

    UNVERIFIED: the shared default tm_J_M = 2.70 was calibrated against a cardiac
    force-pCa target, but that calibration is not confirmed — see the tm_J_M
    entry in _DYNAMIC_DEFAULTS. Do not quote a Hill coefficient from this preset
    without re-fitting.

    Overrides applied to the skeletal baseline:
        tm_k_12  = 0.5 ms⁻¹      — [G] unsourced; the Lehrer & Morris 1982 attribution does not hold (see dict)
        tm_k_01  = 80000 M⁻¹ms⁻¹ — [L] = 8e7. Robertson 1981 Table I gives 3.9e7 for the
                                   cardiac Ca-specific site, so this is 2.05× that; the
                                   "4–8×10⁷" range was never published. Better source:
                                   Pinto 2011 Table 1, human cTn k_on = 0.94e8 M⁻¹s⁻¹ at 21 °C.
        tm_Keq_01    = 750000 M⁻¹    — [G] Kd 1.33 µM. Pinto 2011 JBC 286:1005 Table 1 (WT)
                                   gives pCa50 6.69 for cTn (Kd 0.20 µM) and 6.13 in the
                                   thin filament (Kd 0.74 µM). NO condition yields 1–2 µM;
                                   that range is withdrawn (audit 2026-08-19).
        tm_k_30  = 0.04 ms⁻¹     — [L] Cardiac Ca²⁺ off-rate. Davis 2007 Biophys J 92:3195
                                   Table 1 VERIFIES 42.5 ± 0.3 s⁻¹ — but that is the
                                   ISOLATED cTn row. In a reconstituted thin filament it is
                                   105 s⁻¹, and with S1+ATP 110 s⁻¹ (Pinto: 95.8 / 139.1).
                                   State 3 is an assembled, myosin-accessible site, so the
                                   model is ~2.6× slow for the preparation it represents.
        xb_r12_coeff = 0.175 ms⁻¹ — [L] Kawai et al. 1993 Circ Res 73:35 (ferret
                                     myocardium, V3/beta2 isoform, pCa 4.82, 200 mM
                                     ionic strength, 20 C — SAME conditions as the
                                     skeletal companion, so the preset ratio is not a
                                     temperature artifact). Read as page images
                                     2026-08-19; the PDF has no text layer.
                                     "2πb ~ 5–15 s⁻¹" VERIFIES: f_min "around 1 to 2 Hz"
                                     and "f_min approximates the characteristic
                                     frequency b" (p.39) gives 2πb = 6.3–12.6 s⁻¹.
                                     BUT "3–4× slower" does NOT match the paper, which
                                     says the rate constants are "generally an order of
                                     magnitude smaller in myocardium than in psoas";
                                     its k2 = 48 s⁻¹ vs the skeletal companion's
                                     510 ± 30 s⁻¹ is 10.6×. This preset uses 3.4×
                                     (175/600). Either the skeletal 20–60 s⁻¹ range is
                                     too low (it was never located in its source) or
                                     this value is ~3× too fast. NOT resolved here.
        xb_r23_coeff = 0.065 ms⁻¹ — [L-unverified] "lever arm rate ~2× slower".
                                     Deacon 2012 (recombinant human α-/β-S1, 20 °C) does
                                     report ~2× α/β differences and says the isoforms
                                     "differ by no more than two-fold for ANY" parameter
                                     — but a specifically LEVER-ARM rate 2× slower in β
                                     was not located, and the paper states the working
                                     stroke MAGNITUDE does not change significantly.
                                     Also: the PDF on disk is the erratum reprint at
                                     69:4239–4255, not the cited 69:2261.
        xb_r34_coeff = 0.065 ms⁻¹ — [L] Cardiac ADP release. VERIFIED EXACTLY:
                                   Siemankowski & White 1984 JBC 259:5045 — "the rate
                                   constant for the dissociation of ADP from cardiac
                                   actomyosin-S1, k_AD, is ~65 s⁻¹ at 15 °C" (bovine
                                   ventricle = β-MHC, so isoform AND temperature match
                                   the human-β 15 °C targets). Corroborated by Sung 2015
                                   stopped-flow on human β-cardiac: 72 ± 5 s⁻¹.
                                   STRONGLY temperature dependent: the same paper
                                   extrapolates ~550 s⁻¹ at 38 °C, i.e. Q10 ≈ 2.4.
        xb_r05   = 0.2 ms⁻¹     — DRX→SRX (k−PS in Mijailovich 2021 PMC7852458 Table 1).
                                   200 s⁻¹ matches Mijailovich cardiac canonical model;
                                   with kPSmax=400, k0=5, Hill b=5, gives 97.5% SRX at
                                   rest (Ca→0) and ~33% SRX at pCa 4.5 — close to Linari
                                   2015 saturating myosin recruitment data.
        xb_srx_k0 = 0.005 ms⁻¹   — [G] = 5 s⁻¹ = Mijailovich 2021 kPS0 exactly, which
                                   that paper marks "Assumed". Not tuned here — see dict.
        titin_a  = 55.0 pN        — [I] N2B isoform, stiffer than skeletal N2A;
                                   Granzier & Labeit 2004 Circ Res 94:284
        titin_b  = 0.008 nm⁻¹    — [I] Powers 2018 cardiac-like stiffness (8 µm⁻¹)
        titin_rest = 140.0 nm     — [I] slack at SL 1.85 µm (z_line=925 → L≈138 nm);
                                   Linke 1998 PNAS 95:8052

    Returns:
        (StaticParams, DynamicParams)

    Example:
        static, dynamic = get_cardiac_params()
        dynamic = dynamic.copy(tm_Keq_01=1e6)
    """
    cardiac_overrides = {
        'tm_k_12': 0.5,           # [G] ms⁻¹ blocking→closed, half the skeletal value.
                                  #     NB the SKELETAL value it is half of is itself
                                  #     unsourced (audit 2026-08-19), so this is a fraction
                                  #     of a guess. Lehrer & Morris 1982 re-checked: the PDF
                                  #     contains the word "cardiac" ZERO times.
                                  #   No cardiac-specific measurement of this transition
                                  #   has been located; the halving encodes the general
                                  #   expectation that cardiac thin-filament transitions
                                  #   are slower. Geeves & Lehrer 1994 Biophys J 67:273
                                  #   bound this class of transition at 20–1000 s⁻¹, wide
                                  #   enough that any value inside it is a choice
        'tm_k_01': 80000.0,       # [L] Robertson 1981 Table I cardiac = 3.9e7 (this is 2.05×);
                                  #     "4–8×10⁷" not published. Pinto 2011 human cTn
                                  #     k_on = 0.94e8 M⁻¹s⁻¹ (21 °C) supports 8e7 far better.
        'tm_Keq_01': 750000.0,        # [G] Kd 1.33 µM. Pinto 2011 JBC 286:1005 Table 1 gives
                                  #     Kd 0.20 µM (cTn) / 0.74 µM (thin filament); the
                                  #     "1–2 µM" attribution is withdrawn.
        'tm_k_30': 0.04,          # [L] 40 s⁻¹. Davis 2007 Biophys J 92:3195 Table 1 = 42.5 s⁻¹
                                  #     for ISOLATED cTn (15 °C). In-filament is 105 s⁻¹,
                                  #     +S1+ATP 110 s⁻¹ — the row that matches state 3.
        'xb_r12_coeff': 0.175,    # Process B 3–4× slower; Kawai et al. 1993 Circ Res 73:35
        'xb_r23_coeff': 0.065,    # Lever arm ~2× slower; Deacon et al. 2012 Cell Mol Life Sci 69:2261
        'xb_r34_coeff': 0.065,    # ADP release ~65 s⁻¹; Siemankowski & White 1984 JBC
        'xb_r05': 0.2,            # [G] DRX→SRX; Mijailovich 2021 (PMC7852458 Table 1)
                                  #     k−PS = 200 s⁻¹ — correctly transcribed, but that
                                  #     table marks it "Assumed" (audit 2026-08-19).
        'xb_srx_k0': 0.005,       # [G] = 5 s⁻¹, which is EXACTLY Mijailovich's kPS0
                                  #     ("Assumed"). It is not "empirically calibrated"
                                  #     and not independently tuned — it is the source
                                  #     value adopted unchanged. (The get_cardiac_params
                                  #     docstring previously said "[G] no source; tuned";
                                  #     both descriptions were wrong. Fixed 2026-08-19.)
        'titin_a': 55.0,          # N2B stiffer than N2A; Granzier & Labeit 2004 Circ Res 94:284
        'titin_b': 0.008,         # Powers 2018 cardiac-like stiffness (8 µm⁻¹)
        'titin_rest': 140.0,      # slack at SL 1.85 µm (z_line=925 nm → L≈138 nm)
    }
    return StaticParams(), DynamicParams(**cardiac_overrides)


def get_lethocerus_params() -> Tuple[StaticParams, DynamicParams]:
    """Lethocerus (giant water bug) indirect flight muscle.

    Indirect flight muscle (IFM) is the power source for insect flight, and it is
    built very differently from vertebrate striated muscle. It is ASYNCHRONOUS:
    it contracts many times per nerve impulse, driven not by calcium transients
    but by stretch activation — the muscle is activated by being stretched, which
    lets antagonistic muscle pairs oscillate the thorax at wingbeat frequency far
    above what neural firing could sustain. Calcium acts as a slow enable signal
    rather than a per-contraction trigger.

    Structurally the differences that this preset captures are:
      - a 1:3 thick:thin filament ratio in a hexagonal lattice with thin
        filaments on the edges rather than the interstices (vertebrate is 1:2),
        so each thin filament faces 2 thick filaments rather than 3
      - fourfold rather than threefold crown symmetry, with a 33.75 deg
        azimuthal rotation per crown that does NOT align with the hexagonal
        neighbour directions — so many heads point at no thin filament at all
        and are marked invalid in SarcTopology.xb_valid
      - filaments roughly twice as long as vertebrate ones
      - a much shorter I-band, and a stiffer, shorter connecting filament
        (kettin and projectin, classically the "C-filaments") in place of titin

    Lethocerus thick filaments are NOT arranged in a myosin superlattice; for the
    Drosophila variant that is, see get_drosophila_params().

    IMPORTANT — WHAT THIS PRESET IS AND IS NOT. The GEOMETRY is sourced from the
    structural literature. The KINETICS are not: every xb_* and tm_* rate is
    inherited unchanged from the skeletal defaults, and no attempt has been made
    to fit them to insect myosin. In particular nothing here implements stretch
    activation itself. This preset gives you insect flight muscle GEOMETRY with
    vertebrate skeletal chemistry, which is useful for isolating geometric
    effects and is not a model of a working flight muscle.

    Confidence tiers are as defined in the module docstring: [M] measured,
    [I] inferred, [G] guess. Every value below is a literature-informed starting
    point rather than a fitted one.

    StaticParams overrides:
        actin_geometry       = 'invertebrate'  — 1:3 thick:thin, 2 faces/thin.
            [M] Reedy 1968 J Mol Biol 31:155 (Lethocerus unit cell).
        thick_crown_spacing  = 14.5 nm         — [M] IFM 14.5 nm meridional repeat;
            Reedy 1968; 145 Å crown spacing confirmed by Hu et al. 2016 Sci Adv.
        actin_half_pitch     = 38.7 nm         — [M] IFM long-pitch half-repeat;
            Reedy 1968; Squire et al. 2006 J Mol Biol 361:823.
        mono_per_poly        = 28              — [M] IFM 28/13 actin helix (subunit
        polymer_base_turns   = 13                   axial repeat 2.76nm); Squire 2006
            p.823 (77.4/28 = 2.764nm confirms the pairing).
        n_xb_per_crown       = 4               — [M] IFM 4-fold (32/3) crown symmetry;
            Hu et al. 2016 Sci Adv 2:e1600058 (cryo-EM 6 A; fourfold symmetry imposed).
        crown_rotation_deg   = 33.75            — [I] azimuthal step = 3x360/32 from the
            32/3 helix; matches Hu et al. 2016's directly
            MEASURED 33.98 deg/crown.
        target_zone_wiggle   = 26 deg          — [I] chosen so the model reproduces
            a measured binding COUNT rather than a measured angle; see below.
        n_polymers_per_thin  = 20              — [I] ~1.55 um thin filament
            (20 x 77.4 nm repeat); see filament lengths below.
        n_crowns             = 100             — [I] ~1.5 um half thick filament;
            see filament lengths below. Coupled to n_polymers_per_thin and to the
            z_line you run at — this is what gates the overlap zone.
        thick_bare_zone: NOT overridden — kept at the 58 nm vertebrate value. [G]
            No insect-specific M-line-to-first-crown distance was located, and
            there is no principled basis to guess one, so it is left as-is and
            flagged rather than invented.
        n_superlattice_classes: NOT overridden. Lethocerus has no myosin
            superlattice, so the default of 1 is correct, not a placeholder.
        crown_face_wiggle_deg: [G] left at the vertebrate default; no insect
            calibration exists. Note this is a different acceptance window from
            target_zone_wiggle — see the StaticParams field documentation.

    HOW target_zone_wiggle = 26 deg WAS SET. Reedy et al. (Biophys J 1998, 2004)
    report that ~98% of insect flight muscle crossbridge attachments fall on
    exactly 2 actin monomers per 38.7 nm long-pitch repeat, midway between
    successive troponin complexes. That is a count, not an angle, so the angle was
    solved for: sweeping the window and counting the sites the model actually
    generates gives exactly 1.0 monomer per face per repeat at the vertebrate
    15 deg (too few), and a flat plateau at exactly 2.0 for any window between
    22 and 30 deg. 26 deg is the middle of that plateau. The count is a solid
    structural match; the angle itself is inferred and carries no independent
    measurement.

    Because the same window also sets how many mechanical nodes the thin filament
    has, widening it to 26 deg makes the insect thin filament ~19% more compliant
    per unit length than the vertebrate one at equal thin_k. See the thin_k entry
    in _DYNAMIC_DEFAULTS — a binding-geometry decision has a mechanical side
    effect here, and comparisons across presets should account for it.

    FILAMENT LENGTHS. n_crowns, n_polymers_per_thin and the z_line you choose are
    one coupled setting, not three independent knobs. Insect flight muscle
    filaments are roughly twice the vertebrate length, and it is the THICK
    filament that matters most: the thick-thin overlap zone — which determines
    both the recruitable crossbridge pool and the number of physically meaningful
    tropomyosin sites — is bounded by the thick filament and z_line. At any
    reasonable z_line the thin filament over-covers the thick, so thin length
    alone changes nothing in the overlap.

    n_crowns = 100 gives a full thick filament of
        2 x (thick_bare_zone + (n_crowns-1) x thick_crown_spacing)
        = 2 x (58 + 99 x 14.5) = ~2.99 um
      - [M] Adult Drosophila IFM thick filament measures 3.04 +/- 0.05 um;
        Contompasis, Nyland, Maughan & Vigoreaux 2010 J Mol Biol 395:340
        (Contompasis, Nyland, Maughan & Vigoreaux 2010 J Mol Biol 395:340-348,
        PMID 19917296 -- bare-PMID citation resolved 2026-08-19). It grows from
        ~1.6 um in the early pupa in regulated
        unison with the thin filament.
      - [I] n_crowns = (3040/2 - 58)/14.5 + 1 ~= 101, rounded to 100. Finer
        precision is unwarranted while thick_bare_zone is itself a guess.

    n_polymers_per_thin = 20 gives a thin filament of
        20 x 2 x actin_half_pitch = 20 x 77.4 nm = 1.55 um
      - [M] Adult Drosophila IFM thin filament ~1.5 um, elongating from ~0.8 um
        during pupal development; Mardahl-Dumesnil & Fowler 2001 J Cell Biol
        155:1043.
      - [I] 1.5 um / 77.4 nm per repeat = 19.4, rounded to 20.

    CHOOSING z_line. The anatomical operating point is ~1500-1600 nm, i.e. a
    ~3.0-3.2 um sarcomere. At these filament lengths the thick filament tip sits
    at ~1494 nm, so around z_line = 1544 the connecting filament sits exactly at
    its 50 nm rest length. Running much beyond that extends an exponential spring
    fast; running far below it leaves the connecting filament slack. This is worth
    stating because the coupling is easy to get wrong: with vertebrate-length
    filaments the same z_line would imply a ~700 nm extension and an absurd
    passive force.

    CAVEATS. (1) The filament lengths are Drosophila measurements — no
    Lethocerus-specific values were located. Lethocerus has a shorter sarcomere
    (~2.67 um vs ~3.2 um), so both counts may be somewhat lower for it; treat
    n_crowns ~[95,105] and n_polymers_per_thin ~[19,22] as the honest range. [G]
    (2) thick_bare_zone remains an unsourced vertebrate carry-over. [G]
    (3) n_crowns = 100 roughly doubles total_xbs relative to vertebrate, so
    memory and time per simulation roughly double. Size batches accordingly.

    DynamicParams overrides — ONLY the connecting filament:
        titin_a    = 40.0 pN
        titin_b    = 0.025 nm^-1
        titin_rest = 50.0 nm

    Insect flight muscle has no titin. Its elastic connecting filaments are
    kettin and projectin, and they are short and stiff where vertebrate titin is
    long and compliant — the insect I-band is only ~50 nm, so there is very
    little slack to take up. That short, stiff connection is thought to be part
    of what makes the muscle stretch-activatable at all: strain goes into the
    lattice rather than being absorbed elastically. The model reuses its titin
    spring to represent them, which is a functional substitution, not a claim
    that they are the same protein.

    NO PUBLISHED WORK FITS KETTIN OR PROJECTIN TO A SINGLE-EXPONENTIAL SPRING,
    so all three values are estimates for an underdetermined model:
      - titin_rest = 50.0 nm. [M] Best supported of the three. Kettin's
        N-terminus is embedded in the Z-disc while its C-terminus reaches the end
        of the A-band, because the insect I-band is only ~50 nm long;
        van Straaten et al. 1999 J Mol Biol 285:1549. Cross-checks to the same
        order of magnitude (~40-46 nm) against the passive-tension elastic limit
        in Granzier & Wang 1993 J Gen Physiol 101:235.
      - titin_b = 0.025 nm^-1, plausible range 0.02-0.03. [I] Derived, not
        measured, by two independent routes that happen to agree: proportional
        scaling from the cardiac titin_b/titin_rest ratio, and an elastic-limit
        constraint from Granzier & Wang 1993. Qualitatively consistent with Kulke
        et al. 2001 J Cell Biol 154:1045 reporting Drosophila IFM myofibril
        stiffness about an order of magnitude above rabbit cardiac — though that
        measurement conflates stiffness, scale factor and geometry, so it cannot
        isolate this parameter.
      - titin_a = 40.0 pN, plausible range 20-50 pN. [G] The weakest of the
        three: a multi-step estimate from Linari et al. 2004 Biophys J 87:1101
        passive-force data, combined with an ASSUMED ~50 nm lattice spacing to
        convert stress into per-filament force. The derivation does not clearly
        demand a different order of magnitude from the vertebrate values
        (43 skeletal, 55 cardiac), so it is kept near them rather than trusted
        to better than a factor of two.

    ALSO UNVERIFIED FOR THIS PRESET: the number of connecting filaments per thick
    filament is fixed at 6 in kernels/forces.py, matching the vertebrate sixfold
    arrangement. No source giving that count for a 1:3 insect lattice was located,
    so total passive force may be scaled wrongly here.

    A NOTE ON TERMINOLOGY: kettin and projectin are classically called the
    "C-filaments". That name refers to this axial elastic structure — it is not a
    separate radial element, and it does not imply a distinct lattice-spacing
    constraint. If you want dynamic lattice spacing for an insect run, pass
    K_lat/nu to run() explicitly, knowing that no insect-specific value for
    either has been sourced.

    THIN FILAMENT REGISTRATION IS AUTOMATIC. In the invertebrate lattice each
    thin filament sits at a hexagonal edge midpoint, and the direction of that
    edge determines its threefold actin registration class (Squire et al. 2006
    J Mol Biol 361:823). SarcTopology.create() derives each filament's angular
    and axial phase from its class, so the real lattice's three systematic phases
    appear without any manual setup — do not pass thin_starts trying to emulate
    them. The only remaining free choice is the absolute phase baseline, which
    defaults to 0 (a crystalline lattice, with the class supplying all systematic
    variation); pass thin_starts only to override that baseline.

    thick_starts = [1] * n_thick is the natural "no myosin superlattice" choice
    and matches n_superlattice_classes = 1:

        static, dynamic = get_lethocerus_params()
        topo = SarcTopology.create(
            nrows=4, ncols=3, static_params=static, dynamic_params=dynamic,
            thick_starts=[1] * n_thick,   # no thick superlattice; thin 3-fold is automatic
        )

    Returns:
        Tuple of (static_params, dynamic_params)
    """
    # Tier tags: [M] measured, [I] inferred, [G] guess — see docstring for full
    # sourcing/confidence. Every value is a literature-informed starting point.
    static_overrides = {
        'actin_geometry': 'invertebrate',           # [M] Reedy 1968 JMB 31:155 (Lethocerus IFM)
        'thick_crown_spacing': 14.5,                 # [M] nm; Hu 2016 SciAdv (145 A). NOT Reedy:
                                                     #     "145" appears ZERO times in Reedy 1968
        'actin_half_pitch': 38.7,                    # [M] nm; Squire 2006 JMB 361:823. NOT Reedy:
                                                     #     Reedy's repeat is 380 A = 38.0 nm throughout
        'mono_per_poly': 28,                         # [M] 28/13 actin helix; Squire 2006 p.823
        'polymer_base_turns': 13,                    # [M] "  (rise 77.4/28 = 2.76 nm confirms)
        'n_xb_per_crown': 4,                         # [M] 4-fold crown; Hu 2016 SciAdv
        'crown_rotation_deg': 33.75,                 # [I] 3x360/32 from 32/3 helix; Hu 2016 meas. 33.98
        'target_zone_wiggle': float(np.radians(26.0)),  # [I] set to Reedy 1998/2004 2-monomer target
        'n_crowns': 100,                             # [I] 3.04um Droso thick filament (Contompasis 2010) /2 /14.5; [G] for Leth
        'n_polymers_per_thin': 20,                   # [I] 1.5 um Droso thin filament / 77.4 nm; [G] for Leth
    }
    dynamic_overrides = {
        'titin_a': 40.0,      # [G] pN; weakest — back-of-envelope, Linari 2004; see docstring
        'titin_b': 0.025,     # [I] nm^-1; derived (2 converging chains), Kulke 2001; see docstring
        'titin_rest': 50.0,   # [M] nm; IFM I-band length, van Straaten 1999; see docstring
    }
    return StaticParams(**static_overrides), DynamicParams(**dynamic_overrides)


def get_drosophila_params() -> Tuple[StaticParams, DynamicParams]:
    """Drosophila melanogaster indirect flight muscle.

    The same insect flight muscle geometry as get_lethocerus_params(), plus a
    MYOSIN SUPERLATTICE. Drosophila thick filaments are not all in axial
    register: they fall into three classes, each shifted along the filament axis
    by one third of the crown spacing (Squire et al. 2006 J Mol Biol 361:823).
    The lattice therefore repeats over a larger unit cell than the simple
    hexagonal one, hence "super"-lattice.

    Why it might matter: with every filament in register, all crossbridges in a
    lattice plane meet actin at the same phase, so binding opportunities are
    strongly correlated across filaments. Breaking that register decorrelates
    them, which changes how binding, and therefore cooperativity, propagates
    through the lattice. Whether it changes bulk force appreciably is a genuinely
    open question; this preset exists so it can be tested rather than assumed.

    In the model it acts through per-filament crown_offsets: see
    StaticParams.n_superlattice_classes and SarcTopology.create(). The actin
    threefold registration of the THIN filaments is a separate mechanism that is
    always active in invertebrate geometry. Both descend from the same hexagonal
    unit cell but apply to different filament types, and they can be varied
    independently.

    StaticParams overrides — identical to get_lethocerus_params() except:
        n_superlattice_classes = 3   [M] Squire et al. 2006

    The filament lengths shared with get_lethocerus_params() (n_crowns = 100,
    n_polymers_per_thin = 20) are in fact Drosophila measurements, so they are
    better supported here than they are there. See that function's docstring for
    the sourcing and for how they couple to z_line.

    DynamicParams overrides — identical to get_lethocerus_params()
    (titin_a = 40.0, titin_b = 0.025, titin_rest = 50.0). No literature
    quantitatively distinguishes Drosophila from Lethocerus connecting-filament
    mechanics. Kettin isoform sizes do differ (~500 kDa in Drosophila vs ~700 kDa
    in Lethocerus), but no source translates that into resting stiffness or
    length, so both presets share one estimate. As with Lethocerus, all
    crossbridge and tropomyosin kinetics are unmodified skeletal defaults.

    LATTICE SHAPE PRECONDITION: the threefold colouring only closes properly
    under periodic boundaries if nrows is EVEN and ncols is a MULTIPLE OF 3;
    otherwise filaments of the same class end up adjacent across the periodic
    seam, and SarcTopology.create() raises rather than building it. Known-good
    shapes: 2x3, 4x3, 4x6, 6x6, 6x9, 8x3. Note that 4x4 — the usual go-to size
    elsewhere — does NOT satisfy this and will raise.

        static, dynamic = get_drosophila_params()
        topo = SarcTopology.create(
            nrows=4, ncols=3, static_params=static, dynamic_params=dynamic,
        )   # thin 3-fold registration + thick superlattice both automatic

    Returns:
        Tuple of (static_params, dynamic_params)
    """
    # Tier tags [M]/[I]/[G] and full sourcing: see get_lethocerus_params() docstring
    # (geometry is shared; only n_superlattice_classes differs).
    static_overrides = {
        'actin_geometry': 'invertebrate',           # [M] Reedy 1968 JMB 31:155 (Lethocerus IFM)
        'thick_crown_spacing': 14.5,                 # [M] nm; Hu 2016 SciAdv (145 A). NOT Reedy:
                                                     #     "145" appears ZERO times in Reedy 1968
        'actin_half_pitch': 38.7,                    # [M] nm; Squire 2006 JMB 361:823. NOT Reedy:
                                                     #     Reedy's repeat is 380 A = 38.0 nm throughout
        'mono_per_poly': 28,                         # [M] 28/13 actin helix; Squire 2006 p.823
        'polymer_base_turns': 13,                    # [M] "  (rise 77.4/28 = 2.76 nm confirms)
        'n_xb_per_crown': 4,                         # [M] 4-fold crown; Hu 2016 SciAdv
        'crown_rotation_deg': 33.75,                 # [I] 3x360/32 from 32/3 helix; Hu 2016 meas. 33.98
        'target_zone_wiggle': float(np.radians(26.0)),  # [I] set to Reedy 1998/2004 2-monomer target
        'n_crowns': 100,                             # [I] 3.04um Droso thick filament (Contompasis 2010) /2 /14.5
        'n_polymers_per_thin': 20,                   # [I] 1.5 um Droso thin filament / 77.4 nm
        'n_superlattice_classes': 3,                 # [M] 3-class axial superlattice; Squire 2006
    }
    dynamic_overrides = {
        'titin_a': 40.0,      # [G] pN; see get_lethocerus_params() docstring
        'titin_b': 0.025,     # [I] nm^-1; see get_lethocerus_params() docstring
        'titin_rest': 50.0,   # [M] nm; see get_lethocerus_params() docstring
    }
    return StaticParams(**static_overrides), DynamicParams(**dynamic_overrides)


# Alias for tiered architecture
Constants = DynamicParams
