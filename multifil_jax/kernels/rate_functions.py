"""
Transition rates for the tropomyosin and crossbridge state machines.

Every rate in the model is defined here, one function per transition. The
kernels assemble these into rate matrices; this module is where the biophysics
of "how fast does this step happen, and what does it depend on" lives.

Rates are in ms^-1 (a value of 0.6 means 600 s^-1). Energies are in kT.

THE TROPOMYOSIN UNIT (6 states: position x calcium)
---------------------------------------------------
Tropomyosin lies in the groove of the actin helix, covering the sites myosin
needs. A unit is in one of three POSITIONS (McKillop & Geeves 1993), and its
troponin either has calcium bound or not:

    state = 2*position + has_calcium      position = state >> 1
    0 blocked        1 blocked + Ca       nothing binds
    2 closed         3 closed + Ca        weak attachment only
    4 open           5 open + Ca          the weak -> strong step too

Two kinds of edge, never mixed. POSITION edges (blocked <-> closed <-> open,
no blocked <-> open) at each calcium level, with that level's equilibrium; the
Ising field and the crossbridge lock act here only. CALCIUM edges, one per
position. Calcium does not drive the switch, it biases it: every position
equilibrium is Ca-conditional, so detailed balance FORCES the calcium affinity
at closed and open (tm_ca_affinity_ratios) — calcium binds more tightly the
further the unit is open, which is the measured linkage (strong heads slow Ca
release). A unit with strong heads is biased towards open by (1 + xb_tm_K2)^n
on its exits from open, and a unit holding any head cannot enter blocked (see
transitions._compute_unique_tm_Q_matrices); the functions below return the
unmodified rates.

THE CROSSBRIDGE CYCLE (6 states)
--------------------------------
    0  DRX      disordered relaxed; head is free, ATP hydrolysed, ready to bind
    1  Loose    weakly bound to actin, pre-power-stroke
    2  Tight_1  strongly bound, post-stroke, phosphate released
    3  Tight_2  strongly bound, ADP-release-competent
    4  Free_2   just detached after ADP release; ATP has bound
    5  SRX      super-relaxed; head folded back against the thick filament

The force-producing path is 0 -> 1 -> 2 -> 3 -> 4 -> 0, consuming one ATP per
lap. Each bound state has its own spring configuration — 1 the *_weak set, 2
the *_tight_1 set, 3 the *_strong set; 0, 4 and 5 are detached and carry no
spring at all.

WHERE THE WORKING STROKE ACTUALLY IS
------------------------------------
Mostly on 1 -> 2, with the remainder on 2 -> 3. NOT on 2 -> 3 alone, despite
the Tight_1/Tight_2 naming.

When the *_tight_1 springs equal the *_strong ones, states 2 and 3 are
elastically identical, the whole rest-position change happens on 1 -> 2, and
2 -> 3 is a purely chemical step. At the shipped defaults the *_tight_1 rest
positions differ (the split stroke rows in core/param_table.py), so 2 -> 3
carries part of the lever swing. The axial rest projection g_rest*cos(c_rest)
for Loose / Tight_1 / Tight_2 is 13.55 / 6.60 / 4.76 nm skeletal and insect
(6.95 nm on 1 -> 2, 1.84 nm on 2 -> 3) and 13.55 / 8.49 / 7.14 nm cardiac
(5.07 and 1.34 nm); 2 -> 3 is 21% of the total in every class.

The 2 -> 3 swing stands for the second, smaller lever swing that accompanies
ADP release: ~16 degrees / ~1.5 nm in cardiac myosin (Doran 2023 by cryo-EM;
Woody 2019 measures 1-1.5 nm by single molecule), essentially absent in fast
skeletal myosin (Gollub, Cremo & Cooke 1996).

What separates the two states in every configuration is the ~6 kT free energy
drop that makes the stroke effectively one-way, and the fact that only state 3
can release ADP and detach.

Two force-bearing configurations is a modelling choice. Knupp & Squire 2020 fit
length-step transients, isotonic shortening and the M3 reflection with one
force-bearing plus one low-force attached state, and Eakins 2016 concludes from
X-ray that "no more than two main attached structural states are necessary and
sufficient". Offer & Ranatunga 2013 instead require TWO tension-generating
steps, of 5.6 and 4.6 nm. This model sits between them: two force-bearing
states with a lopsided split.

SRX is not part of that cycle — it is a reserve. Heads parked in SRX have very
low ATPase and cannot bind at all; calcium recruits them out (xb_rate_50).
This gives the thick filament its own activation mechanism, separate from
tropomyosin's, and it is a large part of why force rises so steeply with
calcium. The model's SRX is deeper than the single-tier SRX/DRX equilibrium used
in some published models: a physics-motivated choice, not established precedent.

HOW STRAIN ENTERS THE RATES
---------------------------
Two distinct mechanisms, easy to conflate:

  Energy-difference (Boltzmann) — used for 0->1 and 1->2. The rate depends on the
  elastic energy the head would store in each configuration at its current
  geometry, so a head positioned where the strong state is favourable
  isomerizes faster. Reverse rates follow by detailed balance.

  THIS is the mechanism that carries the working stroke's load dependence, since
  1 -> 2 is the transition that carries the stroke (above). E_diff is a genuine
  free-energy difference, so both the forward rate AND the equilibrium constant
  respond to strain — the textbook Eyring treatment of a strained lever swing,
  and consistent with Caremani 2025 (Front. Physiol.), who find the working
  stroke rate constant depends solely on load.

  Bell / load-dependent — used for 3 -> 4, and for that step ONLY. The rate
  depends on the axial FORCE the head is currently carrying, via
  exp(f*delta_34/kT), where delta_34 is the distance to the transition state.
  The load fed to it is the axial force from the same rotation the force law
  uses (forces.polar_to_filament), not the head's radial spring force.

  Barrier / elastic-energy — used for 2 -> 3. The barrier is the elastic energy
  difference between the flanking configurations, floored at zero, so the uphill
  direction is slowed and the downhill one is strain-free. This is Smith &
  Geeves' treatment of the force-generating isomerisation and it carries NO
  transition-state distance. See xb_rate_23 for the derivation; under it the
  pre-exponential A23 is the rate's CEILING rather than its mid-distribution
  value.

  For 3 -> 4 the sign of delta_34 is NEGATIVE at the shipped defaults: a catch
  bond, in which resistive load SLOWS detachment. See xb_rate_34 for the five
  measurements behind that sign and for the model lineage that took the other
  one.

Reverse rates are derived from forward rates and free-energy differences rather
than being free parameters, so the cycle cannot violate detailed balance and
manufacture energy. The exception is 3->4 and 4->0, which are deliberately
one-way: they are the irreversible, ATP-consuming part of the cycle.

CUSTOMIZING
-----------
Every rate the simulation uses is defined here and nowhere else. The two
generator builders in transitions.py — _compute_unique_tm_Q_matrices and
xb_rate_matrix — call these functions and do nothing but assemble the results
into a matrix, so editing a function body here changes the rate law the
simulation actually runs. Each takes explicit scalar arguments rather than a
parameter object, so dependencies are visible in the signature.

Both TM and crossbridge builders evaluate these on a small set of DISTINCT
configurations rather than once per unit (27 neighbour compositions;
n_xb_bins x 2 geometries), so the arguments arrive as arrays and every function
here must stay elementwise.

The TM functions take a ``mod`` multiplier, which is where cooperativity enters:
the Ising coupling passes exp(+h/2) to the forward rates and exp(-h/2) to their
reverses, so the equilibrium constant shifts by exactly exp(h). Any rate law that
should respond to a site's neighbours reads it; a rate that should not — k_30 —
simply has no such argument.

Naming: ``tm_rate_XY`` / ``xb_rate_XY`` is the rate for state X -> state Y, with
state indices as listed above.


REFERENCES
----------
Every paper cited anywhere in this module, with its title and (where the paper
prints one) its DOI, so any claim here can be checked without access to anything
outside this repository.

Bell GI (1978), "Models for the specific adhesion of cells to cells", Science
    200:618-627.
Caremani M et al. (2025), "Multiple pathways in the actin-myosin cycle of
    energy transduction", Front Physiol 16, doi:10.3389/fphys.2025.1664568.
Davis JP, Norman C, Kobayashi T, Solaro RJ, Swartz DR, Tikunova SB (2007),
    "Effects of thin and thick filament proteins on calcium binding and
    exchange with cardiac troponin C", Biophys J 92:3195-3206,
    doi:10.1529/biophysj.106.095406.
Debold EP (2021), "Recent insights into the relative timing of myosin's
    powerstroke and release of phosphate", Cytoskeleton 78:448-458,
    doi:10.1002/cm.21695.
Doran MH et al. (2023), "Conformational changes linked to ADP release from
    human cardiac myosin bound to actin-tropomyosin", J Gen Physiol
    155:e202213267, doi:10.1085/jgp.202213267.
Eakins CG, Reconditi M, Morris EP, Squire JM (2016), "X-ray diffraction
    evidence for low force actin-attached and rigor-like cross-bridges in the
    contractile cycle", Biology 5:41, doi:10.3390/biology5040041.
Fusi L, Brunello E, Yan Z, Irving M (2016), "Thick filament mechano-sensing is
    a calcium-independent regulatory mechanism in skeletal muscle", Nat Commun
    7:13281, doi:10.1038/ncomms13281.
Geeves MA, Lehrer SS (1994), "Dynamics of the muscle thin filament regulatory
    switch: the size of the cooperative unit", Biophys J 67:273-282.
Gollub J, Cremo CR, Cooke R (1996), "ADP release produces a rotation of the
    neck region of smooth myosin but not skeletal myosin", Nat Struct Biol
    3:796-802.
Kawai M, Zhao Y (1993), "Cross-bridge scheme and force per cross-bridge state
    in skinned rabbit psoas muscle fibers", Biophys J 65:638-651.
Knupp C, Squire JM (2020), "The transient mechanics of muscle require only a
    single force-producing cross-bridge state and a 100 Angstrom working
    stroke", Biology 9:475, doi:10.3390/biology9120475.
McKillop DFA, Geeves MA (1993), "Regulation of the interaction between actin
    and myosin subfragment 1: evidence for three states of the thin filament",
    Biophys J 65:693-701.
Mijailovich SM, Prodanovic M, Poggesi C, Geeves MA, Regnier M (2021),
    "Multiscale modeling of twitch contractions in cardiac trabeculae", J Gen
    Physiol 153:e202012604, doi:10.1085/jgp.202012604.
Millar NC, Homsher E (1990), "The effect of phosphate and calcium on force
    generation in glycerinated rabbit skeletal muscle fibers", J Biol Chem
    265:20234-20240.
Offer G, Ranatunga KW (2013), "A cross-bridge cycle with two tension-generating
    steps simulates skeletal muscle mechanics", Biophys J 105:928-940,
    doi:10.1016/j.bpj.2013.07.009.
Park-Holohan S-J et al. (2021), "Stress-dependent activation of myosin in the
    heart requires thin filament activation and thick filament mechanosensing",
    PNAS 118:e2023706118, doi:10.1073/pnas.2023706118.
Pate E, Cooke R (1989), "A model of crossbridge action: the effects of ATP, ADP
    and Pi", J Muscle Res Cell Motil 10:181-196.
Pinto JR et al. (2011), "Strong cross-bridges potentiate the Ca2+ affinity
    changes produced by hypertrophic cardiomyopathy cardiac troponin C mutants
    in myofilaments", J Biol Chem 286:1005-1013, doi:10.1074/jbc.M110.168583.
Siemankowski RF, White HD (1984), "Kinetics of the interaction between actin,
    ADP, and cardiac myosin-S1", J Biol Chem 259:5045-5053.
Stewart MA, Franks-Skiba K, Chen S, Cooke R (2010), "Myosin ATP turnover rate
    is a mechanism involved in thermogenesis in resting skeletal muscle
    fibers", PNAS 107:430-435, doi:10.1073/pnas.0909468107.
Sung J, Nag S, Mortensen KI, Vestergaard CL, Sutton S, Ruppel K, Flyvbjerg H,
    Spudich JA (2015), "Harmonic force spectroscopy measures load-dependent
    kinetics of individual human beta-cardiac myosin molecules", Nat Commun
    6:7931, doi:10.1038/ncomms8931.
Veigel C, Schmitz S, Wang F, Sellers JR (2005), "Load-dependent kinetics of
    myosin-V can explain its high processivity", Nat Cell Biol 7:861-869,
    doi:10.1038/ncb1287.
Wang Y et al. (2024), "Single-molecule investigation of load-dependent
    actomyosin dissociation kinetics for cardiac and slow skeletal myosin",
    Small, doi:10.1002/smll.202406865.
Woody MS, Winkelmann DA, Capitanio M, Ostap EM, Goldman YE (2019), "Single
    molecule mechanics resolves the earliest events in force generation by
    cardiac myosin", eLife 8:e49266, doi:10.7554/eLife.49266.

"""
import jax.numpy as jnp


# =============================================================================
# TROPOMYOSIN RATE FUNCTIONS
# =============================================================================

def tm_ca_affinity_ratios(Keq_unblock_noca, Keq_unblock_ca, Keq_swing_noca, Keq_swing_ca):
    """Calcium affinity at (blocked, closed, open), relative to blocked: (3,).

    NOT free parameters. Six states and seven edges leave two cycles
    (blocked-closed and closed-open, each with and without Ca), and detailed
    balance around each fixes one ratio:

        K_ca(closed) / K_ca(blocked) = K_B(+Ca) / K_B(-Ca)
        K_ca(open)   / K_ca(closed)  = K_T(+Ca) / K_T(-Ca)

    So calcium enters the unit only as tm_Keq_ca * [Ca]: scaling the blocked
    affinity is indistinguishable from scaling [Ca].
    """
    r_closed = Keq_unblock_ca / Keq_unblock_noca
    return jnp.stack([jnp.ones_like(r_closed), r_closed,
                      r_closed * Keq_swing_ca / Keq_swing_noca])


def tm_rate_ca_on(ca_conc, k_on, ratio, alpha):
    """Calcium binds troponin at a position: k_on * ratio^alpha * [Ca] (ms^-1).

    The only step that reads calcium. alpha in [0, 1] splits the position
    linkage (ratio, from tm_ca_affinity_ratios) between on- and off-rate;
    detailed balance fixes only their quotient. alpha = 0 puts all of it on
    the off-rate, the measured form: strong heads slow calcium release.
    """
    return k_on * ratio ** alpha * ca_conc


def tm_rate_ca_off(k_on, ratio, alpha, Keq_ca):
    """Calcium leaves troponin at a position: k_on * ratio^alpha / (Keq_ca * ratio).

    Detailed balance against tm_rate_ca_on; Keq_ca is the BLOCKED affinity
    (params.tm_Keq_ca), ratio the position's affinity relative to it.
    """
    return k_on * ratio ** alpha / (Keq_ca * ratio)


def tm_rate_forward(k):
    """A forward position step (blocked -> closed, or closed -> open), ms^-1.

    The base rate; the Ising field multiplies it in the generator builder.
    tm_k_unblock: blocked -> closed ~500 /s on a Ca jump (Shitaka 2004, skeletal
    Tn). tm_k_swing: closed -> open, no measured rate; Geeves & Lehrer 1994 bound only
    the sum k+T + k-T, which is fast enough not to limit anything.
    """
    return k


def tm_rate_reverse(k, Keq):
    """The reverse position step: k / Keq at this calcium level (detailed balance)."""
    return k / Keq


# =============================================================================
# CROSSBRIDGE RATE FUNCTIONS
# =============================================================================

def xb_rate_01(permissiveness, r01_coeff, E_weak):
    """Rate 0->1: DRX head attaches weakly to actin.

    The gateway into the force-producing cycle. Tropomyosin gates it, and
    gates r12 separately (see xb_rate_matrix). Two factors multiply:

      attach          1 if the target unit is closed or open, 0 if blocked —
                      a hard gate, so a head cannot bind a blocked site at any
                      strain. The caller multiplies it in; this function is
                      also evaluated ungated (1.0) for detailed balance.
      exp(-E_weak)    Boltzmann penalty for the elastic energy the head must
                      store to reach this site in its weak configuration.
                      Heads far from their unstrained geometry bind
                      exponentially more slowly, which is what confines binding
                      to a narrow axial window around each head's rest position.

    Since E_weak is quadratic in the head's offset, that second factor is a
    Gaussian window centred on the weak rest position. It is the same form Pate &
    Cooke 1989 JMRCM 10:181 use for attachment in Table 2,
    R23(x) = 5 + 500*exp[-0.8*(x - 7.5)^2] s^-1 — a Gaussian of comparable width
    (their elastic constant 0.56 RT/nm^2 gives exp[-0.56*(x - 7.5)^2]) centred on
    their weakly-bound state's free-energy minimum at x = 7.5 nm.

    Args:
        permissiveness: 0 or 1, the attach gate (1.0 for the ungated rate)
        r01_coeff: Pre-exponential binding coefficient (ms^-1) - params.xb_r01_coeff.
                   [G] 305.99 is inherited from an earlier parameterization of
                   this model, where it arose as the product of a binding rate
                   and a duty-cycle scaling. It has no independent literature
                   source and should be treated as a free parameter: it sets the
                   overall attachment timescale and is a natural first candidate
                   when fitting.
        E_weak: Elastic energy of the weak configuration at this geometry (kT)

    Returns:
        Rate r01 (ms^-1)
    """
    r01 = permissiveness * r01_coeff * jnp.exp(-E_weak)
    return jnp.where(jnp.isnan(r01), 0.0, r01)


def xb_rate_10(r01, U_DRX, U_loose):
    """Rate 1->0: weakly bound head detaches.

    Detailed balance against the forward rate: r10 = r01 * exp(U_loose - U_DRX).

    STRAIN CANCELS, and a reader looking at the formula will assume the
    opposite, which is why this paragraph is here. r01 carries exp(-E_weak) and
    U_loose carries +E_weak; detailed balance ties them together:

        r10 = (r01 + 0.005) * exp(U_loose - U_DRX)
            = [C*exp(-E_weak) + 0.005] * exp(U_loose_base + E_weak - U_DRX)
            = C*exp(U_loose_base - U_DRX)                 <- CONSTANT, 41.4 /ms
                                                             at cardiac defaults
              + 0.005*exp(U_loose_base + E_weak - U_DRX)  <- the floor

    So the weak state's LIFETIME is the same at every strain. What strain
    changes is its OCCUPANCY, through r01.

    Computed in log space, since r01 spans many orders of magnitude across the
    strain range and the naive product overflows float32.

    The +0.005 floor inside the logarithm keeps the rate finite where r01 is
    exactly zero. The caller passes the UNGATED r01 (a steric block stops a
    head arriving, not one leaving), so r01 is zero only where exp(-E_weak)
    underflows, far outside the binding window; the floor keeps log(0) = -inf
    out of the rate matrix there.

    DO NOT "SIMPLIFY" THIS FUNCTION. Because the rate is analytically constant,
    the log-space form, the floor and the 10000 ms^-1 cap all look removable.
    Removing the floor changes r10 at OPEN sites by a relative 1.6e-5.

    Args:
        r01: Forward binding rate (ms^-1)
        U_DRX: Free energy of the DRX state (kT)
        U_loose: Free energy of the loose state, including elastic strain (kT)

    Returns:
        Rate r10 (ms^-1), capped at 10000 ms^-1
    """
    upper = 10000.0
    log_r21 = jnp.log(r01 + 0.005) - (U_DRX - U_loose)
    r10 = jnp.exp(log_r21)
    r10 = jnp.minimum(r10, upper)
    return jnp.where(jnp.isnan(r10), upper, r10)


def xb_rate_12(A12, E_diff):
    """Rate 1->2: weak-to-strong isomerization (phosphate release) — THE WORKING STROKE.

    Biologically this is Pi release converting a loosely tethered head into a
    strongly bound one. It is the step that commits a head to force production.

    Mechanically it is also the ONLY transition in the model that moves the
    spring rest configuration, so it carries the entire lever swing and does all
    of the work. The state names suggest the stroke happens later, at 2 -> 3; it
    does not (module docstring).

    Note that lumping the stroke with Pi release side-steps an unsettled
    question rather than answering it: whether the lever swing precedes or
    follows Pi release is actively contested (Debold 2021 Cytoskeleton 78:2 for
    the review; Woody 2019 finds the stroke rate unaffected by 10 mM Pi;
    Caremani 2025 Front. Physiol. argue Pi release is "orthogonal" to the
    progression of the stroke). Because this model has no explicit Pi state, it
    takes no position, which is closer to the current functional consensus than
    an explicitly sequential treatment would be.

    Rate law: r12 = A12 * exp(E_diff / 2), where E_diff = E_weak - E_strong is
    how much elastic energy the head sheds by switching configurations at its
    current position. Positive E_diff means the strong configuration is the more
    relaxed one there, and the isomerization is faster.

    The division by 2 is a symmetric-barrier assumption (alpha = 0.5): the
    transition state is taken to sit halfway between the two configurations, so
    half the energy difference appears in the forward rate and half in the
    reverse. This is a modelling choice with no direct measurement behind it. A
    different alpha would redistribute strain dependence between r12 and r21
    without changing their ratio, which detailed balance fixes.

    E_diff is capped at 30 kT before exponentiating: beyond that the rate is
    already far above anything the timestep can resolve, and float32 overflows.

    Args:
        A12: Pre-exponential rate coefficient (ms^-1) - params.xb_r12_coeff.
             [F] Not a measured isomerization rate. It is tuned so the model
             reproduces the apparent rate of "process B" in sinusoidal analysis
             (2*pi*b ~ 20-60 s^-1 for skeletal at 20-25 C; Kawai & Zhao 1993
             Biophys J 65:638). Be careful with that paper's numbers: its
             286 s^-1 is k2, ATP-induced detachment (process C), NOT this step.
        E_diff: E_weak - E_strong at the head's current geometry (kT); positive
                favours the strong configuration

    Returns:
        Rate r12 (ms^-1)
    """
    r12 = A12 * jnp.exp(jnp.minimum(E_diff, 30.0) / 2.0)
    return jnp.where(jnp.isnan(r12), 0.0, r12)


def xb_rate_21(A12, E_diff, U_loose, U_tight_1):
    """Rate 2->1: strong-to-weak reversal (phosphate rebinding).

    Detailed balance: r21 = r12 * exp(U_tight_1 - U_loose). Since the strong
    state sits well below the loose state in free energy, this reversal is
    strongly suppressed under normal conditions — but it becomes significant for
    heads bound at large strain, where U_tight_1's elastic term climbs. That is
    the model's route for a badly-positioned head to back out without completing
    the cycle and paying an ATP.

    Log-space arithmetic, as with r10, to survive the dynamic range in float32.
    r12 is rebuilt here as log(A12) + E_diff / 2 (xb_rate_12's law and cap)
    rather than passed in: at large strain r12 underflows to exactly 0 while r21
    is at its cap, and log(0) would turn the cap into 0.

    Args:
        A12: Pre-exponential rate coefficient of r12 (ms^-1) - params.xb_r12_coeff
        E_diff: E_weak - E_strong at the head's current geometry (kT), as for
                xb_rate_12
        U_loose: Free energy of the loose state, including strain (kT)
        U_tight_1: Free energy of the tight_1 state, including strain (kT)

    Returns:
        Rate r21 (ms^-1), capped at 10000 ms^-1
    """
    upper = 10000.0
    log_r12 = jnp.log(A12) + jnp.minimum(E_diff, 30.0) / 2.0
    r21 = jnp.exp(log_r12 - (U_loose - U_tight_1))
    return jnp.minimum(r21, upper)


def xb_rate_23(A23, E_tight_1, E_tight_2):
    """Rate 2->3: chemical transition between the two strongly-bound states.

    NOT the whole working stroke, despite the state names — most of the stroke
    is on 1 -> 2; see the module docstring. What this step always does is commit
    the head to the ~6 kT drop that makes the stroke effectively irreversible,
    and move it into the only state from which ADP release and detachment are
    possible. At the shipped (split-stroke) defaults it also carries part of
    the axial swing (module docstring), so it does work and changes force.

    THE ELASTIC-ENERGY FORM, NOT A BELL DISTANCE.

        dE   = E_tight_2 - E_tight_1        elastic, in kT
        r23  = A23 * exp(-max(dE, 0))

    Smith & Geeves 1995 Biophys J 69:528 (Eqs. 2a/2b/3) treat the
    force-generating isomerisation with the elastic energy DIFFERENCE between
    the flanking states, placing the transition state at the top of the
    higher-energy well. There is therefore no transition-state distance to
    parameterise: the strain dependence falls entirely on the uphill direction,
    and the downhill rate is strain-FREE. Smith 1998 (p. 193, Fig. 3 caption)
    confirms the reverse rate carries the same factor when the flanking states
    share elastic energy, and floors the barrier at zero — which is the `max`
    here, and is real physics rather than a numerical guard: a barrier cannot be
    negative, so a downhill step can be strain-independent but never
    strain-ACCELERATED.

    The strain dependence of this step lives in the tight_1/tight_2
    REST-CONFIGURATION difference that sets dE, so the step has no free
    transition-state distance and a ceiling of A23.

    r32 IS DELIBERATELY LEFT AS PLAIN DETAILED BALANCE (xb_rate_32). Substitute
    this rate into it and the strain factor lands on whichever direction climbs,
    in both directions — S&G Eq. 2b exactly, with K_23 preserved. Adding a
    barrier term to r32 as well would double-count it.

    Args:
        A23: Zero-load rate (ms^-1) - params.xb_r23_coeff.
             [G] UNSOURCED, and now the binding uncertainty on this step. The
             usual citation (Millar & Homsher 1990, ~70-100 s^-1) measured k_Pi
             from caged-phosphate photolysis, i.e. Pi release coupled to force
             generation, which in this model is 1 -> 2. A23 is the CEILING of
             the rate, not its mid-distribution value.
        E_tight_1: Elastic energy of state 2's configuration at this geometry (kT)
        E_tight_2: Elastic energy of state 3's configuration at this geometry (kT)

    Returns:
        Rate r23 (ms^-1), capped at 10000 ms^-1

    References:
        Smith DA, Geeves MA (1995), "Strain-dependent cross-bridge cycle for
        muscle", Biophys J 69:524-537, Eqs. 2a/2b/3;
        Smith DA (1998), J Muscle Res Cell Motil 19:189-211, p. 193 Fig. 3.
    """
    upper = 10000.0
    barrier = jnp.maximum(E_tight_2 - E_tight_1, 0.0)
    r23 = A23 * jnp.exp(-barrier)
    return jnp.minimum(r23, upper)


def xb_rate_32(r23, U_tight_1, U_tight_2):
    """Rate 3->2: reverse of the Tight_1 <-> Tight_2 chemical transition.

    A reverse of the smaller part of the stroke at the shipped defaults, and of
    nothing at all in the control configuration where the *_tight_1 springs
    equal the *_strong ones (see xb_rate_23 for both cases).

    Detailed balance: r32 = r23 * exp(U_tight_2 - U_tight_1). At the default free
    energies (U_tight_1 = -15 kT, U_tight_2 = -21 kT) the 6 kT drop gives
    r32 ~ exp(-6) ~ 0.0025 * r23, so the step is effectively one-way. That
    asymmetry is what keeps heads in the ADP-release-competent state long enough
    to bear load and complete the cycle, rather than rattling back and forth.

    THIS LINE IS DELIBERATELY UNCHANGED BY THE MOVE TO THE ENERGY FORM, and it
    is what makes that form correct in both directions. Write dE = E_tight_2 -
    E_tight_1 (elastic, kT) and dU_base for the chemical part of
    U_tight_2 - U_tight_1. Substituting r23 = A23*exp(-max(dE, 0)) here gives

        dE > 0 (2->3 uphill):   r32 = A23*exp(dU_base)             strain-FREE
        dE < 0 (2->3 downhill): r32 = A23*exp(dU_base)*exp(dE)     the barrier

    so the strain factor always lands on whichever direction climbs, and never
    ACCELERATES either — which is Smith & Geeves Eq. 2b, reproduced exactly, with
    K_23 = exp(-(U_tight_2 - U_tight_1)) preserved. Adding a barrier term here
    as well would double-count it.

    THAT THE REVERSE RATE INHERITS THE FORWARD FACTOR IS NOT AN ANOMALY. An
    earlier version of this docstring called it one. Smith 1998's Fig. 3 caption
    (p. 193) states exactly this behaviour, and gives the reason: when the two
    flanking states share elastic energy the barrier is common to both
    directions. It is correct physics, not an artefact of the parameterisation.

    IN THE CONTROL CONFIGURATION (*_tight_1 springs equal to *_strong) both free
    energies carry the SAME elastic term, so dE = 0, both rates are strain-free,
    and K_23 is fixed everywhere: load cannot shift the 2/3 population in either
    direction. AT THE SHIPPED DEFAULTS the elastic terms differ, K_23 becomes
    strain-dependent, and load does redistribute heads between states 2 and 3.
    That is the split stroke doing its job.

    Args:
        r23: Forward working-stroke rate (ms^-1)
        U_tight_1: Free energy of tight_1, including strain (kT)
        U_tight_2: Free energy of tight_2, including strain (kT)

    Returns:
        Rate r32 (ms^-1), capped at 10000 ms^-1

    References:
        Pate E, Cooke R (1989), "A model of crossbridge action: the effects of ATP,
        ADP and Pi", J Muscle Res Cell Motil 10:181-196. (No DOI printed in the
        paper.)
    """
    upper = 10000.0
    log_r43 = jnp.log(r23 + 1e-30) + (U_tight_2 - U_tight_1)
    return jnp.exp(jnp.minimum(log_r43, jnp.log(upper)))


def xb_rate_34(A34, f_strong, delta_catch, delta_slip, slip_frac, k_t):
    """Rate 3->4: ADP release and detachment.

    The post-stroke head releases ADP, binds ATP, and lets go of actin. This is
    the rate-limiting step of the cycle for most myosins, and it consumes the
    ATP: state 4 is the model's accounting point for ATP turnover.

    Two-pathway catch-slip form, two Bell terms in parallel:
        r34 = A34 * ((1 - slip_frac) * exp(f * delta_catch / kT)
                     + slip_frac * exp(f * delta_slip / kT)),
    where f is the head's AXIAL force (positive = resisting). The rate at zero
    load is exactly A34. delta_catch < 0: resistive load SLOWS the catch
    pathway. delta_slip > 0: it speeds the slip pathway, which takes over at
    high load. These are the model's only Bell terms (2 -> 3 uses the
    elastic-energy barrier, see xb_rate_23).

    The rate is slowest at
        f_crit = kT * ln((1 - slip_frac) * |delta_catch| / (slip_frac * delta_slip))
                 / (delta_slip - delta_catch),
    which does not depend on A34. Beyond f_crit detachment speeds up, so the
    force (and with the head's stiffness, the strain) a stretched head reaches
    is bounded for any A34.

    THE SLIP PATHWAY. Guo & Guilford 2006 PNAS 103:9844 (rat skeletal HMM,
    single-molecule step loads 1.8-26.4 pN, perpendicular to actin): lifetimes
    rise to ~6 pN, then "Above 6 pN, bond lifetimes decrease with increasing
    loads" (p. 9846). Their Table 2 fits this form: slip k0 13 / 15 s^-1 and
    x 0.5 / 0.40 nm, catch k0 127 / 176 s^-1 and x -1.5 / -2.5 nm (rigor / ADP),
    so slip_frac ~0.08. Colombini et al. 2007 PNAS 104:9284 (intact frog fibres,
    14 C, fast stretches): "cross-bridge rupture force increased linearly with
    the logarithm" of loading rate (p. 9286), a single Bell barrier of
    x_beta 1.25 nm; and series compliance changes the stretch needed "but does
    not affect the rupture force itself" (p. 9287) -- rupture is set by force,
    not strain.

    Args:
        A34: Zero-load detachment rate (ms^-1) - params.xb_r34_coeff.
             [I] skeletal / [L] cardiac. Siemankowski & White 1984 JBC 259:5045:
             "the rate constant for the dissociation of ADP from cardiac
             actomyosin-S1, k_AD, is ~65 s^-1 at 15 C" (bovine ventricle,
             beta-MHC). For skeletal the same paper gives ">500 s^-1" at 15 C,
             a ONE-SIDED BOUND quoted from their ref (14) -- hence [I], not [M].
             "Cardiac ~10x slower" is a 15 C statement only.
        f_strong: Force carried in the strong state (pN); positive = tensile
        delta_catch: Catch-pathway distance (nm), negative - params.xb_delta_34_catch
        delta_slip: Slip-pathway distance (nm), positive - params.xb_delta_34_slip
        slip_frac: Slip pathway's share of A34 - params.xb_r34_slip_frac
        k_t: Thermal energy kT (pN*nm)

    Returns:
        Rate r34 (ms^-1), capped at 10000 ms^-1

    THE CATCH SIGN. Measurements on the catch side, all of axial load:
      - Sung 2015 Nat Commun 6:7931 (human beta-cardiac, harmonic force
        spectroscopy): "k0 = 87 +/- 7 s^-1 and d = 0.8 +/- 0.1 nm"; resisting
        load slows detachment. The shipped -0.80 nm is this value in this
        kernel's sign convention.
      - Wang 2024 Small (rabbit beta-cardiac and slow skeletal): "both kf and ks
        responded to increasing resistive as well as assistive load by slowing
        the actomyosin unbinding rates ... exhibiting 'catch bond' behavior";
        fitted distances 1.5-1.9 nm, at 10 uM ATP.
      - Marang et al. 2025 PNAS 122:e2504758122 (rabbit fast skeletal), Fig. 6
        caption: d = -0.89 +/- 0.12 nm at 0 mM added Pi, progressing to
        +0.34 nm at 30 mM Pi.
      - Veigel 2005 Nat Cell Biol 7:861 (myosin-V, intramolecular strain):
        resisting load slows ADP release.
      - Pate & Cooke 1989 JMRCM 10:181 Table 2 ASSUME the same sign (R45 = 2 s^-1
        for x >= 3.7 nm, rising to 750 s^-1 for x < 0).
    Premature detachment from earlier force-generating states is the
    strain-gated 3 -> 2 -> 1 -> 0 path; this rate is the exit from the
    post-stroke state. V0 in this model is not detachment-limited: the catch
    sign leaves unloaded shortening velocity unchanged and raises sub-V0 force.

    References:
        Bell 1978 Science 200:618 (the functional form); Siemankowski & White 1984
        JBC 259:5045 (the zero-load rates); Guo & Guilford 2006 PNAS 103:9844 and
        Colombini et al. 2007 PNAS 104:9284 (the slip pathway).
    """
    upper = 10000.0
    r34 = A34 * ((1.0 - slip_frac) * jnp.exp(f_strong * delta_catch / k_t)
                 + slip_frac * jnp.exp(f_strong * delta_slip / k_t))
    return jnp.minimum(r34, upper)


def xb_rate_43():
    """Rate 4->3: re-attachment straight back into the post-stroke state.

    Identically zero. Once ATP has bound and the head has released actin, it
    cannot simply reverse into the strongly-bound post-stroke state — that would
    run the ATPase backwards and let the model recover the energy it just spent.
    Detachment is the irreversible step that makes the cycle a cycle.

    Kept as a function rather than an inline 0.0 so the rate matrix reads
    symmetrically and so an alternative model can override it.

    Returns:
        0.0
    """
    return 0.0


def xb_rate_40(r40_rate):
    """Rate 4->0: recovery stroke, returning the head to DRX.

    ATP is hydrolysed and the lever arm re-primes, so the head is ready to bind
    again. Modelled as a plain first-order rate: nothing here depends on strain,
    since the head is detached and carries no load.

    Args:
        r40_rate: Rate constant (ms^-1) - params.xb_r40.
                  [G] 0.1 is unsourced — it was a hardcoded constant in an
                  earlier parameterization of this model and is retained for
                  continuity. It sets how quickly detached heads become
                  available again, so it caps the achievable cycling rate; worth
                  revisiting if cycling kinetics are ever the fitting target.

    Returns:
        Rate r40 (ms^-1)
    """
    return r40_rate


def xb_rate_04(r04_rate):
    """Rate 0->4: reverse of the recovery stroke.

    A DRX head slipping back into the post-hydrolysis, pre-recovery state. Rare,
    but non-zero — this pair is the one reversible link between the detached
    states, and including it keeps that portion of the cycle thermodynamically
    consistent rather than artificially one-way.

    Args:
        r04_rate: Rate constant (ms^-1) - params.xb_r04.
                  [M] 0.01; Mijailovich 2021 PMC7852458 reports k_-H = 10 s^-1.
                  Paired with xb_r40 = 0.1 this gives an equilibrium constant of
                  10 for hydrolysis, matching the classical measurement that
                  hydrolysis on myosin is only modestly favourable.

    Returns:
        Rate r04 (ms^-1)
    """
    return r04_rate


def xb_rate_50(ca_conc, k0, kmax, b, ca50):
    """Rate 5->0: calcium recruits a head out of the super-relaxed state.

    THICK FILAMENT ACTIVATION. Alongside tropomyosin's thin-filament regulation,
    this is the model's second, independent way for calcium to control force —
    and often the dominant one. Heads in SRX are folded back against the
    backbone with their ATPase suppressed; they cannot bind at any strain. Raise
    calcium and they are released into the available (DRX) pool.

    Hill form:  r50 = k0 + (kmax - k0) * Ca^b / (ca50^b + Ca^b)

    The Hill exponent b is what makes this a steep switch rather than a gradual
    ramp, and it is a large part of why simulated force-pCa curves come out
    steeper than single-site calcium binding could ever produce.

    Do not read this rate as an "SRX release rate" to be balanced against some
    sequestration rate of similar size: xb_r05 (the reverse) is a
    SEQUESTRATION/parking rate, and the two are asymmetric by design.

    Args:
        ca_conc: Calcium concentration (M)
        NOTE: this rate law is taken directly from Mijailovich 2021 JGP
        153:e202012604 Eq. 1, and ALL FOUR of its parameters are marked "Assumed"
        in that paper's Table 1. Nothing in this function is
        measured. Conditions there: rat cardiac trabeculae, 27.2 C.

        k0: Basal rate at zero calcium (ms^-1) - params.xb_srx_k0.
            [G] 0.007 skeletal / 0.005 cardiac. Mijailovich kPS0 = 5 s^-1
            ("Assumed"); cardiac adopts it exactly, skeletal uses 7 s^-1 because
            it is pinned to xb_r05 by construction.
        kmax: Saturating rate at high calcium (ms^-1) - params.xb_srx_kmax.
              [G] 0.4; Mijailovich kPSmax = 400 s^-1, also "Assumed"
              (an assumption, hence [G]).
        b: Hill exponent - params.xb_srx_b.
           [G] 5.0. Mijailovich 2021 Table 1 lists b = 5 but marks it "Assumed" —
           a shape parameter chosen inside a model, not a measurement, so [G] not
           [M]. This gate is a calcium proxy for a process that is MEASURED to be
           stress-driven: Fusi 2016 Nat Commun 7:13281 abstract states that "both
           the extent and kinetics of thick filament activation depend on thick
           filament stress but are independent of intracellular calcium
           concentration in the physiological range" (rabbit psoas), and Linari
           2015 Nature 528:276 that "filament stress controls the transition
           between these two states". So the Ca proxy is CONTRADICTED for
           skeletal. For cardiac it is defensible: Park-Holohan 2021 PNAS
           118:e2023706118 finds folded motors are "not directly switched on ...
           in the absence of thin filament activation" (rat trabeculae, 26 C).
        ca50: Calcium for half-maximal recruitment (M) - params.xb_srx_ca50.
              [G] 1e-6 (pCa 6) is a round number in the physiological range,
              not a measured value for this transition. Mijailovich Table 1
              [Ca2+]50 = 1 uM, "Assumed".

    Returns:
        Rate r50 (ms^-1)
    """
    return k0 + ((kmax - k0) * ca_conc**b) / (ca50**b + ca_conc**b)


def xb_rate_05(r05_rate):
    """Rate 0->5: an available head parks itself in the super-relaxed state.

    Sequestration, not release. A DRX head folds back against the thick filament
    backbone and drops out of the recruitable pool.

    Together with xb_rate_50 this sets the resting SRX occupancy: the fraction
    parked at equilibrium is r05 / (r05 + r50(Ca)), so at low calcium — where
    r50 falls to k0 — the balance tips heavily toward SRX. Skeletal defaults
    (r05 = 0.007, k0 = 0.007) give roughly half the heads parked at rest. The
    cardiac preset uses a much larger r05 (0.2), putting the great majority of
    cardiac heads in reserve at rest and leaving more headroom for calcium to
    recruit — which is where much of cardiac contractile reserve comes from.

    Args:
        r05_rate: Rate constant (ms^-1) - params.xb_r05.
                  [M] skeletal 0.007 (~50% SRX at rest; Stewart 2010 PNAS
                  107:430); cardiac 0.2, matching Mijailovich 2021's
                  k_-PS = 200 s^-1. 200 s^-1 is the literature ceiling for this
                  rate, not a floor to stay under.

    Returns:
        Rate r05 (ms^-1)
    """
    return r05_rate
