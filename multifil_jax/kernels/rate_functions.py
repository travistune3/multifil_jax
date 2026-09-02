"""
Transition rates for the tropomyosin and crossbridge state machines.

Every rate in the model is defined here, one function per transition. The
kernels assemble these into rate matrices; this module is where the biophysics
of "how fast does this step happen, and what does it depend on" lives.

Rates are in ms^-1 (a value of 0.6 means 600 s^-1). Energies are in kT.

THE TROPOMYOSIN CYCLE (4 states)
--------------------------------
Tropomyosin is a coiled-coil that lies in the groove of the actin helix,
physically covering the sites myosin needs to bind. Calcium binding to troponin
lets it move aside. The model resolves that into four states:

    0  Ca-free, blocking      no calcium on troponin, sites covered
    1  Ca-bound, blocking     calcium bound, tropomyosin has not yet moved
    2  closed                 tropomyosin shifted, sites still not accessible
    3  open (permissive)      sites exposed; myosin may bind here

States 1-2-3 are the classical blocked/closed/open picture (McKillop & Geeves
1993); state 0 splits out the calcium-binding step so calcium concentration
enters as a genuine rate rather than a switch. Only state 3 permits binding.

The cycle closes 3 -> 0 rather than running backwards through 1: calcium
dissociates from the open state directly. A site with a crossbridge attached is
locked in state 3 — the bound head physically holds tropomyosin out of the way,
which is the structural basis for crossbridge binding being self-reinforcing.
That lock is NOT expressed in the rates below: whether a site is locked changes
every timestep while the rate matrices do not, so it is applied afterwards as a
[0, 0, 0, 1] override on the locked site's probability vector (see
transitions.thin_transitions). Gating 3->2 and 3->0 in the rates would give the
same result, at the cost of the 27-matrix reduction.

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

THIS DEPENDS ON THE PARAMETERS, and the dependence is worth stating exactly,
because it used to be a structural fact and is now a control path. When the
*_tight_1 springs equal the *_strong ones, states 2 and 3 are elastically
identical, the whole rest-position change happens on 1 -> 2, and 2 -> 3 moves
nothing, does no work and changes no force — a purely chemical step. That was
the model until 2026-09-01 and it is still exactly what the six control values
in core/params.py reproduce.

At the shipped defaults the *_tight_1 rest positions differ (see the split
stroke block in core/params.py), so 2 -> 3 carries part of the lever swing:
the axial rest projection g_rest*cos(c_rest) is 13.5516 / 6.8596 / 4.7604 nm
for Loose / Tight_1 / Tight_2, i.e. 6.692 nm on 1 -> 2 and 2.099 nm — 23.9% of
the 8.791 nm total — on 2 -> 3. Note the projection is NOT linear in the
interpolation fraction the two rest positions were built from (0.75), and
2.099 nm is ~1.4x the ~1.5 nm second swing quoted below, so it is NOT a match
to Doran 2023 or Woody 2019. It is also a different quantity from the 5.837 nm
"effective stroke" measured mechanically in S109; do not mix the two.

What separates the two states in every configuration is the ~6 kT free energy
drop that makes the stroke effectively one-way, and the fact that only state 3
can release ADP and detach.

The two-configuration form was a deliberate design, not an oversight, and the
case for it still stands on its own terms. Fibre mechanics and X-ray modelling
both argue that ONE force-bearing conformation plus one low-force attached
conformation is sufficient: Knupp & Squire 2020 (Biology 9:464) fit length-step
transients, isotonic shortening and the M3 reflection with exactly that, and
could not find parameters for variants carrying two force-producing attached
states; Eakins 2016 (Biology 5:41) concludes from X-ray that "no more than two
main attached structural states are necessary and sufficient", with the weak
state contributing ~4% of tension. In that mapping state 1 is the low-force
attached state and states 2+3 together are the single force-bearing one.

The split stroke breaks that mapping: states 2 and 3 are now elastically
distinct, so this model carries two force-bearing configurations against those
two arguments. It was adopted anyway, on a measurement the two-configuration
form fails outright — force per strong head 1.67 pN against Piazzesi 2007's
~5.7, with the strong-bound FRACTION already right, i.e. a strain-distribution
deficit rather than a recruitment one (S129). Splitting the stroke and taking
the catch-bond sign of xb_delta_34 together carried force from 52% to 93% of
the Kooiker target at literature values, with no fitting.

The known omission WAS the second, much smaller lever swing that accompanies
ADP release: ~16 degrees / ~1.5 nm in cardiac myosin (Doran 2023 JGP
155:e202213267 by cryo-EM; Woody 2019 eLife 8:e49266 measures 1-1.5 nm by
single molecule), essentially absent in fast skeletal myosin (Gollub, Cremo &
Cooke 1996 Nat Struct Biol 3:796). Its structural role is argued to be strain
sensing rather than force generation, and in the two-configuration model it
appeared only as the Bell distance of the detachment step (xb_delta_34), never
as a displacement.

The split stroke is precisely the fix for that omission: the second swing is
now a real displacement, and it is the 2 -> 3 step. That moves the model TOWARD
the dissenting view — Offer & Ranatunga 2013 (Biophys J 105:928-940) require
TWO tension-generating steps, of 5.6 and 4.6 nm, and reject single-step models
on efficiency and on lengthening force-velocity — though not to their
partition: this model's split is 6.69 / 2.10 nm, far more lopsided than theirs.

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

  Bell / load-dependent — used for 2->3 and 3->4. The rate depends on the FORCE
  the head is currently carrying, via exp(+/- f*delta/kT), where delta is the
  distance to the transition state.

  A CAVEAT ON delta_23. Because states 2 and 3 are mechanically identical, the
  net displacement across 2 -> 3 is zero, so the elastic terms cancel in the
  reverse rate and K_23 = r23/r32 is LOAD-INDEPENDENT. That is thermodynamically
  required, not an approximation. But it also means a non-zero delta_23 describes
  a reaction coordinate that travels out and returns to the same place: load
  slows BOTH directions equally and never shifts the population. For a chemical
  isomerization that is tolerable; for a lever-arm swing it has no structural
  referent. The self-consistent choices are (a) delta_23 = 0 with 2 and 3
  mechanically identical, or (b) give state 3 its own rest configuration, in
  which case 0 < delta_23 < the net displacement and K_23 becomes load-dependent
  on its own. The current pairing (identical mechanics, delta_23 = 1.0 nm) is
  neither, and it is not inert: at the ~1.8 pN mean strong-state force it
  suppresses r23 by ~35%, and since state 2 has no detachment exit, that acts as
  a load-gated retention in the pre-ADP-release state — the only place in the
  model where load slows the forward cycle.

  For 3 -> 4 the sign is positive: load accelerates detachment. See xb_rate_34
  for why that sign is contested for cardiac myosin.

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
Capitanio M, Canepari M, Cacciafesta P, Lombardi V, Cicchi R, Maffei M, Pavone
    FS, Bottinelli R (2006), "Two independent mechanical events in the
    interaction cycle of skeletal muscle myosin with actin", PNAS 103:87-92.
Caremani M, Pinzauti F, Powers JD, Governali S, Narayanan T, Stienen GJM,
    Reconditi M, Linari M, Lombardi V, Piazzesi G (2019), "Inotropic
    interventions do not change the resting state of myosin motors during
    cardiac diastole", J Gen Physiol 151:53-65, doi:10.1085/jgp.201812196.
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
Fraser IDC, Marston SB (1995), "In vitro motility analysis of actin-tropomyosin
    regulation by troponin and calcium", J Biol Chem 270:7836-7841.
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
Lehrer SS, Morris EP (1982), "Dual effects of tropomyosin and troponin-
    tropomyosin on actomyosin subfragment 1 ATPase", J Biol Chem 257:8073-8080.
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
Piazzesi G et al. (2007), "Skeletal muscle performance determined by modulation
    of number of myosin motors rather than motor force or stroke size", Cell
    131:784-795, doi:10.1016/j.cell.2007.09.045.
Pinto JR et al. (2011), "Strong cross-bridges potentiate the Ca2+ affinity
    changes produced by hypertrophic cardiomyopathy cardiac troponin C mutants
    in myofilaments", J Biol Chem 286:1005-1013, doi:10.1074/jbc.M110.168583.
Prodanovic M, Irving TC, Mijailovich SM (2019), "Estimation of forces on actin
    filaments in living muscle from X-ray diffraction patterns and mechanical
    data", Int J Mol Sci 20:6044, doi:10.3390/ijms20236044.
Reconditi M, Brunello E, Fusi L, Linari M, Piazzesi G, Lombardi V, Irving M
    (2011), "Motion of myosin head domains during activation and force
    development in skeletal muscle", PNAS 108:7236-7240,
    doi:10.1073/pnas.1018330108.
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
Walcott S, Warshaw DM (2010), "Modeling smooth muscle myosin's two heads: long-
    lived enzymatic roles and phosphorylation-dependent equilibria", Biophys J
    99:1129-1138, doi:10.1016/j.bpj.2010.06.018.
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

def tm_rate_01(ca_conc, k_01_base, mod):
    """Rate 0->1: calcium binds troponin C.

    The only step in the whole model that reads calcium concentration directly.
    Because it is second-order (rate proportional to [Ca2+]), the rate constant
    carries units of M^-1 ms^-1 while the rate itself comes out in ms^-1.

    Everything downstream of this — how steeply force rises with calcium, how
    fast a twitch relaxes — is emergent from this step plus cooperativity plus
    the SRX gate. There is no direct pCa-to-force mapping anywhere.

    Args:
        ca_conc: Calcium concentration (M), i.e. 10**(-pCa)
        k_01_base: Second-order rate constant (M^-1 ms^-1) - params.tm_k_01
        mod: Cooperativity multiplier, exp(+h/2) on the Ising path (1.0 = none)

    Returns:
        Rate k_01 (ms^-1)
    """
    return k_01_base * ca_conc * mod


def tm_rate_10(k_01_base, Keq_01, mod=1.0):
    """Rate 1->0: calcium dissociates from troponin C.

    Fixed by detailed balance rather than being an independent parameter:
    k_10 = k_01_base / Keq_01. Note that k_01_base is used here WITHOUT the
    calcium concentration, so Keq_01 has units of M^-1 and the ratio comes out
    as a first-order rate — this is the standard association/dissociation pair,
    not an approximation.

    Args:
        k_01_base: Second-order forward rate constant (M^-1 ms^-1)
        Keq_01: Association equilibrium constant (M^-1)
        mod: Cooperativity multiplier, exp(-h/2) on the Ising path (1.0 = none).
            The reverse of what tm_rate_01 receives, so the pair shifts Keq_01
            by exactly exp(h) and detailed balance survives.

    Returns:
        Rate k_10 (ms^-1)
    """
    return (k_01_base / Keq_01) * mod


def tm_rate_12(k_12_base, mod):
    """Rate 1->2: tropomyosin shifts from blocking to closed.

    The mechanical consequence of calcium binding: troponin's grip on
    tropomyosin loosens and the strand rolls toward the actin groove. Sites are
    still not open for binding after this step — that is 2->3.

    Args:
        k_12_base: Rate constant (ms^-1) - params.tm_k_12.
                   1.0 skeletal / 0.5 cardiac. NO MEASURED RATE SOURCE EXISTS for
                   this step (audit 2026-08-19). The former citation to Geeves &
                   Lehrer 1994 for "20-1000 s^-1" was withdrawn: that range is not
                   in the paper, and the one rate statement it does make
                   (k+T + k-T >> 500 s^-1) is a ONE-SIDED bound on the closed<->open
                   step, i.e. tm_rate_23, not this one. Fraser & Marston 1995 was
                   also withdrawn: it contains no rate constants at all (it is a
                   steady-state motility paper). The EQUILIBRIUM is constrained --
                   McKillop & Geeves 1993 K_B = [closed]/[blocked] = 0.3 without
                   Ca2+ and >=16 with Ca2+ -- see params.tm_Keq_12.
        mod: Cooperativity multiplier, exp(+h/2) on the Ising path (1.0 = none)

    Returns:
        Rate k_12 (ms^-1)
    """
    return k_12_base * mod


def tm_rate_21(k_12_base, Keq_12, mod=1.0):
    """Rate 2->1: tropomyosin rolls back to the blocking position.

    Detailed balance: k_21 = k_12_base / Keq_12.

    Args:
        k_12_base: Forward rate constant (ms^-1)
        Keq_12: Equilibrium constant (dimensionless)
        mod: Cooperativity multiplier, exp(-h/2) on the Ising path (1.0 = none)

    Returns:
        Rate k_21 (ms^-1)
    """
    return (k_12_base / Keq_12) * mod


def tm_rate_23(k_23_base, mod):
    """Rate 2->3: tropomyosin opens, exposing the binding site.

    The step that actually gates myosin. Sites in state 3 are the only ones a
    crossbridge can attach to (see xb_rate_01's permissiveness argument).

    This is the transition Geeves & Lehrer 1994 (Biophys J 67:273 p.277) actually
    measured. They report only that the switch is in RAPID EQUILIBRIUM:
    k+T + k-T >> 500 s^-1, a one-sided bound on the SUM of this rate and its
    reverse. The model's pair (100 + 1000 = 1100 s^-1) satisfies it. Because only
    the sum and the ratio (tm_Keq_23) are constrained, both rates may be scaled
    together without contradicting anything measured -- a genuine free direction.
    Conditions: rabbit skeletal, reconstituted Tm.Tn.actin + S1, pH 7.0, 20 C.

    Args:
        k_23_base: Rate constant (ms^-1) - params.tm_k_23
        mod: Cooperativity multiplier, exp(+h/2) on the Ising path (1.0 = none)

    Returns:
        Rate k_23 (ms^-1)
    """
    return k_23_base * mod


def tm_rate_32(k_23_base, Keq_23, mod=1.0):
    """Rate 3->2: tropomyosin closes again.

    Detailed balance against tm_rate_23. Note Keq_23 < 1 at rest, so the closed
    state is favoured in the absence of calcium — the open state is the
    excursion, not the resting condition.

    There is no crossbridge gate here. A site with a head attached cannot close
    underneath it, but that lock is applied downstream as a probability-vector
    override rather than as a factor on this rate; see the module docstring.

    Args:
        k_23_base: Forward rate constant (ms^-1)
        Keq_23: Equilibrium constant (dimensionless)
        mod: Cooperativity multiplier, exp(-h/2) on the Ising path (1.0 = none)

    Returns:
        Rate k_32 (ms^-1)
    """
    return (k_23_base / Keq_23) * mod


def tm_rate_30(k_30_base):
    """Rate 3->0: calcium dissociates directly from the open state.

    This is what closes the tropomyosin cycle. Rather than retracing 3->2->1->0,
    the model lets an open site drop straight back to Ca-free-and-blocking,
    which makes the four states a cycle rather than a reversible chain.

    Consequences worth being aware of:
      - Relaxation kinetics are governed largely by this rate, not by the
        forward path. For cardiac muscle it is the slowest TM step.
      - No experiment measures a "3->0" transition; this step is a modelling
        device collapsing "Ca2+ leaves" and "Tm re-blocks" into one irreversible
        event. The measurable proxy is the Ca2+ off-rate from an ASSEMBLED thin
        filament: Davis 2007 gives 105 s^-1 cardiac / 85 s^-1 skeletal in
        filament (vs 42.5 / 7.8 s^-1 for isolated troponin), Pinto 2011 gives
        95.8 s^-1 cardiac. Both presets currently sit outside that band -- cardiac
        low (40 s^-1, the isolated-protein value), skeletal high (200 s^-1,
        unsourced) -- and in the OPPOSITE isoform order to the measurements.
      - Because it is one-way, this step is where the TM cycle stops obeying
        detailed balance. That is deliberate, but it means cooperativity must
        NOT be applied here: boosting the cycle-closing step along with the
        forward steps produces runaway anti-cooperative behaviour rather than
        sharper activation. Hence the missing ``mod`` argument — alone among the
        TM rates, this one cannot be modulated, which makes the mistake
        unrepresentable rather than merely warned against.
      - A bound crossbridge prevents the site from deactivating through this
        step, but that lock lives in the probability-vector override downstream,
        not here; see the module docstring.

    Args:
        k_30_base: Rate constant (ms^-1) - params.tm_k_30

    Returns:
        Rate k_30 (ms^-1)
    """
    return k_30_base


# =============================================================================
# CROSSBRIDGE RATE FUNCTIONS
# =============================================================================

def xb_rate_01(permissiveness, r01_coeff, E_weak):
    """Rate 0->1: DRX head attaches weakly to actin.

    The gateway into the force-producing cycle, and the only crossbridge rate
    that tropomyosin gates directly. Two factors multiply:

      permissiveness  1 if the target site's tropomyosin is open (state 3),
                      0 otherwise — a hard gate, so a head cannot bind a
                      covered site at any strain.
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
        permissiveness: 0 or 1, whether the target site is open (from TM state)
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
    exactly zero, which happens for every head facing a covered site
    (permissiveness = 0). Without it those heads would produce log(0) = -inf and
    poison the rate matrix. The floor is numerical housekeeping with no physical
    effect — but for a different reason since 2026-09-01 than the one that used
    to be written here. It is no longer that a head cannot be bound at a covered
    site because the hard lock forbids tropomyosin from closing over one; the
    lock is finite now and tropomyosin does close. It is that closure DETACHES
    every bound head in the same timestep (kernels/transitions.thin_transitions,
    which runs before thick_transitions), so no state-1 head is ever at a
    covered site when this rate is evaluated.

    DO NOT "SIMPLIFY" THIS FUNCTION. Because the rate is analytically constant,
    the log-space form, the floor and the 10000 ms^-1 cap all look removable.
    Removing the floor changes r10 at OPEN sites by a relative 1.6e-5 — small,
    but not bit-identical, and an exact control path is worth more than the
    three lines. It is a good follow-up on its own.

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


def xb_rate_21(r12, U_loose, U_tight_1):
    """Rate 2->1: strong-to-weak reversal (phosphate rebinding).

    Detailed balance: r21 = r12 * exp(U_tight_1 - U_loose). Since the strong
    state sits well below the loose state in free energy, this reversal is
    strongly suppressed under normal conditions — but it becomes significant for
    heads bound at large strain, where U_tight_1's elastic term climbs. That is
    the model's route for a badly-positioned head to back out without completing
    the cycle and paying an ATP.

    Log-space arithmetic, as with r10, to survive the dynamic range in float32.

    Args:
        r12: Forward isomerization rate (ms^-1)
        U_loose: Free energy of the loose state, including strain (kT)
        U_tight_1: Free energy of the tight_1 state, including strain (kT)

    Returns:
        Rate r21 (ms^-1), capped at 10000 ms^-1
    """
    upper = 10000.0
    log_r32 = jnp.log(r12) - (U_loose - U_tight_1)
    r21 = jnp.exp(log_r32)
    return jnp.minimum(r21, upper)


def xb_rate_23(A23, f_strong, delta23, k_t):
    """Rate 2->3: chemical transition between the two strongly-bound states.

    NOT the whole working stroke, despite the state names — most of the stroke
    is on 1 -> 2; see the module docstring. What this step always does is commit
    the head to the ~6 kT drop that makes the stroke effectively irreversible,
    and move it into the only state from which ADP release and detachment are
    possible.

    WHETHER IT ALSO MOVES THE HEAD DEPENDS ON THE PARAMETERS, and the two cases
    are worth separating because the second one is new (2026-09-01).

    Control case, *_tight_1 springs EQUAL to the *_strong ones. Tight_1 and
    Tight_2 then share a spring configuration and a pair of spring constants, so
    the transition produces no displacement, does no work and changes no force.
    The same Bell factor appears in the reverse rate (xb_rate_32), so K_23 is
    load-independent and load never redistributes heads between the pre- and
    post-ADP-release states: the Huxley-Simmons redistribution between attached
    states is absent from THIS leg and lives entirely on 1 -> 2, via E_diff.
    There is no E_diff term here because the elastic energy is unchanged.

    Shipped case, the split stroke. Tight_1 has its own rest configuration, so
    this step carries 2.099 nm of the 8.791 nm axial swing (23.9%), does work,
    and changes force. K_23 = exp(U_tight_1 - U_tight_2) is then strain-
    dependent, because the two configurations differ elastically, and load DOES
    redistribute heads between them. The elastic energy difference still does
    not appear explicitly in this expression — it enters through U_tight_1 in
    xb_rate_32, and the f fed to the Bell factor here is state 2's force
    (f_tight1), not state 3's.

    Bell model: r23 = A23 * exp(-f * delta_23 / kT). A load resisting the head
    (f > 0) slows it. Physically, external load tilts the energy landscape
    against the transition state, and delta_23 is how far along the reaction
    coordinate that transition state sits.

    delta_23 IS NOW COHERENT, which it was not before. A Bell transition-state
    distance on a step with zero displacement had nothing to be a fraction of;
    on a step carrying ~2.10 nm of swing, delta_23 = 1.0 nm is admissible as a
    transition-state distance precisely because it is less than that
    displacement. That is a new argument for the value, not a derivation of it:
    1.0 nm was inherited from the stroke literature and has never been
    re-derived for this step. See the module docstring's caveat before changing
    it, and note xb_delta_34's coupling to it.

    OPEN, AND OUT OF SCOPE HERE: 1 -> 2 and 2 -> 3 now both carry displacement
    but use two different formalisms for the same kind of physics — r12 is
    A12*exp(E_diff/2), an elastic-energy difference with a symmetric barrier,
    while this step is a Bell distance. Do NOT add an xb_delta_12 to "fix" it:
    r12 does not use the Bell formalism at all and already has strain dependence
    through E_diff, so a Bell distance there would either double-count against
    E_diff or require reformulating the step. The question the split exposes is
    genuine and is new physics.

    Args:
        A23: Zero-load rate (ms^-1) - params.xb_r23_coeff.
             [G] The usual citation for this value (Millar & Homsher 1990,
             ~70-100 s^-1) measured k_Pi from caged-phosphate photolysis, i.e.
             Pi release coupled to force generation. In this model that is the
             1 -> 2 step, not this one, so the citation does not apply here.
             Treat as unsourced pending a value for the chemical transition
             between the two strongly-bound states.
        f_strong: Force carried in the strong state (pN); positive = resisting
        delta23: Transition-state distance (nm) - params.xb_delta_23.
                 [G] 1.0 nm. Pate & Cooke 1989 JMRCM 10:181 and Huxley & Simmons
                 1971 Nature 233:533 (1-2 nm) describe the transition state of
                 the LEVER SWING, which in this model is on 1 -> 2. With zero net
                 displacement across 2 -> 3 this parameter has no structural
                 referent — see the module docstring's caveat.
        k_t: Thermal energy kT (pN*nm)

    Returns:
        Rate r23 (ms^-1), capped at 10000 ms^-1

    References:
        Bell 1978 Science 200:618; Piazzesi 2007 Cell 131:784;
        Reconditi 2011 PNAS 108:7236; Walcott 2010 Biophys J 99:1129.
    """
    upper = 10000.0
    r23 = A23 * jnp.exp(-f_strong * delta23 / k_t)
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

    IN THE CONTROL CONFIGURATION both free energies include the SAME elastic
    term, so it cancels IN THE DIFFERENCE. What that makes strain-independent is
    the RATIO, not this rate: K_23 = r23/r32 is fixed at
    exp(-(U_tight_2 - U_tight_1)) everywhere, so load cannot shift the 2/3
    population in either direction. r32 ITSELF is strongly strain-dependent,
    because it inherits r23's exp(-f*delta_23/kT) factor — measured over
    x in [-4, 21.5] nm it varies by ~8.5e5-fold. Load therefore slows both
    directions of this transition equally, which is the anomaly described in the
    module docstring's caveat on delta_23.

    AT THE SHIPPED DEFAULTS the elastic terms differ — U_tight_1 carries
    E_tight1 and U_tight_2 carries E_strong — so the cancellation is only
    partial, K_23 becomes strain-dependent, and load does redistribute heads
    between states 2 and 3. That is the split stroke doing its job; the anomaly
    above is the control-path statement.

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


def xb_rate_34(A34, f_strong, delta34, k_t):
    """Rate 3->4: ADP release and detachment.

    The post-stroke head releases ADP, binds ATP, and lets go of actin. This is
    the rate-limiting step of the cycle for most myosins, and it consumes the
    ATP: state 4 is the model's accounting point for ATP turnover.

    Slip bond: r34 = A34 * exp(+f * delta_34 / kT). Tensile load ACCELERATES
    detachment. The positive sign is not interchangeable with r23's negative one
    — a catch bond here (load suppressing detachment) would make heads cling
    harder the more they resist, which contradicts the fast unloaded shortening
    velocities and rapid tension redevelopment seen in skeletal muscle.

    Args:
        A34: Zero-load detachment rate (ms^-1) - params.xb_r34_coeff.
             [I] skeletal / [L] cardiac. Siemankowski & White 1984 JBC 259:5045
             VERIFY cardiac exactly: "the rate constant for the dissociation of
             ADP from cardiac actomyosin-S1, k_AD, is ~65 s^-1 at 15 C" (bovine
             ventricle, beta-MHC). For skeletal the same paper gives ">500 s^-1"
             at 15 C, but that is a ONE-SIDED BOUND and is quoted from their
             ref (14), not measured there -- hence [I], not [M]. "Cardiac ~10x
             slower" is a 15 C statement only: k_AD extrapolates to ~550 s^-1 at
             38 C (Q10 ~ 2.4).
        f_strong: Force carried in the strong state (pN); positive = tensile
        delta34: Transition-state distance (nm) - params.xb_delta_34
        k_t: Thermal energy kT (pN*nm)

    Returns:
        Rate r34 (ms^-1), capped at 10000 ms^-1

    THE SIGN IS CONTESTED, AND THIS MODEL'S OWN LINEAGE TAKES THE OTHER ONE.
    Pate & Cooke 1989 JMRCM 10:181, Table 2, make ADP release a decreasing
    function of strain — 2 s^-1 for a highly strained force-producing bridge
    (x >= 3.7 nm) rising to 750 s^-1 once it is dragging (x < 0), a 375-fold
    catch bond. VERIFIED EXACTLY against Table 2 (2026-08-19):
    R45(x) = 2 s^-1 (x >= 3.7); 273.3 - 73.3x (1 <= x < 3.7);
             750 - 550x (0 <= x < 1); 750 s^-1 (x < 0).
    Note their own wording makes this an ASSUMPTION, not a measurement: the
    rates "are assumed to rise dramatically as x approaches 0". They state the
    reasoning directly: "we assume that Mg2+ADP release
    is slow for highly strained crossbridges in the A.M.D state, limiting the
    isometric ATPase. For values of x < 0, any crossbridge which remains attached
    produces a force which inhibits filament sliding, decreasing efficiency."
    That is the classical Huxley g(x): hold on while doing work, let go when
    resisting.

    Two single-molecule studies measure the same sign. Sung 2015 Nat Commun 6:7931
    (human beta-cardiac, harmonic force spectroscopy) VERIFIED: "they average to
    k0 = 87 +/- 7 s^-1 and d = 0.8 +/- 0.1 nm (mean +/- s.e.m., N = 7)"; resisting
    load slows detachment. Wang 2024 Small (rabbit native beta-cardiac and slow
    skeletal) VERIFIED on the sign: "both kf and ks responded to increasing
    resistive as well as assistive load by slowing the actomyosin unbinding rates
    ... exhibiting 'catch bond' behavior".
    CORRECTION 2026-08-19: delta = 0.97 nm was previously attributed to Wang here.
    It is NOT their number -- "0.97" appears zero times in that paper; 0.97 nm is
    Greenberg et al.'s porcine value quoted inside Sung. Wang's own fitted
    distances are 1.53 +/- 0.31, 1.53 +/- 0.17 and 1.87 +/- 1.4 nm. So the measured
    Bell distances are 0.8 nm (Sung) and ~1.5-1.9 nm (Wang) -- both above this
    model's 0.5, and Wang further above than previously stated.
    Two caveats on Wang: the catch behaviour holds for ASSISTIVE as well as
    resistive load, and it was measured at 10 uM ATP to isolate a strong-bound
    rigor-like state, far below physiological ATP.

    The likely resolution is that two detachment routes are being conflated: a
    slow, ADP-release-limited exit from the post-stroke state, which is a catch
    bond (Sung 2015; Wang 2024; and Veigel 2005 Nat Cell Biol 7:861 for myosin-V,
    where resisting load slows ADP release -- VERIFIED: "extrapolation would
    predict the detachment kinetics of the front head to slow down 50-fold and the
    kinetics of the rear head to accelerate"; note this is mouse brain myosin-V, a
    processive motor, and the effect is INTRAMOLECULAR strain between two heads,
    not external load on a muscle crossbridge), and a fast premature detachment
    from early force-generating states, which is a slip bond (Woody 2019;
    Caremani 2025 -- neither checked yet). This model already has the second route
    as the strain-gated 3->2->1->0 path, so this rate is the first one.
    REMOVED 2026-08-19: Capitanio 2006 PNAS 103:87 was cited here for the slip
    route. It does not support that. The paper measures a TWO-STEP working stroke
    (~3.4-5.2 nm then ~1.0-1.3 nm) whose second phase "depends linearly on ATP
    concentration"; "catch bond" and "slip bond" appear zero times in it, and
    every occurrence of "load" is in the Introduction discussing other work. Its
    locator (PNAS 103(1):87-92) is correct; its content does not fit the claim.

    RESOLVED 2026-09-01: the default is now NEGATIVE (-0.80 nm), i.e. a catch
    bond, matching Sung's beta-cardiac measurement in this kernel's own sign
    convention. Marang et al. 2025 PNAS 122:e2504758122 is a fifth measurement
    on the same side, in the same Bell form, in rabbit fast skeletal: Fig. 6
    caption, d = -0.89 +/- 0.12 nm at 0 mM added Pi, progressing to +0.34 nm at
    30 mM Pi — so the slip result quoted from that paper's abstract is its
    high-Pi arm, not a contradiction. Provenance and both isoforms:
    core/params.py, xb_delta_34.

    THE SLIP SIGN'S STATED JUSTIFICATION IS FALSIFIED, not merely unsourced. It
    was that a catch bond "contradicts the fast unloaded shortening velocities".
    Tested by length-controlled ramps (S129): V0 is 0.215 nm/ms shipped and
    0.214 nm/ms with the catch bond — identical. V0 in this model is not
    detachment-limited. What the catch bond does change is sub-V0 force, ~3x the
    power output at 0.1 nm/ms.

    IT IS NOT SEPARABLE FROM delta_23, and how strongly they couple depends on
    the MAGNITUDE of this rate's Bell distance, not just its sign. Measured
    (8x8 lattice, pCa 4.5 vs pCa 9 passive reference, z_line = 1100 nm, 3
    replicates), the cost in active force of setting delta_23 = 0:

                                 skeletal    cardiac
        delta_34 = +0.5 (slip)     -78%        -83%
        delta_34 = -0.5 (catch)    -67%        -45%
        delta_34 = -0.9 (catch)    -46%        -19%

    A catch bond takes over delta_23's job of retaining strained force-bearing
    heads, and the deeper the catch the more of that job it absorbs -- but at
    -0.5 it absorbs only part of it. delta_23 was doing real load-dependent work
    in the old (slip) configuration, which is why zeroing it there cost ~80%.
    The table was measured BEFORE the split stroke, on a two-configuration
    model; the direction holds, the percentages have not been re-measured.
    Sweep the two TOGETHER over both sign and magnitude; neither parameter is
    interpretable alone, and a single-sign test will mislead.

    References:
        Bell 1978 Science 200:618 (the functional form); Siemankowski & White 1984
        JBC 259:5045 (the zero-load rates); Prodanovic 2019 J Gen Physiol 151:1013.
    """
    upper = 10000.0
    r34 = A34 * jnp.exp(f_strong * delta34 / k_t)
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
        in that paper's Table 1 (audit 2026-08-19). Nothing in this function is
        measured. Conditions there: rat cardiac trabeculae, 27.2 C.

        k0: Basal rate at zero calcium (ms^-1) - params.xb_srx_k0.
            [G] 0.007 skeletal / 0.005 cardiac. Mijailovich kPS0 = 5 s^-1
            ("Assumed"); cardiac adopts it exactly, skeletal uses 7 s^-1 because
            it is pinned to xb_r05 by construction.
        kmax: Saturating rate at high calcium (ms^-1) - params.xb_srx_kmax.
              [G] 0.4; Mijailovich kPSmax = 400 s^-1, also "Assumed"
              (retagged from [M] 2026-08-19).
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


# =============================================================================
# ENERGY CALCULATIONS (used by rate functions)
# =============================================================================

def compute_xb_energies(r, theta, g_k_weak, g_r_weak, c_k_weak, c_r_weak,
                        g_k_strong, g_r_strong, c_k_strong, c_r_strong, k_t):
    """Elastic energy a crossbridge would store in each spring configuration.

    Evaluates the two-spring head potential (see core/params.py for the geometry)
    at a head's CURRENT position, once for the weak rest configuration and once
    for the strong one:

        E = [ 0.5*g_k*(r - g_rest)^2  +  0.5*c_k*(theta - c_rest)^2 ] / kT

    Both are computed for every head regardless of which state it is actually in,
    because the rates need the comparison: E_weak controls attachment
    (xb_rate_01), and E_diff = E_weak - E_strong controls the weak-to-strong
    isomerization (xb_rate_12). A head sitting where the strong configuration is
    the relaxed one has large positive E_diff and isomerizes readily; a head
    reaching awkwardly does not.

    E_diff is accumulated term by term rather than as E_weak - E_strong. At
    typical geometries the two energies are large and nearly equal, so the naive
    subtraction loses most of its significant digits in float32 — and E_diff then
    goes straight into an exponential, where that error is amplified.

    In state terms: the weak configuration applies to state 1 (Loose) and the
    strong configuration to states 2 (Tight_1) and 3 (Tight_2). States 0, 4 and
    5 are detached; forces.py bears no force for them and no rate reads a free
    energy for state 4, so no spring configuration applies to them at all.

    Because states 2 and 3 share the strong configuration, ONE E_strong serves
    both — which is why the 2 -> 3 transition is mechanically silent. If state 3
    is ever given its own rest configuration (the ~1.5 nm ADP-linked lever swing
    of cardiac myosin; see the module docstring), this function is where the
    third energy would be computed.

    Args:
        r: Head length, sqrt(axial^2 + lattice_spacing^2) (nm) — NOT the radial
           distance alone
        theta: Head angle from the filament axis, atan2(radial, axial) (radians)
        g_k_weak, g_r_weak: Globular linear spring constant and rest length (weak)
        c_k_weak, c_r_weak: Converter angular spring constant and rest angle (weak)
        g_k_strong, g_r_strong: Globular spring, strong configuration
        c_k_strong, c_r_strong: Converter spring, strong configuration
        k_t: Thermal energy kT (pN*nm)

    Returns:
        E_weak: Elastic energy in the weak configuration (kT)
        E_strong: Elastic energy in the strong configuration (kT)
        E_diff: E_weak - E_strong, computed term-wise for float32 precision (kT)
    """
    # Weak configuration — states 1 (Loose) and 4 (Free_2)
    E_weak = (0.5 * g_k_weak * (r - g_r_weak)**2 +
              0.5 * c_k_weak * (theta - c_r_weak)**2) / k_t

    # Strong configuration — states 2 (Tight_1) and 3 (Tight_2)
    E_strong = (0.5 * g_k_strong * (r - g_r_strong)**2 +
                0.5 * c_k_strong * (theta - c_r_strong)**2) / k_t

    # Term-wise difference — avoids catastrophic cancellation, see docstring
    delta_g_energy = 0.5 * (g_k_weak * (r - g_r_weak)**2 - g_k_strong * (r - g_r_strong)**2)
    delta_c_energy = 0.5 * (c_k_weak * (theta - c_r_weak)**2 - c_k_strong * (theta - c_r_strong)**2)
    E_diff = (delta_g_energy + delta_c_energy) / k_t

    return E_weak, E_strong, E_diff
