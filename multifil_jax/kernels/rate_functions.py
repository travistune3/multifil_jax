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
locked in state 3 (both 3->2 and 3->0 are gated off) — the bound head physically
holds tropomyosin out of the way, which is the structural basis for crossbridge
binding being self-reinforcing.

THE CROSSBRIDGE CYCLE (6 states)
--------------------------------
    0  DRX      disordered relaxed; head is free, ATP hydrolysed, ready to bind
    1  Loose    weakly bound to actin, pre-power-stroke
    2  Tight_1  strongly bound, post-stroke, phosphate released
    3  Tight_2  strongly bound, ADP-release-competent
    4  Free_2   just detached after ADP release; ATP has bound
    5  SRX      super-relaxed; head folded back against the thick filament

The force-producing path is 0 -> 1 -> 2 -> 3 -> 4 -> 0, consuming one ATP per
lap. State 1 uses the weak spring configuration and states 2 and 3 use the
strong one; 0, 4 and 5 are detached and carry no spring at all.

WHERE THE WORKING STROKE ACTUALLY IS
------------------------------------
On 1 -> 2. NOT on 2 -> 3, despite the Tight_1/Tight_2 naming.

There are only two spring configurations in the model, weak and strong, so the
entire rest-position change — and therefore all of the work — happens when a
head isomerizes from Loose to Tight_1. States 2 and 3 share one rest
configuration AND one pair of spring constants (see forces.py, where
`is_strong = (state == 2) | (state == 3)` selects a single parameter set). The
2 -> 3 transition therefore moves nothing, does no work, and changes no force.
It is a purely chemical step. What separates the two states is the ~6 kT free
energy drop that makes the stroke effectively one-way, and the fact that only
state 3 can release ADP and detach.

This is a deliberate design, not an oversight. Fibre mechanics and X-ray
modelling both argue that ONE force-bearing conformation plus one low-force
attached conformation is sufficient: Knupp & Squire 2020 (Biology 9:464) fit
length-step transients, isotonic shortening and the M3 reflection with exactly
that, and could not find parameters for variants carrying two force-producing
attached states; Eakins 2016 (Biology 5:41) concludes from X-ray that "no more
than two main attached structural states are necessary and sufficient", with
the weak state contributing ~4% of tension. In that mapping state 1 is the
low-force attached state and states 2+3 together are the single force-bearing
one.

The known omission is the second, much smaller lever swing that accompanies ADP
release: ~16 degrees / ~1.5 nm in cardiac myosin (Doran 2023 JGP 155:e202213267
by cryo-EM; Woody 2019 eLife 8:e49266 measures 1-1.5 nm by single molecule),
essentially absent in fast skeletal myosin (Gollub, Cremo & Cooke 1996 Nat
Struct Biol 3:796). Its structural role is argued to be strain sensing rather
than force generation, and in this model it appears only as the Bell distance
of the detachment step (xb_delta_34), not as a displacement. Note the dissenting
view: Offer & Ranatunga 2013 (Biophys J 105:1767) require TWO tension-generating
steps of 5.6 and 4.6 nm and reject single-step models on efficiency and on
lengthening force-velocity.

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
Each function takes explicit scalar arguments rather than a parameter object, so
dependencies are visible in the signature. To change a rate law, either sweep
the DynamicParams values these read, or write a replacement with the same
signature.

Naming: ``tm_rate_XY`` / ``xb_rate_XY`` is the rate for state X -> state Y, with
state indices as listed above.
"""
import jax.numpy as jnp


# =============================================================================
# TROPOMYOSIN RATE FUNCTIONS
# =============================================================================

def tm_rate_01(ca_conc, k_01_base, coop_factor):
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
        coop_factor: Cooperativity multiplier (1.0 when not cooperative)

    Returns:
        Rate k_01 (ms^-1)
    """
    return k_01_base * ca_conc * coop_factor


def tm_rate_10(k_01_base, Keq_01):
    """Rate 1->0: calcium dissociates from troponin C.

    Fixed by detailed balance rather than being an independent parameter:
    k_10 = k_01_base / Keq_01. Note that k_01_base is used here WITHOUT the
    calcium concentration, so Keq_01 has units of M^-1 and the ratio comes out
    as a first-order rate — this is the standard association/dissociation pair,
    not an approximation.

    Args:
        k_01_base: Second-order forward rate constant (M^-1 ms^-1)
        Keq_01: Association equilibrium constant (M^-1)

    Returns:
        Rate k_10 (ms^-1)
    """
    return k_01_base / Keq_01


def tm_rate_12(k_12_base, coop_factor):
    """Rate 1->2: tropomyosin shifts from blocking to closed.

    The mechanical consequence of calcium binding: troponin's grip on
    tropomyosin loosens and the strand rolls toward the actin groove. Sites are
    still not open for binding after this step — that is 2->3.

    Args:
        k_12_base: Rate constant (ms^-1) - params.tm_k_12.
                   1.0 skeletal / 0.5 cardiac; Fraser & Bhatt 2019;
                   Geeves & Lehrer 1994 report 20-1000 s^-1 for this class of
                   transition, a range wide enough that the choice within it is
                   effectively a modelling decision.
        coop_factor: Cooperativity multiplier

    Returns:
        Rate k_12 (ms^-1)
    """
    return k_12_base * coop_factor


def tm_rate_21(k_12_base, Keq_12):
    """Rate 2->1: tropomyosin rolls back to the blocking position.

    Detailed balance: k_21 = k_12_base / Keq_12.

    Args:
        k_12_base: Forward rate constant (ms^-1)
        Keq_12: Equilibrium constant (dimensionless)

    Returns:
        Rate k_21 (ms^-1)
    """
    return k_12_base / Keq_12


def tm_rate_23(k_23_base, coop_factor):
    """Rate 2->3: tropomyosin opens, exposing the binding site.

    The step that actually gates myosin. Sites in state 3 are the only ones a
    crossbridge can attach to (see xb_rate_01's permissiveness argument).

    Args:
        k_23_base: Rate constant (ms^-1) - params.tm_k_23
        coop_factor: Cooperativity multiplier

    Returns:
        Rate k_23 (ms^-1)
    """
    return k_23_base * coop_factor


def tm_rate_32(k_23_base, Keq_23, can_leave):
    """Rate 3->2: tropomyosin closes again.

    Detailed balance, times a gate. ``can_leave`` is 0 when a crossbridge is
    attached at this site: an attached head physically obstructs tropomyosin's
    return, so the site cannot close underneath it. This is a hard lock, not a
    slowdown, and it is one of the two ways binding reinforces itself (the other
    being cooperative coupling to neighbours).

    Note Keq_23 < 1 at rest, so the closed state is favoured in the absence of
    calcium — the open state is the excursion, not the resting condition.

    Args:
        k_23_base: Forward rate constant (ms^-1)
        Keq_23: Equilibrium constant (dimensionless)
        can_leave: 1.0 if the site is free, 0.0 if a crossbridge is bound

    Returns:
        Rate k_32 (ms^-1)
    """
    return (k_23_base / Keq_23) * can_leave


def tm_rate_30(k_30_base, can_leave):
    """Rate 3->0: calcium dissociates directly from the open state.

    This is what closes the tropomyosin cycle. Rather than retracing 3->2->1->0,
    the model lets an open site drop straight back to Ca-free-and-blocking,
    which makes the four states a cycle rather than a reversible chain.

    Consequences worth being aware of:
      - Relaxation kinetics are governed largely by this rate, not by the
        forward path. For cardiac muscle it is the slowest TM step.
      - Because it is one-way, this step is where the TM cycle stops obeying
        detailed balance. That is deliberate, but it means cooperativity must
        NOT be applied here: boosting the cycle-closing step along with the
        forward steps produces runaway anti-cooperative behaviour rather than
        sharper activation. The Ising path leaves this rate at its base value
        for exactly that reason.
      - ``can_leave`` gates it to 0 while a crossbridge is attached, same as
        tm_rate_32: a bound head prevents the site from deactivating.

    Args:
        k_30_base: Rate constant (ms^-1) - params.tm_k_30
        can_leave: 1.0 if the site is free, 0.0 if a crossbridge is bound

    Returns:
        Rate k_30 (ms^-1)
    """
    return k_30_base * can_leave


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
    Because U_loose carries the head's elastic energy, a badly strained weak
    bridge detaches quickly — weak binding is only stable near the head's
    unstrained geometry.

    Computed in log space, since r01 spans many orders of magnitude across the
    strain range and the naive product overflows float32.

    The +0.005 floor inside the logarithm keeps the rate finite where r01 is
    exactly zero, which happens for every head facing a covered site
    (permissiveness = 0). Without it those heads would produce log(0) = -inf and
    poison the rate matrix. The floor is numerical housekeeping — it puts a small
    non-zero detachment rate on heads that are not attached in the first place,
    where it has no physical effect.

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

    NOT the working stroke, despite the state names. The stroke is on 1 -> 2 —
    see the module docstring. Tight_1 and Tight_2 share one spring configuration
    and one pair of spring constants, so this transition produces no
    displacement, does no work, and changes no force. What it does is commit the
    head to the ~6 kT drop that makes the stroke effectively irreversible, and
    move it into the only state from which ADP release and detachment are
    possible.

    Bell model: r23 = A23 * exp(-f * delta_23 / kT). A load resisting the head
    (f > 0) slows it. Physically, external load tilts the energy landscape
    against the transition state, and delta_23 is how far along the reaction
    coordinate that transition state sits.

    Because there is no net displacement, the same factor appears in the reverse
    rate (xb_rate_32), so K_23 is load-independent and load never redistributes
    heads between the pre- and post-ADP-release states. The consequence is that
    the Huxley-Simmons load-dependent redistribution between attached states is
    absent from THIS leg; in this model it lives on 1 -> 2, via E_diff. See the
    module docstring's caveat on delta_23 before changing this parameter.

    Unlike r12, there is no E_diff term here — the elastic energy is unchanged by
    the transition, so only the load term matters.

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

    Not a reverse working stroke — nothing moved on the forward step (see
    xb_rate_23).

    Detailed balance: r32 = r23 * exp(U_tight_2 - U_tight_1). At the default free
    energies (U_tight_1 = -15 kT, U_tight_2 = -21 kT) the 6 kT drop gives
    r32 ~ exp(-6) ~ 0.0025 * r23, so the step is effectively one-way. That
    asymmetry is what keeps heads in the ADP-release-competent state long enough
    to bear load and complete the cycle, rather than rattling back and forth.

    Both free energies include the same elastic term (Tight_1 and Tight_2 share a
    spring configuration), so it cancels IN THE DIFFERENCE. What that makes
    strain-independent is the RATIO, not this rate: K_23 = r23/r32 is fixed at
    exp(-(U_tight_2 - U_tight_1)) everywhere, so load cannot shift the 2/3
    population in either direction. r32 ITSELF is strongly strain-dependent,
    because it inherits r23's exp(-f*delta_23/kT) factor — measured over
    x in [-4, 21.5] nm it varies by ~8.5e5-fold. Load therefore slows both
    directions of this transition equally, which is the anomaly described in the
    module docstring's caveat on delta_23.

    Args:
        r23: Forward working-stroke rate (ms^-1)
        U_tight_1: Free energy of tight_1, including strain (kT)
        U_tight_2: Free energy of tight_2, including strain (kT)

    Returns:
        Rate r32 (ms^-1), capped at 10000 ms^-1

    References:
        Hill 1977 Free Energy Transduction in Biology;
        Pate & Cooke 1989 J Muscle Res Cell Motil 10:181.
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
             [M] >=500 s^-1 at near-zero load for fast skeletal myosin
             (Siemankowski & White 1984 JBC 259:5045); cardiac is ~10x slower.
        f_strong: Force carried in the strong state (pN); positive = tensile
        delta34: Transition-state distance (nm) - params.xb_delta_34
        k_t: Thermal energy kT (pN*nm)

    Returns:
        Rate r34 (ms^-1), capped at 10000 ms^-1

    References:
        Bell 1978 Science 200:618; Veigel 2005 Nat Cell Biol 7:861;
        Capitanio 2006 PNAS 103:87; Prodanovic 2019 J Gen Physiol 151:1013.
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
                  [M] 0.01; Mijailovich 2020 PMC7852458 reports k_-H = 10 s^-1.
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
        k0: Basal rate at zero calcium (ms^-1) - params.xb_srx_k0.
            [I] 0.007 skeletal / 0.005 cardiac; Mijailovich 2020 PMC7852458
            reports kPS0 = 5 s^-1.
        kmax: Saturating rate at high calcium (ms^-1) - params.xb_srx_kmax.
              [M] 0.4; Mijailovich 2020 kPSmax = 400 s^-1.
        b: Hill exponent - params.xb_srx_b.
           [M] 5.0; Mijailovich 2020; consistent with the sharp myosin
           recruitment in Linari 2015 Nature 528:276.
        ca50: Calcium for half-maximal recruitment (M) - params.xb_srx_ca50.
              [G] 1e-6 (pCa 6) is a round number in the physiological range,
              not a measured value for this transition.

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
                  107:430); cardiac 0.2, matching Mijailovich 2020's
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
