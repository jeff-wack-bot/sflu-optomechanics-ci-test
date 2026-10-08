"""
Korobko, Ma, Chen & Schnabel, "Quantum expander: precision and bandwidth
enhancement with an internal squeezing", Light Sci. Appl. 8, 118 (2019),
doi:10.1038/s41377-019-0230-2, arXiv:1903.05930.

Mapping onto the graph
----------------------
The paper models a dual-recycled Michelson in resonant sideband extraction
(RSE) by its differential mode, Buonanno & Chen's reduction: one arm cavity
(ITM, ETM) and a signal-extraction (SE) cavity formed by the ITM and the
signal-extraction mirror (SEM), with a chi(2) crystal inside the SE cavity
(supplement Fig. S1 and Eqs. S39-S50). That is
``topologies.signal_recycled(internal_squeezer=True)``, with the paper's SEM
as ``SRM``. The SE path is ``SRM -> SRC.L -> SQZ.to -> SRC.L2 -> ITM`` and
back through ``SQZ.fr``; both squeezer edges are ``edges.SQZEdge``.

* **Crystal.** The paper applies ``S = diag(e^q, e^-q)`` on *each* pass
  (Eqs. S39, S41), ``q`` the single-pass gain, de-amplifying the phase
  (signal) quadrature. An ``SQZEdge`` with ``sqzDB > 0`` de-amplifies the
  quadrature at its angle by ``10^(-sqzDB/20)``, so ``sqzDB = 20 log10(e) q``
  (8.69 dB per unit ``q``) at ``sqzANGdeg = 90`` in both directions -- the
  *same* sign on both edges. See "Crystal orientation" below for why the
  repository's reference internal-squeezing model uses opposite signs and why
  that is a different device.
* **RSE tuning.** The SE cavity is anti-resonant for the signal sidebands at
  DC: a one-way quadrature rotation of pi/2 (the paper's ``phi = pi/2``,
  Sec. S3), put on ``SRC.L`` (the SEM side of the crystal) so that the crystal
  sits in the arm's quadrature frame. ``SRC.L`` and ``SRC.L2`` are each
  ``L_SE / 2`` long.
* **SE-cavity loss** (Fig. S2 only). The paper's loss beamsplitter between
  SEM and crystal (Eq. S39/S41, power loss ``lambda_s`` per pass, vacua
  ``n1`` toward the crystal and ``n2`` toward the SEM) is a mirror ``SEL``
  spliced in between ``SRC.L`` and the crystal, with ``Thr = 1`` and
  ``Lhr = lambda_s`` taken from transmission -- the same device the
  reference model ``sflu.models.coupled_cavity`` uses (``INTSQZL``). Its loss
  inputs are ``SEL.frL.i`` (``n1``) and ``SEL.bkL.i`` (``n2``).
  ``topology()`` adds it to the ``signal_recycled`` graph; with
  ``lambda_s = 0`` it is the identity.
* **Arm.** The ETM carries all the radiation pressure (the paper fixes the
  ITM and doubles the back action on the ETM, Sec. S5). As in KLMTV's
  reduction (see ``sflu.papers.klmtv2001``), the ETM has mass ``m/4``, its
  displacement is ``L h``, and the arm circulates ``P_arm / 2``. That
  reproduces both the paper's ``h_SQL^2 = 8 hbar/(m Omega^2 L^2)`` and the
  shot noise of Sec. S3 with ``|E|^2 = P_arm/(hbar omega0)``, to machine
  precision. The carrier is not injected through the SEM (in RSE it comes in
  through the power-recycling port, which is folded away), so the DC fields
  at the ETM are set directly, in the amplitude quadrature, as
  ``coupled_cavity`` does.
* **GW response.** ``S_h = S_x / (L sinc(Omega L/c))^2``. Eq. S94 prints the
  sinc as a *factor*; it must divide (it is the GW response of an arm, and it
  is what makes every curve in Fig. 3 diverge at the arm FSR, 7.49 kHz).
* **Readout.** Detection loss ``eta`` is the paper's readout beamsplitter
  (Sec. S5 "Detection"), applied after the solve: ``(1 - eta)/eta`` vacuum
  added to the signal-referred noise. Variational readout is an ideal
  frequency-dependent homodyne angle that cancels the amplitude-quadrature
  (back-action) input, the lossless-filter limit of Eq. S95.

Conventions
-----------
**Dark-port frame.** The pi/2 RSE rotation in ``SRC.L`` also rotates the
frame of the light leaving (and entering) the SEM: as solved, the signal
comes out in ``b1``. ``model`` therefore turns the external port back into
the arm's frame, ``O(-pi/2)`` on the way out and on the way in (the
reference model's ``cSEC`` edges do the same). After that, everything is in
this package's convention -- signal in ``b2``, ``zeta = pi/2`` is phase
readout, ``b2 = a2 + K a1`` -- and the paper's ``zeta = pi/2`` (phase
readout) and phase squeezing (``phi_ext = pi/2``) are the same numbers here.
No angle in this paper's figures depends on the sign of ``K``.

**Crystal orientation.** A crystal squeezes a fixed *physical* quadrature,
whichever way the light goes through it. With the squeezer edges adjacent
and the RSE rotation all on one side, both edges see the same frame, so the
physical crystal is the same sign on both: round trip
``S (-r_arm) S`` = gain ``e^{2q}`` per round trip for the amplitude
quadrature and de-amplification for the phase quadrature. This is the
paper's model (Eqs. S39-S41, with ``2 varphi = pi``). The reference
``coupled_cavity`` model puts opposite ``sqzDB`` on ``INTSQZ.armto`` and
``INTSQZ.armfr``, which are likewise adjacent with the pi/2 rotations on the
far side; there the round trip is ``S^-1 (-r_arm) S`` -- no parametric gain at
all, but a squeeze on the way in and the exact inverse on the way out. For a
tuned, lossless SE cavity that only rescales the signal by ``e^q`` and
leaves the noise untouched (``papers/test_korobko2019.py`` shows this). It
corresponds to a crystal displaced by a quarter of the pump's standing-wave
period, not to the quantum expander.

**PSDs** are single-sided, in strain [1/Hz]; the paper's figures plot
``sqrt(S_h)``.

Parameters that are *not stated* in the paper
---------------------------------------------
* The crystal gain of the Fig. 3 expander curves. We use the threshold,
  ``q_th = -ln(R_s)/2 = 0.1077`` (0.935 dB per pass), at which the lossless
  noise touches zero; the curves match Fig. 3 at that value.
* The loss model of Fig. 3: "0.5%, 3%, 10% loss" is taken as *readout* loss
  only (no SE or arm loss, ``T_e = 0``), which matches Fig. 3.
* Fig. S2: 10 dB of phase-squeezed external injection, no radiation
  pressure, and a gain ``q_z`` that puts an exact zero in the input-port
  contribution (as the figure shows). See ``q_input_zero``.
"""
import numpy as np
import scipy.constants as scc

from sflu import edges, elements, readout, solve, topologies
from sflu.lib import MatrixLib

# Table S1, "Baseline GWO" column, and the Fig. 3 / Fig. S2 captions.
PARAMS = dict(
    lambda_m=1550e-9,
    P_arm_W=4e6,        # Table S1 P_arm = P_c/2 (the Fig. 3 caption's "P_c = 4 MW" is P_arm)
    L_m=20e3,           # arm length
    m_kg=200.0,         # each test mass
    L_SE_m=56.0,        # SE cavity length
    T_ITM=0.07,         # power transmissions
    T_SEM=0.35,
    T_ETM=5e-6,
    lambda_s=1500e-6,   # SE cavity loss, single trip (Table S1; Fig. S2)
    eta=0.99,           # detection efficiency (Table S1; Fig. S2)
    sqz_ext_dB=10.0,    # external squeezing e^{2r} (Table S1)
)

# Fig. 3: readout losses of the expander curves (labels "0%, 0.5%, 3%, 10% loss").
FIG3_LOSSES = (0.0, 0.005, 0.03, 0.10)
# Fig. S2 caption: T_ETM raised to 100 ppm "to emphasize the smallness of its influence".
FIGS2_T_ETM = 100e-6

MLIB = MatrixLib(nhom=0)

# O(-pi/2): from the SEM-side frame to the arm frame (see "Dark-port frame").
_TO_ARM_FRAME = np.array([[0.0, 1.0], [-1.0, 0.0]])

IN = "SRM.bk.i.exc"       # the dark port: vacuum or external squeezing enters here
OUT = "SRM.bk.o.tp"
SIGNAL = "ETM.pos.exc"    # x = L h [m]
SE_LOSS = ("SEL.frL.i", "SEL.bkL.i")
ARM_LOSS = "ETM.bk.i"


def omega0(p=PARAMS):
    return 2 * np.pi * scc.c / p["lambda_m"]


def q_threshold(p=PARAMS):
    """Single-pass gain at threshold, ``e^{2q} R_s = 1`` (inferred, not stated).

    It is where the corrected Eq. S18 has a zero (perfect squeezing of the
    output at one frequency) and, equivalently, where the amplitude
    quadrature's sloshing mode reaches oscillation threshold -- the exact
    counterpart of the main text's ``chi -> gamma``.
    """
    return -np.log(1 - p["T_SEM"]) / 4


def q_input_zero(p=PARAMS, lambda_s=None):
    """Gain that puts a zero in the input-port noise with SE loss (Fig. S2).

    Inferred: ``e^{2q} = (1 - lambda_s) / R_s``, the lossy generalisation of
    ``q_threshold`` (the phase-quadrature reflection vanishes when the
    de-amplified round trip, ``e^{-2q} (1-lambda_s)``, equals ``R_s``).
    """
    lam = p["lambda_s"] if lambda_s is None else lambda_s
    return 0.5 * np.log((1 - lam) / np.sqrt(1 - p["T_SEM"]))


def sqz_dB_per_pass(q):
    """``SQZEdge.sqzDB`` for single-pass gain ``q``: ``20 log10(e^q)``."""
    return 20 * np.log10(np.e) * q


def h_SQL(Omega, p=PARAMS):
    """Free-mass SQL of the Michelson, ``sqrt(8 hbar / (m Omega^2 L^2))``."""
    return np.sqrt(8 * scc.hbar / (p["m_kg"] * Omega**2 * p["L_m"]**2))


def sinc_arm(Omega, p=PARAMS):
    w = Omega * p["L_m"] / scc.c
    return np.sin(w) / w


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def omega_s(p=PARAMS):
    """Sloshing frequency, after Eq. 5: ``c sqrt(T_ITM / (4 L_SE L))``."""
    return scc.c * np.sqrt(p["T_ITM"] / (4 * p["L_SE_m"] * p["L_m"]))


def gamma_SE(p=PARAMS):
    """SE coupling rate, after Eq. 5: ``c T_SE / (4 L_SE)``."""
    return scc.c * p["T_SEM"] / (4 * p["L_SE_m"])


def Sh_two_mode(Omega, chi, P_c, p=PARAMS):
    """Eq. 9 (``chi = 0``: Eq. 10), the two-mode approximation, as printed.

    Valid for ``omega_s << c/2L``; here ``omega_s/2pi = 5.96 kHz`` against an
    FSR of 7.49 kHz, so it is a shape guide only. Its prefactor is discussed
    in ``papers/test_korobko2019.py``.
    """
    ws, g = omega_s(p), gamma_SE(p)
    return (scc.hbar * scc.c / (8 * omega0(p) * p["L_m"] * P_c)
            * ((Omega**2 - ws**2)**2 + (g - chi)**2 * Omega**2) / (g * ws**2))


def _S3(Omega, q, p, printed=False):
    """Eqs. S18-S19 (Sec. S3): noise reflection and signal transfer.

    Exact two-cavity model, lossless, no radiation pressure, ``R_e = 1``,
    ``e^{2i phi} = -1``. Returns ``(R_a, |X|)``, with ``X`` the output (in
    units of the vacuum quadrature) per metre of ETM displacement, for
    ``|E|^2 = P_arm / (hbar omega0)``.

    Eq. S18 as printed lacks ``R_s`` on its ``e^{2q}`` term (``printed=True``);
    without it ``|R_a| != 1`` even at ``q = 0`` (1.107 at DC). The corrected
    numerator is the default.
    """
    Ri = np.sqrt(1 - p["T_ITM"])
    Rs = np.sqrt(1 - p["T_SEM"])
    x = np.exp(2j * Omega * p["L_m"] / scc.c)
    y = np.exp(2j * Omega * p["L_SE_m"] / scc.c)
    e2phi = -1
    eq2 = np.exp(2 * q)
    D = eq2 * (x * Ri - 1) + e2phi * y * (x - Ri) * Rs
    Rs_num = 1 if printed else Rs
    Ra = -(e2phi * y * (x - Ri) + eq2 * Rs_num * (x * Ri - 1)) / D
    k = omega0(p) / scc.c
    E = np.sqrt(p["P_arm_W"] / (scc.hbar * omega0(p)))
    X = 2 * k * E * np.exp(q) * np.sqrt(p["T_ITM"] * p["T_SEM"]) / np.abs(D)
    return Ra, X


def R_noise(Omega, q, p=PARAMS, quadrature=2, printed=False):
    """``R_a`` of Eq. S18 for the phase (``2``) or amplitude (``1``) quadrature.

    The amplitude quadrature sees the crystal's gain ``e^{+q}``: Eq. S18 with
    ``q -> -q``.
    """
    return _S3(Omega, q if quadrature == 2 else -q, p, printed)[0]


def Sh_shot(Omega, q, eta=1.0, p=PARAMS):
    """Shot-noise strain PSD: Eqs. S18-S19, readout loss (Eq. S91), Eq. S94.

    ``S_h = [eta |R_a|^2 + 1 - eta] / (eta |X|^2) / (L sinc)^2``, with the
    sinc dividing (see module docstring).
    """
    Ra, X = _S3(Omega, q, p)
    return ((eta * np.abs(Ra)**2 + 1 - eta) / (eta * X**2)
            / (p["L_m"] * sinc_arm(Omega, p))**2)


def Sh_rp(Omega, q, p=PARAMS):
    """Radiation-pressure strain PSD.

    For a lossless, tuned (uncorrelated) system the shot and back-action
    noises are conjugate, ``S_xx S_FF = hbar^2/4`` -- the equality behind the
    QCRB statement of Eqs. 12-14. With the free-mass response of the ETM
    (mass ``m/4``) this gives ``S_h^rp = h_SQL^4 / (4 S_h^shot sinc^4)``,
    ``S_h^shot`` the *lossless* shot noise.
    """
    return h_SQL(Omega, p)**4 / (4 * Sh_shot(Omega, q, 1.0, p) * sinc_arm(Omega, p)**4)


def Sh_phase_readout(Omega, q, eta=1.0, p=PARAMS):
    """Total quantum noise with phase readout, ``S_shot + S_rp`` (Eq. S90)."""
    return Sh_shot(Omega, q, eta, p) + Sh_rp(Omega, q, p)


def Sh_variational(Omega, q, eta=1.0, p=PARAMS):
    """Variational readout that cancels back-action (lossless filter, Eq. S95).

    The homodyne angle removes the amplitude-quadrature input altogether, so
    with ``eta = 1`` only shot noise remains. With readout loss, the tilted
    quadrature costs ``1/sin^2 zeta = 1 + cot^2 zeta``, ``cot zeta`` being the
    ratio of back-action to direct amplitude-quadrature transfer:
    ``S = S_shot(eta) + (1-eta)/eta S_rp / |R_a^(1)|^2``. For the baseline
    (``|R_a^(1)| = 1``) this is the familiar ``S_shot [1/eta + (1-eta) K^2/eta]``.
    """
    Ra1 = R_noise(Omega, q, p, quadrature=1)
    return (Sh_shot(Omega, q, eta, p)
            + (1 - eta) / eta * Sh_rp(Omega, q, p) / np.abs(Ra1)**2)


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def topology():
    """``signal_recycled(internal_squeezer=True)`` with the SE loss ``SEL``
    spliced in between the SEM and the crystal."""
    ifo = topologies.signal_recycled(internal_squeezer=True)
    del ifo.edges[("SQZ.a.i", "SRM.fr.o")]
    del ifo.edges[("SRM.fr.i", "SQZ.a.o")]
    ifo.subgraph_add("SEL", elements.MirrorElement(loss_ports=True),
                     translation_xy=(10, 0), rotation_deg=0)
    ifo.edges.update({
        ("SEL.bk.i", "SRM.fr.o"): "SRC.L",
        ("SQZ.a.i", "SEL.fr.o"): "1",
        ("SEL.fr.i", "SQZ.a.o"): "1",
        ("SRM.fr.i", "SEL.bk.o"): "SRC.L",
    })
    return ifo


def model(F_Hz, q, p=PARAMS, rp=True, T_ETM=0.0, lambda_s=0.0,
          reference_orientation=False, mlib=MLIB):
    """Transfer matrices to the dark port, in the arm frame.

    Parameters
    ----------
    q : float
        Single-pass crystal gain (0: the baseline RSE detector).
    rp : bool
        Radiation pressure on the ETM.
    T_ETM, lambda_s : float
        End-mirror transmission and SE loss per pass.
    reference_orientation : bool
        Opposite ``sqzDB`` on the two squeezer edges, as in
        ``sflu.models.coupled_cavity`` -- not the paper's device; for the
        comparison in the example only.

    Inputs: ``IN`` (dark port), ``SIGNAL``, ``ARM_LOSS``, and ``SE_LOSS``.
    """
    L = p["L_m"]
    M = p["m_kg"] / 4
    suscept = ((lambda F: -1 / (M * (2 * np.pi * F)**2)) if rp
               else (lambda F: np.zeros_like(F)))
    dB = sqz_dB_per_pass(q)
    edge_objs = [
        edges.MirrorEdge("SRM", Thr=p["T_SEM"], mlib=mlib),
        edges.MirrorEdge("ITM", Thr=p["T_ITM"], mlib=mlib),
        edges.RPMirrorEdge("ETM", Thr=T_ETM, suscept=suscept,
                           lambda_m=p["lambda_m"], mlib=mlib),
        edges.MirrorEdge("SEL", Thr=1, Lhr=lambda_s, loss_in_transmission=True,
                         loss_ports=True, mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
        # RSE: one-way pi/2, all of it on the SEM side of the crystal
        edges.LinkEdge("SRC.L", L_m=p["L_SE_m"] / 2, detune_rad=np.pi / 2, mlib=mlib),
        edges.LinkEdge("SRC.L2", L_m=p["L_SE_m"] / 2, mlib=mlib),
        edges.SQZEdge("SQZ.to", sqzDB=dB, sqzANGdeg=90, mlib=mlib),
        edges.SQZEdge("SQZ.fr", sqzDB=-dB if reference_orientation else dB,
                      sqzANGdeg=90, mlib=mlib),
    ]
    # Carrier at the ETM: P_arm/2 in the amplitude quadrature (see docstring).
    E = np.sqrt(p["P_arm_W"] / 2) * mlib.LO(np.pi / 2)
    etm = edge_objs[2]
    dc = {"ETM.fr.i.tp": E, "ETM.fr.o.tp": -etm.r @ E, "ETM.bk.o.tp": etm.t @ E}
    T = solve.solve_ac(
        solve.build(topology()), edge_objs, mlib, F_Hz,
        readout=OUT, inputs={IN, SIGNAL, ARM_LOSS, *SE_LOSS}, resultsDC=dc,
    )
    R = _TO_ARM_FRAME
    return {k: (R @ v @ R if k == IN else R @ v) for k, v in T.items()}


def ba_evading_angle(T):
    """Homodyne angle that cancels the dark-port amplitude quadrature.

    ``cos zeta M11 + sin zeta M21 = 0`` for the input matrix ``M``; for the
    baseline that is ``cot zeta = -K``. ``M21/M11`` is real (tuned cavities)
    up to rounding.
    """
    M = T[IN]
    return np.arctan2(1, -(M[..., 1, 0] / M[..., 0, 0]).real)


def Sh(T, F_Hz, zeta=np.pi / 2, eta=1.0, states=None, p=PARAMS, mlib=MLIB):
    """Strain PSD [1/Hz] at homodyne angle ``zeta`` with readout efficiency
    ``eta``. Returns ``(total, budget)``; the budget has an extra
    ``"readout"`` entry for the detection loss."""
    row = readout.quadrature(mlib, zeta)
    Sx, budget = readout.referred_psd(T, row, SIGNAL, states=states,
                                      lambda_m=p["lambda_m"])
    G2 = np.abs(readout.response(T, row, SIGNAL))**2
    budget["readout"] = (1 - eta) / eta * readout.vacuum_psd(p["lambda_m"]) / G2
    conv = (p["L_m"] * sinc_arm(2 * np.pi * np.asarray(F_Hz), p))**2
    budget = {k: v / conv for k, v in budget.items()}
    return Sx / conv + budget["readout"], budget
