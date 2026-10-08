"""
Buonanno & Chen, "Quantum noise in second generation, signal-recycled laser
interferometric gravitational-wave detectors", PRD 64 042006 (2001),
arXiv:gr-qc/0102012 (v2).

Mapping onto the graph
----------------------
Buonanno & Chen (BC) add a signal-recycling mirror (SRM) to the dark port of
KLMTV's Michelson with Fabry-Perot arms. The differential mode is a single
arm closed by the SRM, which is ``topologies.signal_recycled()``: SRM (input
and output), a short signal-recycling cavity ``SRC.L``, the ITM, the arm
``ARM.L`` and a free radiation-pressure ETM.

* **Arm.** As in ``sflu.papers.klmtv2001``: the end mirror has mass ``m/4``
  (BC Eq. 3.12, ``R_xx = -4/(m Omega^2)``), its displacement is ``x = L h``,
  ``S_h = S_x / L^2``, the circulating power is set so that ``K`` equals
  Eq. 2.13 exactly, and the ITM transmission puts the exact cavity pole at
  ``gamma``. ``gamma = 2 pi 100 Hz`` is used as quoted; BC's ``T = 0.033``
  would give ``2 pi 98.5 Hz``, which only rescales the frequency axis.
* **Carrier.** In the real interferometer the carrier reaches the arm from the
  beamsplitter, not through the SRM. The DC fields are therefore solved on a
  bare ``fp_arm()`` with the same ITM/ETM edges (carrier in the amplitude
  quadrature) and handed to the signal-recycled AC solve.
* **SRM.** ``MirrorEdge("SRM", Thr=tau^2)``, so its amplitude reflectivity is
  ``rho``. The package's mirror has ``-r`` on the front (cavity) face and
  ``+r`` outside, BC's (Eqs. 2.18-2.19) the reverse; the difference is
  absorbed by the arm, which on resonance reflects with ``-1`` from the ITM
  side here and ``+1`` in BC's Eq. 2.11.
* **SR detuning.** BC's ``phi`` is the one-way carrier phase in the SRC
  (``phi = pi/2`` is tuned RSE, ``phi = 0`` is extreme signal recycling). The
  ``SRC.L`` link gets ``detune_rad = -phi`` (``sflu_detune``). Equivalently,
  positive ``phi`` is a positive ``detune_Hz`` in
  ``topologies.detuning_rad``. The SRC has zero length, as BC's
  ``Phi = Omega l / c = 0``.
* **Losses (Sec. V).** Arm loss: the ETM gets ``Lhr`` equal to the round-trip
  loss ``eps T / 2``, with ``T = 4 L gamma / c`` -- BC's ``eps = 2 L_arm / T``
  is defined with the first-order ``T``; the exact ``T`` would put ``O(T)`` =
  1.7% into the arm-loss noise. SRC loss: BC's Eq. 5.4 attenuates the field
  leaving the SRM toward the ITM by ``sqrt(1 - lambda_SR)`` and adds vacuum
  ``p``; the topology is modified so that this direction runs through its own
  link ``SRC.Lin`` with a vacuum input ``SRC.p.exc`` (``lossy_topology``).
  Detection loss (Eq. 5.5) is applied to the solved transfer matrices
  (``with_detection_loss``), with vacuum input ``"PD"``. The input power
  ``I_o`` is held fixed, so arm loss lowers the circulating power as in BC.

Conventions
-----------
Two-photon quadratures as in KLMTV and this package: 1 is amplitude, 2 is
phase. As in KLMTV, the ponderomotive term enters BC's relations with the
opposite sign to this package's (``b2 = a2 + K a1`` here), so the phase
quadrature is flipped.

* **Homodyne.** BC Eq. 2.26 reads ``b1 sin(zeta) + b2 cos(zeta)``: ``zeta = 0``
  is the *phase* quadrature. With the phase flip that is this package's
  quadrature at ``zeta - pi/2`` (``sflu_zeta``).
* **Detuning.** Flipping the phase quadrature reverses every rotation, so
  BC's SRC rotation ``R(phi)`` (Eqs. 2.16-2.17) is the link rotation
  ``-phi``.

Both mappings were found by trying all sign/offset combinations against
Eq. 3.5 (only this one agrees, to 1e-3, for every ``zeta``) and are confirmed
independently by the closed-system resonances of Eq. 4.4: with
``0 < phi < pi/2`` the model shows an optical spring, two real resonances at
the Eq. 4.4 frequencies; ``detune_rad = +phi`` gives an anti-spring.

PSDs are single-sided (BC Eq. 3.3), as in this package.

Paper errors found
------------------
Eq. 5.12, ``N21``: the factor ``cos(beta)`` appears twice. At small loss
(where Eq. 5.13's first-order expansion is exact) the arm-loss noise of the
SFLU model agrees with the ``N`` terms to <1e-3 at ``zeta = 0`` and
``pi/3`` with the corrected form, and is off by 74% and 13% with the printed
one (``Sh_lossy_terms(..., n21="printed")``), independent of the loss level.
In Fig. 8 it changes ``sqrt(S_h)`` at ``zeta = 0`` by up to 0.6%, near the
optical resonance (1.8 gamma); there the model's total sides with the
corrected form (0.17% against 0.44%).
"""
import numpy as np
import scipy.constants as scc

from sflu import edges, readout, solve, topologies
from sflu.lib import MatrixLib, Vnorm_sq

# Table I unless noted.
PARAMS = dict(
    omega0=1.8e15,          # carrier [rad/s]
    L_m=4e3,                # arm length
    m_kg=30.0,              # each of the four test masses
    gamma=2 * np.pi * 100,  # arm half-bandwidth [rad/s], quoted as 2 pi 100
    rho=0.9,                # SRM amplitude reflectivity (Figs. 2, 3, 8)
    phi=np.pi / 2 - 0.47,   # SR detuning, one-way SRC phase (Figs. 2, 3, 8)
    eps=0.01,               # arm loss 2 L_arm / T, Sec. V
    lambda_SR=0.02,         # SRC loss per round trip, Sec. V
    lambda_PD=0.1,          # photodetection loss, Sec. V
)

MLIB = MatrixLib(nhom=0)


def lambda_m(p=PARAMS):
    return 2 * np.pi * scc.c / p["omega0"]


def I_SQL(p=PARAMS):
    """Eq. 2.14; 1.04e4 W with Table I values."""
    return p["m_kg"] * p["L_m"]**2 * p["gamma"]**4 / (4 * p["omega0"])


def h_SQL(Omega, p=PARAMS):
    """Eq. 2.12, single-sided."""
    return np.sqrt(8 * scc.hbar / (p["m_kg"] * Omega**2 * p["L_m"]**2))


def sflu_zeta(bc_zeta):
    """A BC homodyne angle (Eq. 2.26), in this package's convention."""
    return np.asarray(bc_zeta) - np.pi / 2


def sflu_detune(bc_phi):
    """BC's SR detuning ``phi`` as the ``SRC.L`` link's ``detune_rad``."""
    return -np.asarray(bc_phi)


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def kappa(Omega, Io_over_Isql, p=PARAMS):
    """Eq. 2.13."""
    g = p["gamma"]
    return Io_over_Isql * 2 * g**4 / (Omega**2 * (g**2 + Omega**2))


def beta(Omega, p=PARAMS):
    """``arctan(Omega / gamma)``, below Eq. 2.11."""
    return np.arctan(Omega / p["gamma"])


def io_relation(Omega, rho, phi, Io_over_Isql, p=PARAMS):
    """Eqs. 2.21-2.24 with ``Phi = 0``: ``M``, ``C`` (2x2) and ``D`` (2,)."""
    K = kappa(Omega, Io_over_Isql, p)
    b = beta(Omega, p)
    tau2 = 1 - rho**2
    e2b = np.exp(2j * b)
    X = np.cos(2 * phi) + K / 2 * np.sin(2 * phi)
    M = 1 + rho**2 * e2b**2 - 2 * rho * e2b * X
    C11 = (1 + rho**2) * X - 2 * rho * np.cos(2 * b)
    C12 = -tau2 * (np.sin(2 * phi) + K * np.sin(phi)**2)
    C21 = tau2 * (np.sin(2 * phi) - K * np.cos(phi)**2)
    C = np.array([[C11, C12], [C21, C11]])
    D = np.array([-(1 + rho * e2b) * np.sin(phi),
                  -(-1 + rho * e2b) * np.cos(phi)])
    return M, C, D


def Sh(Omega, rho, phi, zeta, Io_over_Isql, p=PARAMS):
    """Eq. 3.5: lossless noise at BC homodyne angle ``zeta``."""
    K = kappa(Omega, Io_over_Isql, p)
    _, C, D = io_relation(Omega, rho, phi, Io_over_Isql, p)
    s, c = np.sin(zeta), np.cos(zeta)
    num = (C[0, 0] * s + C[1, 0] * c)**2 + (C[0, 1] * s + C[1, 1] * c)**2
    den = (1 - rho**2) * np.abs(D[0] * s + D[1] * c)**2
    return h_SQL(Omega, p)**2 / (2 * K) * num / den


def Sh_conventional(Omega, zeta, Io_over_Isql, p=PARAMS):
    """Eq. 3.7: no SRM; ``zeta = 0`` is Eq. 3.8, the "conventional" curves."""
    K = kappa(Omega, Io_over_Isql, p)
    return h_SQL(Omega, p)**2 / (2 * K) * (1 + (np.tan(zeta) - K)**2)


def Sh_ESR(Omega, rho, Io_over_Isql, p=PARAMS):
    """Eqs. 3.25-3.26: extreme signal recycling, ``phi = 0``."""
    K = kappa(Omega, Io_over_Isql, p)
    Kt = K * (1 - rho**2) / (1 + rho**2 - 2 * rho * np.cos(2 * beta(Omega, p)))
    return h_SQL(Omega, p)**2 / 2 * (1 / Kt + Kt)


def Sh_ERSE(Omega, rho, Io_over_Isql, p=PARAMS):
    """Eqs. 3.29-3.30: extreme (tuned) RSE, ``phi = pi/2``."""
    K = kappa(Omega, Io_over_Isql, p)
    Kb = K * (1 - rho**2) / (1 + rho**2 + 2 * rho * np.cos(2 * beta(Omega, p)))
    return h_SQL(Omega, p)**2 / 2 * (1 / Kb + Kb)


def resonances(phi, Io_over_Isql):
    """Eq. 4.4: the two closed-system (``rho = 1``) resonances, ``Omega/gamma``.

    Returns the (possibly complex) pair ``(optical spring, optical)``.
    """
    t = np.tan(phi)
    root = np.sqrt(complex(t**4 - 4 * Io_over_Isql * t))
    return tuple(np.sqrt((t**2 + s * root) / 2) for s in (-1, +1))


def Sh_lossy_terms(Omega, rho, phi, zeta, Io_over_Isql, eps, lambda_SR,
                   lambda_PD, n21="corrected", p=PARAMS):
    """Eqs. 5.7-5.13, split by noise source.

    Returns ``{"a": ..., "p": ..., "q": ..., "n": ...}``, the contributions of
    dark-port vacuum (``C^L``), SRC loss (``P``), detection loss (``Q``) and arm
    loss (``N``) to ``S_h``; their sum is Eq. 5.13. ``n21="printed"`` uses
    Eq. 5.12's ``N21`` as printed, with ``cos(beta)`` twice; ``"corrected"``
    drops the second one (see the module docstring).
    """
    K = kappa(Omega, Io_over_Isql, p)
    b = beta(Omega, p)
    tau = np.sqrt(1 - rho**2)
    eb, e2 = np.exp(1j * b), np.exp(2j * b)
    cb = np.cos(b)
    sp, cp = np.sin(phi), np.cos(phi)
    s2p, c2p = np.sin(2 * phi), np.cos(2 * phi)
    X = c2p + K / 2 * s2p
    a = np.sqrt(1 - lambda_PD)
    lsr = lambda_SR

    # Eq. 5.8
    C11 = a * ((1 + rho**2) * X - 2 * rho * np.cos(2 * b)
               - eps / 4 * (-2 * (1 + e2)**2 * rho + 4 * (1 + rho**2) * cb**2 * c2p
                            + (3 + e2) * K * (1 + rho**2) * s2p)
               + lsr * (e2 * rho - (1 + rho**2) * X / 2))
    C12 = a * tau**2 * (-(s2p + K * sp**2)
                        + eps / 2 * sp * ((3 + e2) * K * sp + 4 * cb**2 * cp)
                        + lsr / 2 * (s2p + K * sp**2))
    C21 = a * tau**2 * ((s2p - K * cp**2)
                        + eps / 2 * cp * ((3 + e2) * K * cp - 4 * cb**2 * sp)
                        + lsr / 2 * (-s2p + K * cp**2))
    # Eq. 5.9
    D1 = a * sp * (-(1 + rho * e2)
                   + eps / 4 * (3 + rho + 2 * rho * e2**2 + e2 * (1 + 5 * rho))
                   + lsr / 2 * e2 * rho)
    D2 = a * cp * (-(-1 + rho * e2)
                   + eps / 4 * (-3 + rho + 2 * rho * e2**2 + e2 * (-1 + 5 * rho))
                   + lsr / 2 * e2 * rho)
    # Eq. 5.10
    q = a * np.sqrt(lsr) * tau
    P11 = q / 2 * (-2 * rho * e2 + 2 * c2p + K * s2p)
    P12 = -q * sp * (2 * cp + K * sp)
    P21 = q * cp * (2 * sp - K * cp)
    # Eq. 5.11
    Q11 = np.sqrt(lambda_PD) * (
        1 / e2 + rho**2 * e2 - rho * (2 * c2p + K * s2p)
        + eps / 2 * rho * (c2p / e2 + e2 * (-2 * rho - 2 * rho * np.cos(2 * b) + c2p + K * s2p)
                           + 2 * c2p + 3 * K * s2p)
        - lsr / 2 * rho * (2 * rho * e2 - 2 * c2p - K * s2p))
    # Eq. 5.12
    n = a * tau
    N11 = n * np.sqrt(eps / 2) * (K * (1 + rho * e2) * sp
                                  + 2 * cb * (cp / eb - rho * eb * (cp + K * sp)))
    N22 = -n * np.sqrt(2 * eps) * (-1 / eb + rho * eb) * cb * cp
    N12 = -n * np.sqrt(2 * eps) * (1 / eb + rho * eb) * cb * sp
    extra = {"corrected": 1, "printed": cb}[n21]
    N21 = n * np.sqrt(eps / 2) * (-K * (1 + rho) * cp
                                  + 2 * cb * (1 / eb + rho * eb) * extra * sp)

    # Eq. 5.13
    s, c = np.sin(zeta), np.cos(zeta)
    pre = h_SQL(Omega, p)**2 / (2 * K * tau**2 * np.abs(D1 * s + D2 * c)**2)

    def pair(A11, A12, A21, A22):
        return pre * (np.abs(A11 * s + A21 * c)**2 + np.abs(A12 * s + A22 * c)**2)

    return {
        "a": pair(C11, C12, C21, C11),
        "p": pair(P11, P12, P21, P11),
        "q": pair(Q11, 0, 0, Q11),
        "n": pair(N11, N12, N21, N22),
    }


def Sh_lossy(*args, **kwargs):
    """Eq. 5.13; arguments as ``Sh_lossy_terms``."""
    return sum(Sh_lossy_terms(*args, **kwargs).values())


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

class SRCLossEdge:
    """The SRM -> ITM leg of a lossy SRC (Eq. 5.4): the link ``SRC.Lin``,
    attenuated by ``sqrt(1 - lambda_SR)``, and the loss vacuum ``SRC.pl``.
    Zero length, like ``SRC.L``."""

    def __init__(self, lambda_SR, detune_rad, mlib=MLIB):
        self.lambda_SR = lambda_SR
        self.detune_rad = detune_rad
        self.mlib = mlib

    def edgesAC(self, *args, **kwargs):
        return {
            "SRC.Lin": np.sqrt(1 - self.lambda_SR) * self.mlib.Mrotation(self.detune_rad),
            "SRC.pl": np.sqrt(self.lambda_SR) * self.mlib.Id,
        }


def lossy_topology():
    """``signal_recycled(loss_ports=True)`` with SRC loss on the SRM -> ITM leg.

    That leg becomes the edge ``SRC.Lin``, and vacuum enters the ITM's back
    face from ``SRC.p.exc`` through ``SRC.pl`` (see ``SRCLossEdge``).
    """
    ifo = topologies.signal_recycled(loss_ports=True)
    ifo.edges[("ITM.bk.i", "SRM.fr.o")] = "SRC.Lin"
    ifo.edges[("ITM.bk.i", "SRC.p.exc")] = "SRC.pl"
    ifo.locations["SRC.p.exc"] = (20, -7)
    return ifo


def _arm_edges(p, L_arm, loss_ports, mlib):
    L, g = p["L_m"], p["gamma"]
    M = p["m_kg"] / 4
    return [
        edges.MirrorEdge("ITM", Thr=topologies.half_bandwidth_T(g / (2 * np.pi), L),
                         mlib=mlib, loss_ports=loss_ports),
        edges.RPMirrorEdge("ETM", Lhr=L_arm,
                           suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=lambda_m(p), mlib=mlib, loss_ports=loss_ports),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
    ]


def _arm_dc(p, L_arm, Io_over_Isql, loss_ports, mlib):
    """Carrier in the arm, entering from the beamsplitter side (the ITM).

    The lossless arm is scaled so ``K`` matches Eq. 2.13 exactly; with loss the
    same input power gives less circulating power, by the ratio of the exact
    cavity gains.
    """
    def dc(Lhr):
        return solve.solve_dc(
            solve.build(topologies.fp_arm(loss_ports=loss_ports)),
            _arm_edges(p, Lhr, loss_ports, mlib), mlib,
            drive={"ITM.bk.i.exc": mlib.LO(np.pi / 2)},
            test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"},
        )
    node = "ETM.fr.i.tp"
    dc0, dcL = dc(0), dc(L_arm)
    gain = Vnorm_sq(dcL[node]) / Vnorm_sq(dc0[node])
    P_in = Io_over_Isql * I_SQL(p) / 4
    return solve.scale_dc(dcL, node, gain * P_in * scc.c / (p["L_m"] * p["gamma"]))


def interferometer(F_Hz, rho, phi, Io_over_Isql, eps=0, lambda_SR=0, lambda_PD=0,
                   p=PARAMS, mlib=MLIB):
    """Transfer matrices of the signal-recycled differential mode.

    ``rho`` and ``phi`` are BC's. Inputs: ``"SRM.bk.i.exc"`` (dark-port vacuum)
    and ``"ETM.pos.exc"`` (``x = L h`` [m]); with losses also the arm loss
    ports, ``"SRC.p.exc"`` and ``"PD"``. Output: the SRM's back face, after
    detection loss.
    """
    lossy = bool(eps or lambda_SR or lambda_PD)
    L_arm = eps * (4 * p["L_m"] * p["gamma"] / scc.c) / 2
    arm = _arm_edges(p, L_arm, lossy, mlib)
    edge_objs = arm + [
        edges.MirrorEdge("SRM", Thr=1 - rho**2, mlib=mlib, loss_ports=lossy),
        edges.LinkEdge("SRC.L", L_m=0, detune_rad=sflu_detune(phi), mlib=mlib),
    ]
    inputs = {"SRM.bk.i.exc", "ETM.pos.exc"}
    if lossy:
        edge_objs.append(SRCLossEdge(lambda_SR, sflu_detune(phi), mlib))
        inputs |= {"SRC.p.exc", "ETM.frL.i", "ITM.frL.i"}
    T = solve.solve_ac(
        solve.build(lossy_topology() if lossy else topologies.signal_recycled()),
        edge_objs, mlib, F_Hz, readout="SRM.bk.o.tp", inputs=inputs,
        resultsDC=_arm_dc(p, L_arm, Io_over_Isql, lossy, mlib),
    )
    return with_detection_loss(T, lambda_PD, mlib) if lossy else T


def with_detection_loss(T, lambda_PD, mlib=MLIB):
    """Eq. 5.5: attenuate every input by ``sqrt(1 - lambda_PD)``, add vacuum ``"PD"``."""
    out = {k: np.sqrt(1 - lambda_PD) * v for k, v in T.items()}
    N = len(T["ETM.pos.exc"])
    out["PD"] = np.sqrt(lambda_PD) * np.broadcast_to(mlib.Id, (N, mlib.dim, mlib.dim))
    return out


def referred(T, zeta, p=PARAMS, mlib=MLIB):
    """``(S_h, budget)`` at package homodyne angle ``zeta`` (``sflu_zeta``)."""
    Sx, budget = readout.referred_psd(T, readout.quadrature(mlib, zeta), "ETM.pos.exc",
                                      lambda_m=lambda_m(p))
    return Sx / p["L_m"]**2, {k: v / p["L_m"]**2 for k, v in budget.items()}


def sqrt_Sh_over_hSQL(T, zeta, p=PARAMS, mlib=MLIB):
    """``sqrt(S_h / S_h^SQL(gamma))``, the y-axis of every BC figure.

    ``zeta`` is in this package's convention; use ``sflu_zeta`` on BC's.
    """
    return np.sqrt(referred(T, zeta, p, mlib)[0]) / h_SQL(p["gamma"], p)
