"""
Buonanno & Chen, "Scaling law in signal recycled laser-interferometer
gravitational-wave detectors", PRD 67 062002 (2003), arXiv:gr-qc/0208048.

Mapping onto the graph
----------------------
BC reduce the differential mode of a signal-recycled Michelson to the
three-mirror cavity SRM-ITM-ETM (``topologies.signal_recycled()``, built by
``sflu.papers.buonanno_chen2002.sr_edges``), and then fold the short SR cavity
and the ITM into one compound input mirror of complex reflectivity ``rho'``
(Eq. 11). The result is a *single detuned cavity*, ``topologies.fp_arm()``,
with

* ITM transmission ``1 - |rho'|^2 = 1 - exp(-4 epsilon L/c)``, i.e.
  ``topologies.half_bandwidth_T(epsilon / 2 pi, L)``, and
* arm detuning ``arg rho' = 2 lambda L / c`` per round trip, i.e. ``ARM.L``
  with ``detune_rad = topologies.detuning_rad(lambda / 2 pi, L)``;

``lambda`` and ``epsilon`` being the exact Eq. 13 values. Both graphs are
built here (``coupled_cavity``, ``single_cavity``) and agree to rounding.

Masses and powers as in ``buonanno_chen2002`` (and KLMTV): one movable ETM of
mass ``m/4`` whose displacement is ``x = L h``, and model arm power ``I_c / 2``,
where ``I_c`` is BC's circulating power per real arm (Eq. 21). This is the
same physics as BC's own effective arm (``I_arm = 2 I_c``, ``mu_arm = m``,
``x = L h / 2``, Eqs. 1-3): the optical phase ``sqrt(P) x`` and the back-action
displacement per unit ``h`` are identical. ``S_h = S_x / L^2``, single-sided.

The SR cavity has zero length (BC neglect ``Omega l / c``). No losses.

Conventions
-----------
Detuning and time convention: see ``buonanno_chen2002`` (``SRC.L`` gets
``detune_rad = -phi``; the graph at ``F`` is BC's ``Omega = -2 pi F``).

**Quadratures.** BC read ``b_zeta = b1 sin(zeta) + b2 cos(zeta)``. There are
two frames:

* the *physical* one of the SR interferometer (Figs. 8, 9): with the arm
  carrier along the amplitude quadrature (``buonanno_chen2002.carrier``), BC's
  ``zeta`` is the package's ``zeta - pi/2`` at ``SRM.bk.o.tp``
  (``physical_zeta``);
* the *tilde* frame of the single cavity (Eqs. 6, 7; Figs. 5, 6, 10), rotated
  from the physical one by ``theta`` (Eqs. 19, 105): ``b~_zeta~`` with
  ``zeta~ = zeta + theta``. In the graph ``single_cavity`` that is the
  package's ``zeta~ - pi/2 - lambda L / c`` at ``ITM.bk.o.tp`` (``tilde_zeta``):
  the extra ``-lambda L / c`` is the one-way detuning that light entering and
  leaving through the detuned ``ARM.L`` link picks up, which BC keep in the
  mirror instead.

All three statements were found numerically and hold to rounding: the
coupled cavity at physical ``zeta`` equals the single cavity at
``zeta + theta_exact`` (noise and signal, ``test_scaling_law``), and both equal
the exact closed form, Eqs. 36 and 99-104, for any ``zeta`` (``test_fig8``).
The ``-pi/2`` is only partly the swapped sin/cos: a ``pi/2``-tuned (RSE) SR
cavity also rotates the dark-port frame, so the carrier reference must be the
arm's, as above, not the dark port's.

The paper uses ``c = 3e8 m/s``; here ``scipy.constants.c``. The difference
(0.07% in ``gamma``) is far below anything plotted.
"""
import numpy as np
import scipy.constants as scc

from sflu import edges, readout, solve, topologies
from sflu.papers import buonanno_chen2002 as bc02

OMEGA0 = 1.8e15  # Sec. IIA

# Sec. IIC and V, and the caption of Fig. 8. I_c is the circulating power per
# real arm. BC config: I_c = 592 kW = 2 I_SQL / T (I_0 = I_SQL at the BS).
# LIGO-II reference: the triangle of Fig. 4, zeta = 1.13 pi.
PARAMS = dict(
    lambda_m=2 * np.pi * scc.c / OMEGA0,
    L_m=4e3,
    T_itm=0.033,
    rho=0.9,
    phi=np.pi / 2 - 0.47,
    m_kg=30.0,
    I_c=592e3,
)
PARAMS["gamma"] = PARAMS["T_itm"] * scc.c / (4 * PARAMS["L_m"])

LIGO2 = dict(PARAMS, T_itm=0.005, rho=0.964, phi=np.pi / 2 - 0.06,
             m_kg=40.0, I_c=840e3, zeta=1.13 * np.pi)
LIGO2["gamma"] = LIGO2["T_itm"] * scc.c / (4 * LIGO2["L_m"])

MLIB = bc02.MLIB


def h_SQL(Omega, p=PARAMS):
    """Eq. 28, single-sided, ``m`` the mass of one mirror."""
    return np.sqrt(8 * scc.hbar / (p["m_kg"] * Omega**2 * p["L_m"]**2))


def iota_c(p=PARAMS):
    """Eq. 20: ``8 omega_0 I_c / (m L c)``."""
    return 8 * OMEGA0 * p["I_c"] / (p["m_kg"] * p["L_m"] * scc.c)


# ---------------------------------------------------------------------------
# the scaling law: (T, rho, phi) <-> (lambda, epsilon)
# ---------------------------------------------------------------------------

def lam_eps_exact(rho, phi, p=PARAMS):
    """Eq. 13 (via Eq. 11): exact ``(lambda, epsilon)``."""
    return bc02.lam_eps(rho, phi, p)


def lam_eps_first_order(rho, phi, p=PARAMS):
    """Eq. 18, leading order in ``T``, with ``gamma = T c / 4L``."""
    g = p["T_itm"] * scc.c / (4 * p["L_m"])
    d = 1 + rho**2 + 2 * rho * np.cos(2 * phi)
    return 2 * rho * g * np.sin(2 * phi) / d, (1 - rho**2) * g / d


def rho_phi(lam, eps, T, L=PARAMS["L_m"]):
    """Eq. 14: the SR mirror and detuning that give ``(lambda, epsilon)`` with
    ITM transmission ``T``. Returns ``(rho, phi)``, ``phi`` in (-pi/2, pi/2]."""
    z = np.exp(2j * (lam + 1j * eps) * L / scc.c)
    sR = np.sqrt(1 - T)
    w = (z - sR) / (1 - sR * z)
    return np.abs(w), np.angle(w) / 2


def theta_first_order(rho, phi):
    """Eqs. 19, 22-23: the rotation from the physical to the tilde frame."""
    return np.arctan((1 - rho) * np.tan(phi) / (1 + rho))


def theta_exact(rho, phi, p=PARAMS):
    """Eqs. 105-106: Eq. 19 with ``rho -> rho sqrt(R)``."""
    rs = rho * np.sqrt(1 - p["T_itm"])
    return np.arctan((1 - rs) * np.tan(phi) / (1 + rs))


# ---------------------------------------------------------------------------
# noise, closed forms (tilde frame)
# ---------------------------------------------------------------------------

def Sh_first_order(Omega, zeta, lam, eps, p=PARAMS):
    """Eq. 37: ``S_h`` of ``b~_zeta``, first order in ``T``."""
    io = iota_c(p)
    s, c = np.sin(zeta), np.cos(zeta)
    pre = (Omega**2 * h_SQL(Omega, p)**2
           / (4 * eps * io * (Omega**2 * c**2 + (eps * c - lam * s)**2)))
    return pre * (
        ((Omega + lam)**2 + eps**2) * ((Omega - lam)**2 + eps**2)
        + 2 * io / Omega**2 * (Omega**2 * (lam - eps * np.sin(2 * zeta))
                               - lam * (eps**2 + lam**2 + 2 * eps**2 * np.cos(2 * zeta))
                               - eps * (eps**2 - lam**2) * np.sin(2 * zeta))
        + io**2 / Omega**4 * (2 * eps**2 * (1 + np.cos(2 * zeta))
                              - 2 * eps * lam * np.sin(2 * zeta) + lam**2))


def CD_exact(Omega, lam, eps, p=PARAMS):
    """Eqs. 100-104: ``C^ex`` (2x2) and ``D^ex`` (2,), all orders in ``T``.

    ``lambda`` and ``epsilon`` must be the exact Eq. 13 values. The common
    factor ``M^ex`` (Eq. 99) cancels in ``S_h`` and is omitted.
    """
    k = p["L_m"] / scc.c
    E2, E4 = np.exp(-2 * eps * k), np.exp(-4 * eps * k)
    pre = Omega**2 / (4 * k**2)
    a = iota_c(p) * k / Omega**2
    s2, c2 = np.sin(2 * lam * k), np.cos(2 * lam * k)
    c2W = np.cos(2 * Omega * k)
    C11 = pre * ((1 - 2 * E2 * c2 * c2W + E4 * np.cos(4 * lam * k))
                 + a * E4 * np.sin(4 * lam * k))
    C12 = pre * (-2 * E2 * s2 * (c2W - E2 * c2) + 2 * a * E4 * s2**2)
    C21 = pre * (2 * E2 * s2 * (c2W - E2 * c2) - 2 * a * (1 - E4 * c2**2))
    sq = np.sqrt((1 - E4) * a)
    D1 = pre * (-2 * E2 * np.exp(1j * Omega * k) * s2) * sq
    D2 = pre * (2 * np.exp(-1j * Omega * k) - 2 * E2 * np.exp(1j * Omega * k) * c2) * sq
    return np.array([[C11, C12], [C21, C11]]), np.array([D1, D2])


def Sh_from_CD(Omega, zeta, C, D, p=PARAMS):
    """Eq. 36: ``S_h`` of ``b~_zeta`` from input-output coefficients."""
    s, c = np.sin(zeta), np.cos(zeta)
    num = (C[0, 0] * s + C[1, 0] * c)**2 + (C[0, 1] * s + C[1, 1] * c)**2
    return h_SQL(Omega, p)**2 * np.abs(num) / np.abs(D[0] * s + D[1] * c)**2


def Sh_exact(Omega, zeta, lam, eps, p=PARAMS):
    """Eqs. 36 and 99-104: the all-orders ``S_h`` of ``b~_zeta``."""
    C, D = CD_exact(Omega, lam, eps, p)
    return Sh_from_CD(Omega, zeta, C, D, p)


def Sh_min_2(Omega, lam, eps, p=PARAMS):
    """Eqs. 80, 82: ``S^min_{h,2} = h_SQL^2 |1 + R_xx K_2^eff|``, the noise if
    the shot/back-action correlations were used optimally (Fig. 5 dashed)."""
    Ic = 8 * OMEGA0 * p["I_c"] / (p["L_m"] * scc.c)  # Eq. 43
    P = ((Omega + lam)**2 + eps**2) * ((Omega - lam)**2 + eps**2)
    K2 = Ic * lam / 4 * (3 * eps**2 + lam**2 - Omega**2) / P
    return h_SQL(Omega, p)**2 * np.abs(1 - 4 / (p["m_kg"] * Omega**2) * K2)


# ---------------------------------------------------------------------------
# the SFLU models
# ---------------------------------------------------------------------------

def physical_zeta(zeta):
    """BC's physical homodyne angle at the SRM, in this package."""
    return bc02.sflu_zeta(zeta)


def tilde_zeta(zeta_t, lam, p=PARAMS):
    """BC's tilde-frame angle ``zeta~`` at the output of ``single_cavity``."""
    return np.asarray(zeta_t) - np.pi / 2 - lam * p["L_m"] / scc.c


def coupled_cavity(F_Hz, rho, phi, p=PARAMS, T_itm=None, mlib=MLIB):
    """The SRM-ITM-ETM graph; dark port ``SRM.bk.i.exc`` -> ``SRM.bk.o.tp``."""
    return bc02.sr_ifo(F_Hz, rho, phi, p["I_c"] / 2, p=p, mlib=mlib, T_itm=T_itm)


def single_cavity(F_Hz, lam, eps, p=PARAMS, mlib=MLIB):
    """The equivalent detuned Fabry-Perot cavity; ``ITM.bk.i.exc`` -> ``ITM.bk.o.tp``."""
    L, M = p["L_m"], p["m_kg"] / 4
    eo = [
        edges.MirrorEdge("ITM", Thr=topologies.half_bandwidth_T(eps / (2 * np.pi), L),
                         mlib=mlib),
        edges.RPMirrorEdge("ETM", suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=p["lambda_m"], mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L,
                       detune_rad=topologies.detuning_rad(lam / (2 * np.pi), L),
                       mlib=mlib),
    ]
    dc = bc02.carrier(topologies.fp_arm, eo, "ITM.bk.i.exc", p["I_c"] / 2, mlib)
    return solve.solve_ac(solve.build(topologies.fp_arm()), eo, mlib, F_Hz,
                          readout="ITM.bk.o.tp",
                          inputs={"ITM.bk.i.exc", "ETM.pos.exc"}, resultsDC=dc)


def Sh_model(T, zeta_sflu, p=PARAMS, mlib=MLIB):
    """``S_h = S_x / L^2`` of the model's quadrature ``zeta_sflu`` (package angle)."""
    Sx, _ = readout.referred_psd(T, readout.quadrature(mlib, zeta_sflu),
                                 "ETM.pos.exc", lambda_m=p["lambda_m"])
    return Sx / p["L_m"]**2
