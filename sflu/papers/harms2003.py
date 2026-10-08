"""
Harms, Chen, Chelkowski, Franzen, Vahlbruch, Danzmann & Schnabel,
"Squeezed-input, optical-spring, signal-recycled gravitational-wave
detectors", PRD 68 042001 (2003), arXiv:gr-qc/0303066 (v2).

Mapping onto the graph
----------------------
The paper's numbers are for "ideal GEO 600": a dual-recycled Michelson with
folded arms and *no arm cavities*, beamsplitter radiation pressure neglected.
Its differential mode is the SRM, a free path of length ``L = 1200 m``
(``Phi = Omega L / c`` one way) and a free mirror. That is
``topologies.signal_recycled()`` with a fully transmissive ITM
(``Thr = 1``): the "arm" is then a plain delay line.

* **SRM and detuning** as in ``sflu.papers.buonanno_chen2001``: amplitude
  reflectivity ``rho``, ``SRC.L`` link phase ``-phi`` (``bc.sflu_detune``),
  zero SRC length.
* **Mass.** The paper's ``h_SQL^2 = 20 hbar / (m Omega^2 L^2)`` (Table II, which
  prints it without the square) is the SQL of a free mass ``m/10`` moved by
  ``x = L h``: the end mirror has mass ``m/10``, ``S_h = S_x / L^2``.
* **Power.** The model reproduces the paper through ``K = Theta / Omega^2``
  alone, and ``K`` of a perfectly reflecting free mirror of mass ``M`` with
  incident power ``P_c`` is ``8 omega_0 P_c / (M c^2 Omega^2)`` -- exactly, the
  delays being common phases (``model_theta`` checks this). ``P_c`` is
  chosen to give the paper's ``Theta``.
* **Which Theta.** Table II gives ``K = 20 P omega_0 / (m c^2 Omega^2)`` with
  ``P = 10 kW`` (Table I), i.e. ``Theta = 703 s^-2``. The figures need half
  that, ``Theta = 10 P omega_0 / (m c^2) = 352 s^-2``: with it the
  optical-spring dip is at 30 Hz, the conventional noise reaches the SQL at
  3 Hz (as the text says) and the shot-noise level is 6.1e-22; with Table II's
  value they are at 43 Hz, 4.2 Hz and 4.3e-22. Table I calls ``P`` the
  "circulating light power", Table II "the input power at the beamsplitter";
  the figures correspond to Table II with 5 kW. ``theta(p, "figures")`` is
  used throughout; ``theta(p, "table")`` is Table II as printed.
* **rho.** Table I lists 0.99 as the SRM *power* reflectivity; the formulas
  take the amplitude ``rho = sqrt(0.99)``, ``tau = 0.1``. (``rho = 0.99`` does
  not reproduce the figures.)

No approximation separates model and paper here: GEO has no arm cavity, so
Eqs. 3-5 are exact for a lossless delay line, and SFLU reproduces them to
machine precision.

Conventions
-----------
Harms et al.'s ``T``, ``M`` and ``s`` (Eqs. 3-5) are Buonanno & Chen's
(BC Eqs. 2.21-2.24) with BC's arm phase ``beta`` replaced by ``Phi``. Their
homodyne angle is different: ``o_zeta = o1 cos(zeta) + o2 sin(zeta)``
(Eq. 6), so ``zeta = pi/2`` is the phase quadrature, which is BC's
``zeta_BC = pi/2 - zeta``. Both were checked numerically
(``papers/test_harms2003.py``). With the opposite sign of the phase
quadrature in this package (as for KLMTV and BC):

* **Homodyne.** Harms's ``zeta`` is this package's ``-zeta``
  (``sflu_zeta``).
* **Squeezing.** Eq. 11 transforms the input by ``D(-lam) S(r) D(lam)`` with
  ``D(lam) = [[cos, sin], [-sin, cos]]``; the squeezed quadrature (variance
  ``e^{-2r}``) is ``-sin(lam) a1 + cos(lam) a2``, so ``lam = 0`` is phase
  squeezing. In this package's ``readout.squeezed`` that is the angle
  ``pi/2 - lam`` (``sflu_squeeze``).

Both found by trying the candidates against Eqs. 7 and 14; the chosen ones
agree to 1e-11, every other to no better than O(1). ``r = 1`` is
``20 / ln 10 = 8.69`` dB.

Eq. 30 has its labels swapped: ``zeta_+``, not ``zeta_-``, is the minimum
(``zeta_opt``).

Fig. 5 plots ``lambda_opt`` as *minus* Eq. 16's angle (unwrapped): Eq. 16
gives 114.6 deg at 10 Hz, the figure 65 deg. ``lambda_opt`` here returns
Eq. 16's value; the test shows the figure's sign explicitly.
"""
import numpy as np
import scipy.constants as scc

from sflu import edges, readout, solve, topologies
from sflu.lib import MatrixLib
from sflu.papers import buonanno_chen2001 as bc

# Table I unless noted.
PARAMS = dict(
    m_kg=5.6,           # each mirror
    L_m=1200.0,         # effective (folded) arm length
    P_W=10e3,           # "circulating light power"
    omega0=1.77e15,     # carrier [rad/s]
    R_SRM=0.99,         # SRM power reflectivity; rho = sqrt(R_SRM)
    phi=0.0055,         # SR detuning [rad], one-way SRC phase (BC's phi)
    r=1.0,              # squeeze parameter, Secs. III and IV
)

MLIB = MatrixLib(nhom=0)


def lambda_m(p=PARAMS):
    return 2 * np.pi * scc.c / p["omega0"]


def rho_tau(p=PARAMS):
    return np.sqrt(p["R_SRM"]), np.sqrt(1 - p["R_SRM"])


def theta(p=PARAMS, which="figures"):
    """``K Omega^2`` [s^-2]: 352 for the figures, 703 per Table II.

    See the module docstring: the figures use half of Table II's coupling.
    """
    factor = {"figures": 10, "table": 20}[which]
    return factor * p["P_W"] * p["omega0"] / (p["m_kg"] * scc.c**2)


def sqz_dB(p=PARAMS):
    """``r`` in dB of squeezing, ``10 log10(e^{2r})``."""
    return 20 * p["r"] / np.log(10)


def h_SQL(Omega, p=PARAMS):
    """Table II (GEO), with the square root Table II leaves out."""
    return np.sqrt(20 * scc.hbar / (p["m_kg"] * Omega**2 * p["L_m"]**2))


def sflu_zeta(zeta):
    """A homodyne angle of Eq. 6, in this package's convention."""
    return -np.asarray(zeta)


def sflu_squeeze(lam):
    """A squeeze angle of Eq. 11, as ``readout.squeezed``'s ``angle``."""
    return np.pi / 2 - np.asarray(lam)


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def io_relation(K, Phi, hsql, rho, phi):
    """Eqs. 3-5: ``(T, M, s)`` with ``T`` of shape ``(N, 2, 2)`` and ``s`` (N, 2).

    Takes ``K``, ``Phi`` and ``h_SQL`` as arrays so that it serves both
    columns of Table II.
    """
    tau2 = 1 - rho**2
    e = np.exp(2j * Phi)
    X = np.cos(2 * phi) + K / 2 * np.sin(2 * phi)
    T11 = e * ((1 + rho**2) * X - 2 * rho * np.cos(2 * Phi))
    T12 = -e * tau2 * (np.sin(2 * phi) + K * np.sin(phi)**2)
    T21 = e * tau2 * (np.sin(2 * phi) - K * np.cos(phi)**2)
    T = np.stack([np.stack([T11, T12], -1), np.stack([T21, T11], -1)], -2)
    M = 1 + rho**2 * e**2 - 2 * rho * e * X
    pre = -np.sqrt(2 * K) / hsql * np.sqrt(tau2)
    s = np.stack([pre * (1 + rho * e) * np.sin(phi),
                  pre * (-1 + rho * e) * np.cos(phi)], -1)
    return T, M, s


def geo(Omega, p=PARAMS, which="figures"):
    """``io_relation`` for ideal GEO 600 (Table II, left column)."""
    rho, _ = rho_tau(p)
    return io_relation(theta(p, which) / Omega**2, Omega * p["L_m"] / scc.c,
                       h_SQL(Omega, p), rho, p["phi"])


def _v(zeta):
    zeta = np.asarray(zeta, dtype=float)
    return np.stack([np.cos(zeta), np.sin(zeta)], -1)


def D(lam):
    """The rotation of Eq. 12, ``[[cos, sin], [-sin, cos]]``."""
    lam = np.asarray(lam, dtype=float)
    c, s = np.cos(lam), np.sin(lam)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


def Sh(T, s, zeta, lam=None, r=0.0):
    """Eq. 14 (Eq. 7 when ``r = 0``): input squeezed by ``r`` at angle ``lam``."""
    v = _v(zeta)
    V = np.eye(2) if lam is None else (
        D(-np.asarray(lam)) @ np.diag([np.exp(2 * r), np.exp(-2 * r)]) @ D(lam))
    vT = np.einsum("...i,...ij->...j", v, T)
    num = np.einsum("...i,...ij,...j->...", vT, V, vT.conj()).real
    return num / np.abs(np.einsum("...i,...i->...", v, s))**2


def lambda_opt(T, zeta):
    """Eq. 16: the squeeze angle that minimises Eq. 14 at readout ``zeta``."""
    c, s = np.cos(zeta), np.sin(zeta)
    # T carries a common phase (Eq. 3), which cancels in the ratio.
    num = T[..., 0, 0] * c + T[..., 1, 0] * s
    den = T[..., 0, 1] * c + T[..., 1, 1] * s
    return np.arctan((-num / den).real)


def Sh_SI(T, s, zeta, r):
    """Eq. 17: optimal frequency-dependent squeezing, ``e^{-2r}`` x Eq. 7."""
    return np.exp(-2 * r) * Sh(T, s, zeta)


def zeta_opt(T, s):
    """Eqs. 28-30: ``(zeta_-, zeta_+)``.

    The paper says ``zeta_-`` is the minimum. As printed (principal square
    root; the arccot branch only shifts by pi) it is ``zeta_+``: it agrees with
    the model's own optimum to 1e-12 rad and reproduces Fig. 5, while
    ``zeta_-`` is the noise maximum.
    """
    def sym(A):
        return (A @ np.swapaxes(A.conj(), -1, -2)).real  # <A A^dag>_sym, A T-like
    Tm = sym(T)
    Sm = (s[..., :, None] * s.conj()[..., None, :]).real
    Q11 = Sm[..., 0, 0] * (Tm[..., 0, 1] + Tm[..., 1, 0]) - Tm[..., 0, 0] * (Sm[..., 0, 1] + Sm[..., 1, 0])
    Q12 = Sm[..., 0, 0] * Tm[..., 1, 1] - Tm[..., 0, 0] * Sm[..., 1, 1]
    Q22 = Tm[..., 1, 1] * (Sm[..., 0, 1] + Sm[..., 1, 0]) - Sm[..., 1, 1] * (Tm[..., 0, 1] + Tm[..., 1, 0])
    root = np.sqrt(-(Q11 * Q22 - Q12**2))
    # arccot(x) = arctan(1/x)
    return tuple(-np.arctan(Q11 / (sgn * root + Q12)) for sgn in (-1, +1))


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def _edges(p, mlib):
    M = p["m_kg"] / 10
    return [
        edges.MirrorEdge("ITM", Thr=1, mlib=mlib),  # absent: GEO has no arm cavity
        edges.RPMirrorEdge("ETM", suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=lambda_m(p), mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=p["L_m"], mlib=mlib),
    ]


def mirror_power(p=PARAMS, which="figures"):
    """Incident power on the ``m/10`` mirror that gives ``theta(p, which)``."""
    return theta(p, which) * (p["m_kg"] / 10) * scc.c**2 / (8 * p["omega0"])


def _dc(p, which, mlib):
    dc = solve.solve_dc(
        solve.build(topologies.fp_arm()), _edges(p, mlib), mlib,
        drive={"ITM.bk.i.exc": mlib.LO(np.pi / 2)},
        test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"},
    )
    return solve.scale_dc(dc, "ETM.fr.i.tp", mirror_power(p, which))


def model_theta(F_Hz, p=PARAMS, which="figures", mlib=MLIB):
    """``K Omega^2`` of the model's bare delay line (no SRM), from its
    transfer matrix: ``b2 = a2 + K a1`` here, as in ``klmtv2001.model_kappa``."""
    T = solve.solve_ac(
        solve.build(topologies.fp_arm()), _edges(p, mlib), mlib, F_Hz,
        readout="ITM.bk.o.tp", inputs={"ITM.bk.i.exc"}, resultsDC=_dc(p, which, mlib),
    )["ITM.bk.i.exc"]
    return (T[:, 1, 0] / T[:, 1, 1]).real * (2 * np.pi * F_Hz)**2


def interferometer(F_Hz, p=PARAMS, which="figures", R_SRM=None, mlib=MLIB):
    """Transfer matrices of ideal GEO 600's differential mode.

    Inputs ``"SRM.bk.i.exc"`` (dark port: vacuum or squeezing) and
    ``"ETM.pos.exc"`` (``x = L h``). ``R_SRM = 0`` gives the conventional
    interferometer (no SRM, and then no SRC phase).
    """
    R = p["R_SRM"] if R_SRM is None else R_SRM
    edge_objs = _edges(p, mlib) + [
        edges.MirrorEdge("SRM", Thr=1 - R, mlib=mlib),
        edges.LinkEdge("SRC.L", L_m=0, detune_rad=bc.sflu_detune(p["phi"]) if R else 0,
                       mlib=mlib),
    ]
    return solve.solve_ac(
        solve.build(topologies.signal_recycled()), edge_objs, mlib, F_Hz,
        readout="SRM.bk.o.tp", inputs={"SRM.bk.i.exc", "ETM.pos.exc"},
        resultsDC=_dc(p, which, mlib),
    )


def sqrt_Sh(T, zeta, states=None, p=PARAMS, mlib=MLIB):
    """``sqrt(S_h)`` [1/sqrt(Hz)], the y-axis of Figs. 2-4.

    ``zeta`` and any squeeze angles are in this package's convention; use
    ``sflu_zeta`` and ``sflu_squeeze`` on the paper's.
    """
    Sx, _ = readout.referred_psd(T, readout.quadrature(mlib, zeta), "ETM.pos.exc",
                                 states=states, lambda_m=lambda_m(p))
    return np.sqrt(Sx) / p["L_m"]


def model_matrices(T):
    """``(T_vac, s)`` from a solved model, the analogues of Eqs. 3-5 in this
    package's quadratures: dark-port transfer matrix ``(N, 2, 2)`` and signal
    column ``(N, 2)`` per metre."""
    return T["SRM.bk.i.exc"], T["ETM.pos.exc"][..., 0]


def _real_direction(w):
    """``w = e^{i theta} w_r`` with ``w_r`` real: return ``w_r`` (up to sign)."""
    ph = np.angle(np.sum(w**2, axis=-1)) / 2
    return (w * np.exp(-1j * ph)[..., None]).real


def model_squeeze_opt(T, zeta):
    """The model's own optimal squeeze angle (``readout.squeezed`` convention)
    at package readout angle ``zeta``.

    The readout sees ``row T a``; the noise is least when the squeezed
    quadrature lies along ``T^dag row^T`` -- a real direction times the
    common phase of ``T``.
    """
    Tv, _ = model_matrices(T)
    w = _real_direction(np.einsum("...i,...ij->...j", _v(zeta), Tv).conj())
    return np.arctan2(w[..., 1], w[..., 0])


def model_zeta_opt(T):
    """The model's own optimal (package) homodyne angle.

    It minimises ``v A v^T / v B v^T`` with ``A = Re(T T^dag)`` and
    ``B = Re(s s^dag)`` (``s`` is *not* a real vector times a phase: its two
    components have different phases), so ``v`` is the generalised
    eigenvector of the smaller eigenvalue of ``A v = mu B v``.
    """
    Tv, s = model_matrices(T)
    A = (Tv @ np.swapaxes(Tv.conj(), -1, -2)).real
    B = (s[..., :, None] * s.conj()[..., None, :]).real
    mu, V = np.linalg.eig(np.linalg.solve(B, A))
    i = np.argmin(mu.real, axis=-1)
    v = np.take_along_axis(V.real, i[..., None, None], axis=-1)[..., 0]
    return np.arctan2(v[..., 1], v[..., 0])
