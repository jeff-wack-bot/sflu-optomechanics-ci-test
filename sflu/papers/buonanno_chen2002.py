"""
Buonanno & Chen, "Signal recycled laser-interferometer gravitational-wave
detectors as optical springs", PRD 65 042001 (2002), arXiv:gr-qc/0107021.

Mapping onto the graph
----------------------
The paper's interferometer is LIGO-II-like: a power-recycled Michelson with
4 km Fabry-Perot arms (ITM transmission ``T = 0.033``) and a signal-recycling
mirror (amplitude reflectivity ``rho``) at the dark port. Its differential
mode is the three-mirror cavity SRM-ITM-ETM, ``topologies.signal_recycled()``:
SRM and ITM are lossless ``MirrorEdge``s, the ETM a perfectly reflecting free
``RPMirrorEdge``, ``ARM.L`` a tuned 4 km link and ``SRC.L`` a zero-length link
carrying only the SR detuning. BC neglect the sideband phase in the SR cavity
(``Omega l / c -> 0``), which a zero-length link realises exactly.

The coordinate is BC's antisymmetric ``x = (x_n1 - x_n2) - (x_e1 - x_e2)``,
whose reduced mass is ``m/4`` (Eq. 3.9, ``R_xx = -4 / (m Omega^2)``, Eq. 3.11),
so the model's end mirror has mass ``m/4`` and its displacement *is* ``x``.
As for KLMTV (``sflu.papers.klmtv2001``), the single arm then carries half the
real arm power: ``P = I_c / 2 = I_0 / T = I_0 c / (4 L gamma)``, with ``I_0`` the
power at the beamsplitter and ``gamma = T c / 4 L`` BC's arm half-width. This
makes the model's radiation-pressure force on ``x`` exactly BC's ``R_FF x``;
the power is set with ``solve.scale_dc``, not via the cavity gain.

The optical spring ``R_FF`` (Eq. 3.26, ``K = -R_FF``) is read from the model's
closed-loop mechanical response. ``ETM.pos.exc`` adds a displacement at the
ETM and ``ETM.pos.tp`` reads the total one, so their ratio is
``1 / (1 - R_xx R_FF)``; ``model_R_FF`` inverts that. The resonances of
Sec. IV are the zeros of ``1 - R_xx R_FF``, found by evaluating the same graph
at *complex* frequency (``model_resonances``); nothing in SFLU assumes the
frequency is real.

**Leading order in T.** Eqs. 3.21-3.26 are first order in ``T`` and in
``Omega L / c``: the SR cavity acts on the arm as a compound input mirror
whose exact reflectivity is BC2003 Eq. 11 (``compound_reflectivity``), and the
paper keeps only the leading-order pole ``Omega_+``. With the printed
``(T, rho, phi)`` the model's optical pole is therefore off by ~2% (194.3 Hz
instead of 191.4 Hz), which the companion paper (``buonanno_chen2003``) is
about. To reproduce *this* paper's figures, the model is built with the SR
parameters ``equivalent_sr(rho, phi)`` whose exact compound mirror has
precisely the paper's pole (BC2003 Eq. 14, the inverse map); that changes
``rho`` and ``phi`` by < 1%. What remains is the ``O((Omega L/c)^2)``
difference between a real cavity and a single pole: ``R_FF_exact``, the
all-orders form, matches the model to rounding.

Conventions
-----------
**Detuning.** BC's one-way SR phase ``phi`` (Eq. 3.13) is the ``SRC.L``
link's ``detune_rad = -phi`` (``sflu_detuning``). The sign is the package's
phase quadrature, which is minus KLMTV's and BC's (``klmtv2001.sflu_angle``):
every rotation of the two-photon quadratures, the SR-cavity propagation
included, maps to minus itself. Checked numerically: ``phi_link = -phi``
reproduces Eq. 3.26 including its sign (``R_FF(0) < 0``, a restoring spring,
for ``phi = pi/2 - 0.47``); ``+phi`` gives ``-R_FF``, and ``pi/2 - phi`` a
different function altogether. (The SRM and ITM reflection signs of
``MirrorEdge`` are both opposite to BC's Eq. 3.15/BC2003 Eq. 9, which cancels.)

**Time.** BC write ``e^{-i Omega t}``; the graph's links delay by
``e^{-2 pi i F L / c}``, i.e. ``e^{+i omega t}``. For real frequency the two are
complex conjugates, and in general the graph at ``F`` is BC's at
``Omega = -2 pi F``. ``model_R_FF`` and ``model_resonances`` take and return
BC's ``Omega``: unstable roots have ``Im Omega > 0``.

**Homodyne angle.** BC read ``b_zeta = b1 sin(zeta) + b2 cos(zeta)`` (Eq. 3.2),
so ``zeta = 0`` is the phase quadrature. With the arm carrier along the
amplitude quadrature (``sr_dc`` rotates the DC fields so), that is the
package's ``readout.quadrature(mlib, zeta - pi/2)`` (``sflu_zeta``), verified
against BC2003's noise formulas in ``buonanno_chen2003``.

The mirror mass ``m = 30 kg`` is BC2001's; this paper's figures depend only on
``I_0 / I_SQL`` (Fig. 6 also on ``m gamma^2``, its normalisation).
"""
import itertools

import numpy as np
import scipy.constants as scc

from sflu import edges, solve, topologies
from sflu.lib import MatrixLib

# Sec. IV.C. gamma: the paper quotes 619 s^-1 (T c / 4L = 618.3); the figures
# use 619. m and the wavelength are not stated here; they are BC2001's.
PARAMS = dict(
    lambda_m=1064e-9,
    L_m=4e3,                # arm length, Sec. IV.C
    T_itm=0.033,            # ITM power transmissivity, Sec. IV.C
    gamma=619.0,            # arm half-width T c / 4L [1/s], Sec. III.A, IV.C
    m_kg=30.0,              # each test mass (BC2001)
    rho=0.9,                # SRM amplitude reflectivity, Figs. 6-11
    phi=np.pi / 2 - 0.47,   # SR detuning, Sec. IV.A
)

MLIB = MatrixLib(nhom=0)


def omega0(p=PARAMS):
    return 2 * np.pi * scc.c / p["lambda_m"]


def I_SQL(p=PARAMS):
    """Text after Eq. 4.7: ``m L^2 gamma^4 / (4 omega_0)``, a beamsplitter power."""
    return p["m_kg"] * p["L_m"]**2 * p["gamma"]**4 / (4 * omega0(p))


def sflu_detuning(phi):
    """BC's one-way SR phase ``phi`` as the ``SRC.L`` link's ``detune_rad``."""
    return -np.asarray(phi)


def sflu_zeta(zeta):
    """BC's homodyne angle (``b1 sin zeta + b2 cos zeta``) in this package."""
    return np.asarray(zeta) - np.pi / 2


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def D(rho, phi):
    """``1 + 2 rho cos 2phi + rho^2``, the denominator of Eqs. 3.21-3.31."""
    return 1 + 2 * rho * np.cos(2 * phi) + rho**2


def Omega_pm(rho, phi, p=PARAMS):
    """Eq. 3.21: the free optical resonances ``(Omega_+, Omega_-)``."""
    g = p["gamma"]
    d = D(rho, phi)
    re = 2 * rho * g * np.sin(2 * phi) / d
    im = -g * (1 - rho**2) / d
    return re + 1j * im, -re + 1j * im


def R_FF(Omega, rho, phi, Io_over_Isql, p=PARAMS):
    """Eq. 3.26: the radiation-pressure force per unit ``x``; ``K = -R_FF``."""
    Op, Om = Omega_pm(rho, phi, p)
    I0 = Io_over_Isql * I_SQL(p)
    return (2 * I0 * omega0(p) / p["L_m"]**2 * rho * np.sin(2 * phi) / D(rho, phi)
            / ((Omega - Op) * (Omega - Om)))


def R_xx(Omega, p=PARAMS):
    """Eq. 3.11: free antisymmetric mode, reduced mass ``m/4``."""
    return -4 / (p["m_kg"] * Omega**2)


def _quartic_roots(rho, phi, Io_over_Isql, p):
    up, um = (x / p["gamma"] for x in Omega_pm(rho, phi, p))
    coef = np.polymul([1, 0, 0], np.polymul([1, -up], [1, -um])).astype(complex)
    coef[-1] += Io_over_Isql / 2 * (up - um)
    return np.roots(coef) * p["gamma"]


def resonances(rho, phi, Io_over_Isql, p=PARAMS, n_steps=60):
    """Eq. 4.7: the four roots of
    ``Omega^2 (Omega - Omega_+)(Omega - Omega_-) + I_0 gamma^3 (Omega_+ - Omega_-) / 2 I_SQL``.

    Returned as ``[mech, mech, opt, opt]``: the paper calls "mechanical" the
    pair that starts at 0 when the power is turned up from nothing and
    "optical" the pair that starts at ``Omega_pm``. That labelling is made here
    by following the roots up in power.
    """
    powers = Io_over_Isql * np.geomspace(1e-8, 1, n_steps)
    r = _quartic_roots(rho, phi, powers[0], p)
    r = r[np.argsort(np.abs(r))]
    for I in powers[1:]:
        new = _quartic_roots(rho, phi, I, p)
        best = min(itertools.permutations(range(4)),
                   key=lambda perm: np.sum(np.abs(new[list(perm)] - r)))
        r = new[list(best)]
    # Within each pair, order by real part (by imaginary part for a purely
    # imaginary pair), so that sweeps give continuous curves.
    key = lambda z: z.real + 1e-6 * z.imag
    return np.concatenate([sorted(r[:2], key=key), sorted(r[2:], key=key)])


def Delta_Omega0_sq(rho, phi, Io_over_Isql, p=PARAMS):
    """Eq. 4.8: the leading-order mechanical resonance, squared."""
    g = p["gamma"]
    return (Io_over_Isql * 2 * rho * g**2 * np.sin(2 * phi) * D(rho, phi)
            / (4 * rho**2 * np.sin(2 * phi)**2 + (1 - rho**2)**2))


# ---------------------------------------------------------------------------
# beyond leading order in T: the exact compound mirror (BC2003)
# ---------------------------------------------------------------------------

def compound_reflectivity(rho, phi, p=PARAMS):
    """BC2003 Eq. 11: the SR cavity and ITM as one mirror, seen from the arm.

    ``rho' = (sqrt R + rho e^{2i phi}) / (1 + sqrt R rho e^{2i phi})``, exact.
    """
    sR = np.sqrt(1 - p["T_itm"])
    z = rho * np.exp(2j * phi)
    return (sR + z) / (1 + sR * z)


def lam_eps(rho, phi, p=PARAMS):
    """BC2003 Eq. 13: the exact detuning and half-width ``(lambda, epsilon)``
    of the equivalent single cavity, ``Omega_+ = lambda - i epsilon``."""
    rp = compound_reflectivity(rho, phi, p)
    k = scc.c / (2 * p["L_m"])
    return k * np.angle(rp), k * np.log(1 / np.abs(rp))


def equivalent_sr(rho, phi, p=PARAMS):
    """The SR parameters whose *exact* compound mirror has this paper's pole.

    BC2003 Eq. 14 applied to ``lambda - i epsilon = Omega_+`` of Eq. 3.21.
    Returns ``(rho_eq, phi_eq)``; both differ from the inputs by < 1% for the
    paper's parameters.
    """
    Op, _ = Omega_pm(rho, phi, p)
    lam, eps = Op.real, -Op.imag
    z = np.exp(2j * (lam + 1j * eps) * p["L_m"] / scc.c)
    sR = np.sqrt(1 - p["T_itm"])
    w = (z - sR) / (1 - sR * z)
    return np.abs(w), np.angle(w) / 2


def R_FF_exact(Omega, lam, eps, P_circ, p=PARAMS):
    """``R_FF`` of a single detuned cavity to all orders in ``Omega L / c``.

    The zero of BC2003 Eq. 99, ``M^ex``, written as ``1 - R_xx R_FF``:
    ``R_FF = omega_0 I_c sin(2 lambda L/c) / (c^2 sin y_+ sin y_-)`` with
    ``y_pm = (Omega pm lambda + i epsilon) L / c`` and ``I_c = 2 P_circ`` the real
    arm power. Expanding the sines recovers Eq. 3.26.
    """
    k = p["L_m"] / scc.c
    yp = (Omega + lam + 1j * eps) * k
    ym = (Omega - lam + 1j * eps) * k
    return (omega0(p) * 2 * P_circ * np.sin(2 * lam * k)
            / (scc.c**2 * np.sin(yp) * np.sin(ym)))


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def circulating_power(Io_over_Isql, p=PARAMS):
    """The model arm's power, ``I_c / 2 = I_0 c / (4 L gamma)`` (see above)."""
    return Io_over_Isql * I_SQL(p) * scc.c / (4 * p["L_m"] * p["gamma"])


def sr_edges(rho, phi, p=PARAMS, mlib=MLIB, T_itm=None):
    """Edge objects of the SRM-ITM-ETM cavity; ``phi`` in BC's convention."""
    M = p["m_kg"] / 4
    return [
        edges.MirrorEdge("SRM", Thr=1 - rho**2, mlib=mlib),
        edges.MirrorEdge("ITM", Thr=p["T_itm"] if T_itm is None else T_itm,
                         mlib=mlib),
        edges.RPMirrorEdge("ETM", suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=p["lambda_m"], mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=p["L_m"], mlib=mlib),
        edges.LinkEdge("SRC.L", L_m=0, detune_rad=sflu_detuning(phi), mlib=mlib),
    ]


def carrier(topology, edge_objs, drive, P_circ, mlib=MLIB):
    """DC fields with ``P_circ`` on the ETM and the arm carrier along the
    amplitude quadrature (BC's frame; any common carrier phase is a choice)."""
    dc = solve.solve_dc(solve.build(topology()), edge_objs, mlib,
                        drive={drive: mlib.LO(np.pi / 2)},
                        test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"})
    dc = solve.scale_dc(dc, "ETM.fr.i.tp", P_circ)
    E = dc["ETM.fr.i.tp"]
    rot = mlib.Mrotation(-np.arctan2(E[1, 0].real, E[0, 0].real))
    return {k: rot @ v for k, v in dc.items()}


def sr_ifo(F_Hz, rho, phi, P_circ, readout="SRM.bk.o.tp",
           inputs=("SRM.bk.i.exc", "ETM.pos.exc"), p=PARAMS, mlib=MLIB, T_itm=None):
    """Transfer matrices of the signal-recycled differential mode.

    ``rho``, ``phi`` in BC's convention; ``P_circ`` the model arm's power.
    The dark port is ``SRM.bk.i.exc`` in, ``SRM.bk.o.tp`` out; ``ETM.pos.exc``
    is the displacement ``x = L h``.
    """
    eo = sr_edges(rho, phi, p, mlib, T_itm)
    dc = carrier(topologies.signal_recycled, eo, "SRM.bk.i.exc", P_circ, mlib)
    return solve.solve_ac(solve.build(topologies.signal_recycled()), eo, mlib,
                          F_Hz, readout=readout, inputs=set(inputs), resultsDC=dc)


def closed_loop(Omega, rho, phi, P_circ, p=PARAMS, mlib=MLIB):
    """``x / x_exc = 1 / (1 - R_xx R_FF)`` at BC's (complex) ``Omega``."""
    Omega = np.asarray(Omega, dtype=complex)
    T = sr_ifo(-Omega / (2 * np.pi), rho, phi, P_circ, readout="ETM.pos.tp",
               inputs=("ETM.pos.exc",), p=p, mlib=mlib)
    return np.asarray(T["ETM.pos.exc"]).reshape(Omega.shape)


def model_R_FF(Omega, rho, phi, P_circ, p=PARAMS, mlib=MLIB):
    """The optical spring of the model, ``R_FF = (1 - 1 / G) / R_xx``."""
    G = closed_loop(Omega, rho, phi, P_circ, p, mlib)
    return (1 - 1 / G) / R_xx(np.asarray(Omega, dtype=complex), p)


def newton(f, guesses, scale, tol=1e-10):
    """Zeros of an analytic ``f`` near ``guesses``, all at once.

    ``f`` maps an array of complex frequencies to values; each step calls it
    once, at ``2 len(guesses)`` points (the second half for the derivative).
    """
    x = np.array(guesses, dtype=complex)
    n = len(x)
    for _ in range(50):
        h = 1e-7 * np.maximum(np.abs(x), scale)
        v = f(np.concatenate([x, x + h]))
        dx = v[:n] * h / (v[n:] - v[:n])
        x = x - dx
        if np.max(np.abs(dx)) < tol * scale:
            return x
    raise RuntimeError("newton did not converge")


def model_resonances(rho, phi, P_circ, guesses, p=PARAMS, mlib=MLIB):
    """Zeros of the model's ``1 - R_xx R_FF`` near ``guesses`` (BC's Omega).

    The model has more roots than the quartic -- copies of the optical pair
    one free spectral range apart -- so it needs the guesses; the quartic's
    roots (``resonances``) serve.
    """
    return newton(lambda W: 1 / closed_loop(W, rho, phi, P_circ, p, mlib),
                  guesses, p["gamma"])


def model_optical_pole(rho, phi, guess, p=PARAMS, mlib=MLIB):
    """The model's free optical resonance ``lambda - i epsilon``: the pole of
    its ``R_FF`` (independent of power) nearest ``guess``."""
    P = circulating_power(1.0, p)
    return newton(lambda W: 1 / model_R_FF(W, rho, phi, P, p, mlib),
                  [guess], p["gamma"])[0]
