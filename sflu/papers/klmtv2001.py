"""
Kimble, Levin, Matsko, Thorne & Vyatchanin, "Conversion of conventional
gravitational-wave interferometers into quantum nondemolition interferometers
by modifying their input and/or output optics", PRD 65 022002 (2001),
arXiv:gr-qc/0008026.

Mapping onto the graph
----------------------
KLMTV's interferometer is a Michelson with Fabry-Perot arms, carrier on
resonance, no signal recycling. Its differential mode is a single
Fabry-Perot cavity, so the model is ``topologies.fp_arm()``: a lossless ITM
and a perfectly reflecting, free end mirror with radiation pressure.

The four test masses of mass ``m`` move the differential coordinate
``x = L h`` with reduced mass ``m/4`` (KLMTV footnote to Eq. 13). The model's
end mirror therefore has mass ``m/4``, its displacement *is* ``L h``, and
``S_h = S_x / L^2``. KLMTV's ``K`` (Eq. 18) depends on the light only through
the arm power, so the single cavity carries input power ``I_o / 4`` with the
matching circulating power ``(I_o / 4) c / (L gamma)``; then ``K`` agrees
exactly with Eq. 18. (Not ``4 P / T``: that differs at ``O(T)``, ~1.7% here,
which is enough to spoil the back-action cancellation of variational
readout.) For the same reason the ITM transmission puts the cavity's exact
pole at ``gamma`` (``topologies.half_bandwidth_T``) rather than using KLMTV's
first-order ``T = 4 L gamma / c``.

Conventions
-----------
In this package the ponderomotive term enters as ``b2 = a2 + K a1`` (up to
an overall phase); KLMTV write ``b2 = a2 - K a1``. The two differ only by
the sign of the phase quadrature, so every KLMTV angle -- squeeze angle
``lambda``, homodyne angle ``zeta`` -- maps to minus itself here. ``sflu_angle``
does the conversion. Filter-cavity detunings ``xi`` need no conversion:
``topologies.detuning_rad`` already uses KLMTV's sign.

PSDs are single-sided, as in KLMTV Eq. 22.
"""
import numpy as np
import scipy.constants as scc

from sflu import edges, readout, solve, topologies
from sflu.lib import MatrixLib

# Table I and Sec. II. omega_o: the text's 1.78e15 rad/s (1.06 um); Table I
# rounds it to 1.8e15. Nothing plotted depends on it -- every figure is
# normalised by I_SQL and h_SQL.
PARAMS = dict(
    lambda_m=1064e-9,
    L_m=4e3,               # arm length, Eq. 11
    m_kg=30.0,             # each test mass, Table I
    gamma=2 * np.pi * 100,  # arm half-bandwidth [rad/s], Eq. 11
    sqz_dB=10.0,           # e^{-2R} = 0.1, Table I
)

MLIB = MatrixLib(nhom=0)


def omega0(p=PARAMS):
    return 2 * np.pi * scc.c / p["lambda_m"]


def I_SQL(p=PARAMS):
    """Eq. 19: the input power at which the noise touches the SQL at gamma."""
    return p["m_kg"] * p["L_m"]**2 * p["gamma"]**4 / (4 * omega0(p))


def h_SQL(Omega, p=PARAMS):
    """Eq. 20, single-sided."""
    return np.sqrt(8 * scc.hbar / (p["m_kg"] * Omega**2 * p["L_m"]**2))


def sflu_angle(klmtv_angle):
    """A KLMTV squeeze or homodyne angle, in this package's convention."""
    return -np.asarray(klmtv_angle)


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def kappa(Omega, Io_over_Isql, p=PARAMS):
    """Eq. 18, the optomechanical coupling constant."""
    g = p["gamma"]
    return Io_over_Isql * 2 * g**4 / (Omega**2 * (g**2 + Omega**2))


def Phi(Omega, Io_over_Isql, p=PARAMS):
    """Eq. 45, ``arccot K``: the ponderomotive squeeze angle."""
    return np.arctan2(1, kappa(Omega, Io_over_Isql, p))


def Sh_conventional(Omega, Io_over_Isql, p=PARAMS):
    """Eq. 27: phase readout, vacuum input."""
    K = kappa(Omega, Io_over_Isql, p)
    return h_SQL(Omega, p)**2 / 2 * (1 / K + K)


def Sh_squeezed_input(Omega, Io_over_Isql, lam, sqz_dB, p=PARAMS):
    """Eq. 46: squeezed vacuum at angle ``lam``, phase readout."""
    R2 = sqz_dB / 10 * np.log(10)  # 2R
    return (Sh_conventional(Omega, Io_over_Isql, p)
            * (np.cosh(R2) - np.cos(2 * (lam + Phi(Omega, Io_over_Isql, p)))
               * np.sinh(R2)))


def Sh_variational(Omega, Io_over_Isql, p=PARAMS):
    """Eq. 58: vacuum input, homodyne at ``zeta = Phi``; no back-action."""
    return h_SQL(Omega, p)**2 / (2 * kappa(Omega, Io_over_Isql, p))


def Sh_squeezed_variational(Omega, Io_over_Isql, sqz_dB, p=PARAMS):
    """Eq. 73: phase-squeezed input and ``zeta = Phi``."""
    return Sh_variational(Omega, Io_over_Isql, p) * 10**(-sqz_dB / 10)


def Sh_general(Omega, K, zeta, lam, sqz_dB, p=PARAMS):
    """Eq. 71: squeeze angle ``lam`` and homodyne angle ``zeta``, both arbitrary.

    Takes ``K`` itself rather than the power, so that it can be evaluated with
    the coupling a model actually has (``model_kappa``) and angles tuned for a
    different one -- which is what a filter cavity designed from Eq. 18 does.
    """
    R2 = sqz_dB / 10 * np.log(10)
    Kt = K - 1 / np.tan(zeta)
    Phit = np.arctan2(1, Kt)
    return (h_SQL(Omega, p)**2 / (2 * K) * (1 + Kt**2)
            * (np.exp(-R2) + np.sinh(R2) * (1 - np.cos(2 * (Phit + lam)))))


def output_filters(Io_over_Isql):
    """Eq. 89: detunings ``xi`` and half-widths ``delta/gamma`` of the two
    output filter cavities that realise ``zeta(Omega) = Phi(Omega)``.

    Returns ``((xi_I, delta_I/gamma), (xi_II, delta_II/gamma))``.
    """
    P = 8 * Io_over_Isql  # 4 Lambda^4 / gamma^4, Eq. 84
    Q = (1 + np.sqrt(1 + P**2)) / 2
    Ap = (Q + np.sqrt(Q)) / P
    Am = (Q - np.sqrt(Q)) / P
    xi_I = 1 / (2 * Ap) + np.sqrt(1 + 1 / (2 * Ap)**2)
    xi_II = 1 / (2 * Am) - np.sqrt(1 + 1 / (2 * Am)**2)
    d_I = np.sqrt(P / (8 * xi_I * np.sqrt(Q)))
    d_II = np.sqrt(P / (8 * -xi_II * np.sqrt(Q)))
    return (xi_I, d_I), (xi_II, d_II)


def input_filters():
    """Eq. 90 as printed: input filters for squeezed input at ``I_o = I_SQL``.

    These are the ``I_o = I_SQL`` output filters with the detunings negated.
    Used as printed, with squeezing at ``theta = pi/2`` ahead of them, the
    squeezing lands at the wrong angle. The *un*-negated filters --
    ``output_filters(1.0)`` -- give exactly ``lambda = -Phi``, which is what
    ``papers/test_klmtv2001.py`` shows. Our reading: a cavity rotates a
    state's squeeze angle by ``+alpha`` but a readout angle by ``-alpha``, so
    one set of filters serves both purposes, ``theta - alpha = Phi`` at the
    output and ``theta + alpha = -Phi`` at the input agreeing mod pi when
    ``theta = pi/2``. The sign flip in Eq. 90 counts that difference twice.
    The statement is convention-independent: changing the sign of the phase
    quadrature reverses every angle and every rotation together.
    """
    Q = (1 + np.sqrt(65)) / 2
    Ap = -(Q + np.sqrt(Q)) / 8
    Am = -(Q - np.sqrt(Q)) / 8
    xi_I = 1 / (2 * Ap) - np.sqrt(1 + 1 / (2 * Ap)**2)
    xi_II = 1 / (2 * Am) + np.sqrt(1 + 1 / (2 * Am)**2)
    return ((xi_I, np.sqrt(1 / (-xi_I * np.sqrt(Q)))),
            (xi_II, np.sqrt(1 / (xi_II * np.sqrt(Q)))))


def filter_angle(Omega, filters, p=PARAMS):
    """The homodyne angle a pair of lossless filters realises with fixed
    ``theta = pi/2`` readout: ``pi/2 - sum_{J,+-} arctan(xi_J +- Omega/delta_J)``.

    This is the *full* sum. KLMTV Eq. 88 and Eq. 81 write half of it; with the
    half-sum their Eq. 89 parameters miss ``Phi`` by up to pi/4, while the full
    sum -- the actual reflection phase of a detuned cavity, ``2 arctan`` --
    recovers it to machine precision. See ``papers/test_klmtv2001.py``.
    """
    g = p["gamma"]
    total = sum(np.arctan(xi + s * Omega / (d * g))
                for xi, d in filters for s in (+1, -1))
    return np.pi / 2 - total


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def arm(F_Hz, Io_over_Isql, p=PARAMS, mlib=MLIB):
    """Transfer matrices of the differential arm, input -> output port.

    Inputs: ``"ITM.bk.i.exc"`` (vacuum or squeezing enters here) and
    ``"ETM.pos.exc"`` (differential displacement ``x = L h`` [m]).
    """
    L, g = p["L_m"], p["gamma"]
    M = p["m_kg"] / 4
    edge_objs = [
        edges.MirrorEdge("ITM", Thr=topologies.half_bandwidth_T(g / (2 * np.pi), L),
                         mlib=mlib),
        edges.RPMirrorEdge("ETM", suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=p["lambda_m"], mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
    ]
    dc = solve.solve_dc(
        solve.build(topologies.fp_arm()), edge_objs, mlib,
        drive={"ITM.bk.i.exc": mlib.LO(np.pi / 2)},
        test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"},
    )
    P_in = Io_over_Isql * I_SQL(p) / 4
    dc = solve.scale_dc(dc, "ETM.fr.i.tp", P_in * scc.c / (L * g))
    return solve.solve_ac(
        solve.build(topologies.fp_arm()), edge_objs, mlib, F_Hz,
        readout="ITM.bk.o.tp", inputs={"ITM.bk.i.exc", "ETM.pos.exc"},
        resultsDC=dc,
    )


def model_kappa(T):
    """The coupling ``K`` of a solved model, read off its transfer matrix.

    ``b2 = a2 + K a1`` here, so ``K`` is the ratio of the a1 -> b2 and a2 -> b2
    elements. It agrees with Eq. 18 to ~2e-5 at low frequency and ~2e-3 at
    10 gamma; the difference is KLMTV's single-pole approximation of the arm.
    That is negligible almost everywhere, but not for squeezed variational
    readout, where 10 dB of anti-squeezing amplifies any error in the
    back-action cancellation.
    """
    M = T["ITM.bk.i.exc"]
    return (M[..., 1, 0] / M[..., 1, 1]).real


def filter_cavity(F_Hz, name, xi, delta_over_gamma, p=PARAMS, mlib=MLIB, L_m=None):
    """A lossless detuned filter cavity in reflection (KLMTV Fig. 2).

    Input ``<name>1.bk.i.exc``, output ``<name>1.bk.o.tp``. KLMTV take the
    filters as long as the arms.
    """
    L_m = p["L_m"] if L_m is None else L_m
    delta_Hz = delta_over_gamma * p["gamma"] / (2 * np.pi)
    edge_objs = [
        edges.MirrorEdge(f"{name}1", Thr=topologies.half_bandwidth_T(delta_Hz, L_m),
                         mlib=mlib),
        edges.MirrorEdge(f"{name}2", mlib=mlib),
        edges.LinkEdge(f"{name}.L", L_m=L_m,
                       detune_rad=topologies.detuning_rad(xi * delta_Hz, L_m),
                       mlib=mlib),
    ]
    return solve.solve_ac(
        solve.build(topologies.filter_cavity(name)), edge_objs, mlib, F_Hz,
        readout=f"{name}1.bk.o.tp", inputs={f"{name}1.bk.i.exc"},
    )


def with_output_filters(T_arm, F_Hz, filters, p=PARAMS, mlib=MLIB):
    """The arm followed by two output filter cavities (KLMTV Fig. 2)."""
    (xi_I, d_I), (xi_II, d_II) = filters
    T = topologies.cascade(T_arm, filter_cavity(F_Hz, "FCI", xi_I, d_I, p, mlib),
                           "FCI1.bk.i.exc")
    return topologies.cascade(T, filter_cavity(F_Hz, "FCII", xi_II, d_II, p, mlib),
                              "FCII1.bk.i.exc")


def with_input_filters(T_arm, F_Hz, filters, p=PARAMS, mlib=MLIB):
    """Two input filter cavities ahead of the arm (KLMTV Fig. 1).

    Squeezed light now enters at ``"FCI1.bk.i.exc"``.
    """
    (xi_I, d_I), (xi_II, d_II) = filters
    T = topologies.cascade(filter_cavity(F_Hz, "FCI", xi_I, d_I, p, mlib),
                           filter_cavity(F_Hz, "FCII", xi_II, d_II, p, mlib),
                           "FCII1.bk.i.exc")
    return topologies.cascade(T, T_arm, "ITM.bk.i.exc")


def sqrt_Sh_over_hSQL(T, zeta, states=None, p=PARAMS, mlib=MLIB):
    """``sqrt(S_h) / h_SQL(gamma)``, the y-axis of KLMTV Fig. 4.

    ``zeta`` and any squeeze angles in ``states`` are in this package's
    convention; use ``sflu_angle`` on KLMTV's.
    """
    Sx, _ = readout.referred_psd(T, readout.quadrature(mlib, zeta), "ETM.pos.exc",
                                 states=states, lambda_m=p["lambda_m"])
    return np.sqrt(Sx) / p["L_m"] / h_SQL(p["gamma"], p)
