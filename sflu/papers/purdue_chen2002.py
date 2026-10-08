"""
P. Purdue and Y. Chen, "Practical speed meter designs for quantum
nondemolition gravitational-wave interferometers", PRD 66 122004 (2002),
arXiv:gr-qc/0208049.

Mapping onto the graph
----------------------
The interferometer (Purdue & Chen Figs. 2-4) is a Michelson whose dark port
leads, through a 45-degree *extraction mirror* (transmission ``T_o``), into a
second long cavity, the *sloshing cavity* (input mirror ``T_s``, length ``L``,
perfect end mirror). The extraction mirror's transmitted port is the readout;
its fourth port is shut by a *port-closing mirror*. In the practical version
(Fig. 3) each arm also has an ITM, and a matching *RSE mirror* (both ``T_i``)
sits between the beamsplitter and the extraction mirror.

Only differential sidebands reach the sloshing cavity and the readout, so the
model is the differential mode alone, one arm long:

``ETM`` -- ``ARM.L`` -- [``ITM`` -- ``RSEL`` -- ``RSE``] -- ``EXM`` (beamsplitter)

with ``EXM`` reflecting arm <-> sloshing cavity (``SLM`` -- ``SLC.L`` -- ``SLE``)
and transmitting arm <-> readout and sloshing <-> port-closing mirror
(``PCM``). The bracketed RSE cavity is present only with ``rse=True``; with
no loss it is transparent to sidebands (paper Sec. II), so the lossless
figures leave it out. ``RSEL`` is a reflection-free pass-through that carries
the RSE-cavity loss of Appendix B (one loss per pass, each direction).

The paper reduces this to a three-mirror chain -- port coupler (``4 T_o``),
arm, sloshing mirror, sloshing cavity, perfect mirror -- by coupled-mode
arguments. That chain is ``chain_model`` below, but it is only a
leading-order equivalent: with the input coupled to the arm alone, an
``O(T_o)`` position response survives and spoils the flat ``kappa`` below
~0.1 ``omega_opt`` (shown in the Fig. 6 example). In the real topology the
input reaches the arm both directly and via the port-closing mirror, and
the two cancel exactly at ``omega = 0`` -- the beamsplitter graph is an exact
speed meter (``kappa`` flat to 2e-5 down to 1e-3 ``omega_opt``). It also keeps
the moving ETM a perfect reflector carrying the paper's ``W_circ``, so the
figures use it.

*Radiation pressure and signal.* Only the ETM carries carrier light, so it is
the only ``RPMirrorEdge``; the sloshing and extraction optics see sidebands
only (paper Sec. II). Exactly as in ``sflu.papers.klmtv2001``, the four test
masses ``m`` are lumped into the ETM with mass ``m/4``, its displacement is the
differential arm length ``x = L h``, and it carries half the per-arm
circulating power, ``W_circ / 2``. The carrier lives in the common mode, which
this graph does not contain, so the graph is never DC-solved: the ETM's carrier
fields are set from ``W_circ`` directly.

*Exact pole placement.* The paper's ``delta = c T_o / L`` (Eq. 2) and
``Omega = c sqrt(T_s) / 2L`` (Eq. 1) are leading order. With them the model's
sloshing frequency comes out 0.8% low -- the effect of the ``Omega'``
correction in the paper's footnote to Sec. III A -- which moves the noise
minimum visibly. ``tuned_transmissions`` instead picks ``T_o`` and ``T_s`` so
that the model's own coupling ``kappa(omega)``, fitted to the form of Eq. 14,
has exactly the paper's ``delta`` and ``Omega``. Then Eq. 14 holds to 2e-3 up
to 10 ``omega_opt``.

*Tuning.* Every long cavity is on resonance (paper Sec. II). With this
package's mirror signs (``fr.r = -r``, ``bk.r = +r``) that needs a quarter-wave
(``pi/2`` one-way) phase on the sloshing cavity and on the port-closing link.
With the closing link at 0 instead, the two leaks of the extraction mirror
cancel and the readout is cut off; with the sloshing cavity at 0 it is
anti-resonant and the device is a position meter.

Conventions
-----------
Purdue & Chen use KLMTV's conventions: ``q2 = (p2 - kappa p1) e^{2i psi} + ...``
(Eq. 12). This package has ``b2 = a2 + K a1``, so every homodyne or squeeze
angle of the paper maps to minus itself (``sflu_angle``); checked numerically
in ``papers/test_purdue_chen2002.py``, where the other sign fails by orders of
magnitude. Their ``kappa`` is this package's ``K``.

PSDs are single-sided; ``h_SQL^2 = 8 hbar / (m omega^2 L^2)`` (Eq. 15) and the
figures' y-axis is ``sqrt(S_h) / h_SQL(2 pi 100 Hz)``.
"""
import numpy as np
import scipy.constants as scc
from wield.control.SFLU import optics

from sflu import edges, elements, readout, solve, topologies
from sflu.lib import MatrixLib

# Table I. The figures are built from the dimensionless parametrisation
# (delta = 2 omega_opt, Omega^2 = omega_opt^2 + delta^2/2, kappa_max = 5),
# not from Table I's rounded T_s, T_o, so those are not used.
PARAMS = dict(
    omega0=1.78e15,        # carrier [rad/s], Table I
    m_kg=40.0,             # each of the four test masses, Table I
    L_m=4e3,               # arms and sloshing cavity, Table I
    f_opt_Hz=100.0,        # omega_opt = 2 pi 100 Hz, Table I / Eq. 21
    delta_over_wopt=2.0,   # delta = 2 omega_opt, Table I (Eq. 29: 1.977)
    OmegaI3_over_wopt3=20.0,  # Omega_I^3 = 20 omega_opt^3: kappa_max = 5, Eq. 24
    sqz_dB=10.0,           # e^{-2R} = 0.1, Sec. IV A
    T_i=0.005,             # ITM and RSE mirror, Table I
    losses=dict(           # Table IV
        arm=2e-5, slosh=2e-5, ext=2e-5, RSE=2e-5, close=2e-5,
        OPC=3e-3,          # local oscillator + photodiode + circulator
        F=5e-3,            # filter cavities incl. mode mismatch
    ),
    # Sec. IV B: the two output filters for Eq. 49 (4 km long, as the arms).
    filters=((1.7355, 91.57), (-1.1133, 114.3)),  # (xi_J, delta_J / 2pi [Hz])
)

MLIB = MatrixLib(nhom=0)


def lambda_m(p=PARAMS):
    return 2 * np.pi * scc.c / p["omega0"]


def w_opt(p=PARAMS):
    return 2 * np.pi * p["f_opt_Hz"]


def h_SQL(omega, p=PARAMS):
    """Eq. 15, single-sided."""
    return np.sqrt(8 * scc.hbar / (p["m_kg"] * omega**2 * p["L_m"]**2))


def W_circ(OmegaI3, p=PARAMS):
    """Eq. 23 inverted: the per-arm circulating power for a given ``Omega_I^3``."""
    return OmegaI3 * p["m_kg"] * p["L_m"] * scc.c / (16 * p["omega0"])


def Omega_slosh(delta, p=PARAMS):
    """Eq. 21 inverted: ``Omega^2 = omega_opt^2 + delta^2 / 2``."""
    return np.sqrt(w_opt(p)**2 + delta**2 / 2)


def sflu_angle(paper_angle):
    """A Purdue-Chen (KLMTV-convention) homodyne or squeeze angle, here."""
    return -np.asarray(paper_angle)


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper
# ---------------------------------------------------------------------------

def calL(omega, delta, Omega):
    """Eq. 9."""
    return Omega**2 - omega**2 - 1j * omega * delta


def psi(omega, delta, Omega):
    """Eq. 13, as the phase of ``e^{i psi}`` with ``e^{2 i psi} = -L*/L``."""
    return np.angle(1j * np.conj(calL(omega, delta, Omega))
                    / np.abs(calL(omega, delta, Omega)))


def kappa(omega, W, delta, Omega, p=PARAMS):
    """Eq. 14."""
    return (16 * p["omega0"] * delta * W
            / (p["m_kg"] * scc.c * p["L_m"] * np.abs(calL(omega, delta, Omega))**2))


def kappa_max(W, delta, p=PARAMS):
    """Eq. 24, ``kappa`` at ``omega_opt`` with ``Omega`` from Eq. 21."""
    OmegaI3 = 16 * p["omega0"] * W / (p["m_kg"] * p["L_m"] * scc.c)
    return OmegaI3 / (delta * (w_opt(p)**2 + delta**2 / 4))


def kappa_eq49(omega, p=PARAMS):
    """Eq. 49: the SISM's ``kappa`` for ``delta = 2 omega_opt`` and
    ``Omega_I^3 delta = 4 omega_opt^4``."""
    wo = w_opt(p)
    return 4 * wo**4 / ((omega**2 - wo**2)**2 + 8 * wo**4)


def Sh_lossless(omega, K, cotPhi, p=PARAMS):
    """Eq. 19: ``h_SQL^2 [(cot Phi - kappa)^2 + 1] / (2 kappa)``."""
    return h_SQL(omega, p)**2 * ((cotPhi - K)**2 + 1) / (2 * K)


def Sh_squeezed(omega, K, cotPhi, lam, sqz_dB, printed=False, p=PARAMS):
    """Eq. 46: squeeze angle ``lam``, homodyne ``cot Phi``, both arbitrary.

    As printed the prefactor is ``h_SQL^2 / kappa``; it should be
    ``h_SQL^2 / (2 kappa)``, as in Eqs. 12b and 19 (the same slip is in
    Eqs. 41b, 42 and 44). ``printed=True`` gives the printed version. For
    ``lam = pi/2`` the corrected form reduces exactly to the lossless part of
    Eq. 59.
    """
    R2 = sqz_dB / 10 * np.log(10)  # 2R
    Kt = K - cotPhi                # Eq. 45
    Psit = np.arctan2(1, Kt)
    pref = h_SQL(omega, p)**2 / K * (1 if printed else 0.5)
    return (pref * (1 + Kt**2)
            * (np.exp(-R2) + np.sinh(R2) * (1 - np.cos(2 * (Psit + lam)))))


def epsilon_AES(omega, delta, Omega, p=PARAMS):
    """Eq. B8: ``eps_arm + eps_ext + eps_slosh Omega^2 / omega^2``."""
    e = p["losses"]
    return e["arm"] + e["ext"] + e["slosh"] * Omega**2 / omega**2


def loss_factors(omega, delta, Omega, external=("OPC",), p=PARAMS):
    """Table III: ``{N: (E^S_N, E^R_N)}`` for the internal losses plus the
    ``external`` ones named (``"OPC"``, ``"F"``).

    ``T_o = delta L / c`` (Eq. 2) and ``delta_i = c T_i / 4L``, as in the paper.
    """
    e, L, Ti = p["losses"], p["L_m"], p["T_i"]
    To = delta * L / scc.c
    di = scc.c * Ti / (4 * L)
    bi = np.arctan(omega / di)
    aL = np.abs(calL(omega, delta, Omega))
    eip = np.exp(1j * psi(omega, delta, Omega))
    eAES = epsilon_AES(omega, delta, Omega, p)
    rse_S = np.sqrt(e["RSE"] * Ti / (4 * To) * (1 + omega**2 / di**2)) * omega * delta / aL
    rse_R = np.sqrt(e["RSE"] * To / Ti) / (omega * delta)
    out = {
        "AES": (np.sqrt(eAES / To) * omega * delta / aL,
                -eip / 2 * np.sqrt(eAES / To)),
        "close": (np.sqrt(e["close"]) * (Omega**2 - omega**2) / aL,
                  -1j * eip / 2 * np.sqrt(e["close"])),
        "RSE_in": (rse_S, eip * np.exp(-1j * bi) * rse_R
                   * (omega * (di + delta) + 1j * Omega**2)),
        "RSE_out": (rse_S, eip * np.exp(1j * bi) * rse_R
                    * (omega * (di - delta) - 1j * Omega**2)),
    }
    for name in external:
        out[name] = (np.sqrt(e[name]) * np.ones_like(omega), 0 * omega)
    return out


def Sh_lossy(omega, K, cotPhi, sqz_dB, factors, p=PARAMS):
    """Eq. 59 (Eq. 57 for ``sqz_dB = 0``): phase-squeezed input
    (``lambda = pi/2``), homodyne ``cot Phi``, with the loss factors of
    ``loss_factors``. ``K`` is ``kappa*``, Eq. 14 at the lossy ``W*_circ``.
    """
    R2 = sqz_dB / 10 * np.log(10)
    total = (cotPhi - K)**2 * np.exp(R2) + np.exp(-R2)
    for ES, ER in factors.values():
        total = total + np.abs(ES * cotPhi - ER * K)**2 + ES**2
    return h_SQL(omega, p)**2 / (2 * K) * total


def filter_angle(omega, filters=None, p=PARAMS):
    """App. A (A1, A8): the homodyne angle two lossless filters realise with
    ``theta = pi/2``: ``pi/2 - sum_{J,+-} arctan(xi_J +- omega/delta_J)``.
    With the Sec. IV B filters, ``cot`` of it is Eq. 49's ``kappa``.
    """
    filters = p["filters"] if filters is None else filters
    total = sum(np.arctan(xi + s * omega / (2 * np.pi * f))
                for xi, f in filters for s in (+1, -1))
    return np.pi / 2 - total


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def _bs_ports(ifo, name):
    """External input/output on the beamsplitter's readout port ``bkA``."""
    ifo.locations.update({f"{name}.bkA.i.exc": (-25, -12), f"{name}.bkA.o.tp": (-25, 12)})
    ifo.edges.update({(f"{name}.bkA.i", f"{name}.bkA.i.exc"): "1",
                      (f"{name}.bkA.o.tp", f"{name}.bkA.o"): "1"})


def speed_meter(rse=False, loss_ports=False):
    """The differential mode of the three-cavity speed meter (Figs. 2, 3).

    Edge names: ``ETM`` (RP mirror), ``EXM`` (``BSEdge``), ``SLM``, ``SLE``,
    ``PCM`` mirror edges; links ``ARM.L``, ``SLC.L``, ``PCM.L`` (port-closing
    path) and ``SHORT`` (zero-length joins). With ``rse=True`` also ``ITM``,
    ``RSE`` and the pass-through loss ``RSEL`` (needs ``loss_ports=True`` to
    expose ``RSEL.frL.i`` / ``RSEL.bkL.i``, the RSE_in / RSE_out vacua).

    The readout is ``EXM.bkA.o.tp``, the input ``EXM.bkA.i.exc``. Vacuum also
    enters through the transmissions of ``ETM``, ``SLE`` and ``PCM``
    (``<name>.bk.i``) when those are given any.
    """
    ifo = optics.GraphElement()
    ifo.subgraph_add("ETM", elements.RPMirrorElement(), translation_xy=(60, 0),
                     rotation_deg=0)
    ifo.subgraph_add("EXM", elements.BeamSplitterElement(), translation_xy=(-30, 0),
                     rotation_deg=0)
    ifo.subgraph_add("SLM", elements.MirrorElement(), translation_xy=(-30, 30),
                     rotation_deg=90)
    ifo.subgraph_add("SLE", elements.MirrorElement(), translation_xy=(-30, 70),
                     rotation_deg=90)
    ifo.subgraph_add("PCM", elements.MirrorElement(), translation_xy=(-30, -30),
                     rotation_deg=270)
    if rse:
        # ITM front face toward the ETM, so the ITM-ETM arm is resonant for
        # the carrier (fr.r = -r against the ETM's -1), and the two back faces
        # (bk.r = +r) facing each other, so the short ITM-RSE cavity is
        # resonant too. The pair then transmits sidebands with unit gain and
        # no quadrature rotation. (Turning the ITM round leaves the lossless
        # model unchanged but makes the arm anti-resonant, which wrongly
        # enhances the RSE-cavity loss by ~1/T_i.)
        ifo.subgraph_add("ITM", elements.MirrorElement(), translation_xy=(30, 0),
                         rotation_deg=0)
        ifo.subgraph_add("RSEL", elements.MirrorElement(loss_ports=loss_ports),
                         translation_xy=(10, 0), rotation_deg=0)
        ifo.subgraph_add("RSE", elements.MirrorElement(), translation_xy=(-10, 0),
                         rotation_deg=180)
        ifo.edges.update({
            ("ETM.fr.i", "ITM.fr.o"): "ARM.L", ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
            ("RSEL.fr.i", "ITM.bk.o"): "SHORT", ("ITM.bk.i", "RSEL.fr.o"): "SHORT",
            ("RSE.bk.i", "RSEL.bk.o"): "SHORT", ("RSEL.bk.i", "RSE.bk.o"): "SHORT",
            ("EXM.frA.i", "RSE.fr.o"): "SHORT", ("RSE.fr.i", "EXM.frA.o"): "SHORT",
        })
    else:
        ifo.edges.update({
            ("ETM.fr.i", "EXM.frA.o"): "ARM.L", ("EXM.frA.i", "ETM.fr.o"): "ARM.L",
        })
    ifo.edges.update({
        # extraction mirror: frA = arm, frB = sloshing (reflection pair),
        # bkA = readout, bkB = port-closing mirror
        ("SLM.fr.i", "EXM.frB.o"): "SHORT", ("EXM.frB.i", "SLM.fr.o"): "SHORT",
        ("SLE.fr.i", "SLM.bk.o"): "SLC.L", ("SLM.bk.i", "SLE.fr.o"): "SLC.L",
        ("PCM.fr.i", "EXM.bkB.o"): "PCM.L", ("EXM.bkB.i", "PCM.fr.o"): "PCM.L",
    })
    _bs_ports(ifo, "EXM")
    return ifo


SIGNAL = "ETM.pos.exc"
PORT = "EXM.bkA.i.exc"


def _edges(To, Ts, W, losses, p, mlib, rse):
    L = p["L_m"]
    M = p["m_kg"] / 4
    lp = losses is not None
    e = losses or {}
    T_e = e.get("arm", 0) + e.get("ext", 0)  # AES: same form, lumped at the ETM
    objs = [
        edges.RPMirrorEdge("ETM", Thr=T_e, suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=lambda_m(p), mlib=mlib),
        edges.BSEdge("EXM", Thr=To, mlib=mlib),
        edges.MirrorEdge("SLM", Thr=Ts, mlib=mlib),
        edges.MirrorEdge("SLE", Thr=e.get("slosh", 0), mlib=mlib),
        edges.MirrorEdge("PCM", Thr=e.get("close", 0), mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
        edges.LinkEdge("SLC.L", L_m=L, detune_rad=np.pi / 2, mlib=mlib),
        edges.LinkEdge("PCM.L", L_m=0, detune_rad=np.pi / 2, mlib=mlib),
        edges.LinkEdge("SHORT", L_m=0, mlib=mlib),
    ]
    if rse:
        objs += [
            edges.MirrorEdge("ITM", Thr=p["T_i"], mlib=mlib),
            edges.MirrorEdge("RSE", Thr=p["T_i"], mlib=mlib),
            edges.MirrorEdge("RSEL", Thr=1, Lhr=e.get("RSE", 0), loss_in_transmission=True,
                             loss_ports=lp, mlib=mlib),
        ]
    # The carrier, set by hand (see module docstring): W/2 on the ETM.
    E = np.sqrt(W / 2) * mlib.LO(np.pi / 2)
    dc = {"ETM.fr.i.tp": E, "ETM.fr.o.tp": -np.sqrt(1 - T_e) * E}
    return objs, dc


def model(F_Hz, W, delta, Omega, losses=None, rse=None, transmissions=None,
          p=PARAMS, mlib=MLIB):
    """Transfer matrices of the speed meter to its readout.

    Parameters
    ----------
    W : per-arm circulating power ``W_circ`` (or ``W*_circ``) [W].
    delta, Omega : extraction rate and sloshing frequency [rad/s].
    losses : dict like ``PARAMS["losses"]`` or None for lossless. The internal
        ones (arm, ext, slosh, close, RSE) go into the graph; external ones
        are applied with ``with_output_loss``.
    rse : include the ITM/RSE pair; defaults to ``losses is not None``.

    Inputs: ``PORT`` (vacuum or squeezing), ``SIGNAL`` (``x = L h`` [m]), and
    with losses ``ETM.bk.i``, ``SLE.bk.i``, ``PCM.bk.i``, ``RSEL.frL.i``,
    ``RSEL.bkL.i``.
    """
    rse = (losses is not None) if rse is None else rse
    To, Ts = transmissions or tuned_transmissions(delta, Omega, p)
    objs, dc = _edges(To, Ts, W, losses, p, mlib, rse)
    inputs = {PORT, SIGNAL}
    if losses is not None:
        inputs |= {"ETM.bk.i", "SLE.bk.i", "PCM.bk.i"}
        if rse:
            inputs |= {"RSEL.frL.i", "RSEL.bkL.i"}
    return solve.solve_ac(
        solve.build(speed_meter(rse=rse, loss_ports=losses is not None)), objs, mlib,
        F_Hz, readout="EXM.bkA.o.tp", inputs=inputs, resultsDC=dc,
    )


def model_kappa(T):
    """The model's ``kappa`` (= this package's ``K``), from ``b2 = a2 + K a1``."""
    M = T[PORT]
    return (M[..., 1, 0] / M[..., 1, 1]).real


_TUNED = {}


def tuned_transmissions(delta, Omega, p=PARAMS, mlib=MLIB, iterations=5):
    """``(T_o, T_s)`` giving the model exactly the paper's ``delta``, ``Omega``.

    Starts from Eqs. 1-2 and iterates: fit ``1 / kappa_model`` to Eq. 14's
    form ``(Omega^2 - omega^2)^2 + omega^2 delta^2`` (a quadratic in
    ``omega^2``) over 0.1-3 ``omega_opt``, read off the model's ``delta`` and
    ``Omega``, and rescale ``T_o ~ delta``, ``T_s ~ Omega^2``. Converges to
    1e-12 in four steps.
    """
    key = (delta, Omega, p["L_m"])
    if key in _TUNED:
        return _TUNED[key]
    L = p["L_m"]
    W = W_circ(p["OmegaI3_over_wopt3"] * w_opt(p)**3, p)
    C = p["m_kg"] * scc.c * L / (16 * p["omega0"] * W)
    w = np.geomspace(0.1, 3, 30) * w_opt(p)
    To, Ts = delta * L / scc.c, (2 * L * Omega / scc.c)**2
    for _ in range(iterations):
        K = model_kappa(model(w / (2 * np.pi), W, delta, Omega, transmissions=(To, Ts),
                              p=p, mlib=mlib))
        a0, a1, a2 = np.polynomial.polynomial.polyfit(w**2, 1 / K, 2)
        d = C / a2
        O = (a0 * d / C)**0.25
        To *= delta / d
        Ts *= (Omega / O)**2
    _TUNED[key] = (To, Ts)
    return To, Ts


def chain_model(F_Hz, W, delta, Omega, p=PARAMS, mlib=MLIB):
    """The paper's coupled-mode reduction as a three-mirror chain.

    ``CPL`` (moving, output coupler, ``T = 1 - e^{-4 delta L/c}``, i.e. ``4 T_o``
    to first order) -- ``ARM.L`` -- ``SLM`` -- ``SLC.L`` -- ``SLE`` (perfect),
    with ``T_s`` putting the chain's exact poles at the roots of Eq. 9 and
    carrier ``W/2`` incident on ``CPL``. Readout ``CPL.bk.o.tp``, inputs
    ``CPL.bk.i.exc`` and ``CPL.pos.exc``.

    This is *not* an exact speed meter. At ``omega = 0`` the sloshing cavity
    returns the light to the arm with the wrong sign, so the arm is
    anti-resonant rather than dark, and an input field still leaks into it
    with amplitude ``~ T/4`` relative to the resonant build-up. That residual
    position sensitivity is negligible near ``omega_opt`` but dominates
    ``kappa`` below ~0.1 ``omega_opt``. In the real topology the second leak
    through the port-closing mirror cancels it exactly; see the Fig. 6
    example. The moving coupler also makes ``kappa`` ``O(T)`` (~5%) low,
    since the sideband it generates is ``r E 2 k x`` rather than ``E 2 k x``.
    """
    L, c = p["L_m"], scc.c
    Tc = -np.expm1(-4 * delta * L / c)
    w = np.sqrt(Omega**2 - delta**2 / 4)
    rs = np.cos(2 * w * L / c) / np.cosh(delta * L / c)
    M = p["m_kg"] / 4
    ifo = optics.GraphElement()
    ifo.subgraph_add("CPL", elements.RPMirrorElement(), translation_xy=(0, 0),
                     rotation_deg=180)
    ifo.subgraph_add("SLM", elements.MirrorElement(), translation_xy=(40, 0),
                     rotation_deg=0)
    ifo.subgraph_add("SLE", elements.MirrorElement(), translation_xy=(80, 0),
                     rotation_deg=0)
    ifo.edges.update({
        ("SLM.fr.i", "CPL.fr.o"): "ARM.L", ("CPL.fr.i", "SLM.fr.o"): "ARM.L",
        ("SLE.fr.i", "SLM.bk.o"): "SLC.L", ("SLM.bk.i", "SLE.fr.o"): "SLC.L",
    })
    topologies._ports(ifo, "CPL")
    objs = [
        edges.RPMirrorEdge("CPL", Thr=Tc, suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=lambda_m(p), mlib=mlib),
        edges.MirrorEdge("SLM", Thr=1 - rs**2, mlib=mlib),
        edges.MirrorEdge("SLE", mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
        edges.LinkEdge("SLC.L", L_m=L, detune_rad=np.pi / 2, mlib=mlib),
    ]
    r = np.sqrt(1 - Tc)
    E = np.sqrt(W / 2) * mlib.LO(np.pi / 2)
    dc = {"CPL.fr.i.tp": E, "CPL.fr.o.tp": -r * E}
    return solve.solve_ac(
        solve.build(ifo), objs, mlib, F_Hz, readout="CPL.bk.o.tp",
        inputs={"CPL.bk.i.exc", "CPL.pos.exc"}, resultsDC=dc,
    )


def with_output_loss(T, eps, name, mlib=MLIB):
    """A lossy element ahead of the homodyne (the paper's OPC or F terms):
    every transfer matrix times ``sqrt(1 - eps)``, plus a vacuum input
    ``name`` entering with ``sqrt(eps)``."""
    out = {k: np.sqrt(1 - eps) * v for k, v in T.items()}
    n = len(next(iter(T.values())))
    out[name] = np.broadcast_to(np.sqrt(eps) * mlib.Id, (n, mlib.dim, mlib.dim))
    return out


def filter_cavity(F_Hz, name, xi, delta_Hz, L_m=None, p=PARAMS, mlib=MLIB):
    """A lossless detuned filter cavity in reflection, as in Sec. IV B / App. A.

    Input ``<name>1.bk.i.exc``, output ``<name>1.bk.o.tp``. Length 4 km
    (the paper does not state it; it matters only through ``delta``).
    """
    L_m = p["L_m"] if L_m is None else L_m
    objs = [
        edges.MirrorEdge(f"{name}1", Thr=topologies.half_bandwidth_T(delta_Hz, L_m),
                         mlib=mlib),
        edges.MirrorEdge(f"{name}2", mlib=mlib),
        edges.LinkEdge(f"{name}.L", L_m=L_m,
                       detune_rad=topologies.detuning_rad(xi * delta_Hz, L_m), mlib=mlib),
    ]
    return solve.solve_ac(
        solve.build(topologies.filter_cavity(name)), objs, mlib, F_Hz,
        readout=f"{name}1.bk.o.tp", inputs={f"{name}1.bk.i.exc"},
    )


def with_output_filters(T, F_Hz, filters=None, p=PARAMS, mlib=MLIB):
    """The speed meter followed by the two output filters of Sec. IV B."""
    filters = p["filters"] if filters is None else filters
    (xi_I, f_I), (xi_II, f_II) = filters
    T = topologies.cascade(T, filter_cavity(F_Hz, "FCI", xi_I, f_I, p=p, mlib=mlib),
                           "FCI1.bk.i.exc")
    return topologies.cascade(T, filter_cavity(F_Hz, "FCII", xi_II, f_II, p=p, mlib=mlib),
                              "FCII1.bk.i.exc")


def sqrt_Sh_over_hSQL(T, zeta, states=None, signal=SIGNAL, p=PARAMS, mlib=MLIB):
    """``sqrt(S_h) / h_SQL(omega_opt)``, the y-axis of Figs. 8, 10, 12.

    ``zeta`` (and squeeze angles in ``states``) in this package's convention;
    use ``sflu_angle`` on the paper's.
    """
    Sx, _ = readout.referred_psd(T, readout.quadrature(mlib, zeta), signal,
                                 states=states, lambda_m=lambda_m(p))
    return np.sqrt(Sx) / p["L_m"] / h_SQL(w_opt(p), p)
