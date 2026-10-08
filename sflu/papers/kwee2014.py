"""
Kwee, Miller, Isogai, Barsotti & Evans, "Decoherence and degradation of
squeezed states in quantum filter cavities", PRD 90 062006 (2014),
arXiv:1704.03531.

Mapping onto the graph
----------------------
The paper's chain (its Fig. 1) is squeezer -> injection loss -> filter cavity
(FC) -> interferometer -> readout loss -> balanced homodyne. Here every link of
that chain is a separately solved SFLU graph, joined with
``topologies.cascade``:

``INJ``   ``lossy_link``: a pick-off mirror of power transmission
          ``1 - Lambda_inj^2`` whose other input (``INJ.v.exc``) admits vacuum.
          Squeezed light enters at ``INJ.i.exc``.
``FC``    ``topologies.filter_cavity("FC", loss_ports=True)``: input mirror
          ``t_in^2``, end mirror carrying the round-trip loss
          ``Lambda_rt^2`` (vacuum enters at ``FC2.frL.i``), 16 m link.
``TEL``   ``lossy_link`` without loss: a "telescope" whose link edge changes
          basis from the squeezer's modes to the local oscillator's.
``IFO``   ``topologies.fp_arm()``: a tuned Fabry-Perot cavity with a free end
          mirror, standing in for the signal-recycled interferometer.
``RO``    ``lossy_link``: readout loss ``Lambda_ro^2``, vacuum at
          ``RO.v.exc``.

**Mode mismatch, with one higher-order mode.** Kwee et al. let every mode but
the FC's fundamental ``U_0`` see a non-resonant FC (``r_fc = 1``) and lump all
of them into ``t_mm = sum_{n>=1} a_n b_n^*``. With a single higher-order mode
their "upper bound" (Eq. 25 with equality) is *exact*: three unit vectors in
a two-dimensional mode space -- squeezer, FC and LO modes -- fixed by
``|<U_sqz|U_0>|^2 = a_0^2``, ``|<U_lo|U_sqz>|^2 = c_0^2`` and the phase
``phi_mm``. So the model uses ``MatrixLib(nhom=1)`` and works in the
*squeezer's* basis ``{U_sqz, U_sqz_perp}``:

* the FC link is ``M^-1 L M`` with ``M = mlib.MrotationMM(1 - a_0^2, 0)``,
  the change from the squeezer's basis to the cavity's (``LinkEdge`` ``MM_fr``
  / ``MM_to``). Mirrors act identically on both modes, so the FC reflects
  ``M^-1 diag(r_fc, r_HOM) M``. The cavity's higher-order mode gets a one-way
  Gouy phase of ``pi/2``, i.e. it is exactly anti-resonant and
  ``r_HOM = 1`` to within ``Lambda_rt^2/4``;
* the telescope's link is ``mlib.MrotationMM(1 - c_0^2, phi_mm)``, which
  takes the squeezer's basis to the LO's: ``|<U_lo|U_sqz>|^2 = c_0^2``, and
  the LO's overlap with the cavity mode, ``b_0``, is Eq. 25's for the given
  ``phi_mm``. The *sign* of ``phi_mm`` was fixed by comparing with the closed
  form; ``-phi_mm`` gives the mirror-image family, whose best member is at
  -9.0 dB instead of -8.3 dB at low frequency. (Kwee's ``phi_mm`` is a label
  for an unknown phase, and their reflectivity is conjugated -- see
  Conventions -- so the sign has no further meaning; what matters is that
  their "bounds" are over ``phi_mm`` in ``[0, pi]`` only.)
* from there on the field is in the LO's basis. The interferometer carrier is
  in the fundamental, so the interferometer acts on the LO mode exactly as
  Eq. 38 has it (``T_ifo`` applied after the projection onto ``U_lo``), and
  ``readout.quadrature``, which reads the fundamental only, *is* the
  projection onto ``U_lo``.
* the squeezer's state (``readout.squeezed``) squeezes the fundamental and
  leaves the higher-order mode in vacuum; that vacuum, and the vacuum entering
  every loss port, is what Kwee lump into ``Lambda_2`` (Eq. 41).

**Interferometer.** Only its ``K`` (Eq. 29) enters Fig. 2. Kwee's
signal-recycled interferometer has ``gamma_ifo`` and ``Omega_SQL`` of Eqs.
32-33; the model is a single cavity with that half-width (exact pole,
``topologies.half_bandwidth_T``) and a free end mirror of mass ``m/2``. That
is the reduced mass of one arm's two free test masses; Kwee's
``Omega_SQL,0`` (Eq. 31) is KLMTV's, and a single cavity of mass ``m/2``
reproduces it with the *real* arm power (KLMTV's own mapping, mass ``m/4``
with half the power, is equivalent: ``K`` depends on power over mass only).
The circulating power is set from ``Omega_SQL`` (``circulating_power``); it
comes out at exactly ``P_arm = 800 kW`` (since ``t_sr^2 = (1+r_sr)(1-r_sr)``),
which is the check on the ``m/2``. The model's ``K`` (``model_kappa``) then
matches Eq. 29 to 3e-3 below 1 kHz -- the difference between the exact pole
and Kwee's single-pole ``gamma_ifo`` -- and the SFLU curves are normalised
by the model's own ``1 + K^2``, so even that largely cancels.

Conventions
-----------
Kwee use the KLMTV/BnC quadratures with ``T_ifo = [[1, 0], [-K, 1]]``, signal
in quadrature 2; this package has ``b2 = a2 + K a1``. All of Fig. 2 is at
squeeze angle ``phi_sqz = 0`` (quadrature 2 squeezed) and readout angle
``zeta = 0`` (quadrature 2 read), which are ``pi/2`` here
(``readout.squeezed(..., pi/2)``, ``readout.quadrature(mlib, pi/2)``); the
sign flip between the conventions does not affect ``pi/2``, nor a symmetric
jitter about it.

**Filter-cavity detuning: a sign slip in the paper.** Kwee define
``Delta omega_fc = omega_fc - omega_0`` (resonance minus carrier) and write
the round-trip propagator as ``exp(-i Phi)``, ``Phi = (Omega - Delta
omega_fc) 2 L_fc / c`` (Eqs. 3-4). But their time convention is
``exp(-i omega t)`` (Eq. A1), under which a delay multiplies an amplitude by
``exp(+i omega tau)``: the physical propagator is ``exp(+i Phi)``. Their
Eq. 3 is therefore the complex conjugate of the physical reflectivity, and
the cavity that reproduces their Fig. 2 has the *carrier above the
resonance* by 48 Hz -- KLMTV's ``xi > 0``, and ``topologies.detuning_rad``'s
positive sign. So the FC link gets ``detuning_rad(+Delta omega_fc / 2 pi)``;
the other sign rotates the squeezing the wrong way (+7 dB at 50 Hz). The
closed form with ``exp(+i Phi)`` (``r_fc(..., physical=True)``) needs
``-Delta omega_fc`` to reproduce the figure, independently of this package.
Nothing in the paper's results changes: only the word "resonance minus
carrier" is reversed.

Kwee's homodyne row ``(sin zeta, cos zeta)`` has unit norm, and their
``N-hat`` (Eq. 43, normalised as in Eq. A21) is in units of vacuum, single-
or double-sided alike. Fig. 2 plots ``10 log10 [N-hat / (1 + K^2)]``, "relative
to coherent vacuum": the denominator is the noise of the same lossless
interferometer with vacuum at its input, *not* of the lossy chain. The SFLU
curves are normalised the same way, by the solved interferometer's own
vacuum noise (``vacuum_reference``).
"""
import numpy as np
import scipy.constants as scc
from wield.control.SFLU import optics

from sflu import edges, elements, readout, solve, topologies
from sflu.lib import MatrixLib

# Tables I and II. gamma_fc = 2 pi 61.4 Hz in Table II is *derived* from
# t_in^2 and Lambda_rt^2 (Eq. 10); it is recomputed here, not stored.
PARAMS = dict(
    omega0=2 * np.pi * 282e12,  # Table I
    L_m=3995.0,          # arm length, Table I
    T_arm=0.014,         # ITM transmission, Table I
    t_sr2=0.35,          # SRM transmission, Table I
    P_arm_W=800e3,       # arm power, Table I
    m_kg=40.0,           # each test mass, Table I
    L_fc_m=16.0,         # Table II
    t_in2=66.3e-6,       # FC input transmission, Table II
    loss_rt=16e-6,       # FC round-trip loss Lambda_rt^2, Table II
    detune_fc_Hz=48.0,   # Delta omega_fc / 2 pi, Table II
    loss_inj=0.05,       # Lambda_inj^2, Table II
    loss_ro=0.05,        # Lambda_ro^2, Table II
    mm_fc=0.02,          # Lambda_mmFC^2 = 1 - a_0^2, Table II
    mm_lo=0.05,          # Lambda_mmLO^2 = 1 - c_0^2, Table II
    dzeta=30e-3,         # frequency-independent phase noise [rad RMS], Table II
    dL_fc_m=0.3e-12,     # FC length noise [m RMS], Table II
    sqz_dB=9.1,          # injected squeezing, Table II
)

MLIB = MatrixLib(nhom=1)


def lambda_m(p=PARAMS):
    return 2 * np.pi * scc.c / p["omega0"]


def gamma_ifo(p=PARAMS):
    """Eqs. 30 and 32: interferometer half-width [rad/s]."""
    r = np.sqrt(1 - p["t_sr2"])
    return (1 + r) / (1 - r) * p["T_arm"] * scc.c / (4 * p["L_m"])


def Omega_SQL(p=PARAMS):
    """Eqs. 31 and 33 [rad/s]."""
    r, t = np.sqrt(1 - p["t_sr2"]), np.sqrt(p["t_sr2"])
    W0 = 8 / scc.c * np.sqrt(p["P_arm_W"] * p["omega0"] / (p["m_kg"] * p["T_arm"]))
    return t / (1 + r) * W0


def f_FSR(p=PARAMS):
    """``c / 2 L_fc``, used by Kwee as a *rate* [1/s] (Eqs. 8, 10)."""
    return scc.c / (2 * p["L_fc_m"])


def gamma_fc(t_in2, loss_rt, p=PARAMS):
    """Eq. 10: FC half-width [rad/s]."""
    return (t_in2 + loss_rt) / 2 * f_FSR(p)


def epsilon(t_in2, loss_rt):
    """Eq. 8."""
    return 2 * loss_rt / (t_in2 + loss_rt)


def detune_jitter_Hz(p=PARAMS):
    """Eq. 62: RMS detuning noise ``omega_0 dL / L_fc``, in Hz."""
    return p["omega0"] * p["dL_fc_m"] / p["L_fc_m"] / (2 * np.pi)


def ideal_fc(p=PARAMS):
    """The lossless optimum, Eqs. 51-52 and 54: ``Delta = gamma = Omega_SQL/sqrt 2``.

    Returns ``(t_in2, detune_Hz)``. Fig. 2's single-mechanism curves (all but
    "filter cavity losses") use this cavity; the paper does not say so, but
    only this cavity reproduces them (Table II's, with its loss removed,
    mis-rotates by ~14 deg at low frequency).
    """
    g = Omega_SQL(p) / np.sqrt(2)
    return 2 * g / f_FSR(p), g / (2 * np.pi)


def fig2_configs(p=PARAMS):
    """The curves of Fig. 2, as keyword sets for ``noise`` and ``model_noise``.

    Each is a dict of: FC ``t_in2``, ``loss_rt``, ``detune_Hz``; ``loss_inj``,
    ``loss_ro``; mismatches ``mm_fc``, ``mm_lo`` and phase ``phi_mm``; RMS
    jitters ``dphi_sqz`` (squeeze angle), ``dzeta`` (readout angle) and
    ``ddetune_Hz`` (FC detuning). Mismatch curves are families over
    ``phi_mm = k pi / 5``, ``k = 0..5``, given as lists.
    """
    t_id, d_id = ideal_fc(p)
    ideal = dict(t_in2=t_id, loss_rt=0.0, detune_Hz=d_id, loss_inj=0.0, loss_ro=0.0,
                 mm_fc=0.0, mm_lo=0.0, phi_mm=0.0,
                 dphi_sqz=0.0, dzeta=0.0, ddetune_Hz=0.0)
    phis = PHI_MM
    table_fc = dict(t_in2=p["t_in2"], loss_rt=p["loss_rt"], detune_Hz=p["detune_fc_Hz"])
    return {
        "Ideal system": [ideal],
        # The table's 30 mrad is "dzeta", but the plotted curve is a jitter
        # of the squeeze angle (see the example page).
        "Frequency independent phase noise": [dict(ideal, dphi_sqz=p["dzeta"])],
        "Injection/Readout losses": [dict(ideal, loss_inj=p["loss_inj"],
                                          loss_ro=p["loss_ro"])],
        "Mode-mismatch": [dict(ideal, mm_fc=p["mm_fc"], mm_lo=p["mm_lo"], phi_mm=f)
                          for f in phis],
        "Frequency dependent phase noise": [dict(ideal, ddetune_Hz=detune_jitter_Hz(p))],
        "Filter cavity losses": [dict(ideal, **table_fc)],
        "All mechanisms": [dict(ideal, **table_fc, loss_inj=p["loss_inj"],
                                loss_ro=p["loss_ro"], mm_fc=p["mm_fc"],
                                mm_lo=p["mm_lo"], phi_mm=f, dphi_sqz=p["dzeta"],
                                ddetune_Hz=detune_jitter_Hz(p))
                           for f in phis],
    }


# The mismatch phases of Fig. 2's families. Not stated in the paper; k pi / 5
# reproduces the low-frequency values of all six cyan curves.
PHI_MM = np.arange(6) * np.pi / 5


# ---------------------------------------------------------------------------
# closed forms, transcribed from the paper (Kwee's quadrature convention)
# ---------------------------------------------------------------------------

A2 = np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2)  # Eq. 11
A2i = np.linalg.inv(A2)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def squeezer(sigma, phi):
    """Eq. 1: ``S(sigma, phi) = R(phi) diag(e^sigma, e^-sigma) R(-phi)``."""
    return rot(phi) @ np.diag([np.exp(sigma), np.exp(-sigma)]) @ rot(-phi)


def kappa(Omega, p=PARAMS):
    """Eq. 29."""
    g = gamma_ifo(p)
    return (Omega_SQL(p) / Omega)**2 * g**2 / (Omega**2 + g**2)


def r_fc(Omega, t_in2, loss_rt, detune_Hz, p=PARAMS, physical=False):
    """Eqs. 7-10: FC amplitude reflectivity at sideband frequency ``Omega``.

    ``physical=True`` returns the complex conjugate: the reflectivity with the
    propagator ``exp(+i Phi)`` that Kwee's own time convention implies (see
    the module docstring).
    """
    g = gamma_fc(t_in2, loss_rt, p)
    xi = (Omega - 2 * np.pi * detune_Hz) / g
    r = (epsilon(t_in2, loss_rt) - 1 + 1j * xi) / (1 + 1j * xi)
    return np.conj(r) if physical else r


def two_photon(t_plus, t_minus):
    """Eq. 12 / Eq. 23: ``A2 diag(t_+, t_-^*) A2^-1``."""
    D = np.zeros(np.shape(t_plus) + (2, 2), dtype=complex)
    D[..., 0, 0] = t_plus
    D[..., 1, 1] = np.conj(t_minus)
    return A2 @ D @ A2i


def mismatch_overlaps(mm_fc, mm_lo, phi_mm):
    """Eqs. 25-27: ``(t_00, t_mm)`` from ``a_0^2``, ``c_0^2`` and ``phi_mm``."""
    a0, c0 = np.sqrt(1 - mm_fc), np.sqrt(1 - mm_lo)
    b0 = a0 * c0 + np.sqrt(mm_fc * mm_lo) * np.exp(1j * phi_mm)
    t00 = a0 * np.conj(b0)
    return t00, c0 - t00


def _N_hat(Omega, t_in2, loss_rt, detune_Hz, loss_inj, loss_ro, mm_fc, mm_lo,
           phi_mm, phi_sqz=0.0, zeta=0.0, p=PARAMS, physical=False):
    """Eqs. 38-43 at fixed parameters, normalised as in Eq. A21."""
    sigma = p["sqz_dB"] / (20 * np.log10(np.e))
    tau_inj, tau_ro = np.sqrt(1 - loss_inj), np.sqrt(1 - loss_ro)
    t00, tmm = mismatch_overlaps(mm_fc, mm_lo, phi_mm)
    tp = t00 * r_fc(Omega, t_in2, loss_rt, detune_Hz, p, physical) + tmm
    tm = t00 * r_fc(-Omega, t_in2, loss_rt, detune_Hz, p, physical) + tmm
    K = kappa(Omega, p)
    T_ifo = np.zeros(np.shape(Omega) + (2, 2))
    T_ifo[..., 0, 0] = T_ifo[..., 1, 1] = 1
    T_ifo[..., 1, 0] = -K                                       # Eq. 28
    T1 = tau_ro * T_ifo @ two_photon(tp, tm) @ (tau_inj * squeezer(sigma, phi_sqz))
    Lam2 = np.sqrt(np.maximum(0, 1 - tau_inj**2 * (np.abs(tp)**2 + np.abs(tm)**2) / 2))  # Eq. 41
    T2 = tau_ro * T_ifo * Lam2[..., None, None]
    b = np.array([np.sin(zeta), np.cos(zeta)])
    N = (np.sum(np.abs(b @ T1)**2, axis=-1) + np.sum(np.abs(b @ T2)**2, axis=-1)
         + loss_ro)                                             # Eq. 43, T3
    return N


def noise(Omega, cfg, p=PARAMS, physical=False):
    """``N-hat(zeta = 0)``, Eq. 43, with phase noise added by Eq. 61.

    ``cfg`` is one entry of ``fig2_configs``. The jitters are averaged by the
    paper's finite difference, ``(N(X + dX) + N(X - dX))/2 - N`` per
    parameter, summed.
    """
    keys = ("t_in2", "loss_rt", "detune_Hz", "loss_inj", "loss_ro",
            "mm_fc", "mm_lo", "phi_mm")
    kw = {k: cfg[k] for k in keys}
    N0 = _N_hat(Omega, **kw, p=p, physical=physical)
    N = N0.copy()
    for name, dX in (("phi_sqz", cfg["dphi_sqz"]), ("zeta", cfg["dzeta"]),
                     ("detune_Hz", cfg["ddetune_Hz"])):
        if dX == 0:
            continue
        pm = []
        for s in (+1, -1):
            kws = dict(kw)
            kws[name] = kws.get(name, 0.0) + s * dX
            pm.append(_N_hat(Omega, **kws, p=p, physical=physical))
        N += (pm[0] + pm[1]) / 2 - N0
    return N


def relative_dB(Omega, cfg, p=PARAMS, physical=False):
    """Fig. 2's y-axis: ``10 log10 [N-hat / (1 + K^2)]``."""
    return 10 * np.log10(noise(Omega, cfg, p, physical) / (1 + kappa(Omega, p)**2))


# ---------------------------------------------------------------------------
# the SFLU model
# ---------------------------------------------------------------------------

def lossy_link(name):
    """A one-way link through a pick-off mirror: ``<name>.i.exc`` in,
    ``<name>.o.tp`` out, vacuum entering at ``<name>.v.exc``.

    Edges: a ``MirrorEdge`` ``<name>`` used in transmission (power
    transmission ``1 - loss``, so the vacuum port couples with amplitude
    ``sqrt(loss)``) and a ``LinkEdge`` ``<name>.L`` ahead of it, which may
    carry a mode basis change.
    """
    ifo = optics.GraphElement()
    ifo.subgraph_add(name, elements.MirrorElement(), translation_xy=(0, 0))
    ifo[name].locations.update({"i.exc": (12, -12), "v.exc": (-12, 12),
                                "o.tp": (-12, -12)})
    ifo[name].edges.update({
        ("bk.i", "i.exc"): ".L",
        ("fr.i", "v.exc"): "1",
        ("o.tp", "fr.o"): "1",
    })
    return ifo


def link_stage(F_Hz, name, loss=0.0, MM=1, mlib=MLIB):
    """Solve a ``lossy_link``: inputs ``<name>.i.exc``, ``<name>.v.exc``."""
    edge_objs = [
        edges.MirrorEdge(name, Thr=1 - loss, mlib=mlib),
        edges.LinkEdge(f"{name}.L", L_m=0, MM_to=MM, mlib=mlib),
    ]
    return solve.solve_ac(solve.build(lossy_link(name)), edge_objs, mlib, F_Hz,
                          readout=f"{name}.o.tp",
                          inputs={f"{name}.i.exc", f"{name}.v.exc"})


def circulating_power(p=PARAMS, M_kg=None):
    """Circulating power giving a single cavity of mirror mass ``M_kg`` and
    half-width ``gamma_ifo`` Kwee's ``Omega_SQL`` (Eq. 33).

    For one free mirror, ``K = 8 P omega_0 / (M c L) gamma / (Omega^2 (gamma^2 +
    Omega^2))``, so ``Omega_SQL^2 = 8 P omega_0 / (M c L gamma)``. With
    ``M = m/2`` this equals ``P_arm`` to the accuracy of the paper's "Omega_SQL
    = t_sr/(1+r_sr) Omega_SQL,0" (exact for the identity used).
    """
    M_kg = p["m_kg"] / 2 if M_kg is None else M_kg
    return Omega_SQL(p)**2 * M_kg * scc.c * p["L_m"] * gamma_ifo(p) / (8 * p["omega0"])


def ifo_stage(F_Hz, p=PARAMS, mlib=MLIB):
    """The interferometer: a tuned cavity with Kwee's ``gamma_ifo`` and ``K``.

    Input ``ITM.bk.i.exc``, output ``ITM.bk.o.tp``; ``ETM.pos.exc`` is the
    end-mirror displacement.
    """
    L, g = p["L_m"], gamma_ifo(p)
    M = p["m_kg"] / 2
    lam = lambda_m(p)
    edge_objs = [
        edges.MirrorEdge("ITM", Thr=topologies.half_bandwidth_T(g / (2 * np.pi), L),
                         lambda_m=lam, mlib=mlib),
        edges.RPMirrorEdge("ETM", suscept=lambda F: -1 / (M * (2 * np.pi * F)**2),
                           lambda_m=lam, mlib=mlib),
        edges.LinkEdge("ARM.L", L_m=L, mlib=mlib),
    ]
    dc = solve.solve_dc(
        solve.build(topologies.fp_arm()), edge_objs, mlib,
        drive={"ITM.bk.i.exc": mlib.LO(np.pi / 2)},
        test_points={"ETM.fr.i.tp", "ETM.fr.o.tp"},
    )
    dc = solve.scale_dc(dc, "ETM.fr.i.tp", circulating_power(p))
    return solve.solve_ac(
        solve.build(topologies.fp_arm()), edge_objs, mlib, F_Hz,
        readout="ITM.bk.o.tp", inputs={"ITM.bk.i.exc", "ETM.pos.exc"},
        resultsDC=dc,
    )


def model_kappa(T_ifo):
    """``K`` of the solved interferometer, ``b2 = a2 + K a1`` here."""
    M = T_ifo["ITM.bk.i.exc"]
    return (M[..., 1, 0] / M[..., 1, 1]).real


def fc_stage(F_Hz, t_in2, loss_rt, detune_Hz, mm_fc=0.0, p=PARAMS, mlib=MLIB):
    """The filter cavity in reflection, in the squeezer's mode basis.

    Inputs ``FC1.bk.i.exc`` and ``FC2.frL.i`` (round-trip loss vacuum);
    output ``FC1.bk.o.tp``. ``detune_Hz`` is Kwee's ``Delta omega_fc / 2 pi``,
    which is physically the carrier *above* resonance (module docstring). The cavity's fundamental overlaps the
    squeezer's mode with power ``1 - mm_fc``; its higher-order mode is
    anti-resonant.
    """
    L = p["L_fc_m"]
    MM = mlib.MrotationMM(mm_fc, 0.0)
    edge_objs = [
        edges.MirrorEdge("FC1", Thr=t_in2, lambda_m=lambda_m(p), mlib=mlib,
                         loss_ports=True),
        edges.MirrorEdge("FC2", Lhr=loss_rt, lambda_m=lambda_m(p), mlib=mlib,
                         loss_ports=True),
        edges.LinkEdge("FC.L", L_m=L, detune_rad=topologies.detuning_rad(detune_Hz, L),
                       gouy_rad=np.pi / 2, MM_fr=MM, MM_to=mlib.Minv(MM), mlib=mlib),
    ]
    return solve.solve_ac(
        solve.build(topologies.filter_cavity("FC", loss_ports=True)), edge_objs, mlib,
        F_Hz, readout="FC1.bk.o.tp", inputs={"FC1.bk.i.exc", "FC2.frL.i"},
    )


def chain(F_Hz, cfg, T_ifo, p=PARAMS, mlib=MLIB, detune_Hz=None):
    """The whole chain INJ -> FC -> TEL -> IFO -> RO, input ``INJ.i.exc``.

    ``T_ifo`` is ``ifo_stage``'s result (it is the same for every curve).
    ``detune_Hz`` overrides ``cfg["detune_Hz"]`` (for detuning jitter).
    """
    detune_Hz = cfg["detune_Hz"] if detune_Hz is None else detune_Hz
    T = topologies.cascade(
        link_stage(F_Hz, "INJ", loss=cfg["loss_inj"], mlib=mlib),
        fc_stage(F_Hz, cfg["t_in2"], cfg["loss_rt"], detune_Hz, cfg["mm_fc"], p, mlib),
        "FC1.bk.i.exc")
    T = topologies.cascade(
        T, link_stage(F_Hz, "TEL", MM=mlib.MrotationMM(cfg["mm_lo"], cfg["phi_mm"]),
                      mlib=mlib),
        "TEL.i.exc")
    T = topologies.cascade(T, T_ifo, "ITM.bk.i.exc")
    T = topologies.cascade(T, link_stage(F_Hz, "RO", loss=cfg["loss_ro"], mlib=mlib),
                           "RO.i.exc")
    # A frequency-independent path (here RO's vacuum port) comes back from
    # solve_ac without the frequency axis; give every entry one, so that
    # readout.noise_budget's per-input PSDs can be summed.
    N = np.shape(F_Hz)
    return {k: np.broadcast_to(v, N + np.shape(v)[-2:]) for k, v in T.items()}


def _gauss(dX, n=5):
    """Gauss-Hermite nodes and weights for a Gaussian of RMS ``dX``.

    ``n = 2`` gives nodes ``+-dX`` with equal weights: Eq. 61's two-point
    average.
    """
    if dX == 0:
        return np.zeros(1), np.ones(1)
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return dX * x, w / w.sum()


def model_noise(T, cfg, p=PARAMS, mlib=MLIB, n=5):
    """Readout noise of a solved chain in units of vacuum, at Kwee's ``zeta = 0``.

    Squeeze-angle and readout-angle jitter are averaged over a Gaussian with
    an ``n``-point Gauss-Hermite rule (``n = 2`` is Eq. 61's finite
    difference). They cost no re-solve: only the input state and the readout
    row change.
    """
    q = readout.vacuum_psd(lambda_m(p))
    out = 0
    for dphi, wp in zip(*_gauss(cfg["dphi_sqz"], n)):
        state = {"INJ.i.exc": readout.squeezed(mlib, p["sqz_dB"], np.pi / 2 + dphi)}
        for dz, wz in zip(*_gauss(cfg["dzeta"], n)):
            row = readout.quadrature(mlib, np.pi / 2 + dz)
            budget = readout.noise_budget(T, row, state, lambda_m(p))
            out = out + wp * wz * np.sum(list(budget.values()), axis=0) / q
    return out


def vacuum_reference(T_ifo, p=PARAMS, mlib=MLIB):
    """Noise of the lossless interferometer with vacuum input: ``1 + K^2``."""
    q = readout.vacuum_psd(lambda_m(p))
    budget = readout.noise_budget(T_ifo, readout.quadrature(mlib, np.pi / 2),
                                  lambda_m=lambda_m(p))
    return np.sum(list(budget.values()), axis=0) / q


def model_relative_dB(F_Hz, cfg, T_ifo, p=PARAMS, mlib=MLIB, n=5):
    """Fig. 2's y-axis from the SFLU model.

    FC detuning jitter re-solves the chain at each Gauss-Hermite detuning.
    Several jitters are averaged on the product grid (Eq. 61 adds them
    instead; the difference is fourth order in the jitters).
    """
    N = 0
    for dd, w in zip(*_gauss(cfg["ddetune_Hz"], n)):
        T = chain(F_Hz, cfg, T_ifo, p, mlib, detune_Hz=cfg["detune_Hz"] + dd)
        N = N + w * model_noise(T, cfg, p, mlib, n)
    return 10 * np.log10(N / vacuum_reference(T_ifo, p, mlib))
