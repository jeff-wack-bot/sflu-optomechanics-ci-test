r"""
P. Purdue and Y. Chen, *Practical speed meter designs for quantum
nondemolition gravitational-wave interferometers*,
[Phys. Rev. D 66, 122004 (2002)](https://doi.org/10.1103/PhysRevD.66.122004),
[arXiv:gr-qc/0208049](https://arxiv.org/abs/gr-qc/0208049).

**What the paper shows.** The dark port of a Michelson is coupled to a second
long cavity, the *sloshing cavity*. Signal sidebands move into it and come back
to the arms with their sign flipped. The position signal then cancels, and the
output measures the test masses' *velocity*. The input-output relation has
KLMTV's form (Eq. 12),

$$
q_2 = (p_2 - \kappa p_1) e^{2i\psi} + \sqrt{2\kappa}\,\frac{h}{h_\mathrm{SQL}} e^{i\psi},
\qquad
\kappa = \frac{16\omega_0\delta W_\mathrm{circ}}{m c L\,|\mathcal L(\omega)|^2},
\qquad
\mathcal L = \Omega^2 - \omega^2 - i\omega\delta ,
$$

but now $\kappa$ is *flat* at low frequency instead of growing as $1/\omega^2$.
A single frequency-independent homodyne angle, $\cot\Phi = \kappa_\mathrm{max}$,
therefore removes back-action noise over the whole low-frequency band.

**The model.** The model is the differential mode of the paper's Fig. 2/3
topology, built from SFLU edges: a free ETM of mass $m/4$ carrying $W/2$, the
4 km arm, a beamsplitter as the extraction mirror, the sloshing cavity, and the
port-closing mirror. For the lossy model the ITM/RSE pair is added too.
``sflu.papers.purdue_chen2002`` describes the mapping. In short: no new
library code is needed. The extraction and sloshing transmissions are chosen
so that the model's own $\kappa(\omega)$ has exactly the paper's $\delta$ and
$\Omega$. The leading-order Eq. 1 would put $\Omega$ 0.8% low, which is the
paper's own footnoted $\Omega'$ correction.

**Sign convention.** Purdue & Chen follow KLMTV ($q_2 = p_2 - \kappa p_1$).
This package has $b_2 = a_2 + K a_1$, so their homodyne and squeeze angles
enter negated (``sflu_angle``). The Fig. 8 example checks this.

**Reproduced:** Fig. 6 ($\kappa$), Fig. 8 (lossless, several $\delta$),
Fig. 10 (squeezed input, fixed vs frequency-dependent readout, also with real
filter cavities) and Fig. 12 (all four lossy curves). Every curve agrees with
the paper's closed form to better than 1%.

**Found along the way:**

* Eqs. 41b, 42, 44 and 46 drop a factor 2: they should read
  $h_\mathrm{SQL}^2/(2\kappa)$, as in Eqs. 12, 19 and 59. SFLU agrees with the
  corrected Eq. 46 for every squeeze and homodyne angle tried. As printed, Eq. 46
  is $\sqrt2$ too high, and Fig. 10 itself was evidently drawn with the
  corrected form.
* The paper's coupled-mode picture reduces the device to a three-mirror
  chain. That chain is only a leading-order equivalent: it keeps a residual
  position response that ruins the flat $\kappa$ below about $0.1\,\omega_\mathrm{opt}$.
  The port-closing mirror is what makes the real topology an exact speed meter.
* Figs. 14 and 15 are not reproduced, because their plotted curves do not match the
  parameters in their captions.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import purdue_chen2002 as pc

P = pc.PARAMS
MLIB = pc.MLIB
WOPT = pc.w_opt()
SQZ = P["sqz_dB"]
W_OSM = pc.W_circ(P["OmegaI3_over_wopt3"] * WOPT**3)  # 8.36 MW, Eq. 30
W_SISM = W_OSM * 10**(-SQZ / 10)                         # 836 kW, Sec. IV A
SQUEEZED = {pc.PORT: pc.readout.squeezed(MLIB, SQZ, pc.sflu_angle(np.pi / 2))}


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def homodyne(cotPhi):
    """The paper's homodyne angle, given as ``cot Phi``, in SFLU's convention."""
    return pc.sflu_angle(np.arctan2(1, cotPhi))


def sql_line(ax, x):
    ax.loglog(x, 1 / x, c="k", lw=1, label="SQL")


def test_fig6(tpath_join):
    r"""Fig. 6: the coupling constant $\kappa(\omega)$, and why the closing mirror matters.

    The power is fixed at $\Omega_I^3 = 20\,\omega_\mathrm{opt}^3$ and
    $\delta = 0.5, 2, 4\,\omega_\mathrm{opt}$, with $\Omega^2 = \omega_\mathrm{opt}^2
    + \delta^2/2$. The model's $\kappa$ is read off its transfer matrix
    (``model_kappa``) and compared with Eq. 14. It is flat at low frequency,
    with plateaus of 8, 4.4 and 1.
    """
    x = np.geomspace(0.1, 10, 200)
    w = x * WOPT
    F = w / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for dr in (0.5, 2.0, 4.0):
        d = dr * WOPT
        O = pc.Omega_slosh(d)
        K = pc.model_kappa(pc.model(F, W_OSM, d, O))
        Kp = pc.kappa(w, W_OSM, d, O)
        ax.loglog(x, K, lw=2, alpha=0.7, label=f"$\\delta = {dr:g}\\,\\omega_\\mathrm{{opt}}$")
        ax.loglog(x, Kp, ls="--", c="k", lw=0.8)
        print(f"delta = {dr:3g} w_opt: kappa(0.1 w_opt) = {K[0]:.3g}, "
              f"max |SFLU/Eq.14 - 1| = {rel_err(K, Kp):.2e}")
        assert rel_err(K, Kp) < 3e-3
    assert np.allclose([pc.kappa(0, W_OSM, dr * WOPT, pc.Omega_slosh(dr * WOPT))
                        for dr in (0.5, 2, 4)], [7.9, 4.44, 0.99], rtol=0.01)

    # The paper's coupled-mode reduction is a chain: port coupler (4 T_o),
    # arm, sloshing mirror, sloshing cavity. Take it literally as an SFLU
    # graph (``chain_model``) and the moving mirror doubles as the coupler.
    # The trouble is the chain's input reaches only the arm. At omega = 0 the
    # sloshing cavity hands the light back with the wrong sign, so the arm is
    # anti-resonant rather than dark. A field of relative size ~T/4 is left
    # in it, and that field gives a *position* response. Near omega_opt it
    # hardly matters; below ~0.1 omega_opt it takes over (|K| plotted, as K
    # is no longer real). In the real topology the input also reaches the arm
    # via the port-closing mirror, and the two paths cancel exactly. Its
    # kappa stays flat to 2e-5 down to 1e-3 omega_opt.
    d = 2 * WOPT
    O = pc.Omega_slosh(d)
    xl = np.geomspace(1e-3, 10, 200)
    Tc = pc.chain_model(xl * WOPT / (2 * np.pi), W_OSM, d, O)
    Mc = Tc["CPL.bk.i.exc"]
    Kc = np.abs(Mc[:, 1, 0] / Mc[:, 1, 1])
    Kb = pc.model_kappa(pc.model(xl * WOPT / (2 * np.pi), W_OSM, d, O))
    ax.loglog(xl[xl >= 0.1], Kc[xl >= 0.1], ls=":", c="C1", lw=1.5,
              label="$\\delta = 2\\,\\omega_\\mathrm{opt}$, three-mirror chain")
    Kp = pc.kappa(xl * WOPT, W_OSM, d, O)
    print(f"chain: |K|/Eq.14 = {Kc[xl >= 0.1][0] / Kp[xl >= 0.1][0]:.3f} at 0.1 w_opt, "
          f"{Kc[0] / Kp[0]:.1f} at 1e-3 w_opt; full topology: "
          f"{rel_err(Kb[xl < 0.1], Kp[xl < 0.1]):.1e} off below 0.1 w_opt")
    assert rel_err(Kb[xl < 0.1], Kp[xl < 0.1]) < 1e-4
    assert Kc[0] / Kp[0] > 10
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 100)
    ax.set_xlabel("$\\omega / \\omega_\\mathrm{opt}$")
    ax.set_ylabel("$\\kappa$")
    ax.set_title("Purdue & Chen Fig. 6 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig6.pdf"))


def test_fig8(tpath_join):
    r"""Fig. 8: lossless speed meter, noise for several extraction rates $\delta$.

    The power is fixed at $W_\mathrm{circ} = 8.36$ MW. For each $\delta$,
    $\Omega$ follows from Eq. 21 and the homodyne angle is $\cot\Phi =
    \kappa_\mathrm{max}(\delta)$ (Eq. 24): 8.53, 5 and 3.12 for
    $\delta/\omega_\mathrm{opt} = 1.5, 2, 2.5$. The closed form is Eq. 19. The
    paper's fourth (dotted) curve, "$\kappa_\mathrm{max} > \cot\Phi$", comes
    without parameters and is omitted.
    """
    x = np.geomspace(0.1, 10, 200)
    w = x * WOPT
    F = w / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sql_line(ax, x)
    ax.axvline(1, c="k", lw=0.5)
    y01 = {}
    for dr, ls in ((1.5, "--"), (2.0, "-"), (2.5, ":")):
        d = dr * WOPT
        O = pc.Omega_slosh(d)
        To, Ts = pc.tuned_transmissions(d, O)
        print(f"delta = {dr} w_opt: T_o = {To:.5f} (Eq. 2: {d * P['L_m'] / pc.scc.c:.5f}), "
              f"T_s = {Ts:.4e} (Eq. 1: {(2 * P['L_m'] * O / pc.scc.c)**2:.4e})")
        T = pc.model(F, W_OSM, d, O)
        cot = pc.kappa_max(W_OSM, d)
        y = pc.sqrt_Sh_over_hSQL(T, homodyne(cot))
        paper = np.sqrt(pc.Sh_lossless(w, pc.kappa(w, W_OSM, d, O), cot)) / pc.h_SQL(WOPT)
        ax.loglog(x, y, lw=2, alpha=0.7, ls=ls,
                  label=f"$\\delta = {dr:g}\\,\\omega_\\mathrm{{opt}}$, $\\cot\\Phi = {cot:.3g}$")
        ax.loglog(x, paper, ls="--", c="k", lw=0.8)
        print(f"   max |SFLU/Eq.19 - 1| = {rel_err(y, paper):.2e}")
        assert rel_err(y, paper) < 3e-3
        y01[dr] = y[0]

        # The sign convention: the paper's homodyne angle used un-negated
        # reads the wrong combination, and the curve misses by a large factor.
        wrong = pc.sqrt_Sh_over_hSQL(T, np.arctan2(1, cot))
        assert rel_err(wrong, paper) > 1
        if dr == 2.0:
            i = np.argmin(y)
            print(f"   minimum {y[i]:.3f} at {x[i]:.2f} w_opt")
            assert abs(y[i] - 0.26) < 0.01 and 1.2 < x[i] < 1.4

    # Spot values read off the paper's figure at omega = 0.1 omega_opt.
    print("y(0.1 w_opt):", {k: round(float(v), 2) for k, v in y01.items()})
    assert np.allclose([y01[1.5], y01[2.0], y01[2.5]], [5.9, 3.82, 4.2], rtol=0.03)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.1, 10)
    ax.set_xlabel("$\\omega / \\omega_\\mathrm{opt}$")
    ax.set_ylabel("$[S_h(f) / S_\\mathrm{SQL}(100\\,\\mathrm{Hz})]^{1/2}$")
    ax.set_title("Purdue & Chen Fig. 8 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig8.pdf"))


def test_fig10(tpath_join):
    r"""Fig. 10: squeezed input, fixed vs frequency-dependent homodyne.

    10 dB of phase-squeezed vacuum ($\lambda = \pi/2$) lets the circulating
    power drop tenfold, to 836 kW, for the same noise (Eq. 43). With the fixed
    angle $\cot\Phi = 0.5$ the curve is the $\delta = 2\,\omega_\mathrm{opt}$
    curve of Fig. 8. With the frequency-dependent angle $\cot\Phi = \kappa(\omega)$
    (Eqs. 48-49) all back-action is removed and the high-frequency noise falls
    by $e^{-2R}$.

    The closed form is Eq. 46. It is shown corrected (dashed black, prefactor
    $h_\mathrm{SQL}^2/2\kappa$) and as printed (dotted grey, $h_\mathrm{SQL}^2/\kappa$).
    SFLU follows the corrected form, and so does the paper's own figure.
    """
    x = np.geomspace(0.1, 20, 200)
    w = x * WOPT
    F = w / (2 * np.pi)
    d = 2 * WOPT
    O = pc.Omega_slosh(d)
    T = pc.model(F, W_SISM, d, O)
    K = pc.kappa(w, W_SISM, d, O)
    assert rel_err(K, pc.kappa_eq49(w)) < 1e-12  # Eq. 49 is Eq. 14 here

    # Tolerance 1e-2: up to 20 omega_opt (2 kHz) the arm is no longer short
    # compared with the signal wavelength (omega L / c = 0.17), and the model's
    # kappa leaves the single-pole Eq. 14 by ~1%.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sql_line(ax, x)
    curves = {}
    for label, cot in (("fixed angle, $\\cot\\Phi = 0.5$", 0.5 + 0 * w),
                       ("frequency-dependent, $\\cot\\Phi = \\kappa(\\omega)$",
                        pc.kappa_eq49(w))):
        y = pc.sqrt_Sh_over_hSQL(T, homodyne(cot), SQUEEZED)
        paper = np.sqrt(pc.Sh_squeezed(w, K, cot, np.pi / 2, SQZ)) / pc.h_SQL(WOPT)
        printed = np.sqrt(pc.Sh_squeezed(w, K, cot, np.pi / 2, SQZ, printed=True)) / pc.h_SQL(WOPT)
        ax.loglog(x, y, lw=2, alpha=0.7, label=label)
        ax.loglog(x, paper, ls="--", c="k", lw=0.8)
        ax.loglog(x, printed, ls=":", c="0.5", lw=1)
        print(f"{label:50s} |SFLU/Eq.46 - 1| = {rel_err(y, paper):.2e}; "
              f"vs Eq. 46 as printed {rel_err(y, printed):.2f}")
        assert rel_err(y, paper) < 1e-2
        assert abs(rel_err(y, printed) - (1 - 1 / np.sqrt(2))) < 5e-3
        curves[label] = y

    # Eq. 46 for general angles: squeeze angle lambda and homodyne angle Phi
    # are both arbitrary, and the corrected formula holds for all of them.
    for lam in (0.3, 1.0, 2.0):
        for cot in (0.2, 1.0, 3.0):
            y = pc.sqrt_Sh_over_hSQL(
                T, homodyne(cot),
                {pc.PORT: pc.readout.squeezed(MLIB, SQZ, pc.sflu_angle(lam))})
            paper = np.sqrt(pc.Sh_squeezed(w, K, cot, lam, SQZ)) / pc.h_SQL(WOPT)
            assert rel_err(y, paper) < 1e-2, (lam, cot)
    print("Eq. 46 (corrected) holds for lambda in (0.3, 1, 2), cot Phi in (0.2, 1, 3)")

    # The frequency-dependent angle made by real optics: the speed meter's
    # output passes the two 4 km filter cavities of Sec. IV B, then a fixed
    # homodyne at theta = pi/2. The filter detunings xi_J are used with the
    # sign of topologies.detuning_rad, as for KLMTV. Negating them puts the
    # rotation the wrong way, which the last assertion checks.
    Tf = pc.with_output_filters(T, F)
    yf = pc.sqrt_Sh_over_hSQL(Tf, pc.sflu_angle(np.pi / 2), SQUEEZED)
    fd = curves["frequency-dependent, $\\cot\\Phi = \\kappa(\\omega)$"]
    ax.loglog(x, yf, ls="-.", lw=1.2, c="C3", label="frequency-dependent via filter cavities")
    print(f"filter cavities vs ideal FD angle: max rel. diff. {rel_err(yf, fd):.2e}")
    assert rel_err(yf, fd) < 1e-3
    flipped = tuple((-xi, f) for xi, f in P["filters"])
    yflip = pc.sqrt_Sh_over_hSQL(pc.with_output_filters(T, F, flipped),
                                 pc.sflu_angle(np.pi / 2), SQUEEZED)
    assert rel_err(yflip, fd) > 1

    # Spot values read off the paper's figure.
    at = lambda y, x0: float(np.interp(np.log(x0), np.log(x), y))
    print("FD curve at 2, 10, 20 w_opt:", [round(at(fd, x0), 3) for x0 in (2, 10, 20)])
    assert np.allclose([at(fd, 2), at(fd, 10), at(fd, 20)], [0.23, 1.1, 2.2], rtol=0.03)
    ax.set_xlim(0.1, 20)
    ax.set_ylim(0.1, 10)
    ax.set_xlabel("$\\omega / \\omega_\\mathrm{opt}$")
    ax.set_ylabel("$[S_h(f) / S_\\mathrm{SQL}(100\\,\\mathrm{Hz})]^{1/2}$")
    ax.set_title("Purdue & Chen Fig. 10 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig10.pdf"))


def test_fig12(tpath_join):
    r"""Fig. 12: lossy speed meters.

    The internal losses of Table IV, each $2\times10^{-5}$, go into the graph:

    * arm and extraction-mirror loss as ETM transmission;
    * sloshing loss as the sloshing end mirror's transmission;
    * the port-closing mirror's transmission;
    * one loss per pass, in each direction, inside the ITM-RSE cavity.

    That last one needs the ITM and RSE mirror ($T_i = 0.005$) in the graph.
    The readout losses (OPC, 0.003) and the filter loss (F, 0.005, SVSM
    only) are applied as a loss ahead of the homodyne. The closed form is
    Eq. 57 (OSM) or Eq. 59 (SISM, SVSM), using the Table III loss factors
    with the frequency-dependent $\varepsilon_\mathrm{AES}$ of Eq. B8.

    * Lossless SM and OSM: $W = 8.36$ MW, $\cot\Phi = 5$.
    * SISM: 836 kW, 10 dB phase squeezing, $\cot\Phi = 0.5$.
    * SVSM: as SISM, but with $\cot\Phi = \kappa^*(\omega)$ and filter loss.
    """
    f = np.geomspace(3, 1000, 200)
    w = 2 * np.pi * f
    d = 2 * WOPT
    O = pc.Omega_slosh(d)
    e = P["losses"]
    T0 = pc.model(f, W_OSM, d, O)
    TO = pc.with_output_loss(pc.model(f, W_OSM, d, O, losses=e), e["OPC"], "OPC")
    TS = pc.with_output_loss(pc.model(f, W_SISM, d, O, losses=e), e["OPC"], "OPC")
    TV = pc.with_output_loss(TS, e["F"], "F")
    Ko = pc.kappa(w, W_OSM, d, O)
    Ks = pc.kappa(w, W_SISM, d, O)
    fac = pc.loss_factors(w, d, O)
    facF = pc.loss_factors(w, d, O, external=("OPC", "F"))
    curves = [
        ("lossless SM", "-", pc.sqrt_Sh_over_hSQL(T0, homodyne(5)),
         pc.Sh_lossless(w, Ko, 5)),
        ("OSM", "--", pc.sqrt_Sh_over_hSQL(TO, homodyne(5)),
         pc.Sh_lossy(w, Ko, 5, 0, fac)),
        ("SISM", ":", pc.sqrt_Sh_over_hSQL(TS, homodyne(0.5), SQUEEZED),
         pc.Sh_lossy(w, Ks, 0.5, SQZ, fac)),
        ("SVSM", "-.", pc.sqrt_Sh_over_hSQL(TV, homodyne(Ks), SQUEEZED),
         pc.Sh_lossy(w, Ks, Ks, SQZ, facF)),
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(f, WOPT / w, c="0.6", lw=1, label="SQL")

    # The paper's loss formulas are leading order, but they hold well here:
    # every curve agrees with SFLU to ~0.5%. The first attempt did not. With
    # the ITM turned round, the ITM-ETM arm is anti-resonant for the carrier.
    # The lossless model does not notice, since the ITM-RSE pair is
    # transparent either way, but the RSE-cavity loss is then enhanced by
    # ~1/T_i, and the noise comes out three times higher at 100 Hz. The
    # paper's analysis, and the module, have the arm resonant.
    spots = {}
    for label, ls, y, Sh in curves:
        paper = np.sqrt(Sh) / pc.h_SQL(WOPT)
        ax.loglog(f, y, lw=2, alpha=0.7, ls=ls, label=label)
        ax.loglog(f, paper, ls="--", c="k", lw=0.8)
        print(f"{label:12s} max |SFLU/paper - 1| = {rel_err(y, paper):.2e}")
        assert rel_err(y, paper) < 1e-2, label
        spots[label] = [float(np.interp(np.log(f0), np.log(f), y)) for f0 in (3, 10, 130, 1000)]

    # Spot values at 3, 10, 130 and 1000 Hz, from Eqs. 57/59 evaluated
    # independently of this module; they agree with the paper's figure by eye
    # (at 130 Hz the lossy curves sit just above the lossless minimum).
    expected = {"lossless SM": [12.8, 3.82, 0.260, 5.64],
                "OSM": [58.9, 6.57, 0.275, 5.65],
                "SISM": [22.4, 4.24, 0.271, 5.65],
                "SVSM": [21.6, 3.90, 0.269, 1.15]}
    for label, vals in expected.items():
        print(f"{label:12s} at 3, 10, 130, 1000 Hz:", np.round(spots[label], 3))
        assert np.allclose(spots[label], vals, rtol=0.02), label
    # The paper's text: losses raise the SISM noise by 73% at 3 Hz.
    print(f"SISM / lossless at 3 Hz: {spots['SISM'][0] / spots['lossless SM'][0]:.2f}")
    ax.set_xlim(3, 1000)
    ax.set_ylim(0.2, 50)
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$[S_h(f) / S_\\mathrm{SQL}(100\\,\\mathrm{Hz})]^{1/2}$")
    ax.set_title("Purdue & Chen Fig. 12 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig12.pdf"))
