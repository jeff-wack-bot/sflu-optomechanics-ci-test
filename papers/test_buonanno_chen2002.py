"""
A. Buonanno and Y. Chen,
*Signal recycled laser-interferometer gravitational-wave detectors as optical
springs*,
[Phys. Rev. D 65, 042001 (2002)](https://doi.org/10.1103/PhysRevD.65.042001),
[arXiv:gr-qc/0107021](https://arxiv.org/abs/gr-qc/0107021).

**What the paper shows.** In a *detuned* signal-recycled interferometer the
light pushes back on the mirrors in proportion to how far they have moved:
the radiation-pressure force has a part $R_{FF}\\,x$, and the test masses feel
a frequency-dependent spring $K(\\Omega) = -R_{FF}(\\Omega)$ (Eq. 3.26),

$$
R_{FF}(\\Omega) = \\frac{2 I_0 \\omega_0}{L^2}\\,
\\frac{\\rho \\sin 2\\phi}{1 + 2\\rho\\cos 2\\phi + \\rho^2}\\,
\\frac{1}{(\\Omega - \\Omega_+)(\\Omega - \\Omega_-)} ,
$$

where $\\Omega_\\pm$ are the free optical resonances of the coupled
arm/signal-recycling cavity (Eq. 3.21). A tuned interferometer
($\\phi = 0, \\pi/2$) has no spring. With the spring, the antisymmetric mode
of the four test masses (reduced mass $m/4$, $R_{xx} = -4/m\\Omega^2$) becomes
an oscillator with two pairs of resonances, the zeros of
$1 - R_{xx} R_{FF}$ (Eq. 4.7): a "mechanical" pair that leaves zero frequency
as the power goes up, and an "optical" pair that starts at $\\Omega_\\pm$. For
$0 < \\phi < \\pi/2$ the mechanical pair is always slightly unstable. This is the
origin of the two dips in BC2001's noise curves, and of the optical spring in
Advanced LIGO.

**The model.** The differential mode is the three-mirror cavity
SRM-ITM-ETM (``sflu.topologies.signal_recycled``) with a free ETM of mass
$m/4$. The optical spring is read from the graph's closed-loop mechanical
response, ETM displacement drive to ETM displacement, which is
$1/(1 - R_{xx}R_{FF})$; the resonances are the zeros of its inverse, found by
solving the same graph at *complex* frequency. ``sflu.papers.buonanno_chen2002``
documents the mapping: power $I_0/T$ in the model arm, SR detuning
``detune_rad = -phi``, and the time-convention flip (the graph at $F$ is BC's
$\\Omega = -2\\pi F$).

**Leading order in $T$.** BC's formulas are first order in the ITM
transmission and treat the arm as a single pole. A real coupled cavity with
the printed $(T, \\rho, \\phi)$ has its optical pole at 194.3 Hz, not
191.4 Hz -- the subject of the companion paper (``test_buonanno_chen2003``).
To reproduce *these* figures, each model is built with the $(\\rho, \\phi)$
that give its exact compound mirror the paper's $\\Omega_+$
(``equivalent_sr``, a < 1% change). The printed parameters are shown too.

**Reproduced:** Figs. 6, 7 (the spring), 9 (resonances vs power), 10 and 11
(growth rates vs detuning). Fig. 13 needs a feedback servo, which is not
part of the optical model, and is not reproduced.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import buonanno_chen2002 as bc

P = bc.PARAMS
GAMMA = P["gamma"]
RHO, PHI = P["rho"], P["phi"]
P_ISQL = bc.circulating_power(1.0)   # model arm power for I_0 = I_SQL

# Figs. 6 and 7: 5 Hz to 5 kHz.
F_HZ = np.geomspace(5, 5000, 200)
OMEGA = 2 * np.pi * F_HZ


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def spring():
    """``R_FF`` from the model (equivalent and printed SR parameters) and Eq. 3.26."""
    rho_eq, phi_eq = bc.equivalent_sr(RHO, PHI)
    return (bc.model_R_FF(OMEGA, rho_eq, phi_eq, P_ISQL),
            bc.model_R_FF(OMEGA, RHO, PHI, P_ISQL),
            bc.R_FF(OMEGA, RHO, PHI, 1.0))


def test_fig6(tpath_join):
    """BC Fig. 6: the magnitude of the optical spring, $\\rho = 0.9$,
    $\\phi = \\pi/2 - 0.47$, $I_0 = I_\\mathrm{SQL}$.

    The paper's axis is "$\\log|R_{FF}|$, arbitrary units". It is the natural
    log of $|R_{FF}|/m\\gamma^2$ with $m = 30$ kg: that reproduces the plotted
    plateau (-2.07), peak (-0.71 at 191 Hz) and roll-off (-6 at 1.4 kHz),
    while $\\log_{10}$ would give a peak only 0.59 above the plateau instead
    of 1.36.
    """
    sflu, printed, paper = spring()
    lam, eps = bc.Omega_pm(RHO, PHI)[0].real, -bc.Omega_pm(RHO, PHI)[0].imag
    exact = bc.R_FF_exact(OMEGA, lam, eps, P_ISQL)

    # The model and Eq. 3.26 agree to 0.2% below 1 kHz. Above that they part
    # by up to 6% at 5 kHz: Eq. 3.26 treats the arm as a single pole, and at
    # 5 kHz $\Omega L/c = 0.4$. The all-orders form, ``R_FF_exact``, accounts
    # for the whole difference.
    low = F_HZ < 1000
    print(f"SFLU vs Eq. 3.26: max rel. err. {rel_err(sflu[low], paper[low]):.2e} "
          f"below 1 kHz, {rel_err(sflu, paper):.2e} up to 5 kHz")
    print(f"SFLU vs all-orders R_FF: max rel. err. {rel_err(sflu, exact):.2e}")
    assert rel_err(sflu[low], paper[low]) < 3e-3
    assert rel_err(sflu, paper) < 0.07
    assert rel_err(sflu, exact) < 1e-8

    # With the printed $(T, \rho, \phi)$ the spring is the same shape, but its
    # resonance sits at 194.3 Hz rather than 191.4 Hz (dotted). Near the peak
    # that is a 12% difference: the paper's $O(T)$ error, not the model's.
    print(f"printed (T, rho, phi) vs Eq. 3.26: max rel. err. {rel_err(printed, paper):.2e}")

    # The same statement for the pole of the model's $R_{FF}$, found by
    # solving the graph at complex frequency: the equivalent SR parameters
    # put it exactly on Eq. 3.21's $\Omega_+$ (191.4 - 25.0i Hz); the
    # printed ones put it on BC2003's exact Eq. 13 (194.3 - 25.4i Hz).
    Op = bc.Omega_pm(RHO, PHI)[0]
    pole_eq = bc.model_optical_pole(*bc.equivalent_sr(RHO, PHI), Op)
    pole_pr = bc.model_optical_pole(RHO, PHI, Op)
    lam_x, eps_x = bc.lam_eps(RHO, PHI)
    print(f"model pole / 2 pi: {pole_eq / (2 * np.pi):.3f} Hz (Eq. 3.21: "
          f"{Op / (2 * np.pi):.3f}); printed: {pole_pr / (2 * np.pi):.3f} Hz")
    assert abs(pole_eq - Op) / GAMMA < 1e-8
    assert abs(pole_pr - (lam_x - 1j * eps_x)) / GAMMA < 1e-8

    norm = P["m_kg"] * GAMMA**2
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogx(F_HZ, np.log(np.abs(sflu) / norm), lw=2, label="SFLU")
    ax.semilogx(F_HZ, np.log(np.abs(printed) / norm), ls=":", lw=1.5,
                label="SFLU, printed $(T, \\rho, \\phi)$")
    ax.semilogx(F_HZ, np.log(np.abs(paper) / norm), ls="--", c="k", lw=0.8,
                label="Eq. 3.26")
    ax.set_xlim(5, 5000)
    ax.set_ylim(-6, 0)
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\ln(|R_{FF}| / m\\gamma^2)$")
    ax.set_title("BC 2002 Fig. 6 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig6.pdf"))

    # Spot values read off the paper's figure.
    y = np.log(np.abs(sflu) / norm)
    i_peak = np.argmax(y)
    assert abs(y[0] - (-2.07)) < 0.02
    assert abs(y[i_peak] - (-0.71)) < 0.02 and abs(F_HZ[i_peak] - 191) < 4
    assert abs(np.interp(1400, F_HZ, y) - (-6.0)) < 0.05


def test_fig7(tpath_join):
    """BC Fig. 7: the phase of the optical spring, same parameters.

    $\\arg R_{FF} = -180°$ at low frequency means $K = -R_{FF} > 0$: a
    restoring spring. That sign is what fixes the SR detuning convention
    (``sflu_detuning``): the opposite sign of ``detune_rad`` gives the
    anti-restoring spring of $\\phi \\to -\\phi$.
    """
    sflu, printed, paper = spring()
    ph = np.degrees(np.angle(sflu))
    ph_paper = np.degrees(np.angle(paper))
    ph_printed = np.degrees(np.angle(printed))
    err = np.max(np.abs(ph - ph_paper))
    print(f"SFLU vs Eq. 3.26: max phase difference {err:.2f} deg")
    assert err < 0.5

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogx(F_HZ, ph, lw=2, label="SFLU")
    ax.semilogx(F_HZ, ph_printed, ls=":", lw=1.5,
                label="SFLU, printed $(T, \\rho, \\phi)$")
    ax.semilogx(F_HZ, ph_paper, ls="--", c="k", lw=0.8, label="Eq. 3.26")
    ax.set_xlim(5, 5000)
    ax.set_ylim(-180, 0)
    ax.set_xlabel("$f$ [Hz]")
    ax.set_ylabel("$\\arg R_{FF}$ [deg]")
    ax.set_title("BC 2002 Fig. 7 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig7.pdf"))

    # Read off the paper: -180 deg at low f, about -95 deg at the 191 Hz
    # resonance, approaching 0 above 1 kHz.
    assert ph[0] < -179
    assert abs(np.interp(191, F_HZ, ph) - (-95)) < 3
    assert abs(np.interp(1000, F_HZ, ph) - (-3.0)) < 0.5


def test_fig9(tpath_join):
    """BC Fig. 9: the resonances in the complex frequency plane as $I_0$ goes
    from 0 to $I_\\mathrm{SQL}$, $\\rho = 0.9$.

    For $\\phi = \\pi/2 - 0.47$ (solid) the mechanical pair leaves the origin
    and drifts into the unstable upper half-plane, ending at
    $\\pm 75.2 + 5.4i$ Hz; the optical pair moves from $\\Omega_\\pm/2\\pi$ to
    $\\pm 176.9 - 30.4i$ Hz. For $\\phi = \\pi/2 + 0.47$ (dotted) the spring is
    anti-restoring: the mechanical pair splits along the imaginary axis, one
    root growing without oscillating. Here $f = \\Omega / 2\\pi$ in BC's
    $e^{-i\\Omega t}$ convention, so the upper half-plane is unstable.
    """
    powers = np.geomspace(1e-4, 1, 40)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for phi, ls, color in ((np.pi / 2 - 0.47, "-", "C0"), (np.pi / 2 + 0.47, ":", "C3")):
        rho_eq, phi_eq = bc.equivalent_sr(RHO, phi)
        paper = np.array([bc.resonances(RHO, phi, I) for I in powers])
        sflu = np.array([bc.model_resonances(rho_eq, phi_eq, bc.circulating_power(I), r)
                         for I, r in zip(powers, paper)])
        err = np.max(np.abs(sflu - paper)) / GAMMA
        print(f"phi = pi/2 {phi - np.pi / 2:+.2f}: max |SFLU - Eq. 4.7| = {err:.1e} gamma")
        assert err < 2e-4
        for k in range(4):
            fs, fp = sflu[:, k] / (2 * np.pi), paper[:, k] / (2 * np.pi)
            ax.plot(fs.real, fs.imag, ls=ls, c=color, lw=2.5,
                    label=f"$\\phi = \\pi/2 {phi - np.pi / 2:+.2f}$" if k == 0 else None)
            ax.plot(fp.real, fp.imag, ls="--", c="k", lw=0.8)
            ax.plot(fs.real[-1], fs.imag[-1], "o", c=color, ms=4)
        if phi < np.pi / 2:
            end_restoring = sflu[-1] / (2 * np.pi)
        else:
            end_anti = sflu[-1] / (2 * np.pi)
    ax.axhline(0, c="k", lw=0.8)
    ax.text(-200, 60, "unstable half-plane")
    ax.text(-200, -65, "stable half-plane")
    ax.set_xlim(-220, 220)
    ax.set_ylim(-80, 80)
    ax.set_xlabel("Re $f$ [Hz]")
    ax.set_ylabel("Im $f$ [Hz]")
    ax.set_title("BC 2002 Fig. 9 (solid/dotted: SFLU, dashed: paper)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(tpath_join("fig9.pdf"))

    # End points at I_0 = I_SQL, read off the paper's figure (and its text).
    mech, opt = end_restoring[:2], end_restoring[2:]
    assert np.allclose(sorted(np.abs(mech.real)), [75.2, 75.2], atol=0.3)
    assert np.allclose(mech.imag, 5.4, atol=0.1)
    assert np.allclose(np.abs(opt.real), 176.9, atol=0.3)
    assert np.allclose(opt.imag, -30.4, atol=0.1)
    assert np.allclose(sorted(end_anti[:2].imag), [-68.9, 64.1], atol=0.1)
    assert np.allclose(np.abs(end_anti[:2].real), 0, atol=1e-6)
    assert np.allclose(np.abs(end_anti[2:].real), 202.3, atol=0.3)


def growth_rates(rho, phis):
    """Model and Eq. 4.7 resonances [mech, mech, opt, opt] over a detuning sweep."""
    paper = np.array([bc.resonances(rho, phi, 1.0) for phi in phis])
    sflu = np.array([bc.model_resonances(*bc.equivalent_sr(rho, phi), P_ISQL, r)
                     for phi, r in zip(phis, paper)])
    return sflu, paper


def test_fig10(tpath_join):
    """BC Fig. 10: imaginary parts of the resonances against the detuning,
    $\\rho = 0.9$, $I_0 = I_\\mathrm{SQL}$.

    For $0 < \\phi < \\pi/2$ the mechanical pair (dashed in the paper) is
    always unstable, $\\mathrm{Im}\\,\\Omega > 0$; for $\\pi/2 < \\phi < \\pi$ it
    splits into one growing and one decaying non-oscillating mode. The two
    optical roots share one imaginary part. Tuned detunings
    ($\\phi = 0, \\pi/2, \\pi$) are skipped: there the spring vanishes and the
    mechanical roots merge at zero.
    """
    phis = np.linspace(0.02, np.pi - 0.02, 100)
    phis = phis[np.abs(phis - np.pi / 2) > 0.02]
    sflu, paper = growth_rates(RHO, phis)

    # The optical roots near phi = pi/2 have half-widths up to ~20 gamma,
    # where a real arm (free spectral range 236 gamma) is no longer a single
    # pole; the comparison is made where the paper's plot shows them.
    mech_err = np.max(np.abs(sflu[:, :2] - paper[:, :2])) / GAMMA
    shown = np.abs(paper[:, 2:].imag) < GAMMA
    opt_err = np.max(np.abs(sflu[:, 2:] - paper[:, 2:])[shown]) / GAMMA
    print(f"max |SFLU - Eq. 4.7|: mechanical {mech_err:.1e} gamma, "
          f"optical (|Im| < gamma) {opt_err:.1e} gamma")
    assert mech_err < 1e-3
    assert opt_err < 1e-3

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(phis, sflu[:, 2].imag / GAMMA, c="C4", lw=2, label="optical resonance")
    for k in (0, 1):
        ax.plot(phis, sflu[:, k].imag / GAMMA, c="C2", lw=2,
                label="mechanical resonance" if k == 0 else None)
    for k in range(3):
        ax.plot(phis, paper[:, k].imag / GAMMA, ls="--", c="k", lw=0.8)
    ax.axhline(0, c="k", lw=1)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$\\phi$")
    ax.set_ylabel("Im$(\\Omega)/\\gamma$")
    ax.set_title("BC 2002 Fig. 10 (solid: SFLU, dashed: paper)")
    ax.legend(loc="upper center")
    ax.grid(True, alpha=0.3)
    fig.savefig(tpath_join("fig10.pdf"))

    # Read off the paper: the mechanical growth rate peaks at ~0.51 gamma
    # near phi = 0.5, and at +0.76 / -0.82 gamma near phi = 2.45.
    mech = np.max(sflu[:, :2].imag, axis=1) / GAMMA
    left = phis < np.pi / 2
    assert abs(np.max(mech[left]) - 0.51) < 0.01
    assert abs(phis[left][np.argmax(mech[left])] - 0.5) < 0.1
    assert abs(np.max(mech[~left]) - 0.765) < 0.01
    assert abs(np.min(sflu[~left, :2].imag) / GAMMA - (-0.823)) < 0.01


def test_fig11(tpath_join):
    """BC Fig. 11: the largest growth rate in the detuning range of interest,
    $1 \\le \\phi \\le 1.5$, for $\\rho = 0.8$ to $0.98$, $I_0 = I_\\mathrm{SQL}$.

    The worst case, $0.19\\gamma \\approx 120\\,\\mathrm{s^{-1}}$ at
    $\\rho = 0.8$, $\\phi = 1$, is the paper's "e-folding time of about 8 ms";
    a servo much faster than that is easy (Sec. V).
    """
    phis = np.linspace(1, 1.5, 40)
    fig, ax = plt.subplots(figsize=(6, 5))
    spots = {0.8: (0.193, 0.056, 0.0078), 0.9: (0.169, 0.028, 0.0037),
             0.95: (0.148, 0.014, 0.0018), 0.98: (0.132, 0.0055, 0.0007)}
    for rho, ls in zip(spots, ("-", "--", "-.", "-")):
        sflu, paper = growth_rates(rho, phis)
        rate = np.max(sflu.imag, axis=1) / GAMMA
        rate_paper = np.max(paper.imag, axis=1) / GAMMA
        err = np.max(np.abs(rate - rate_paper))
        print(f"rho = {rho}: max |SFLU - Eq. 4.7| growth rate {err:.1e} gamma")
        assert err < 1e-4
        ax.plot(phis, rate, ls=ls, lw=2, label=f"$\\rho = {rho}$")
        ax.plot(phis, rate_paper, ls="--", c="k", lw=0.8)
        # Values read off the paper at phi = 1, 1.2, 1.5.
        got = np.interp([1.0, 1.2, 1.5], phis, rate)
        assert np.allclose(got, spots[rho], rtol=0.05, atol=2e-4), (rho, got)
    ax.set_xlim(1, 1.5)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("$\\phi$")
    ax.set_ylabel("Im$(\\Omega)/\\gamma$")
    ax.set_title("BC 2002 Fig. 11 (solid: SFLU, dashed: paper)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(tpath_join("fig11.pdf"))
