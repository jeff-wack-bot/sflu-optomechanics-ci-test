"""
A. Buonanno and Y. Chen,
*Quantum noise in second generation, signal-recycled laser interferometric
gravitational-wave detectors*,
[Phys. Rev. D 64, 042006 (2001)](https://doi.org/10.1103/PhysRevD.64.042006),
[arXiv:gr-qc/0102012](https://arxiv.org/abs/gr-qc/0102012).

**What the paper shows.** A signal-recycling mirror (SRM) of amplitude
reflectivity $\\rho$ at the dark port of KLMTV's interferometer, detuned by
the one-way SRC phase $\\phi$, rotates the two quadratures into each other
on every pass. Radiation pressure then couples back into the measured
quadrature: the interferometer becomes an *optical spring*, with the
input-output relation (Eqs. 2.20-2.24)

$$
\\mathbf b = \\frac{1}{M}\\Big[e^{2i\\beta}\\,\\mathbf C\\,\\mathbf a
  + \\sqrt{2\\mathcal K}\\,\\tau e^{i\\beta}\\,\\mathbf D\\,\\frac{h}{h_\\mathrm{SQL}}\\Big],
\\qquad
M = 1 + \\rho^2 e^{4i\\beta} - 2\\rho e^{2i\\beta}
    \\Big(\\cos 2\\phi + \\frac{\\mathcal K}{2}\\sin 2\\phi\\Big).
$$

With LIGO-II-like parameters the noise has two resonant dips -- the
optical-spring resonance and the shifted optical resonance -- and beats the
standard quantum limit by about a factor 2 over a band $\\Delta f/f \\sim 1$.

**The model.** ``sflu.topologies.signal_recycled()``: SRM, SRC, ITM, arm and
a free end mirror of mass $m/4$, with the arm power matched to Eq. 2.13
exactly as in the KLMTV example. The carrier is put into the arm directly
(it comes from the beamsplitter, not through the SRM). The mapping is in
``sflu.papers.buonanno_chen2001``.

**Conventions.** BC's homodyne angle reads $b_1\\sin\\zeta + b_2\\cos\\zeta$,
so $\\zeta = 0$ is the phase quadrature; it is this package's angle
$\\zeta - \\pi/2$ (``sflu_zeta``). BC's SR detuning $\\phi$ is the SRC link's
``detune_rad = -phi`` (``sflu_detune``). Both signs come from the opposite
sign of the phase quadrature (as for KLMTV) and were fixed numerically:
only this combination reproduces Eq. 3.5 at every $\\zeta$, and it gives an
optical spring at the resonances of Eq. 4.4 (Fig. 5 below).

**Reproduced:** Figs. 2, 3, 4, 5, 6 and 8, every curve except Fig. 2's "LIGO
II correl. neglected", which comes from an unpublished semiclassical
calculation. **Paper error:** Eq. 5.12's $N_{21}$ has a spurious
$\\cos\\beta$ (Fig. 8 below).
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import buonanno_chen2001 as bc

P = bc.PARAMS
GAMMA = P["gamma"]
RHO, PHI = P["rho"], P["phi"]

# The x-axis of every figure: Omega/gamma from 0.1 to 10.
X = np.geomspace(0.1, 10, 201)
OMEGA = X * GAMMA
F_HZ = OMEGA / (2 * np.pi)


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def sqrt_paper(Sh):
    return np.sqrt(Sh) / bc.h_SQL(GAMMA)


def axes(title, ylim=(0.1, 10), top_hz=False):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(X, 1 / X, c="k", lw=1, label="SQL")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$\\Omega / \\gamma$")
    ax.set_ylabel("$\\sqrt{S_h(\\Omega) / S_h^\\mathrm{SQL}(\\gamma)}$")
    ax.set_title(title, pad=30 if top_hz else None)
    ax.grid(True, which="both", alpha=0.3)
    if top_hz:
        top = ax.secondary_xaxis("top", functions=(lambda x: 100 * x, lambda f: f / 100))
        top.set_xlabel("f [Hz]  ($\\gamma = 2\\pi \\cdot 100$ Hz)")
    return fig, ax


def overlay(ax, label, sflu, paper, tol, **kw):
    """Plot an SFLU curve with its closed form, and assert they agree."""
    ax.loglog(X, sflu, lw=2, alpha=0.7, label=label, **kw)
    ax.loglog(X, paper, ls="--", c="k", lw=0.8)
    err = rel_err(sflu, paper)
    print(f"{label:45s} max |SFLU/paper - 1| = {err:.2e}")
    assert err < tol, label


def conventional(ax):
    """The "conventional" curve of every figure: no SRM, I_o = I_SQL, Eq. 3.8.

    The SFLU version is the same model with a perfectly transmissive SRM and
    no SRC phase (with ``rho = 0`` a nonzero ``phi`` would only rotate the
    output quadratures).
    """
    T = bc.interferometer(F_HZ, 0.0, 0.0, 1.0)
    overlay(ax, "conventional (Eq. 3.8)", bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(0)),
            sqrt_paper(bc.Sh_conventional(OMEGA, 0, 1.0)), 3e-3, c="C3")


def test_fig2(tpath_join):
    """BC Fig. 2: the two quadratures of a detuned SR interferometer.

    $\\rho = 0.9$, $\\phi = \\pi/2 - 0.47$, $I_o = I_\\mathrm{SQL}$, no losses.
    Reading $b_2$ ($\\zeta = 0$) gives a deep dip at the optical-spring
    resonance and a shallower one at the optical resonance; $b_1$
    ($\\zeta = \\pi/2$) gives a broad trough between them.
    """
    T = bc.interferometer(F_HZ, RHO, PHI, 1.0)
    fig, ax = axes("BC Fig. 2 (solid: SFLU, dashed: paper)")
    conventional(ax)
    # The residual differences, ~1e-3, are the paper's single-pole arm
    # (Eq. 2.11) against the model's exact 4 km cavity, as in KLMTV.
    for label, zeta in [("quadrature $b_1$, $\\zeta = \\pi/2$ (Eq. 3.5)", np.pi / 2),
                        ("quadrature $b_2$, $\\zeta = 0$ (Eq. 3.5)", 0.0)]:
        overlay(ax, label, bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(zeta)),
                sqrt_paper(bc.Sh(OMEGA, RHO, PHI, zeta, 1.0)), 3e-3)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig2.pdf"))

    # Spot values read off Fig. 2 (and the paper's Eq. 3.5): the b2 dips of
    # ~0.2 at the optical-spring resonance and ~0.47 at the optical one, the
    # bump of ~0.95 between them, and the b1 trough of ~0.5-0.6.
    xs = np.array([0.775, 1.0, 1.81])
    Ts = bc.interferometer(xs * GAMMA / (2 * np.pi), RHO, PHI, 1.0)
    b2 = bc.sqrt_Sh_over_hSQL(Ts, bc.sflu_zeta(0))
    b1 = bc.sqrt_Sh_over_hSQL(Ts, bc.sflu_zeta(np.pi / 2))
    print("b2 at", xs, "=", b2, "  b1 =", b1)
    assert np.allclose(b2, [0.199, 0.947, 0.466], rtol=0.02)
    assert np.allclose(b1, [0.506, 0.614, 0.505], rtol=0.02)


def test_fig3(tpath_join):
    """BC Fig. 3: the same interferometer, read at four homodyne angles.

    The dips barely move with $\\zeta$: their positions are set by the
    resonances of the optomechanical system, not by the readout.
    """
    T = bc.interferometer(F_HZ, RHO, PHI, 1.0)
    fig, ax = axes("BC Fig. 3 (solid: SFLU, dashed: paper)")
    conventional(ax)
    curves = {}
    for name, zeta in [("0", 0), ("\\pi/6", np.pi / 6), ("\\pi/3", np.pi / 3),
                       ("\\pi/2", np.pi / 2)]:
        curves[name] = bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(zeta))
        overlay(ax, f"$\\zeta = {name}$ (Eq. 3.5)", curves[name],
                sqrt_paper(bc.Sh(OMEGA, RHO, PHI, zeta, 1.0)), 3e-3)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig3.pdf"))

    # Read off Fig. 3: zeta = pi/6 dips to ~0.35 near Omega = 0.75 gamma.
    band = (X > 0.6) & (X < 0.9)
    c = curves["\\pi/6"][band]
    i = np.argmin(c)
    print(f"zeta = pi/6 minimum {c[i]:.3f} at {X[band][i]:.3f} gamma")
    assert abs(c[i] - 0.35) < 0.05
    assert abs(X[band][i] - 0.75) < 0.05


def test_fig4(tpath_join):
    """BC Fig. 4: the two undetuned limits.

    Left, extreme signal recycling ($\\phi = 0$, Eq. 3.26): the SRC is
    resonant for the signal, which narrows the band. Right, extreme RSE
    ($\\phi = \\pi/2$, Eq. 3.30): it is anti-resonant, which broadens it. In
    both, $\\rho$ only rescales $\\mathcal K$ and the noise stays above the SQL.
    The signal ends up in $b_2$ for ESR but in $b_1$ for ERSE -- a quarter
    turn per pass, twice -- so the two panels are read at $\\zeta = 0$ and
    $\\zeta = \\pi/2$ respectively.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, phi, zeta, Sh) in zip(axs, [
            ("ESR, $\\phi = 0$ (Eq. 3.26)", 0.0, 0.0, bc.Sh_ESR),
            ("ERSE, $\\phi = \\pi/2$ (Eq. 3.30)", np.pi / 2, np.pi / 2, bc.Sh_ERSE)]):
        plt.sca(ax)
        ax.loglog(X, 1 / X, c="k", lw=1, label="SQL")
        conventional(ax)
        # ERSE widens the detection band to ~gamma (1 + rho) / (1 - rho), where
        # the paper's single-pole arm is least accurate: the difference grows
        # as Omega^2 and reaches 3e-3 at 10 gamma for rho = 0.9, hence 5e-3.
        for rho in (0.7, 0.8, 0.9):
            T = bc.interferometer(F_HZ, rho, phi, 1.0)
            overlay(ax, f"{name}, $\\rho = {rho}$", bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(zeta)),
                    sqrt_paper(Sh(OMEGA, rho, 1.0)), 5e-3)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(0.1, 10)
        ax.set_xlabel("$\\Omega / \\gamma$")
        ax.set_ylabel("$\\sqrt{S_h(\\Omega) / S_h^\\mathrm{SQL}(\\gamma)}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("BC Fig. 4 (solid: SFLU, dashed: paper)")
    fig.savefig(tpath_join("fig4.pdf"))

    # Eqs. 3.26 and 3.30 are Eq. 3.5 at phi = 0 and pi/2: a check that the
    # angle assignments above are the paper's.
    assert rel_err(bc.Sh(OMEGA, 0.9, 0.0, 0.0, 1.0), bc.Sh_ESR(OMEGA, 0.9, 1.0)) < 1e-12
    assert rel_err(bc.Sh(OMEGA, 0.9, np.pi / 2, np.pi / 2, 1.0),
                   bc.Sh_ERSE(OMEGA, 0.9, 1.0)) < 1e-12


def test_fig5(tpath_join):
    """BC Fig. 5: three detunings at $\\rho = 0.95$, $\\zeta = 0$, $I_o = I_\\mathrm{SQL}$.

    The vertical lines are the resonances of the closed system ($\\rho = 1$),
    Eq. 4.4. For $\\phi = \\pi/2 - 0.59$ they form a complex pair and only
    the real part is drawn.
    """
    fig, ax = axes("BC Fig. 5 (solid: SFLU, dashed: paper)", ylim=(0.1, 100), top_hz=True)
    conventional(ax)
    for d in (0.19, 0.39, 0.59):
        phi = np.pi / 2 - d
        T = bc.interferometer(F_HZ, 0.95, phi, 1.0)
        overlay(ax, f"$\\phi = \\pi/2 - {d}$ (Eq. 3.5)", bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(0)),
                sqrt_paper(bc.Sh(OMEGA, 0.95, phi, 0, 1.0)), 3e-3)
        for r in bc.resonances(phi, 1.0):
            ax.axvline(r.real, c="k", lw=0.5)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig5.pdf"))

    # Eq. 4.4 against the model's own resonances: with the SRM almost closed
    # (rho = 0.9999) the signal response peaks sharply at the resonances. This
    # is what fixes the sign of the detuning independently of Eq. 3.5:
    # 0 < phi < pi/2 must be an optical spring with two real resonances, and
    # with detune_rad = +phi instead the low one disappears.
    xf = np.geomspace(0.1, 10, 4001)
    for d in (0.19, 0.39, 0.47):
        phi = np.pi / 2 - d
        T = bc.interferometer(xf * GAMMA / (2 * np.pi), 0.9999, phi, 1.0)
        resp = np.abs(bc.readout.response(T, bc.readout.quadrature(bc.MLIB, bc.sflu_zeta(0)),
                                          "ETM.pos.exc"))
        peaks = xf[1:-1][(resp[1:-1] > resp[:-2]) & (resp[1:-1] > resp[2:])]
        ana = np.array([r.real for r in bc.resonances(phi, 1.0)])
        print(f"phi = pi/2 - {d}: model resonances {peaks}, Eq. 4.4 {ana}")
        assert len(peaks) == 2 and np.allclose(peaks, ana, rtol=2e-3)
    # The values the notes give for Fig. 2's detuning: 0.775 and 1.809.
    assert np.allclose([r.real for r in bc.resonances(PHI, 1.0)], [0.7754, 1.8095], atol=1e-3)


def test_fig6(tpath_join):
    """BC Fig. 6: low power, $I_o = 10^{-4} I_\\mathrm{SQL}$, $\\rho = 0.95$, $\\zeta = 0$.

    With little radiation pressure only the optical resonance is left, at
    $\\Omega = \\gamma\\tan\\phi$ (Eq. 4.6, vertical lines).
    """
    fig, ax = axes("BC Fig. 6 (solid: SFLU, dashed: paper)", ylim=(10, 1000), top_hz=True)
    for d in (0.19, 0.39, 0.59, 0.79, 0.99):
        phi = np.pi / 2 - d
        T = bc.interferometer(F_HZ, 0.95, phi, 1e-4)
        sflu = bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(0))
        overlay(ax, f"$\\phi = \\pi/2 - {d}$ (Eq. 3.5)", sflu,
                sqrt_paper(bc.Sh(OMEGA, 0.95, phi, 0, 1e-4)), 3e-3)
        ax.axvline(np.tan(phi), c="k", lw=0.5)
        # the dip sits at tan(phi), to the grid spacing (2.3%)
        assert abs(X[np.argmin(sflu)] / np.tan(phi) - 1) < 0.03
    ax.legend(fontsize=7, loc="upper left")
    fig.savefig(tpath_join("fig6.pdf"))


def test_fig8(tpath_join):
    """BC Fig. 8: losses. Arm loss $\\epsilon = 0.01$, SRC loss
    $\\lambda_\\mathrm{SR} = 0.02$, detection loss $\\lambda_\\mathrm{PD} = 0.1$.

    Eq. 5.13 is first order in $\\epsilon$ and $\\lambda_\\mathrm{SR}$; the SFLU
    model is exact, so here they agree to ~0.5% rather than 1e-3. Losses
    mostly fill in the sharp $b_2$ dip.
    """
    eps, lsr, lpd = P["eps"], P["lambda_SR"], P["lambda_PD"]
    T = bc.interferometer(F_HZ, RHO, PHI, 1.0, eps, lsr, lpd)
    T0 = bc.interferometer(F_HZ, RHO, PHI, 1.0)
    fig, ax = axes("BC Fig. 8 (solid: SFLU, dashed: paper)", top_hz=True)
    conventional(ax)
    for q, zeta, c in [("b_1", np.pi / 2, ("C0", "C1")), ("b_2", 0.0, ("C2", "C4"))]:
        overlay(ax, f"${q}$ losses (Eq. 5.13)", bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(zeta)),
                sqrt_paper(bc.Sh_lossy(OMEGA, RHO, PHI, zeta, 1.0, eps, lsr, lpd)), 6e-3,
                c=c[0])
        overlay(ax, f"${q}$ no losses (Eq. 3.5)", bc.sqrt_Sh_over_hSQL(T0, bc.sflu_zeta(zeta)),
                sqrt_paper(bc.Sh(OMEGA, RHO, PHI, zeta, 1.0)), 3e-3, c=c[1], ls=":")
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig8.pdf"))

    # The typo in Eq. 5.12 (next paragraph) changes the b2 total by up to 0.6%
    # near the optical resonance. Where it matters most, the model sides with
    # the corrected N21.
    b2 = bc.referred(T, bc.sflu_zeta(0))[0]
    fixed = bc.Sh_lossy(OMEGA, RHO, PHI, 0, 1.0, eps, lsr, lpd)
    printed = bc.Sh_lossy(OMEGA, RHO, PHI, 0, 1.0, eps, lsr, lpd, n21="printed")
    i = np.argmax(np.abs(printed / fixed - 1))
    e_fix, e_pr = np.sqrt(b2[i] / fixed[i]) - 1, np.sqrt(b2[i] / printed[i]) - 1
    print(f"at {X[i]:.2f} gamma: SFLU vs corrected {e_fix:.2e}, vs printed {e_pr:.2e}")
    assert abs(e_fix) < abs(e_pr) / 2

    # Read off Fig. 8: the b2 dip near 0.78 gamma fills in from ~0.2 to ~0.32.
    lossy = bc.sqrt_Sh_over_hSQL(T, bc.sflu_zeta(0))
    band = (X > 0.6) & (X < 0.9)
    print(f"lossy b2 minimum {lossy[band].min():.3f}")
    assert 0.30 < lossy[band].min() < 0.37

    # Each noise source separately: the SFLU budget per input against the
    # matching group of terms in Eq. 5.13 -- C^L (dark-port vacuum), P (SRC
    # loss), Q (detection loss), N (arm loss). Each group is only first order
    # in eps and lambda_SR, so at Fig. 8's losses they agree to a few percent
    # (P to 4%, N to 7-14%), and the error falls in proportion as the losses
    # are reduced. The check is therefore made at 1/100 of eps and lambda_SR
    # (lambda_PD, which Eq. 5.13 treats exactly, is kept).
    #
    # There the arm-loss term exposes the typo in Eq. 5.12 clearly: with N21 as
    # printed, with cos(beta) twice, it is off by 74% at zeta = 0 and 13% at
    # pi/3, however small the loss; with the second cos(beta) removed it agrees
    # like the others. (At zeta = pi/2 N21 does not enter.)
    e_s, l_s = eps / 100, lsr / 100
    Ts = bc.interferometer(F_HZ, RHO, PHI, 1.0, e_s, l_s, lpd)
    sources = {"a": ["SRM.bk.i.exc"], "p": ["SRC.p.exc"], "q": ["PD"],
               "n": ["ETM.frL.i", "ITM.frL.i"]}
    for zeta in (0.0, np.pi / 3, np.pi / 2):
        _, budget = bc.referred(Ts, bc.sflu_zeta(zeta))
        terms = bc.Sh_lossy_terms(OMEGA, RHO, PHI, zeta, 1.0, e_s, l_s, lpd)
        printed = bc.Sh_lossy_terms(OMEGA, RHO, PHI, zeta, 1.0, e_s, l_s, lpd,
                                    n21="printed")
        for k, inputs in sources.items():
            err = rel_err(sum(budget[i] for i in inputs), terms[k])
            print(f"zeta = {zeta:.3f}, term {k}: max |SFLU/Eq. 5.13 - 1| = {err:.2e}")
            assert err < 5e-3
        err_printed = rel_err(sum(budget[i] for i in sources["n"]), printed["n"])
        print(f"zeta = {zeta:.3f}, term n with N21 as printed: {err_printed:.2e}")
        if zeta != np.pi / 2:
            assert err_printed > 0.1
