"""
A. Buonanno and Y. Chen,
*Scaling law in signal recycled laser-interferometer gravitational-wave
detectors*,
[Phys. Rev. D 67, 062002 (2003)](https://doi.org/10.1103/PhysRevD.67.062002),
[arXiv:gr-qc/0208048](https://arxiv.org/abs/gr-qc/0208048).

**What the paper shows.** The short signal-recycling cavity and the ITM act
on the arm as a single mirror of complex reflectivity (Eq. 11)

$$
\\tilde\\rho' = \\frac{\\sqrt R + \\rho\\,e^{2i\\phi}}{1 + \\sqrt R\\,\\rho\\,e^{2i\\phi}}
= e^{2i(\\lambda + i\\epsilon)L/c} ,
$$

so a signal-recycled interferometer *is* a single detuned cavity with
detuning $\\lambda$ and half-width $\\epsilon$ (Eq. 13). Its quantum noise and
its optical spring depend on the ITM transmission $T$, the SR mirror $\\rho$
and the SR detuning $\\phi$ only through $\\lambda$, $\\epsilon$ and the arm
power: any $(T, \\rho, \\phi)$ with the same $\\tilde\\rho'$ is the same
detector (the "scaling law"), up to a rotation $\\theta$ of the output
quadratures (Eq. 105). The paper also gives the input-output relation to all
orders in $T$ (Eqs. 99-104) and shows that the usual first-order formulas
(Eqs. 18, 37) are off by up to tens of percent near the optical-spring dip.

**The model.** Both pictures are built as graphs: the coupled cavity
SRM-ITM-ETM (``sflu.topologies.signal_recycled``) and the single detuned
cavity (``sflu.topologies.fp_arm`` with a detuned arm link and an ITM chosen by
``half_bandwidth_T``). The paper's statement becomes a numerical identity
between two different graphs. ``sflu.papers.buonanno_chen2003`` documents the
mapping, in particular the two quadrature frames (physical and tilde).

**Reproduced:** Fig. 8 (exact vs first order, both panels), Fig. 4 (the
scaling law, with poles measured from the model), Fig. 5 (the solid
$S_{h,2}$ curves). The equivalence itself is shown in a section of its own.
Figs. 6, 9 and 10 use the same machinery and are not repeated; Fig. 7 is a
closed-form ratio with no optical content.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc

from sflu.papers import buonanno_chen2003 as b3
from sflu.papers import buonanno_chen2002 as bc02

F_HZ = np.geomspace(20, 500, 200)
OMEGA = 2 * np.pi * F_HZ


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def freq_axis(ax):
    """The paper's frequency axis: 20 to 500 Hz, ticks at 20, 40, 100, 200, 400."""
    ax.set_xlim(20, 500)
    ticks = [20, 40, 100, 200, 400]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("$f$ [Hz]")


def test_fig8(tpath_join):
    """BC 2003 Fig. 8: noise of the physical output quadratures, exact (SFLU)
    against first order in $T$ (paper, dotted).

    Left: the BC configuration $T = 0.033$, $\\rho = 0.9$,
    $\\phi = \\pi/2 - 0.47$, $m = 30$ kg, $I_c = 592$ kW, quadratures $b_1$
    (light) and $b_2$ (dark). Right: the LIGO-II reference, $T = 0.005$,
    $\\rho = 0.964$, $\\phi = \\pi/2 - 0.06$, $m = 40$ kg, $I_c = 840$ kW,
    $\\zeta = 1.13\\pi$.

    The figure labels the curves $\\tilde b_{1,2}$, but the plotted values are
    those of the *physical* quadratures $b_\\zeta$, i.e. the tilde-frame
    formulas at $\\tilde\\zeta = \\zeta + \\theta$; in the tilde frame the
    right panel would be 3.5 times higher at 20 Hz than printed.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, p, quads in ((axs[0], b3.PARAMS, ((np.pi / 2, "$b_1$", "rosybrown"),
                                              (0.0, "$b_2$", "k"))),
                         (axs[1], b3.LIGO2, ((1.13 * np.pi, "$b_{1.13\\pi}$", "C0"),))):
        rho, phi = p["rho"], p["phi"]
        lam_x, eps_x = b3.lam_eps_exact(rho, phi, p)
        lam_1, eps_1 = b3.lam_eps_first_order(rho, phi, p)
        th_x, th_1 = b3.theta_exact(rho, phi, p), b3.theta_first_order(rho, phi)
        print(f"T = {p['T_itm']}: (lambda, epsilon)/2pi = ({lam_x / 2 / np.pi:.2f}, "
              f"{eps_x / 2 / np.pi:.2f}) Hz exact (Eq. 13), "
              f"({lam_1 / 2 / np.pi:.2f}, {eps_1 / 2 / np.pi:.2f}) Hz first order (Eq. 18)")
        T = b3.coupled_cavity(F_HZ, rho, phi, p)
        ax.loglog(F_HZ, b3.h_SQL(OMEGA, p), c="0.6", lw=0.8)
        for zeta, label, color in quads:
            sflu = np.sqrt(b3.Sh_model(T, b3.physical_zeta(zeta), p))
            exact = np.sqrt(b3.Sh_exact(OMEGA, zeta + th_x, lam_x, eps_x, p))
            first = np.sqrt(b3.Sh_first_order(OMEGA, zeta + th_1, lam_1, eps_1, p))
            # The coupled-cavity graph and the exact Eqs. 99-104 agree to
            # rounding. The first-order Eq. 37 does not: near the sharp
            # optical-spring dip of $b_2$ it is 28% high, as the paper's own
            # Fig. 8 shows (dashed above solid at 75 Hz).
            e_x, e_1 = rel_err(sflu, exact), rel_err(first, sflu)
            print(f"  {label:16s} SFLU vs exact (Eqs. 99-104): {e_x:.1e};"
                  f"  first order (Eq. 37) vs SFLU: {e_1:.3f}")
            assert e_x < 1e-8
            assert e_1 < (0.3 if p is b3.PARAMS else 0.01)
            ax.loglog(F_HZ, sflu, c=color, lw=2.5, alpha=0.8, label=f"{label} SFLU")
            ax.loglog(F_HZ, exact, ls="--", c="k", lw=0.8)
            ax.loglog(F_HZ, first, ls=":", c=color, lw=1.5,
                      label=f"{label} first order (Eq. 37)")
            if p is b3.PARAMS and zeta == 0:
                b2_left = sflu, first
            elif p is b3.PARAMS:
                b1_left = sflu
            else:
                right = sflu
        freq_axis(ax)
        ax.set_ylim(3e-25, 1e-22)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("$S_h^{1/2}$ [Hz$^{-1/2}$]")
    axs[0].set_title("BC 2003 Fig. 8 left (solid: SFLU, dashed: paper)", fontsize=10)
    axs[1].set_title("Fig. 8 right, LIGO-II reference", fontsize=10)
    fig.tight_layout()
    fig.savefig(tpath_join("fig8.pdf"))

    # Read off the paper's figure. Left, b_2: dip of ~4e-25 near 73-75 Hz
    # (exact), bump 2.3e-24 at 110 Hz (exact) vs 2.1e-24 (first order),
    # second minimum ~1e-24 at 180 Hz. Left, b_1: 4.8e-23 at 20 Hz.
    # Right: 4.5e-23 at 20 Hz, dip ~1e-24 near 55-60 Hz, 2.8e-24 at 110 Hz,
    # 1.7e-24 at 220 Hz.
    at = lambda y, f: np.interp(f, F_HZ, y)
    sflu2, first2 = b2_left
    i = np.argmin(sflu2)
    assert 70 < F_HZ[i] < 77 and sflu2[i] < 6e-25
    assert np.isclose(at(sflu2, 110), 2.3e-24, rtol=0.05)
    assert np.isclose(at(first2, 110), 2.1e-24, rtol=0.05)
    assert np.isclose(at(sflu2, 180), 1.0e-24, rtol=0.05)
    assert np.isclose(at(b1_left, 20), 4.8e-23, rtol=0.05)
    assert np.isclose(at(right, 20), 4.5e-23, rtol=0.05)
    assert np.isclose(at(right, 110), 2.8e-24, rtol=0.05)
    assert np.isclose(at(right, 220), 1.7e-24, rtol=0.05)
    j = np.argmin(right[F_HZ < 100])
    assert 53 < F_HZ[j] < 62 and right[j] < 1.2e-24


def test_scaling_law(tpath_join):
    """The equivalence itself: three coupled cavities and one single cavity.

    Take the BC configuration's exact $(\\lambda, \\epsilon)$. For ITM
    transmissions $T = 0.005$, $0.033$ and $0.05$, Eq. 14 gives the SR mirror
    and detuning that reproduce it; each is built as a separate SRM-ITM-ETM
    graph. A fourth graph is a plain detuned Fabry-Perot cavity with the same
    $\\lambda$ and $\\epsilon$, no SR mirror at all. Read out in the tilde frame
    ($\\zeta_\\mathrm{phys} = \\tilde\\zeta - \\theta$ for each coupled cavity),
    all four give the same noise *and* the same signal response, to rounding.
    """
    p = b3.PARAMS
    lam, eps = b3.lam_eps_exact(p["rho"], p["phi"], p)
    single = b3.single_cavity(F_HZ, lam, eps, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for zt, ls in ((0.0, "-"), (np.pi / 2, "-")):
        ref = b3.Sh_model(single, b3.tilde_zeta(zt, lam, p), p)
        ax.loglog(F_HZ, np.sqrt(ref), c="0.7", lw=6,
                  label="single detuned cavity" if zt == 0 else None)
        for T_itm in (0.005, 0.033, 0.05):
            q = dict(p, T_itm=T_itm)
            rho, phi = b3.rho_phi(lam, eps, T_itm, p["L_m"])
            coupled = b3.coupled_cavity(F_HZ, rho, phi, q)
            th = b3.theta_exact(rho, phi, q)
            S = b3.Sh_model(coupled, b3.physical_zeta(zt - th), q)
            err = rel_err(S, ref)
            print(f"zeta~ = {zt:.2f}, T = {T_itm}: rho = {rho:.4f}, "
                  f"phi = pi/2 {phi - np.pi / 2:+.4f}; coupled vs single cavity {err:.1e}")
            assert err < 1e-9
            ax.loglog(F_HZ, np.sqrt(S), lw=1.2,
                      label=f"SR, $T = {T_itm}$" if zt == 0 else None)
        exact = b3.Sh_exact(OMEGA, zt, lam, eps, p)
        assert rel_err(ref, exact) < 1e-8
        ax.loglog(F_HZ, np.sqrt(exact), ls="--", c="k", lw=0.8,
                  label="Eqs. 36, 99-104" if zt == 0 else None)

    # The signal vectors too, not only the noise: the coupled cavity's
    # response to $x$, rotated by $\theta + \lambda L / c$, is the single
    # cavity's (the second term is the detuned link's one-way phase).
    rho, phi = p["rho"], p["phi"]
    coupled = b3.coupled_cavity(F_HZ, rho, phi, p)
    rot = b3.MLIB.Mrotation(b3.theta_exact(rho, phi, p) - lam * p["L_m"] / scc.c)
    sig_c = rot @ coupled["ETM.pos.exc"]
    assert rel_err(sig_c[:, :, 0], single["ETM.pos.exc"][:, :, 0]) < 1e-9

    freq_axis(ax)
    ax.set_ylim(3e-25, 1e-22)
    ax.set_ylabel("$S_h^{1/2}$ [Hz$^{-1/2}$]")
    ax.set_title("Scaling law: $\\tilde b_1$, $\\tilde b_2$ (solid: SFLU, dashed: paper)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("scaling_law.pdf"))


def test_fig4(tpath_join):
    """BC 2003 Fig. 4: $\\rho$ and $\\phi - \\pi/2$ against $T$ at fixed
    $(\\lambda, \\epsilon)/2\\pi = (194.48, 25.42)$, $(228.10, 69.13)$ and
    $(900, 30)$ Hz (Eq. 14).

    Each marker is an SRM-ITM-ETM graph built with those $(T, \\rho, \\phi)$,
    whose free optical resonance $\\lambda - i\\epsilon$ is then *measured*:
    the pole of its optical spring $R_{FF}$, located by solving the graph at
    complex frequency. Every marker lands on its target to $10^{-9}$. The
    square is the BC configuration, the triangle LIGO-II (which needs
    $\\rho = 0.964$; the text's $0.96$ gives (223, 76) Hz).
    """
    targets = [((194.48, 25.42), "-", "C0"), ((228.10, 69.13), ":", "C3"),
               ((900.0, 30.0), "--", "C2")]
    T_line = np.linspace(1e-4, 0.05, 200)
    T_mark = np.array([0.005, 0.015, 0.025, 0.035, 0.045])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for (lh, eh), ls, color in targets:
        lam, eps = 2 * np.pi * lh, 2 * np.pi * eh
        rho, phi = b3.rho_phi(lam, eps, T_line)
        ax1.plot(T_line, rho, ls=ls, c=color, lw=2, label=f"({lh:g}, {eh:g}) Hz")
        ax2.plot(T_line, phi - np.pi / 2, ls=ls, c=color, lw=2)
        for T_itm in T_mark:
            r, f = b3.rho_phi(lam, eps, T_itm)
            pole = bc02.model_optical_pole(r, f, lam - 1j * eps,
                                           dict(b3.PARAMS, T_itm=T_itm))
            err = abs(pole - (lam - 1j * eps)) / abs(lam - 1j * eps)
            assert err < 1e-9, (lh, T_itm, err)
            ax1.plot(T_itm, r, "o", mfc="none", c="k", ms=6)
            ax2.plot(T_itm, f - np.pi / 2, "o", mfc="none", c="k", ms=6)
        print(f"({lh}, {eh}) Hz: SFLU poles at Eq. 14's (rho, phi) hit the target")
    for ax in (ax1, ax2):
        ax.plot(0.033, 0.9 if ax is ax1 else -0.47, "s", c="k")
        ax.plot(0.005, 0.964 if ax is ax1 else -0.06, "^", c="k")
        ax.set_xlim(0, 0.05)
        ax.set_xlabel("$T$")
        ax.grid(True, alpha=0.3)
    ax1.set_ylim(0.76, 1)
    ax1.set_ylabel("$\\rho$")
    ax2.set_ylabel("$\\phi - \\pi/2$")
    ax1.legend(title="$(\\lambda, \\epsilon)/2\\pi$", fontsize=8)
    ax1.set_title("BC 2003 Fig. 4 (lines: Eq. 14, circles: SFLU poles)", fontsize=10)
    fig.tight_layout()
    fig.savefig(tpath_join("fig4.pdf"))

    # The square and triangle sit on their curves.
    r, f = b3.rho_phi(2 * np.pi * 194.48, 2 * np.pi * 25.42, 0.033)
    assert abs(r - 0.9) < 2e-3 and abs(f - (np.pi / 2 - 0.47)) < 2e-3
    r, f = b3.rho_phi(2 * np.pi * 228.10, 2 * np.pi * 69.13, 0.005)
    assert abs(r - 0.964) < 1e-3 and abs(f - (np.pi / 2 - 0.06)) < 1e-3


def test_fig5(tpath_join):
    """BC 2003 Fig. 5: $S_{h,2}$ in the tilde frame for
    $\\lambda = 2\\pi\\cdot191.3$ Hz, $\\epsilon = 2\\pi\\cdot25.0$ Hz, $m = 30$ kg,
    $I_c = 300$ kW (light) and 600 kW (dark), with the free-mass SQL.

    The model is the single detuned cavity. Its $\\tilde b_2$ noise agrees with
    the exact Eqs. 99-104 to rounding, but differs from the paper's
    first-order Eq. 37 (dashed) by up to 8%. App. C explains why: to second
    order the exact result is Eq. 37 in a quadrature rotated by
    $\\lambda L/c = 0.016$ rad -- and indeed the model read out at
    $\\tilde\\zeta = \\lambda L/c$ matches Eq. 37 to 0.2%. The thin dashed curves
    are the paper's $S^\\mathrm{min}_{h,2}$ (Eq. 82), a bound with no readout
    to model.
    """
    lam, eps = 2 * np.pi * 191.3, 2 * np.pi * 25.0
    rot = lam * b3.PARAMS["L_m"] / scc.c
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(F_HZ, b3.h_SQL(OMEGA), c="k", lw=2, label="$h_\\mathrm{SQL}$")
    dips = {}
    for Ic, color in ((300e3, "turquoise"), (600e3, "b")):
        p = dict(b3.PARAMS, I_c=Ic)
        T = b3.single_cavity(F_HZ, lam, eps, p)
        S = b3.Sh_model(T, b3.tilde_zeta(0, lam, p), p)
        S_rot = b3.Sh_model(T, b3.tilde_zeta(rot, lam, p), p)
        paper = b3.Sh_first_order(OMEGA, 0, lam, eps, p)
        e_x = rel_err(S, b3.Sh_exact(OMEGA, 0, lam, eps, p))
        e_1 = rel_err(np.sqrt(S), np.sqrt(paper))
        e_r = rel_err(np.sqrt(S_rot), np.sqrt(paper))
        print(f"I_c = {Ic / 1e3:.0f} kW: SFLU vs exact {e_x:.1e}; vs Eq. 37 {e_1:.3f}; "
              f"at zeta~ = lambda L/c vs Eq. 37 {e_r:.1e}")
        assert e_x < 1e-8 and e_1 < 0.1 and e_r < 3e-3
        ax.loglog(F_HZ, np.sqrt(S), c=color, lw=2.5, label=f"$I_c$ = {Ic / 1e3:.0f} kW")
        ax.loglog(F_HZ, np.sqrt(paper), ls="--", c="k", lw=0.8)
        ax.loglog(F_HZ, np.sqrt(b3.Sh_min_2(OMEGA, lam, eps, p)), ls="--", c=color, lw=1)
        i = np.argmin(S)
        dips[Ic] = F_HZ[i], np.sqrt(S[i])
    freq_axis(ax)
    ax.set_ylim(2e-25, 2e-23)
    ax.set_ylabel("$S_h^{1/2}$ [Hz$^{-1/2}$]")
    ax.set_title("BC 2003 Fig. 5 (solid: SFLU, dashed: paper)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig5.pdf"))

    # The optical-spring dips read off the paper: ~53 Hz (300 kW) and ~78 Hz
    # (600 kW), both near 4.5e-25.
    assert abs(dips[300e3][0] - 52.6) < 1.5 and dips[300e3][1] < 6e-25
    assert abs(dips[600e3][0] - 78) < 1.5 and dips[600e3][1] < 6e-25
