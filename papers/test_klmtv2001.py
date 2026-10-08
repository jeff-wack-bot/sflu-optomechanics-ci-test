"""
H. J. Kimble, Yu. Levin, A. B. Matsko, K. S. Thorne and S. P. Vyatchanin,
*Conversion of conventional gravitational-wave interferometers into quantum
nondemolition interferometers by modifying their input and/or output optics*,
[Phys. Rev. D 65, 022002 (2001)](https://doi.org/10.1103/PhysRevD.65.022002),
[arXiv:gr-qc/0008026](https://arxiv.org/abs/gr-qc/0008026).

**What the paper shows.** In a conventional interferometer, radiation pressure
makes the light that leaves the dark port *ponderomotively squeezed*: back-action
noise and shot noise become correlated. The input-output relation (Eq. 16) is

$$
b_1 = a_1 e^{2i\\beta}, \\qquad
b_2 = (a_2 - \\mathcal K a_1) e^{2i\\beta}
      + \\sqrt{2\\mathcal K}\\,\\frac{h}{h_\\mathrm{SQL}} e^{i\\beta},
\\qquad
\\mathcal K = \\frac{(I_o/I_\\mathrm{SQL})\\, 2\\gamma^4}{\\Omega^2(\\gamma^2+\\Omega^2)} .
$$

Reading the phase quadrature $b_2$ ignores the correlation, and the noise
can at best touch the standard quantum limit $h_\\mathrm{SQL}$. KLMTV show three
ways past it that change only the input and output optics:

* **squeezed input**: squeezed vacuum at the frequency-dependent angle
  $\\lambda = -\\Phi(\\Omega)$, with $\\Phi = \\operatorname{arccot}\\mathcal K$;
* **variational output**: homodyne readout at $\\zeta = \\Phi(\\Omega)$, which
  removes back-action altogether;
* **squeezed variational**: both together.

The frequency dependence comes from detuned *filter cavities*, which is the
idea Advanced LIGO's frequency-dependent squeezing descends from.

**The model.** The differential mode is one Fabry-Perot arm with a free end
mirror of mass $m/4$ (``sflu.topologies.fp_arm``). The mapping, and the
two places it is more careful than a literal reading of the paper, are
described in ``sflu.papers.klmtv2001``. In short: the arm power is matched
to KLMTV's $\\mathcal K$ exactly, and the ITM transmission puts the cavity's
*exact* pole at $\\gamma$. Without those two choices the variational curves
come out wrong by a factor of 30 at low frequency, because back-action
cancellation subtracts two numbers of size $\\mathcal K \\sim 10^3$.

**Sign convention.** In this package the coupling enters as
$b_2 = a_2 + \\mathcal K a_1$. Every KLMTV squeeze or homodyne angle is
therefore negated before use (``sflu_angle``).

**Reproduced:** Fig. 4 (all six curves, with ideal angles and again with real
filter cavities) and Fig. 10 (the filter-cavity parameters). Fig. 14 is not
reproduced: its thermal-noise lines come with no formula.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import klmtv2001 as kl

P = kl.PARAMS
GAMMA = P["gamma"]
SQZ = P["sqz_dB"]

# The x-axis of Fig. 4: Omega/gamma from 0.1 to 10.
X = np.geomspace(0.1, 10, 201)  # odd, so that Omega = gamma is on the grid
OMEGA = X * GAMMA
F_HZ = OMEGA / (2 * np.pi)


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def test_fig4(tpath_join):
    """KLMTV Fig. 4: lossless noise curves, with ideal frequency-dependent angles.

    Each SFLU curve is drawn solid, and the paper's closed form for it dashed on
    top. The angles are applied ideally, frequency by frequency, which is what
    Fig. 4 assumes. "Ideally" means tuned to the coupling the model actually
    has, ``model_kappa``. That differs from KLMTV's single-pole Eq. 18 by only
    $2\times10^{-5}$ at low frequency, but squeezed variational readout
    notices even that; see the next example.
    """
    T1 = kl.arm(F_HZ, 1.0)
    T10 = kl.arm(F_HZ, 10.0)
    phi1 = np.arctan2(1, kl.model_kappa(T1))
    phi10 = np.arctan2(1, kl.model_kappa(T10))
    sqz = kl.readout.squeezed
    mlib = kl.MLIB
    zeta_phase = kl.sflu_angle(np.pi / 2)

    # Each entry: label, SFLU curve, KLMTV closed form (Eq. number).
    curves = [
        ("conventional, $I_o = I_\\mathrm{SQL}$ (Eq. 29)",
         kl.sqrt_Sh_over_hSQL(T1, zeta_phase),
         kl.Sh_conventional(OMEGA, 1.0)),
        ("squeezed input, $\\lambda = -\\Phi$ (Eq. 49)",
         kl.sqrt_Sh_over_hSQL(
             T1, zeta_phase,
             {"ITM.bk.i.exc": sqz(mlib, SQZ, kl.sflu_angle(-phi1))}),
         kl.Sh_squeezed_input(OMEGA, 1.0, -phi1, SQZ)),
        ("squeezed input, fixed $\\lambda = -\\pi/4$ (Eq. 52)",
         kl.sqrt_Sh_over_hSQL(
             T1, zeta_phase,
             {"ITM.bk.i.exc": sqz(mlib, SQZ, kl.sflu_angle(-np.pi / 4))}),
         kl.Sh_squeezed_input(OMEGA, 1.0, -np.pi / 4, SQZ)),
        ("variational output, $I_o = 10 I_\\mathrm{SQL}$ (Eq. 58)",
         kl.sqrt_Sh_over_hSQL(T10, kl.sflu_angle(phi10)),
         kl.Sh_variational(OMEGA, 10.0)),
        ("squeezed variational, $I_o = 10 I_\\mathrm{SQL}$ (Eq. 73)",
         kl.sqrt_Sh_over_hSQL(
             T10, kl.sflu_angle(phi10),
             {"ITM.bk.i.exc": sqz(mlib, SQZ, kl.sflu_angle(np.pi / 2))}),
         kl.Sh_squeezed_variational(OMEGA, 10.0, SQZ)),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(X, kl.h_SQL(OMEGA) / kl.h_SQL(GAMMA), c="k", lw=1, label="$h_\\mathrm{SQL}$")
    for label, sflu, Sh in curves:
        paper = np.sqrt(Sh) / kl.h_SQL(GAMMA)
        line, = ax.loglog(X, sflu, lw=2, alpha=0.7, label=label)
        ax.loglog(X, paper, ls="--", c="k", lw=0.8)
        print(f"{label:60s} max |SFLU/paper - 1| = {rel_err(sflu, paper):.2e}")
        assert rel_err(sflu, paper) < 3e-3, label
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.03, 10)
    ax.set_xlabel("$\\Omega / \\gamma$")
    ax.set_ylabel("$\\sqrt{S_h} / h_\\mathrm{SQL}(\\gamma)$")
    ax.set_title("KLMTV Fig. 4 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig4.pdf"))

    # The check values KLMTV's figure can be read against, at Omega = gamma:
    # conventional and h_SQL both touch 1, squeezed input sits at sqrt(0.1).
    i = np.argmin(np.abs(X - 1))
    assert abs(curves[0][1][i] - 1) < 2e-3
    assert abs(curves[1][1][i] - np.sqrt(0.1)) < 2e-3


def test_fig4_with_filter_cavities(tpath_join):
    """The same curves, with the angles made by real filter cavities.

    Fig. 4 applies the frequency-dependent angles by hand. KLMTV realise them
    with two detuned filter cavities of the arm's length, whose detunings
    $\\xi_J$ and half-widths $\\delta_J$ are given in closed form: Eq. 89 for the
    output filters, Eq. 90 for the input filters. Here those cavities are SFLU
    graphs cascaded with the arm, and the readout is a fixed homodyne.
    """
    out_filters = kl.output_filters(10.0)
    # The input filters as Eq. 90 prints them put the squeezing at the wrong
    # angle (dotted curve, below). The output filters for the same power do
    # the job exactly. A detuned cavity rotates a squeezed state one way and a
    # readout quadrature the other, so one set of filters serves both; Eq. 90
    # negates the detunings, which accounts for that difference twice. See
    # ``kl.input_filters``.
    in_filters = kl.output_filters(1.0)
    in_printed = kl.input_filters()
    T10 = kl.with_output_filters(kl.arm(F_HZ, 10.0), F_HZ, out_filters)
    T1 = kl.with_input_filters(kl.arm(F_HZ, 1.0), F_HZ, in_filters)
    sqz = kl.readout.squeezed
    mlib = kl.MLIB
    fixed = kl.sflu_angle(np.pi / 2)  # theta = pi/2 for both, Eqs. 85 and 91

    # Squeezed variational readout is the exception, and it is instructive.
    # The filters are designed from KLMTV's single-pole formulas, while the
    # model's arm and filter cavities are exact; at low frequency the two differ
    # at the 1e-5 level (``model_kappa``). With 10 dB of anti-squeezing that is
    # enough to leave visible back-action below about 0.3 gamma: a real limit
    # on how precisely the filters must be built, not a numerical artefact.
    # That curve is therefore checked against Eq. 73 only above 0.3 gamma,
    # and its low-frequency excess is reported. Eq. 71 (``Sh_general``), fed
    # the model's own K, accounts for most of it.
    curves = [
        ("variational output via output filters",
         kl.sqrt_Sh_over_hSQL(T10, fixed),
         kl.Sh_variational(OMEGA, 10.0)),
        ("squeezed variational via output filters",
         kl.sqrt_Sh_over_hSQL(T10, fixed, {"ITM.bk.i.exc": sqz(mlib, SQZ, fixed)}),
         kl.Sh_squeezed_variational(OMEGA, 10.0, SQZ)),
        ("squeezed input via input filters",
         kl.sqrt_Sh_over_hSQL(T1, fixed, {"FCI1.bk.i.exc": sqz(mlib, SQZ, fixed)}),
         kl.Sh_squeezed_input(OMEGA, 1.0, -kl.Phi(OMEGA, 1.0), SQZ)),
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(X, kl.h_SQL(OMEGA) / kl.h_SQL(GAMMA), c="k", lw=1, label="$h_\\mathrm{SQL}$")
    for label, sflu, Sh in curves:
        paper = np.sqrt(Sh) / kl.h_SQL(GAMMA)
        ax.loglog(X, sflu, lw=2, alpha=0.7, label=label)
        ax.loglog(X, paper, ls="--", c="k", lw=0.8)
        band = X > 0.3 if label.startswith("squeezed variational") else X > 0
        err = rel_err(sflu[band], paper[band])
        print(f"{label:45s} max |SFLU/paper - 1| = {err:.2e}")
        assert err < 3e-3, label
    printed = kl.sqrt_Sh_over_hSQL(
        kl.with_input_filters(kl.arm(F_HZ, 1.0), F_HZ, in_printed), fixed,
        {"FCI1.bk.i.exc": sqz(mlib, SQZ, fixed)})
    ax.loglog(X, printed, ls=":", lw=1.5, label="squeezed input via Eq. 90 as printed")
    assert np.max(printed / curves[2][1]) > 5
    sv, sv_paper = curves[1][1], np.sqrt(curves[1][2]) / kl.h_SQL(GAMMA)
    excess = np.max(sv / sv_paper) - 1
    print(f"squeezed variational exceeds Eq. 73 by up to {excess:.1%} below 0.3 gamma")
    assert excess < 0.15
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.03, 10)
    ax.set_xlabel("$\\Omega / \\gamma$")
    ax.set_ylabel("$\\sqrt{S_h} / h_\\mathrm{SQL}(\\gamma)$")
    ax.set_title("KLMTV with filter cavities (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig4_filters.pdf"))


def test_fig10(tpath_join):
    """KLMTV Fig. 10: the output filter parameters against power.

    The x-axis is $(\\Lambda/\\gamma)^4 = 2 I_o / I_\\mathrm{SQL}$. The lower
    panel checks the claim the parameters exist to satisfy: that the two
    filters turn a fixed homodyne into $\\zeta(\\Omega) = \\Phi(\\Omega)$.
    """
    r = np.geomspace(0.3, 30, 200)
    pars = np.array([kl.output_filters(ri / 2) for ri in r])  # (N, 2, 2)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    ax.semilogx(r, pars[:, 0, 0], label="$\\xi_I$")
    ax.semilogx(r, pars[:, 1, 0], label="$\\xi_{II}$")
    ax.semilogx(r, pars[:, 0, 1], ls="--", label="$\\delta_I / \\gamma$")
    ax.semilogx(r, pars[:, 1, 1], ls="--", label="$\\delta_{II} / \\gamma$")
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlim(0.3, 30)
    ax.set_ylim(-1, 2.5)
    ax.set_xlabel("$(\\Lambda / \\gamma)^4 = 2 I_o / I_\\mathrm{SQL}$")
    ax.set_title("KLMTV Fig. 10: output filter parameters (Eq. 89)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Spot values against the table in the paper's Fig. 10, at I_o = I_SQL.
    (xi_I, d_I), (xi_II, d_II) = kl.output_filters(1.0)
    assert np.allclose([xi_I, xi_II, d_I, d_II],
                       [1.7671, -0.2772, 0.5156, 1.3018], atol=1e-4)

    # KLMTV Eqs. 81 and 88 have the filters rotate the quadrature by *half*
    # the summed arctangents. Evaluated as printed, the Eq. 89 parameters then
    # miss Phi by up to pi/4. With the full sum -- the true reflection phase of
    # a detuned cavity, 2 arctan(detuning / half-width) -- they hit it exactly.
    # The SFLU filter cavities in the previous example make the full rotation,
    # which is why they reproduce Eq. 58.
    for Io in (0.15, 1.0, 3.2, 15.0):
        filt = kl.output_filters(Io)
        full = kl.filter_angle(OMEGA, filt)
        half = np.pi / 2 - (np.pi / 2 - full) / 2
        target = kl.Phi(OMEGA, Io)
        wrap = lambda a: (a + np.pi / 2) % np.pi - np.pi / 2
        assert np.max(np.abs(wrap(full - target))) < 1e-9
        ax2.semilogx(X, np.degrees(wrap(half - target)),
                     label=f"half sum, $I_o = {Io:g}\\,I_\\mathrm{{SQL}}$")
    ax2.axhline(0, c="k", lw=1, label="full sum (all $I_o$)")
    ax2.set_xlabel("$\\Omega / \\gamma$")
    ax2.set_ylabel("$\\zeta - \\Phi$ [deg]")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(tpath_join("fig10.pdf"))
