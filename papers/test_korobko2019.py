r"""
M. Korobko, Y. Ma, Y. Chen and R. Schnabel,
*Quantum expander: precision and bandwidth enhancement with an internal
squeezing*,
[Light Sci. Appl. 8, 118 (2019)](https://doi.org/10.1038/s41377-019-0230-2),
[arXiv:1903.05930](https://arxiv.org/abs/1903.05930).

**What the paper shows.** In a detector with resonant sideband extraction the
arm cavity and the signal-extraction (SE) cavity form two coupled modes. A
degenerate parametric amplifier (a $\chi^{(2)}$ crystal) inside the SE
cavity, de-amplifying the signal quadrature, squeezes the output noise in
just the band where the coupled cavities lose signal. In the main text's
two-mode approximation (Eq. 9)

$$
S_h(\Omega) \propto
\frac{(\Omega^2-\omega_s^2)^2 + (\gamma-\chi)^2\Omega^2}{\gamma\,\omega_s^2},
\qquad
\gamma_q = \frac{\omega_s^2}{\gamma - \chi},
$$

so the shot-noise-limited bandwidth $\gamma_q$ grows without bound as the
crystal's gain $\chi$ approaches the SE coupling rate $\gamma$ (threshold),
while the low-frequency sensitivity is unchanged: the "quantum expander".

**The model.** The differential mode, with the crystal as a pair of one-way
``SQZEdge``s inside ``sflu.topologies.signal_recycled(internal_squeezer=True)``.
The mapping is in ``sflu.papers.korobko2019``; the points that matter:

* The paper's quantitative model is the exact two-cavity one of its
  supplement (Sec. S3, S5), and that is what the SFLU graph is. Its
  single-pass gain $q$ appears as $20\log_{10}e^{q}$ dB on *each* squeezer
  edge, with the *same* sign in both directions (see the last example).
* The SE cavity carries a one-way quadrature rotation of $\pi/2$ (RSE). That
  rotates the dark port's frame too, so the model turns it back into the
  arm frame; then phase readout is $\zeta = \pi/2$, as in the paper.
* The ETM has mass $m/4$ and moves by $L h$; the arm circulates
  $P_\mathrm{arm}/2$. Then $h_\mathrm{SQL}^2 = 8\hbar/(m\Omega^2L^2)$ and the
  shot noise agrees with Sec. S3 to machine precision.

**Not stated in the paper, inferred here.** (1) The crystal gain of the Fig. 3
expander curves: we use the threshold $e^{2q}R_s = 1$, i.e.
$q_\mathrm{th} = -\ln R_s/2 = 0.1077$ per pass (0.935 dB). (2) That the
"0.5%, 3%, 10% loss" of Fig. 3 is *readout* loss only. Both reproduce the
figure to reading accuracy; neither is written in the paper. Fig. S2's gain,
its external squeezing and the absence of radiation pressure are inferred too.

**Errata found.** Eq. S18 drops $R_s$ from its $e^{2q}$ term (the printed
form is not unitary even at $q = 0$); Eq. S94 multiplies by the arm's
$\operatorname{sinc}^2$ where it must divide; the prefactor of Eqs. 9-11 is
low by a factor 2 even with the Fig. 3 caption's $P_c = 4$ MW. Details below.

**Reproduced:** Fig. 3 (all curves, to $10^{-10}$ against the paper's exact
formulas) and, approximately, Fig. S2. Figs. S3 and S4 need the expander gain
and filter-cavity parameters, which the paper does not give.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import korobko2019 as ko

P = ko.PARAMS
Q_TH = ko.q_threshold()

# Fig. 3 / S2 x-axis: 10 Hz to just below the arm FSR (7.49 kHz).
F_HZ = np.geomspace(10, 7400, 200)
OMEGA = 2 * np.pi * F_HZ
U = 1e-24  # Fig. 3 plots sqrt(S_h) in units of 1e-24 / sqrt(Hz)


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def crossing(x, y, level):
    """The x values where y crosses level (log-linear interpolation)."""
    s = np.sign(y - level)
    i = np.nonzero(s[1:] != s[:-1])[0]
    lx, ly = np.log(x), np.log(y)
    t = (np.log(level) - ly[i]) / (ly[i + 1] - ly[i])
    return np.exp(lx[i] + t * (lx[i + 1] - lx[i]))


def test_fig3(tpath_join):
    r"""Korobko et al. Fig. 3: the quantum expander against the baseline detector.

    Table S1 parameters: 20 km arms, 4 MW per arm, 200 kg mirrors,
    $T_\mathrm{ITM} = 0.07$, $T_\mathrm{SEM} = 0.35$, a 56 m SE cavity, 1550 nm.
    The baseline (blue) is the RSE detector without a crystal, read out in
    the phase quadrature, lossless and with 10% readout loss (gray). The red
    curves are the expander at threshold with 0, 0.5, 3 and 10% readout loss
    and variational readout; below ~50 Hz they are the paper's green
    "+Variational readout" curves, above it variational readout no longer
    matters.

    The paper's closed forms (dashed): the exact two-cavity shot noise of
    Eqs. S18-S19 (with $R_s$ restored in S18), readout loss as in Eq. S91,
    the GW response of Eq. S94 (with the sinc dividing), and radiation
    pressure from the lossless relation $S_{xx}S_{FF} = \hbar^2/4$ behind
    Eqs. 12-14. Variational readout is the ideal, frequency-dependent
    homodyne angle that cancels back-action, the lossless-filter limit of
    Eq. S95; the paper's filter cavity parameters are not given.
    """
    T0 = ko.model(F_HZ, 0.0)
    Tq = ko.model(F_HZ, Q_TH)
    zeta_q = ko.ba_evading_angle(Tq)
    reds = ["#67000d", "#cb181d", "#ef3b2c", "#fc9272"]

    # label, SFLU S_h, paper S_h, style
    curves = [("baseline, phase readout",
               ko.Sh(T0, F_HZ)[0], ko.Sh_phase_readout(OMEGA, 0.0),
               dict(c="#3030c0")),
              ("baseline, phase readout, 10% loss",
               ko.Sh(T0, F_HZ, eta=0.9)[0], ko.Sh_phase_readout(OMEGA, 0.0, 0.9),
               dict(c="gray"))]
    for loss, c in zip(ko.FIG3_LOSSES, reds):
        curves.append((f"expander + var. readout, {100 * loss:g}% loss",
                       ko.Sh(Tq, F_HZ, zeta=zeta_q, eta=1 - loss)[0],
                       ko.Sh_variational(OMEGA, Q_TH, 1 - loss),
                       dict(c=c)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(F_HZ, ko.h_SQL(OMEGA) / U, c="k", ls=":", lw=1, label="SQL")
    for label, sflu, paper, style in curves:
        ax.loglog(F_HZ, np.sqrt(sflu) / U, lw=2, alpha=0.8, label=label, **style)
        ax.loglog(F_HZ, np.sqrt(paper) / U, ls="--", c="k", lw=0.8)
        err = rel_err(np.sqrt(sflu), np.sqrt(paper))
        print(f"{label:42s} max |SFLU/paper - 1| = {err:.1e}")
        assert err < 1e-8, label
    ax.set_xlim(10, 7400)
    ax.set_ylim(0.2, 5)
    ax.set_yticks([0.2, 0.3, 0.5, 1, 2, 3, 5], ["0.2", "0.3", "0.5", "1", "2", "3", "5"])
    ax.set_yticks([], minor=True)
    ax.set_xlabel("$\\Omega / 2\\pi$ [Hz]")
    ax.set_ylabel("$\\sqrt{S_h}$ [$10^{-24}/\\sqrt{\\mathrm{Hz}}$]")
    ax.set_title("Korobko et al. Fig. 3 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("korobko2019_fig3.pdf"))

    # Values read off Fig. 3, in units of 1e-24 / sqrt(Hz). Baseline: 3.0 at
    # 10 Hz, minimum about 0.47 between 50 and 100 Hz, 1.2 at 1 kHz, 5 at
    # about 4.4 kHz. SQL 1.63 at 10 Hz.
    amp = {lab: np.sqrt(s) / U for lab, s, _, _ in curves}
    base = amp["baseline, phase readout"]
    assert abs(base[0] - 3.0) < 0.1
    assert 50 < F_HZ[np.argmin(base)] < 100 and abs(base.min() - 0.47) < 0.02
    assert abs(np.interp(1000, F_HZ, base) - 1.2) < 0.05
    assert abs(crossing(F_HZ, base, 5.0)[0] - 4400) < 200
    assert abs(ko.h_SQL(OMEGA[0]) / U - 1.63) < 0.01
    # Lossless expander: flat 0.45 at low frequency, below 0.2 from about 2.5
    # to 3.7 kHz, back up to 5 between 6.5 and 7 kHz.
    qe = amp["expander + var. readout, 0% loss"]
    assert abs(qe[0] - 0.45) < 0.01
    lo, hi = crossing(F_HZ, qe, 0.2)[:2]
    print(f"lossless expander below 0.2 from {lo:.0f} to {hi:.0f} Hz")
    assert abs(lo - 2500) < 200 and abs(hi - 3700) < 200
    assert 6500 < crossing(F_HZ, qe, 5.0)[-1] < 7000
    # 0.5% loss: minimum about 0.435 near 2 kHz. Variational readout at 10 Hz:
    # about 0.50, 0.69 and 1.0 for 0.5, 3 and 10% loss.
    qe05 = amp["expander + var. readout, 0.5% loss"]
    assert abs(qe05.min() - 0.435) < 0.01 and 1500 < F_HZ[np.argmin(qe05)] < 2500
    for loss, read in [(0.5, 0.50), (3, 0.69), (10, 1.0)]:
        assert abs(amp[f"expander + var. readout, {loss:g}% loss"][0] / read - 1) < 0.1

    # The noise zero of the lossless expander, on a fine grid: the exact model
    # puts it at 3.26 kHz. The two-mode picture would put it at the sloshing
    # frequency, 5.96 kHz, which is not small against the 7.49 kHz FSR here.
    f_fine = np.linspace(3100, 3400, 301)
    f0 = f_fine[np.argmin(ko.Sh_shot(2 * np.pi * f_fine, Q_TH))]
    print(f"noise zero at {f0:.0f} Hz; omega_s/2pi = {ko.omega_s() / 2 / np.pi:.0f} Hz")
    assert abs(f0 - 3255) < 5

    # The main text's Eq. 10 (and so Eqs. 9, 11) with P_c = 4 MW, as in the
    # Fig. 3 caption, sits a factor ~0.59 below the plotted baseline shot
    # noise at 10 Hz. 1.23 of that is the two-mode bandwidth (477 Hz instead
    # of the exact 389 Hz); the remaining factor 2 is in the prefactor. With
    # Table S1's P_c = 2 P_arm = 8 MW it is a factor 4. Eqs. 9-11 are good
    # for shapes, not levels.
    r = ko.Sh_two_mode(OMEGA[0], 0.0, 4e6) / ko.Sh_shot(OMEGA[0], 0.0)
    print(f"Eq. 10 (P_c = 4 MW) / exact shot noise at 10 Hz = {r:.3f}")
    assert abs(r - 0.59) < 0.01


def test_figS2(tpath_join):
    r"""Korobko et al. Fig. S2 (approximate): where the noise comes from.

    The fraction of the total quantum noise contributed by each vacuum input:
    the dark port, the SE-cavity loss (1500 ppm per pass, both vacua), the
    detection loss (1%) and the arm (ETM transmission raised to 100 ppm, as
    in the caption).

    The caption does not give the crystal gain, nor say whether external
    squeezing and radiation pressure are included. Inferred: 10 dB of phase
    squeezing is injected (the detection fraction of ~0.08 at low frequency
    needs an input ~10 dB below vacuum), there is no radiation pressure (the
    input fraction is flat to 10 Hz), and the gain puts an exact zero in the
    input contribution, $e^{2q} = (1-\lambda_s)/R_s$ -- the lossy version of
    the threshold. The loss matrices of Sec. S5 ($\mathcal L_{b1,b2}$, $\mathcal L_{d1,d2}$) are not usable
    as a check: they mix $\lambda_s$ and $\sqrt{\lambda_s}$ and drop factors of
    $R_s$, $T_s$; so this figure is compared to read-off values only.
    """
    q = ko.q_input_zero()
    T = ko.model(F_HZ, q, rp=False, T_ETM=ko.FIGS2_T_ETM, lambda_s=P["lambda_s"])
    states = {ko.IN: ko.readout.squeezed(ko.MLIB, P["sqz_ext_dB"], np.pi / 2)}
    total, b = ko.Sh(T, F_HZ, eta=P["eta"], states=states)
    frac = {
        "Input": b[ko.IN] / total,
        "SE cavity": sum(b[k] for k in ko.SE_LOSS) / total,
        "Detection": b["readout"] / total,
        "Arm cavity": b[ko.ARM_LOSS] / total,
    }
    styles = {"Input": dict(c="#b00000", ls="-"),
              "SE cavity": dict(c="#c000c0", ls="--"),
              "Detection": dict(c="#0000a0", ls="-."),
              "Arm cavity": dict(c="#00a000", ls=":")}

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, v in frac.items():
        ax.loglog(F_HZ, v, lw=2, label=k, **styles[k])
    ax.set_xlim(10, 7400)
    ax.set_ylim(5e-4, 2)
    ax.set_xlabel("$\\Omega / 2\\pi$ [Hz]")
    ax.set_ylabel("fraction of total quantum noise")
    ax.set_title("Korobko et al. Fig. S2 (SFLU only; approximate)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("korobko2019_figS2.pdf"))

    print(f"q = {q:.4f} ({ko.sqz_dB_per_pass(q):.3f} dB per pass), "
          f"q_th = {Q_TH:.4f}")
    for f in (10, 1000, 5000):
        i = np.argmin(np.abs(F_HZ - f))
        print(f"{f:5d} Hz: " + ", ".join(f"{k} {v[i]:.3g}" for k, v in frac.items()))
    assert np.allclose(sum(frac.values()), 1)

    # Agrees with the figure: the input contribution has a zero near 3 kHz
    # and recovers to ~0.5-0.7 at the FSR; the arm contributes ~5e-3 at low
    # frequency and falls off; detection is ~0.1 at low frequency and ~0.5
    # in the kHz band, where the input noise is squeezed away.
    i_zero = np.argmin(frac["Input"])
    assert 2800 < F_HZ[i_zero] < 3600 and frac["Input"][i_zero] < 1e-4
    assert frac["Input"][-1] > 0.4
    assert abs(frac["Arm cavity"][0] / 4.7e-3 - 1) < 0.25
    assert abs(frac["Detection"][0] / 0.08 - 1) < 0.25
    assert 0.4 < np.interp(2000, F_HZ, frac["Detection"]) < 0.7

    # Disagrees: the figure has the SE-cavity loss at 0.14 of the total at low
    # frequency; here it is ~0.003. Physically the SE cavity is anti-resonant
    # for low-frequency sidebands (RSE), so loss inside it barely couples
    # there; 0.14 would need ~50 times more loss. Above 1 kHz, where the SE
    # cavity takes part in the sloshing mode, the two agree within ~30%
    # (0.3-0.4 here, ~0.5 in the figure). We attribute the low-frequency
    # difference to the printed loss matrices; we cannot check that.
    assert frac["SE cavity"][0] < 0.01
    assert 0.25 < np.interp(2000, F_HZ, frac["SE cavity"]) < 0.6


def test_crystal_orientation(tpath_join):
    r"""Why both squeezer edges carry the same sign, and the reference model's
    opposite sign.

    The crystal squeezes a fixed physical quadrature whichever way light
    crosses it. Here both edges sit at the same point with the RSE rotation on
    one side, so they share a frame, and the paper's crystal (Eqs. S39, S41:
    $\mathcal S$ on both passes) is the same ``sqzDB`` on both. The repository's
    reference internal-squeezing model (``sflu.models.coupled_cavity``) puts
    *opposite* ``sqzDB`` on its two adjacent edges. Then a round trip of the
    SE cavity is $\mathcal S^{-1}(\pm r)\,\mathcal S$: the light is squeezed
    going in and un-squeezed coming out, with no parametric gain in the loop.
    For a tuned, lossless SE cavity without radiation pressure that only
    amplifies the signal by $e^{q}$ and leaves the output vacuum alone:
    exactly the baseline curve divided by $e^{2q}$, with no frequency
    dependence -- not an expander. (It is the configuration of a crystal a
    quarter-period of the pump standing wave away.)

    Also shown: the expander computed from Eq. S18 exactly as printed. Its
    missing $R_s$ makes the noise reflection non-unitary, $|R_a| = 1.107$ at DC
    for $q = 0$, and at $q_\mathrm{th}$ it shows no expander at all: no
    noise zero, everything above the baseline.
    """
    Tb = ko.model(F_HZ, 0.0, rp=False)
    Tq = ko.model(F_HZ, Q_TH, rp=False)
    Tr = ko.model(F_HZ, Q_TH, rp=False, reference_orientation=True)
    Sb, Sq, Sr = (ko.Sh(T, F_HZ)[0] for T in (Tb, Tq, Tr))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(F_HZ, np.sqrt(Sb) / U, c="#3030c0", lw=2, alpha=0.8,
              label="no crystal")
    ax.loglog(F_HZ, np.sqrt(Sq) / U, c="#cb181d", lw=2, alpha=0.8,
              label="same sign both ways (quantum expander)")
    ax.loglog(F_HZ, np.sqrt(Sr) / U, c="#e08000", lw=2, alpha=0.8,
              label="opposite signs (as in coupled_cavity)")
    ax.loglog(F_HZ, np.sqrt(ko.Sh_shot(OMEGA, Q_TH)) / U, c="k", ls="--", lw=0.8)
    ax.loglog(F_HZ, np.sqrt(np.exp(-2 * Q_TH) * Sb) / U, c="k", ls="--", lw=0.8)
    Ra_pr, X = ko._S3(OMEGA, Q_TH, P, printed=True)
    printed = np.abs(Ra_pr)**2 / X**2 / (P["L_m"] * ko.sinc_arm(OMEGA))**2
    ax.loglog(F_HZ, np.sqrt(printed) / U, c="gray", ls=":", lw=1.5,
              label="expander from Eq. S18 as printed")
    ax.set_xlim(10, 7400)
    ax.set_ylim(0.01, 10)
    ax.set_xlabel("$\\Omega / 2\\pi$ [Hz]")
    ax.set_ylabel("$\\sqrt{S_h}$ (shot noise) [$10^{-24}/\\sqrt{\\mathrm{Hz}}$]")
    ax.set_title("Squeezer orientation, $q = q_\\mathrm{th}$ (solid: SFLU, dashed: closed form)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("korobko2019_orientation.pdf"))

    e1 = rel_err(Sq, ko.Sh_shot(OMEGA, Q_TH))
    e2 = rel_err(Sr, np.exp(-2 * Q_TH) * Sb)
    print(f"same sign vs Eqs. S18-S19:        max rel. err. {e1:.1e}")
    print(f"opposite signs vs e^-2q baseline: max rel. err. {e2:.1e}")
    assert e1 < 1e-8 and e2 < 1e-12
    Ra0 = np.abs(ko.R_noise(np.array([2 * np.pi * 1e-3]), 0.0, printed=True))[0]
    print(f"|R_a(DC)| at q = 0 from Eq. S18 as printed: {Ra0:.4f}")
    assert abs(Ra0 - 2 / (1 + np.sqrt(1 - P["T_SEM"]))) < 1e-6
