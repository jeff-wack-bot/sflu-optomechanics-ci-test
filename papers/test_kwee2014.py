r"""
P. Kwee, J. Miller, T. Isogai, L. Barsotti and M. Evans,
*Decoherence and degradation of squeezed states in quantum filter cavities*,
[Phys. Rev. D 90, 062006 (2014)](https://doi.org/10.1103/PhysRevD.90.062006),
[arXiv:1704.03531](https://arxiv.org/abs/1704.03531).

**What the paper shows.** A detuned filter cavity (FC) turns
frequency-independent squeezing into the frequency-dependent squeezing a
radiation-pressure-limited interferometer needs: the squeeze angle must follow
$\tan\alpha_p = \mathcal K$. Kwee et al. build a two-photon model of the whole
chain -- squeezer, injection loss, FC, mode mismatch, interferometer, readout
loss, homodyne -- and quantify everything that degrades the result: FC
round-trip loss (which both adds vacuum and *dephases* the squeezed state),
injection and readout loss, mode mismatch (part of the squeezed light bypasses
the FC), and phase noise, both frequency-independent and from FC length
noise. The noise is a sum over the paper's three vacuum paths (Eq. 43),

$$
\widehat N(\zeta) = \sum_{n=1}^3 \big|(\sin\zeta,\ \cos\zeta)\,\mathbf T_n\big|^2,
\qquad
\mathbf T_1 = \tau_\mathrm{ro}\,\mathbf T_\mathrm{ifo}
  \big(t_{00}\mathbf T_\mathrm{fc} + \mathbf T_\mathrm{mm}\big)
  \tau_\mathrm{inj}\mathbf S(\sigma, \phi),
$$

and their single figure, Fig. 2, plots it relative to coherent vacuum,
$10\log_{10}[\widehat N / (1 + \mathcal K^2)]$, for a 16 m FC on Advanced LIGO.

**The model.** Each element of the chain is an SFLU graph, solved separately
and joined with ``sflu.topologies.cascade``: a pick-off mirror for injection
loss, ``topologies.filter_cavity``, a mode-matching "telescope", a
Fabry-Perot arm with a free end mirror for the interferometer
(``topologies.fp_arm``, with Kwee's $\gamma_\mathrm{ifo}$ and
$\Omega_\mathrm{SQL}$), and a second pick-off for readout loss. The details
are in ``sflu.papers.kwee2014``.

**Mode mismatch is modelled with one higher-order mode** (``MatrixLib(nhom=1)``).
Kwee lump every mode but the FC's fundamental into one bypass amplitude
$t_\mathrm{mm}$; with a single higher-order mode their parametrisation (Eqs.
25-27) is not an upper bound but exact. The FC link is conjugated by
``mlib.MrotationMM(1 - a_0^2, 0)``, the change from the squeezer's mode basis
to the cavity's, and gives the cavity's higher-order mode a Gouy phase that
makes it anti-resonant. The telescope's link carries
``mlib.MrotationMM(1 - c_0^2, phi_mm)``, the change to the local oscillator's
basis. From there on the fundamental *is* the LO mode, so the interferometer's
carrier and the homodyne (``readout.quadrature``) both see exactly what Kwee's
Eq. 38 has them see. No mismatch formula is coded into the model: the bypass,
the extra vacuum and their interference all come out of the graph.

**Two places where the paper and its figure disagree with its text**, both
established below: the "frequency-independent phase noise" curve is a jitter
of the squeeze angle, not of the readout angle the text names; and the FC
detuning has the opposite sign to the paper's definition
$\Delta\omega_\mathrm{fc} = \omega_\mathrm{fc} - \omega_0$ (a sign slip in
Eq. 3, which leaves every result intact).

**Reproduced:** Fig. 2, every curve, to 0.002 dB with the paper's
phase-noise average and 0.05 dB with an exact Gaussian one.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import kwee2014 as kw

P = kw.PARAMS

# The x-axis of Fig. 2: 1 Hz to 10 kHz.
F_HZ = np.geomspace(1, 1e4, 200)
OMEGA = 2 * np.pi * F_HZ

COLORS = {
    "Ideal system": "b",
    "Frequency independent phase noise": "limegreen",
    "Injection/Readout losses": "r",
    "Mode-mismatch": "c",
    "Frequency dependent phase noise": "m",
    "Filter cavity losses": "gold",
    "All mechanisms": "0.3",
}


def at(F, curve):
    r"""The value of a curve at frequency ``F`` [Hz] (nearest grid point)."""
    return curve[np.argmin(np.abs(F_HZ - F))]


def test_fig2(tpath_join):
    r"""Kwee Fig. 2: the squeezing each degradation mechanism leaves.

    Every single-mechanism curve is the ideal system plus that one mechanism;
    "All mechanisms" has them all. The mode-mismatch families are drawn for
    the six values $\phi_\mathrm{mm} = k\pi/5$, $k = 0..5$; as in the paper,
    the band between the extremes of "All mechanisms" is shaded. The paper gives neither the
    $\phi_\mathrm{mm}$ grid nor the FC of the single-mechanism curves; the
    grid $k\pi/5$ and the lossless optimum
    $\Delta\omega_\mathrm{fc} = \gamma_\mathrm{fc} = \Omega_\mathrm{SQL}/\sqrt 2$
    (Eqs. 51-52) reproduce them. "Filter cavity losses" and "All mechanisms"
    use Table II's cavity.
    """
    T_ifo = kw.ifo_stage(F_HZ)

    # First the interferometer. Its end mirror has mass m/2, the reduced mass
    # of one arm, and the circulating power that gives Kwee's Omega_SQL comes
    # out at Table I's 800 kW exactly. Its K then matches Eq. 29 to 3e-3 below
    # 1 kHz; above, K is below 1e-4 and the arm's free spectral range makes
    # the difference irrelevant.
    P_circ = kw.circulating_power()
    K_err = np.max(np.abs(kw.model_kappa(T_ifo) / kw.kappa(OMEGA) - 1)[F_HZ < 1e3])
    print(f"circulating power {P_circ / 1e3:.3f} kW, max |K/K_Eq29 - 1| = {K_err:.1e}")
    assert abs(P_circ / P["P_arm_W"] - 1) < 1e-12
    assert K_err < 5e-3

    # Each SFLU curve is computed twice. Phase noise is averaged over a
    # Gaussian distribution of the jittered parameter: once with Eq. 61's
    # two-point rule (n = 2, which is what the paper plots), once with a
    # 5-point Gauss-Hermite rule, the exact Gaussian average for a
    # polynomial of degree 9. The two differ only for FC length noise, by up
    # to 0.05 dB near 50 Hz, where the noise depends strongly on the
    # detuning; the solid curves are the exact average.
    configs = kw.fig2_configs()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    results = {}
    for label, family in configs.items():
        c = COLORS[label]
        sflu = np.array([kw.model_relative_dB(F_HZ, cfg, T_ifo) for cfg in family])
        sflu2 = np.array([kw.model_relative_dB(F_HZ, cfg, T_ifo, n=2) for cfg in family])
        paper = np.array([kw.relative_dB(OMEGA, cfg) for cfg in family])
        results[label] = sflu
        err2 = np.max(np.abs(sflu2 - paper))
        err = np.max(np.abs(sflu - paper))
        print(f"{label:35s} max |SFLU - paper| = {err2:.4f} dB (Eq. 61 average), "
              f"{err:.4f} dB (Gaussian average)")
        assert err2 < 0.005, label
        assert err < 0.06, label
        lw = 2 if len(family) == 1 else 1
        for i, (s, pp) in enumerate(zip(sflu, paper)):
            ax.semilogx(F_HZ, s, c=c, lw=lw, label=label if i == 0 else None)
            ax.semilogx(F_HZ, pp, ls="--", c="k", lw=0.6)
        if label == "All mechanisms":
            ax.fill_between(F_HZ, sflu.min(0), sflu.max(0), color=c, alpha=0.25, lw=0)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(-10, 0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Quantum noise relative to coherent vacuum [dB]")
    ax.set_title("Kwee et al. Fig. 2 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig2.pdf"))

    # Spot checks against values read off the paper's figure (to ~0.1 dB).
    # At low frequency the squeezing is 9.1 dB in the ideal system; 5% + 5%
    # loss leaves 8.0 dB there and 6.8 dB at high frequency, where the FC
    # reflects with no rotation and the full anti-squeezing no longer
    # matters. The mismatch family spans -8.3 to -5.6 dB at low frequency.
    # FC loss limits the low-frequency end to -3.3 dB, worst (-1.85 dB) near
    # the 48 Hz detuning. All mechanisms together leave 6 dB at high
    # frequency and between 2.2 and 2.7 dB at low frequency.
    r = results
    checks = [
        ("ideal, 1 Hz", at(1, r["Ideal system"][0]), -9.10, 0.05),
        ("phase noise, 1 Hz", at(1, r["Frequency independent phase noise"][0]), -8.85, 0.05),
        ("inj/ro, 1 Hz", at(1, r["Injection/Readout losses"][0]), -8.00, 0.05),
        ("inj/ro, 10 kHz", at(1e4, r["Injection/Readout losses"][0]), -6.81, 0.05),
        ("mismatch low, 1 Hz", at(1, r["Mode-mismatch"].min(0)), -8.31, 0.1),
        ("mismatch high, 1 Hz", at(1, r["Mode-mismatch"].max(0)), -5.58, 0.1),
        ("mismatch, 10 kHz", at(1e4, r["Mode-mismatch"][0]), -7.78, 0.05),
        ("length noise, 1 Hz", at(1, r["Frequency dependent phase noise"][0]), -6.62, 0.1),
        ("length noise, peak", r["Frequency dependent phase noise"][0].max(), -5.85, 0.15),
        ("FC loss, 1 Hz", at(1, r["Filter cavity losses"][0]), -3.33, 0.05),
        ("FC loss, peak", r["Filter cavity losses"][0].max(), -1.85, 0.05),
        ("all, 1 Hz, lower", at(1, r["All mechanisms"].min(0)), -2.70, 0.1),
        ("all, 1 Hz, upper", at(1, r["All mechanisms"].max(0)), -2.20, 0.1),
        ("all, peak, upper", r["All mechanisms"].max(), -0.73, 0.1),
        ("all, 10 kHz", at(1e4, r["All mechanisms"][0]), -5.95, 0.05),
    ]
    for name, value, read, tol in checks:
        print(f"  {name:22s} SFLU {value:6.2f} dB, read off Fig. 2 {read:6.2f} dB")
        assert abs(value - read) < tol, name


def test_mismatch_phase(tpath_join):
    r"""The mode-mismatch family, against the unknown phase $\phi_\mathrm{mm}$.

    Kwee fix the squeezer-FC and squeezer-LO mode overlaps and leave the
    phase of the bypass amplitude, $\phi_\mathrm{mm}$ in Eq. 25, unknown; the
    figure's "upper and lower bounds" are taken over
    $\phi_\mathrm{mm} \in [0, \pi]$. With the single higher-order mode of
    this model, $\phi_\mathrm{mm}$ is the phase of ``MrotationMM`` in the
    telescope, and the closed form is followed exactly around the whole
    circle. The other half circle, $(\pi, 2\pi)$, is not a mirror image: it
    contains *better* configurations, down to $-9.0$ dB at low frequency,
    where the bypassing light happens to sit at the right angle. So the
    paper's lower bound is not a bound over all mode-matching phases.
    """
    T_ifo = kw.ifo_stage(F_HZ)
    phis = np.linspace(0, 2 * np.pi, 41)
    base = kw.fig2_configs()["Mode-mismatch"][0]
    idx = [np.argmin(np.abs(F_HZ - f)) for f in (1, 50, 100)]
    fig, ax = plt.subplots(figsize=(6, 4))
    sflu = np.array([kw.model_relative_dB(F_HZ, dict(base, phi_mm=f), T_ifo)[idx]
                     for f in phis])
    paper = np.array([kw.relative_dB(OMEGA[idx], dict(base, phi_mm=f)) for f in phis])
    err = np.max(np.abs(sflu - paper))
    print(f"max |SFLU - paper| over phi_mm = {err:.4f} dB")
    assert err < 0.005
    for j, f in enumerate((1, 50, 100)):
        ax.plot(phis / np.pi, sflu[:, j], lw=2, label=f"{f} Hz")
        ax.plot(phis / np.pi, paper[:, j], ls="--", c="k", lw=0.8)
    ax.axvspan(0, 1, color="c", alpha=0.15, label="range plotted in Fig. 2")
    ax.set_xlim(0, 2)
    ax.set_xlabel("$\\phi_\\mathrm{mm} / \\pi$")
    ax.set_ylabel("relative to coherent vacuum [dB]")
    ax.set_title("Kwee mode mismatch vs. $\\phi_\\mathrm{mm}$ (solid: SFLU, dashed: paper)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(tpath_join("mismatch_phase.pdf"))

    low = sflu[:, 0]
    in_fig = phis <= np.pi + 1e-9
    print(f"1 Hz: [0, pi] spans {low[in_fig].min():.2f}..{low[in_fig].max():.2f} dB, "
          f"full circle {low.min():.2f}..{low.max():.2f} dB")
    assert abs(low[in_fig].min() - -8.31) < 0.02
    assert low.min() < -8.9


def test_phase_noise_quadrature(tpath_join):
    r"""Which angle the "frequency-independent phase noise" jitters.

    Kwee's text represents it by a jitter $\delta\zeta = 30$ mrad of the
    homodyne *readout* angle (Table II). But in Fig. 2's normalisation that
    costs nothing at low frequency: there the signal quadrature carries
    radiation-pressure noise $\mathcal K^2$ times larger than the squeezed
    shot noise, and a small readout rotation mixes in only the
    unamplified, squeezed, other quadrature. The plotted curve is flat at
    $-8.85$ dB all the way down, which is what a 30 mrad jitter of the
    *squeeze* angle $\phi_\mathrm{sqz}$ gives: at low frequency the FC hands
    the interferometer amplitude squeezing, and a rotation of it lets the
    anti-squeezing into the radiation-pressure noise. Both are shown, SFLU and
    closed form; Fig. 2 is reproduced with the squeeze-angle jitter.
    """
    T_ifo = kw.ifo_stage(F_HZ)
    ideal = kw.fig2_configs()["Ideal system"][0]
    cases = [
        ("squeeze angle $\\delta\\phi_\\mathrm{sqz}$ = 30 mrad (as plotted)",
         dict(ideal, dphi_sqz=P["dzeta"])),
        ("readout angle $\\delta\\zeta$ = 30 mrad (as in the text)",
         dict(ideal, dzeta=P["dzeta"])),
        ("no phase noise", ideal),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    out = []
    for label, cfg in cases:
        sflu = kw.model_relative_dB(F_HZ, cfg, T_ifo, n=2)
        paper = kw.relative_dB(OMEGA, cfg)
        err = np.max(np.abs(sflu - paper))
        print(f"{label:60s} max |SFLU - paper| = {err:.4f} dB")
        assert err < 0.005
        ax.semilogx(F_HZ, sflu, lw=2, label=label)
        ax.semilogx(F_HZ, paper, ls="--", c="k", lw=0.8)
        out.append(sflu)
    ax.set_xlim(1, 1e4)
    ax.set_ylim(-9.3, -8.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("relative to coherent vacuum [dB]")
    ax.set_title("Kwee phase noise (solid: SFLU, dashed: paper)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(tpath_join("phase_noise.pdf"))

    # Fig. 2's green curve reads -8.85 dB at 1 Hz. Only the squeeze-angle
    # jitter gets there; readout-angle jitter leaves the ideal -9.10 dB.
    # Both agree at high frequency, where K -> 0 and the two rotations
    # are equivalent.
    sqz, ro, none = out
    assert abs(at(1, sqz) - -8.85) < 0.02
    assert abs(at(1, ro) - at(1, none)) < 0.01
    assert abs(at(1e4, sqz) - at(1e4, ro)) < 0.01


def test_detuning_sign(tpath_join):
    r"""The sign of the FC detuning.

    Kwee define $\Delta\omega_\mathrm{fc} = \omega_\mathrm{fc} - \omega_0$,
    resonance minus carrier, and propagate with $e^{-i\Phi}$,
    $\Phi = (\Omega - \Delta\omega_\mathrm{fc})\,2L_\mathrm{fc}/c$ (Eqs. 3-4).
    In their $e^{-i\omega t}$ convention (Eq. A1) a delay $\tau$ multiplies an
    amplitude by $e^{+i\omega\tau}$, so the physical round trip is
    $e^{+i\Phi}$: Eq. 3 is the complex conjugate of the true reflectivity.
    With the physical propagator the figure needs
    $\Delta\omega_\mathrm{fc} = -2\pi\cdot 49.5$ Hz, the carrier *above*
    the resonance. The SFLU model, whose ``topologies.detuning_rad`` uses
    KLMTV's sign ($\xi > 0$: carrier above resonance), independently needs
    the same: Kwee's printed $+\Delta\omega_\mathrm{fc}$ fed to
    ``detuning_rad`` with *its* sign. The paper's numbers are unaffected;
    only the words "resonance minus carrier" are reversed.
    """
    T_ifo = kw.ifo_stage(F_HZ)
    ideal = kw.fig2_configs()["Ideal system"][0]
    flip = dict(ideal, detune_Hz=-ideal["detune_Hz"])
    # Blue: the right rotation; red: the wrong one.
    curves = [
        ("Eq. 3 as printed, $+\\Delta\\omega_\\mathrm{fc}$",
         kw.relative_dB(OMEGA, ideal), dict(c="C0", lw=4, alpha=0.4)),
        ("Eq. 3 as printed, $-\\Delta\\omega_\\mathrm{fc}$",
         kw.relative_dB(OMEGA, flip), dict(c="C3", lw=4, alpha=0.4)),
        ("Eq. 3 with $e^{+i\\Phi}$, $-\\Delta\\omega_\\mathrm{fc}$",
         kw.relative_dB(OMEGA, flip, physical=True), dict(c="C0", ls=":", lw=2)),
        ("SFLU, carrier above resonance",
         kw.model_relative_dB(F_HZ, ideal, T_ifo), dict(c="navy", ls="--", lw=1)),
        ("SFLU, carrier below resonance",
         kw.model_relative_dB(F_HZ, flip, T_ifo), dict(c="darkred", ls="--", lw=1)),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, y, style in curves:
        ax.semilogx(F_HZ, y, label=label, **style)
    ax.set_xlim(1, 1e4)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("relative to coherent vacuum [dB]")
    ax.set_title("Kwee ideal system: sign of the FC detuning", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(tpath_join("detuning_sign.pdf"))

    printed, wrong, physical, sflu, sflu_wrong = (c[1] for c in curves)
    assert np.max(np.abs(physical - printed)) < 1e-9
    assert np.max(np.abs(sflu - printed)) < 0.005
    print(f"wrong sign: up to {wrong.max():+.1f} dB (closed form), "
          f"{sflu_wrong.max():+.1f} dB (SFLU)")
    assert wrong.max() > 5 and sflu_wrong.max() > 5
