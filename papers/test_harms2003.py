"""
J. Harms, Y. Chen, S. Chelkowski, A. Franzen, H. Vahlbruch, K. Danzmann and
R. Schnabel,
*Squeezed-input, optical-spring, signal-recycled gravitational-wave detectors*,
[Phys. Rev. D 68, 042001 (2003)](https://doi.org/10.1103/PhysRevD.68.042001),
[arXiv:gr-qc/0303066](https://arxiv.org/abs/gr-qc/0303066).

**What the paper shows.** Buonanno & Chen's detuned signal-recycled
interferometer (``test_buonanno_chen2001``) combined with KLMTV's
squeezed input and variational readout, for GEO 600. The input-output
relation is BC's (Eqs. 2-5), with BC's arm phase $\\beta$ replaced by the
GEO delay $\\Phi = \\Omega L/c$:

$$
\\mathbf o = \\frac{1}{M}\\big[\\mathbf T\\,\\mathbf i + \\mathbf s\\,h\\big],
\\qquad
S_h = \\frac{v\\,\\mathbf T\\,\\mathcal D(-\\lambda)\\mathcal S(2r)\\mathcal D(\\lambda)\\,
            \\mathbf T^\\dagger v^T}{|v\\cdot\\mathbf s|^2},
\\quad v = (\\cos\\zeta, \\sin\\zeta).
$$

Squeezing the input at the optimal, frequency-dependent angle (Eq. 16)
lowers the noise by exactly $e^{-2r}$ at every frequency, optical spring
notwithstanding (Eq. 17); optimising the homodyne angle as well gives a
further lower envelope (Eq. 30).

**The model.** ``sflu.topologies.signal_recycled()`` with a fully
transmissive ITM: GEO has no arm cavities, so the "arm" is a 1200 m delay
line to a free mirror of mass $m/10$ (which reproduces Table II's
$h_\\mathrm{SQL}$). There is no approximation between model and paper, and
they agree to machine precision. Mapping and conventions are in
``sflu.papers.harms2003``.

**Two factors of 2 in the paper's tables.** Table II's
$h_\\mathrm{SQL} = 20\\hbar/(m\\Omega^2L^2)$ is $h_\\mathrm{SQL}^2$. And Table II's
coupling $\\mathcal K = 20P\\omega_0/(mc^2\\Omega^2)$ with Table I's
$P = 10$ kW is twice what the figures use: they need
$\\mathcal K\\Omega^2 = 352\\ \\mathrm{s}^{-2}$, i.e. $P = 5$ kW in Table II
(Fig. 2 below shows both).

**Conventions.** Harms et al. read $o_1\\cos\\zeta + o_2\\sin\\zeta$
($\\zeta = \\pi/2$ is the phase quadrature, unlike BC); here that is the
angle $-\\zeta$. Their squeeze angle $\\lambda$ ($\\lambda = 0$ is phase
squeezing) is ``readout.squeezed``'s $\\pi/2 - \\lambda$. The SR detuning maps
as for BC.

**Reproduced:** Figs. 2-5. The thin curves of Figs. 3 and 4 are drawn at
evenly spaced angles, since the paper does not give its values. **Paper
inconsistency:** with Eq. 30 as printed, $\\zeta_-$ is the *maximum* of the
noise and $\\zeta_+$ the minimum, the reverse of the text.
"""
import numpy as np
import matplotlib.pyplot as plt

from sflu.papers import harms2003 as hm
from sflu.papers import buonanno_chen2001 as bc

P = hm.PARAMS
R = P["r"]
DB = hm.sqz_dB()
sqz = hm.readout.squeezed
MLIB = hm.MLIB
IN = "SRM.bk.i.exc"


def rel_err(a, b):
    return np.max(np.abs(a / b - 1))


def grid(f_lo, f_hi, n=200):
    f = np.geomspace(f_lo, f_hi, n)
    return f, 2 * np.pi * f


def axes(title, xlim, ylim):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Linear spectral noise density [$1/\\sqrt{\\mathrm{Hz}}$]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def overlay(ax, f, label, sflu, paper, tol=1e-9, in_legend=True, **kw):
    """Plot an SFLU curve with its closed form, and assert they agree."""
    ax.loglog(f, sflu, label=label if in_legend else None, **{"lw": 2, "alpha": 0.7, **kw})
    ax.loglog(f, paper, ls="--", c="k", lw=0.8)
    err = rel_err(sflu, paper)
    print(f"{label:45s} max |SFLU/paper - 1| = {err:.2e}")
    assert err < tol, label


def test_fig2(tpath_join):
    """Harms et al. Fig. 2: ideal GEO 600, coherent vacuum input.

    The conventional interferometer ($\\rho = 0$) with its shot-noise and
    radiation-pressure parts, and the detuned SR interferometer read at
    $\\zeta = \\pi/2$ (phase) and $\\zeta = 0$ (amplitude). Both beat the SQL
    near the optical-spring resonance at 30 Hz; the second minimum near
    200 Hz is the optical resonance of the SR cavity.
    """
    f, W = grid(1, 500)
    hs = hm.h_SQL(W)
    K = hm.theta() / W**2

    # First, the model's coupling is the one intended, for every frequency:
    # K Omega^2 is constant, the delays being common phases.
    assert rel_err(hm.model_theta(f), hm.theta()) < 1e-9

    fig, ax = axes("Harms et al. Fig. 2 (solid: SFLU, dashed: paper)", (1, 500), (3e-24, 3e-21))
    ax.loglog(f, hs, c="k", lw=1, label="SQL")
    Tc = hm.interferometer(f, R_SRM=0)
    overlay(ax, f, "conventional, $\\rho = 0$", hm.sqrt_Sh(Tc, hm.sflu_zeta(np.pi / 2)),
            np.sqrt(hs**2 / 2 * (K + 1 / K)))
    ax.loglog(f, hs / np.sqrt(2 * K), c="gray", ls=":", lw=1, label="shot noise")
    ax.loglog(f, hs * np.sqrt(K / 2), c="gray", ls="-.", lw=1, label="radiation pressure")
    T = hm.interferometer(f)
    Tp, _, s = hm.geo(W)
    curves = {}
    for name, zeta in [("\\pi/2", np.pi / 2), ("0", 0.0)]:
        curves[name] = hm.sqrt_Sh(T, hm.sflu_zeta(zeta))
        overlay(ax, f, f"SR, $\\zeta = {name}$ (Eq. 7)", curves[name],
                np.sqrt(hm.Sh(Tp, s, zeta)))

    # The same at the coupling of Table II as printed: the optical-spring dip
    # moves to 43 Hz and the shot noise drops to 4.3e-22 -- not the figure.
    Tt = hm.interferometer(f, which="table")
    Ttp, _, st = hm.geo(W, which="table")
    overlay(ax, f, "SR, $\\zeta = \\pi/2$, $\\mathcal{K}$ of Table II", hm.sqrt_Sh(Tt, hm.sflu_zeta(np.pi / 2)),
            np.sqrt(hm.Sh(Ttp, st, np.pi / 2)), c="C5", lw=1.5)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig2.pdf"))

    # Spot values read off Fig. 2.
    fs = np.array([30.4, 71.0, 232.0, 500.0, 192.0])
    Ts = hm.interferometer(fs)
    b2 = hm.sqrt_Sh(Ts, hm.sflu_zeta(np.pi / 2))
    b1 = hm.sqrt_Sh(Ts, hm.sflu_zeta(0))
    print("zeta = pi/2 at", fs[:4], "Hz:", b2[:4], "  zeta = 0 at 500, 192 Hz:", b1[3:])
    assert np.allclose(b2[:4], [2.2e-23, 1.1e-22, 5.6e-23, 1.3e-22], rtol=0.06)
    assert np.allclose(b1[3:], [3.0e-22, 6.1e-23], rtol=0.06)
    shot = hs[0] / np.sqrt(2 * K[0])
    assert abs(shot / 6.1e-22 - 1) < 0.01
    f_sql = np.sqrt(hm.theta()) / (2 * np.pi)  # K = 1
    print(f"conventional touches the SQL at {f_sql:.2f} Hz (text: 3 Hz)")
    assert abs(f_sql - 3) < 0.05
    dip_table = f[np.argmin(np.where(f > 10, hm.sqrt_Sh(Tt, hm.sflu_zeta(np.pi / 2)), np.inf))]
    print(f"with Table II's K the optical-spring dip is at {dip_table:.1f} Hz")
    assert 40 < dip_table < 46

    # Harms et al.'s T, M, s are BC's with beta -> Phi, and their zeta is
    # BC's pi/2 - zeta. Check it on the Advanced LIGO column of Table II,
    # where Phi = beta exactly and K, h_SQL are BC's.
    Wb = np.geomspace(0.1, 10, 50) * bc.PARAMS["gamma"]
    rho, phi = 0.9, np.pi / 2 - 0.47
    Ta, _, sa = hm.io_relation(bc.kappa(Wb, 1.0), bc.beta(Wb), bc.h_SQL(Wb), rho, phi)
    for zeta in (0.0, 0.4, np.pi / 2):
        assert rel_err(hm.Sh(Ta, sa, zeta), bc.Sh(Wb, rho, phi, np.pi / 2 - zeta, 1.0)) < 1e-12


def test_fig3(tpath_join):
    """Harms et al. Fig. 3: squeezed input ($r = 1$), $\\zeta = \\pi/2$.

    Thin curves: frequency-independent squeeze angles $\\lambda$ (Eq. 14;
    seven, evenly spaced, as the paper does not list them). Each one shifts
    both resonances. Their lower envelope is the frequency-dependent optimum,
    the coherent-vacuum curve (dashed in the paper, bold here) lowered by
    $e^{-r}$ (Eq. 17). The SFLU envelope uses the squeeze angle the *model*
    finds optimal, frequency by frequency; that it reproduces Eq. 17 checks
    Eq. 16 as well.
    """
    f, W = grid(10, 600)
    T = hm.interferometer(f)
    Tp, _, s = hm.geo(W)
    zeta = np.pi / 2
    z = hm.sflu_zeta(zeta)
    fig, ax = axes("Harms et al. Fig. 3 (solid: SFLU, dashed: paper)", (10, 600), (2e-24, 1.2e-21))
    ax.loglog(f, hm.h_SQL(W), c="k", lw=1, label="SQL")
    overlay(ax, f, "coherent vacuum (Eq. 7)", hm.sqrt_Sh(T, z), np.sqrt(hm.Sh(Tp, s, zeta)),
            c="C0", lw=2.5)
    for k in range(7):
        lam = k * np.pi / 7
        overlay(ax, f, f"fixed $\\lambda = {k}\\pi/7$ (Eq. 14)",
                hm.sqrt_Sh(T, z, {IN: sqz(MLIB, DB, hm.sflu_squeeze(lam))}),
                np.sqrt(hm.Sh(Tp, s, zeta, lam, R)), in_legend=k == 0, c="C7", lw=0.8)
    a_model = hm.model_squeeze_opt(T, z)
    overlay(ax, f, "optimal $\\lambda(\\Omega)$ (Eq. 17)", hm.sqrt_Sh(T, z, {IN: sqz(MLIB, DB, a_model)}),
            np.sqrt(hm.Sh_SI(Tp, s, zeta, R)), c="C3", lw=2.5)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig3.pdf"))

    # The model's optimal angle is Eq. 16's, mod pi.
    wrap = lambda a: (a + np.pi / 2) % np.pi - np.pi / 2
    assert np.max(np.abs(wrap(a_model - hm.sflu_squeeze(hm.lambda_opt(Tp, zeta))))) < 1e-9
    # Read off Fig. 3: the envelope's minimum, ~8e-24 at 30 Hz.
    env = np.exp(-R) * hm.sqrt_Sh(hm.interferometer(np.array([30.4])), z)
    assert abs(env[0] / 8e-24 - 1) < 0.05


def test_fig4(tpath_join):
    """Harms et al. Fig. 4: optimal squeezing, and then the optimal homodyne angle.

    Thin curves: Eq. 17 at fixed $\\zeta = 0, 20, \\ldots, 160$ degrees (the
    paper's values are not given). Dashed in the paper: one "arbitrary but
    fixed" angle, which matches $\\zeta = 65^\\circ$. Bold: the envelope, with
    $\\zeta$ optimal at every frequency (Eq. 30). The SFLU envelope uses the
    model's own optimal homodyne and squeeze angles.
    """
    f, W = grid(10, 600)
    T = hm.interferometer(f)
    Tp, _, s = hm.geo(W)
    fig, ax = axes("Harms et al. Fig. 4 (solid: SFLU, dashed: paper)", (10, 600), (2e-24, 1.2e-21))
    ax.loglog(f, hm.h_SQL(W), c="k", lw=1, label="SQL")
    for k, deg in enumerate(range(0, 180, 20)):
        zeta = np.radians(deg)
        z = hm.sflu_zeta(zeta)
        overlay(ax, f, f"fixed $\\zeta = {deg}^\\circ$ (Eq. 17)",
                hm.sqrt_Sh(T, z, {IN: sqz(MLIB, DB, hm.model_squeeze_opt(T, z))}),
                np.sqrt(hm.Sh_SI(Tp, s, zeta, R)), in_legend=k == 0, c="C7", lw=0.8)
    z65 = hm.sflu_zeta(np.radians(65))
    overlay(ax, f, "$\\zeta = 65^\\circ$ (Eq. 17)",
            hm.sqrt_Sh(T, z65, {IN: sqz(MLIB, DB, hm.model_squeeze_opt(T, z65))}),
            np.sqrt(hm.Sh_SI(Tp, s, np.radians(65), R)), c="C0", lw=2.5)
    zm = hm.model_zeta_opt(T)
    _, zeta_plus = hm.zeta_opt(Tp, s)
    env = hm.sqrt_Sh(T, zm, {IN: sqz(MLIB, DB, hm.model_squeeze_opt(T, zm))})
    overlay(ax, f, "optimal $\\zeta(\\Omega)$ (Eqs. 17, 30)", env,
            np.sqrt(hm.Sh_SI(Tp, s, zeta_plus, R)), c="C3", lw=2.5)
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(tpath_join("fig4.pdf"))

    # Spot values of the envelope, read off Fig. 4 (and computed from Eq. 30).
    fs = np.array([10, 20, 26, 30, 40, 100, 220, 600.0])
    Ts = hm.interferometer(fs)
    zs = hm.model_zeta_opt(Ts)
    env_s = hm.sqrt_Sh(Ts, zs, {IN: sqz(MLIB, DB, hm.model_squeeze_opt(Ts, zs))})
    print("envelope at", fs, "Hz:", env_s)
    assert np.allclose(env_s, [1.54e-22, 1.85e-23, 2.43e-24, 7.1e-24, 1.5e-23,
                               2.22e-23, 1.83e-23, 6.0e-23], rtol=0.05)
    # and of the zeta = 65 deg curve: dip 2.6e-24 at 25.9 Hz, 1.03e-22 at 44 Hz
    d = hm.sqrt_Sh(hm.interferometer(np.array([25.9, 44.0])), z65) * np.exp(-R)
    assert np.allclose(d, [2.6e-24, 1.03e-22], rtol=0.1)


def test_fig5(tpath_join):
    """Harms et al. Fig. 5: the optimal homodyne and squeeze angles.

    $\\zeta_\\mathrm{opt}$ from Eq. 30 against the model's own optimum, and
    $\\lambda_\\mathrm{opt}$ from Eq. 16 at that $\\zeta$.

    Two remarks on the paper. Eq. 30 calls $\\zeta_-$ the minimum; as printed
    (principal square root; the branch of arccot does not matter mod $\\pi$) it
    is $\\zeta_+$, and $\\zeta_-$ is the noise *maximum*. And the figure plots
    $-\\lambda$, not $\\lambda$: Eq. 16 gives $114.6^\\circ$ at 10 Hz where the
    figure shows $65^\\circ$. Its jump by $180^\\circ$ near 30 Hz is a branch
    change of the arctangent, not a physical rotation -- a squeeze angle is
    defined mod $180^\\circ$.
    """
    f, W = grid(10, 600)
    T = hm.interferometer(f)
    Tp, _, s = hm.geo(W)
    zeta_minus, zeta_plus = hm.zeta_opt(Tp, s)
    z_model = -hm.model_zeta_opt(T)  # back to the paper's convention
    wrap = lambda a: (a + np.pi / 2) % np.pi - np.pi / 2
    err = np.max(np.abs(wrap(z_model - zeta_plus)))
    print(f"max |zeta_opt(SFLU) - zeta_+ (Eq. 30)| = {err:.2e} rad")
    assert err < 1e-8
    assert np.all(hm.Sh(Tp, s, zeta_minus) > hm.Sh(Tp, s, zeta_plus))

    deg_z = np.degrees(zeta_plus) % 180
    # the figure's branch of -lambda: arctan2 of Eq. 16's ratio (with T's
    # common phase removed), then -lambda mod 360
    c, sn = np.cos(zeta_plus), np.sin(zeta_plus)
    ph = np.exp(-2j * W * P["L_m"] / hm.scc.c)
    num = ((Tp[:, 0, 0] * c + Tp[:, 1, 0] * sn) * ph).real
    den = ((Tp[:, 0, 1] * c + Tp[:, 1, 1] * sn) * ph).real
    deg_l = np.degrees(-np.arctan2(-num, den)) % 360
    lam_model = np.pi / 2 - hm.model_squeeze_opt(T, hm.sflu_zeta(zeta_plus))
    assert np.max(np.abs(wrap(lam_model - hm.lambda_opt(Tp, zeta_plus)))) < 1e-9

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogx(f, np.degrees(z_model) % 180, lw=2, alpha=0.7, label="$\\zeta_\\mathrm{opt}$, SFLU")
    ax.semilogx(f, deg_z, ls="--", c="k", lw=0.8)
    ax.semilogx(f, np.degrees(-lam_model) % 180 + 180 * (deg_l >= 180), lw=2, alpha=0.7,
                label="$-\\lambda_\\mathrm{opt}$, SFLU (figure's branch)")
    ax.semilogx(f, deg_l, ls="--", c="k", lw=0.8)
    ax.set_xlim(10, 600)
    ax.set_ylim(0, 250)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("angle [deg]")
    ax.set_title("Harms et al. Fig. 5 (solid: SFLU, dashed: paper)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(tpath_join("fig5.pdf"))

    # Read off Fig. 5: zeta_opt 22.5 deg at 10 Hz, a maximum of 146.5 deg near
    # 100 Hz, 94 deg at 600 Hz; -lambda 65 deg at 10 Hz, 184 deg at 600 Hz.
    i100 = np.argmax(deg_z)
    print(f"zeta_opt: {deg_z[0]:.1f} deg at 10 Hz, max {deg_z[i100]:.1f} at {f[i100]:.0f} Hz, "
          f"{deg_z[-1]:.1f} at 600 Hz; -lambda: {deg_l[0]:.1f}, {deg_l[-1]:.1f}")
    assert abs(deg_z[0] - 22.5) < 0.5 and abs(deg_z[i100] - 146.5) < 0.5 and abs(deg_z[-1] - 94) < 1
    assert 80 < f[i100] < 120
    assert abs(deg_l[0] - 65) < 1 and abs(deg_l[-1] - 184) < 1
